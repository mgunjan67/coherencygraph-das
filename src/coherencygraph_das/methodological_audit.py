from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
import time
import importlib.metadata
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
import h5py
from scipy import linalg
from scipy.interpolate import CubicSpline
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from torch import nn

from .config import sha256, write_json
from .data import reliable_mask
from .hybrid_search import _dataset_and_split, _load_amendment
from .models import LinearSpectralOperator, ModernSpectralOperator, spectral_basis
from .review_revision import (
    _batches,
    _climatology_prediction,
    _complex,
    _context_predictions,
    _event_bootstrap,
    _flatten_rows,
    _gauge_lengths,
    _generalized_weight,
    _held_lag_mask,
    _mask_lag_features,
    _predict,
    _ratio,
    _restore_rows,
    _sample_mse,
    _sample_nrmse,
    _scaler,
    _training_augmentation,
)
from .spectral import effective_rank, normalize_coherency, spectral_mixture_operator


AMENDMENT_FILE = "protocol_amendment_06_coordinate_and_classical_audit.yaml"


def _load_audit_amendment(cfg: dict) -> tuple[dict, Path]:
    path = Path(cfg["_root"]) / "configs" / AMENDMENT_FILE
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def _audit_paths(cfg: dict, amendment: dict) -> tuple[Path, Path, Path]:
    root = Path(cfg["_root"])
    reports = root / amendment["outputs"]["directory"]
    models = root / amendment["outputs"]["model_directory"]
    figures = root / amendment["outputs"]["figure_directory"]
    for path in (reports, models, figures):
        path.mkdir(parents=True, exist_ok=True)
    return reports, models, figures


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_model(
    family: str,
    features: int,
    dataset,
    amendment: dict,
    q_bins: int,
) -> nn.Module:
    train = amendment["training"]
    decoder = amendment["decoder_audit"]
    common = dict(
        features=features,
        bands=dataset.targets.shape[2],
        q_bins=int(q_bins),
        lags=list(map(int, dataset.lags)),
        coordinate_mode="channel_lag",
        channel_spacing_m=float(decoder["channel_spacing_m"]),
        maximum_lag=int(decoder["maximum_supervised_lag_channels"]),
    )
    if family == "linear_psd":
        return LinearSpectralOperator(**common)
    return ModernSpectralOperator(
        **common,
        hidden_dim=int(train["hidden_dim"]),
        layers=int(train["layers"]),
        dropout=float(train["dropout"]),
        family=family,
        heads=int(train["attention_heads"]),
    )


def _development_partition(dataset, development: np.ndarray, seed: int, root: Path) -> tuple[np.ndarray, np.ndarray]:
    split = pd.read_csv(root / "reports" / "hybrid_search" / "architecture_test_split.csv", dtype={"event_id": str})
    groups = dict(zip(split.event_id.astype(str), split.source_group_id.astype(str), strict=True))
    event_group = {
        str(event): groups.get(str(event), f"event-{event}")
        for event in np.unique(dataset.event_ids[development])
    }
    validation_groups = {
        group
        for group in set(event_group.values())
        if int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 5 == 0
    }
    validation = development[
        np.asarray([event_group[str(dataset.event_ids[index])] in validation_groups for index in development])
    ]
    training = development[~np.isin(development, validation)]
    if len(np.unique(dataset.event_ids[validation])) < 10:
        ordered = sorted(set(event_group.values()), key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
        validation_groups = set(ordered[: max(10, len(ordered) // 5)])
        validation = development[
            np.asarray([event_group[str(dataset.event_ids[index])] in validation_groups for index in development])
        ]
        training = development[~np.isin(development, validation)]
    return training, validation


def _train_epochs(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    event_ids: np.ndarray,
    loss_mask: torch.Tensor,
    amendment: dict,
    seed: int,
    epochs: int,
    augment: bool,
) -> list[float]:
    train = amendment["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        epoch_losses = []
        for batch_indices in _batches(indices, event_ids, int(train["batch_events"]), rng):
            optimizer.zero_grad(set_to_none=True)
            batch_x = x[batch_indices]
            if augment:
                batch_x = _training_augmentation(
                    batch_x,
                    rng,
                    float(train["block_dropout_probability"]),
                    int(train["maximum_missing_blocks"]),
                )
            prediction, _ = model(batch_x)
            expanded = loss_mask[None, None, :, :, None].expand_as(prediction)
            loss = ((prediction - y[batch_indices]) ** 2)[expanded].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def _fit_locked_model(
    family: str,
    values: np.ndarray,
    dataset,
    development: np.ndarray,
    loss_mask: np.ndarray,
    amendment: dict,
    seed: int,
    q_bins: int,
    root: Path,
    augment: bool = False,
) -> tuple[nn.Module, dict]:
    """Select an epoch on an internal development fold, then refit on all development data."""

    _seed(seed)
    train_cfg = amendment["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(values.astype(np.float32)).to(device)
    y = torch.from_numpy(dataset.targets.astype(np.float32)).to(device)
    mask = torch.from_numpy(loss_mask).to(device)
    inner_train, inner_validation = _development_partition(dataset, development, seed, root)
    model = _new_model(family, values.shape[-1], dataset, amendment, q_bins).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
    )
    rng = np.random.default_rng(seed)
    best_state = None
    best_epoch = 1
    best_score = np.inf
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(int(train_cfg["train_epochs"])):
        model.train()
        batch_losses = []
        for batch_indices in _batches(inner_train, dataset.event_ids, int(train_cfg["batch_events"]), rng):
            optimizer.zero_grad(set_to_none=True)
            batch_x = x[batch_indices]
            if augment:
                batch_x = _training_augmentation(
                    batch_x,
                    rng,
                    float(train_cfg["block_dropout_probability"]),
                    int(train_cfg["maximum_missing_blocks"]),
                )
            prediction, _ = model(batch_x)
            expanded = mask[None, None, :, :, None].expand_as(prediction)
            loss = ((prediction - y[batch_indices]) ** 2)[expanded].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation_prediction, _ = _predict(model, x, inner_validation)
        score = float(_sample_nrmse(validation_prediction, dataset.targets[inner_validation], loss_mask).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(batch_losses)), "validation_nrmse": score})
        if score < best_score - 1e-5:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= int(train_cfg["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("internal development fold failed to select an epoch")

    _seed(seed)
    final = _new_model(family, values.shape[-1], dataset, amendment, q_bins).to(device)
    refit_losses = _train_epochs(
        final, x, y, development, dataset.event_ids, mask, amendment, seed, best_epoch, augment
    )
    return final, {
        "family": family,
        "seed": int(seed),
        "q_bins": int(q_bins),
        "coordinate_mode": "channel_lag",
        "internal_validation_nrmse": float(best_score),
        "locked_epochs": int(best_epoch),
        "parameters": int(sum(parameter.numel() for parameter in final.parameters())),
        "training_seconds": float(time.perf_counter() - started),
        "history": history,
        "refit_losses": refit_losses,
    }


def _rich_sample_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    pred = _complex(prediction)
    truth = _complex(target)
    keep = np.broadcast_to(mask[None, None, :, :], pred.shape)
    rows = {name: [] for name in ["nrmse", "mse", "real_mae", "imag_mae", "sign_accuracy", "magnitude_mae", "phase_mae"]}
    for p, t, selected in zip(pred, truth, keep, strict=True):
        pv, tv = p[selected], t[selected]
        error = pv - tv
        mse = float(np.mean(np.abs(error) ** 2))
        rows["mse"].append(mse)
        rows["nrmse"].append(float(np.sqrt(mse) / max(np.sqrt(np.mean(np.abs(tv) ** 2)), np.finfo(float).eps)))
        rows["real_mae"].append(float(np.mean(np.abs(pv.real - tv.real))))
        rows["imag_mae"].append(float(np.mean(np.abs(pv.imag - tv.imag))))
        rows["sign_accuracy"].append(float(np.mean(np.sign(pv.real) == np.sign(tv.real))))
        rows["magnitude_mae"].append(float(np.mean(np.abs(np.abs(pv) - np.abs(tv)))))
        reliable_phase = np.abs(tv) >= 0.05
        phase = np.angle(pv[reliable_phase] * np.conj(tv[reliable_phase])) if reliable_phase.any() else np.array([np.nan])
        rows["phase_mae"].append(float(np.nanmean(np.abs(phase))))
    return {key: np.asarray(value) for key, value in rows.items()}


def _metric_rows(name: str, role: str, prediction: np.ndarray, dataset, indices: np.ndarray, mask: np.ndarray) -> list[dict]:
    metrics = _rich_sample_metrics(prediction, dataset.targets[indices], mask)
    return [
        {
            "model": name,
            "role": role,
            "event_id": str(dataset.event_ids[sample]),
            "route": str(dataset.routes[sample]),
            **{metric: float(values[row]) for metric, values in metrics.items()},
        }
        for row, sample in enumerate(indices)
    ]


def _fft_q(q_bins: int) -> np.ndarray:
    return np.sort(np.fft.fftfreq(q_bins) * 2.0 * np.pi)


def _operator_from_probabilities(probabilities: np.ndarray, positions: np.ndarray, q_bins: int) -> np.ndarray:
    return spectral_mixture_operator(probabilities, positions, _fft_q(q_bins))


def run_decoder_coordinate_audit(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    reports, _, _ = _audit_paths(cfg, amendment)
    decoder = amendment["decoder_audit"]
    lags = list(map(int, cfg["spectral"]["channel_lags"]))
    maximum = int(decoder["maximum_supervised_lag_channels"])
    spacing = float(decoder["channel_spacing_m"])
    rows = []
    rng = np.random.default_rng(20260906)
    for bins in decoder["diagnostic_q_bins"]:
        bins = int(bins)
        corrected_cos, corrected_sin, _, _ = spectral_basis(bins, lags, mode="channel_lag")
        legacy_cos, legacy_sin, _, _ = spectral_basis(bins, lags, mode="legacy_channel_lag")
        basis = torch.complex(corrected_cos, corrected_sin).numpy()
        weights = rng.dirichlet(np.ones(bins))
        positions = np.arange(maximum + 1)
        matrix = _operator_from_probabilities(weights, positions, bins)
        alias = minimum = min(maximum % bins, (-maximum) % bins)
        rows.append(
            {
                "q_bins": bins,
                "period_channels": bins,
                "period_metres": bins * spacing,
                "largest_supervised_lag": maximum,
                "shortest_circular_equivalent_lag": int(minimum),
                "harmful_alias_over_supervised_range": bool(bins <= 2 * maximum),
                "legacy_vs_fft_basis_max_difference": float(
                    np.max(np.abs(torch.complex(legacy_cos, legacy_sin).numpy() - basis))
                ),
                "random_operator_min_eigenvalue": float(np.linalg.eigvalsh(matrix).min()),
                "random_operator_max_diagonal_error": float(np.max(np.abs(np.diag(matrix) - 1.0))),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(reports / "decoder_grid_alias_audit.csv", index=False)

    reference_cos, reference_sin, _, _ = spectral_basis(257, lags, mode="channel_lag")
    reference = torch.complex(reference_cos, reference_sin).numpy()
    coordinate_rows = []
    for mode in decoder["coordinate_parameterizations"]:
        cosine, sine, separation, q = spectral_basis(
            257,
            lags,
            mode=str(mode),
            channel_spacing_m=spacing,
            maximum_lag=maximum,
        )
        current = torch.complex(cosine, sine).numpy()
        coordinate_rows.append(
            {
                "coordinate_mode": str(mode),
                "d_model_min": float(separation.min()),
                "d_model_max": float(separation.max()),
                "q_min": float(q.min()),
                "q_max": float(q.max()),
                "phase_basis_max_difference_from_channel_lag": float(np.max(np.abs(current - reference))),
            }
        )
    pd.DataFrame(coordinate_rows).to_csv(reports / "decoder_coordinate_equivalence.csv", index=False)
    receipt = {
        "amendment_sha256": sha256(amendment_path),
        "legacy_problem": "For M=65, the original odd grid satisfies gamma(d+65)=-gamma(d); lag 89 is therefore constrained to minus lag 24.",
        "corrected_definition": "d_model is integer channel lag and q=2*pi*k/M rad/channel on FFT-centred integer harmonics.",
        "physical_equivalence": "d_m=d_channel*Delta_x and q_m=q_channel/Delta_x (rad/m) produce the identical phase matrix.",
        "final_q_bins": int(decoder["final_q_bins"]),
        "choice_rule": decoder["final_choice_rule"],
        "alias_free_through_supervised_lag": bool(int(decoder["final_q_bins"]) > 2 * maximum),
    }
    path = reports / "decoder_coordinate_audit.json"
    write_json(path, receipt)
    return path


def _toeplitz_from_sparse_lags(values: np.ndarray, lags: np.ndarray, size: int) -> np.ndarray:
    known_lags = np.concatenate([[0], lags]).astype(float)
    known_values = np.concatenate([[1.0 + 0.0j], values])
    all_lags = np.arange(size, dtype=float)
    gamma = np.interp(all_lags, known_lags, known_values.real) + 1j * np.interp(
        all_lags, known_lags, known_values.imag
    )
    gamma[0] = 1.0
    return linalg.toeplitz(gamma, np.conj(gamma))


def _nearest_psd_toeplitz(values: np.ndarray, lags: np.ndarray, iterations: int, rank: int | None = None) -> np.ndarray:
    # A 257-point Hermitian circulant embedding is alias-free through lag 89.
    # Fourier eigenvalue clipping is the exact PSD projection for that embedding
    # and avoids thousands of dense 90x90 eigendecompositions.
    period = max(257, 2 * int(max(lags)) + 1)
    known = np.concatenate([[0], np.asarray(lags)]).astype(float)
    observed = np.concatenate([[1.0 + 0.0j], values])
    positive = np.arange((period + 1) // 2)
    gamma_positive = np.interp(positive, known, observed.real) + 1j * np.interp(
        positive, known, observed.imag
    )
    gamma = np.zeros(period, dtype=np.complex128)
    gamma[: len(positive)] = gamma_positive
    gamma[-(len(positive) - 1) :] = np.conj(gamma_positive[1:][::-1])
    spectrum = np.real(np.fft.fft(gamma))
    for _ in range(max(1, int(iterations))):
        spectrum = np.maximum(spectrum, 0.0)
        if rank is not None and int(rank) < len(spectrum):
            keep = np.argsort(spectrum)[-int(rank) :]
            restricted = np.zeros_like(spectrum)
            restricted[keep] = spectrum[keep]
            spectrum = restricted
        spectrum *= period / max(spectrum.sum(), np.finfo(float).eps)
        gamma = np.fft.ifft(spectrum)
        gamma[0] = 1.0
        spectrum = np.real(np.fft.fft(gamma))
    fitted = gamma[np.asarray(lags, dtype=int)]
    return fitted


def _project_prediction(prediction: np.ndarray, lags: np.ndarray, method: str, amendment: dict, parameter: float | int | None = None) -> np.ndarray:
    complex_prediction = _complex(prediction)
    output = np.empty_like(complex_prediction)
    q_bins = int(amendment["classical_baselines"]["spectral_nnls_bins"])
    q = _fft_q(q_bins)
    phase = np.exp(1j * np.asarray(lags)[:, None] * q[None, :])
    design = np.vstack([phase.real, phase.imag, 10.0 * np.ones((1, q_bins))])
    for index in np.ndindex(complex_prediction.shape[:-1]):
        values = complex_prediction[index]
        if method == "nearest_psd":
            fitted = _nearest_psd_toeplitz(
                values, lags, int(amendment["classical_baselines"]["nearest_psd_iterations"])
            )
        elif method == "low_rank":
            fitted = _nearest_psd_toeplitz(
                values,
                lags,
                int(amendment["classical_baselines"]["nearest_psd_iterations"]),
                rank=int(parameter),
            )
        elif method == "spectral_nnls":
            response = np.concatenate([values.real, values.imag, [10.0]])
            weights, _ = nnls(design, response, maxiter=20 * q_bins)
            weights = weights / max(weights.sum(), np.finfo(float).eps)
            fitted = phase @ weights
        else:
            raise ValueError(method)
        output[index] = fitted
    return np.stack([output.real, output.imag], axis=-1).astype(np.float32)


def _save_model_predictions(
    family: str,
    values: np.ndarray,
    dataset,
    development: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    cfg: dict,
    amendment: dict,
    models: Path,
    q_bins: int,
    prefix: str,
    augment: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    prediction_members = {"calibration": [], "architecture_test": []}
    probability_members = {"calibration": [], "architecture_test": []}
    training_rows = []
    for seed in amendment["training"]["final_seeds"]:
        checkpoint = models / f"{prefix}_{family}_q{q_bins}_seed{seed}.pt"
        history_path = models / f"{prefix}_{family}_q{q_bins}_seed{seed}_history.json"
        if checkpoint.exists() and history_path.exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = _new_model(family, values.shape[-1], dataset, amendment, q_bins).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            record = {
                "family": family,
                "seed": int(seed),
                "q_bins": int(q_bins),
                "coordinate_mode": "channel_lag",
                "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "training_seconds": 0.0,
                "cached_checkpoint": True,
            }
        else:
            model, record = _fit_locked_model(
                family, values, dataset, development, mask, amendment, int(seed), q_bins, Path(cfg["_root"]), augment
            )
            torch.save(model.state_dict(), checkpoint)
            history = record.pop("history")
            refit = record.pop("refit_losses")
            write_json(history_path, {"selection": history, "refit_losses": refit})
            record["cached_checkpoint"] = False
        record["checkpoint_sha256"] = sha256(checkpoint)
        training_rows.append(record)
        x = torch.from_numpy(values.astype(np.float32)).to(next(model.parameters()).device)
        for role, indices in [("calibration", calibration), ("architecture_test", test)]:
            prediction, probabilities = _predict(model, x, indices)
            prediction_members[role].append(prediction)
            probability_members[role].append(probabilities)
    predictions = {role: np.mean(values_, axis=0) for role, values_ in prediction_members.items()}
    probabilities = {role: np.mean(values_, axis=0) for role, values_ in probability_members.items()}
    for role in predictions:
        np.save(models / f"{prefix}_{family}_{role}_predictions.npy", predictions[role])
        np.save(models / f"{prefix}_{family}_{role}_probabilities.npy", probabilities[role])
    return predictions, probabilities, training_rows


def run_corrected_baselines(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models, _ = _audit_paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    normalized, mean, std = _scaler(dataset, roles)
    np.savez(models / "corrected_feature_scaler.npz", mean=mean, std=std)
    mask = reliable_mask(cfg)
    development = np.flatnonzero(roles == "model_development")
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    q_bins = int(amendment["decoder_audit"]["final_q_bins"])
    rows: list[dict] = []
    training_rows = []
    prediction_store: dict[tuple[str, str], np.ndarray] = {}
    probability_store: dict[tuple[str, str], np.ndarray] = {}

    for family in ["linear_psd", *amendment["training"]["learned_families"]]:
        prediction_paths = {
            role: models / f"corrected_{family}_{role}_predictions.npy"
            for role in ["calibration", "architecture_test"]
        }
        probability_paths = {
            role: models / f"corrected_{family}_{role}_probabilities.npy"
            for role in ["calibration", "architecture_test"]
        }
        if all(path.exists() for path in [*prediction_paths.values(), *probability_paths.values()]):
            predictions = {role: np.load(path) for role, path in prediction_paths.items()}
            probabilities = {role: np.load(path) for role, path in probability_paths.items()}
            training = []
        else:
            predictions, probabilities, training = _save_model_predictions(
                str(family), normalized, dataset, development, calibration, test, mask, cfg, amendment, models, q_bins, "corrected"
            )
        label = {
            "linear_psd": "Linear PSD decoder",
            "conv1d_psd": "CNN PSD",
            "bigru_psd": "BiGRU PSD",
            "bissm_psd": "State-space PSD",
            "gat_bissm_psd": "Local-state PSD",
        }[str(family)]
        training_rows.extend(training)
        for role, indices in [("calibration", calibration), ("architecture_test", test)]:
            prediction_store[(role, label)] = predictions[role]
            probability_store[(role, label)] = probabilities[role]
            rows.extend(_metric_rows(label, role, predictions[role], dataset, indices, mask))
        print(f"completed corrected baseline: {family}", flush=True)

    gauges = _gauge_lengths(dataset)
    for role, indices in [("calibration", calibration), ("architecture_test", test)]:
        prediction_store[(role, "Climatology")] = _climatology_prediction(dataset, roles, indices, gauges)
        prediction_store[(role, "Persistence")] = _context_predictions(dataset, indices)

    x_train, y_train = _flatten_rows(normalized[development], dataset.targets[development])
    ridge = Ridge(alpha=float(amendment["classical_baselines"]["ridge_alpha"])).fit(x_train, y_train)
    for role, indices in [("calibration", calibration), ("architecture_test", test)]:
        x_role, _ = _flatten_rows(normalized[indices], dataset.targets[indices])
        ridge_prediction = _restore_rows(
            ridge.predict(x_role), len(indices), dataset.targets.shape[1], dataset.targets.shape[2:]
        ).astype(np.float32)
        prediction_store[(role, "Ordinary ridge")] = ridge_prediction
        prediction_store[(role, "Ridge + nearest PSD/Toeplitz")] = _project_prediction(
            ridge_prediction, dataset.lags, "nearest_psd", amendment
        )
        prediction_store[(role, "Convex nonnegative spectral fit")] = _project_prediction(
            ridge_prediction, dataset.lags, "spectral_nnls", amendment
        )

    # Development-only selection for persistence-derived shrinkage and rank.
    context_development = _context_predictions(dataset, development)
    shrinkage_scores = []
    for value in amendment["classical_baselines"]["shrinkage_candidates"]:
        candidate = context_development.copy()
        candidate *= 1.0 - float(value)
        shrinkage_scores.append({"shrinkage": float(value), "development_nrmse": float(_sample_nrmse(candidate, dataset.targets[development], mask).mean())})
    selected_shrinkage = min(shrinkage_scores, key=lambda row: row["development_nrmse"])["shrinkage"]
    rank_scores = []
    # Rank audit uses a deterministic subset for tractable 90x90 projections.
    audit_development = development[:: max(1, len(development) // 48)]
    context_subset = _context_predictions(dataset, audit_development)
    for rank in amendment["classical_baselines"]["low_rank_candidates"]:
        candidate = _project_prediction(context_subset, dataset.lags, "low_rank", amendment, int(rank))
        rank_scores.append({"rank": int(rank), "development_nrmse": float(_sample_nrmse(candidate, dataset.targets[audit_development], mask).mean())})
    selected_rank = min(rank_scores, key=lambda row: row["development_nrmse"])["rank"]
    pd.DataFrame(shrinkage_scores).to_csv(reports / "classical_shrinkage_development_selection.csv", index=False)
    pd.DataFrame(rank_scores).to_csv(reports / "classical_rank_development_selection.csv", index=False)

    for role, indices in [("calibration", calibration), ("architecture_test", test)]:
        context = _context_predictions(dataset, indices)
        prediction_store[(role, "Shrinkage persistence")] = context * (1.0 - float(selected_shrinkage))
        prediction_store[(role, "Low-rank persistence")] = _project_prediction(
            context, dataset.lags, "low_rank", amendment, int(selected_rank)
        )

    existing = {(row["role"], row["model"]) for row in rows}
    for (role, name), prediction in prediction_store.items():
        if (role, name) in existing:
            continue
        indices = calibration if role == "calibration" else test
        rows.extend(_metric_rows(name, role, prediction, dataset, indices, mask))
        np.save(models / f"baseline_{name.lower().replace(' ', '_').replace('/', '_').replace('+', 'plus')}_{role}.npy", prediction)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(reports / "corrected_baseline_event_route_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(reports / "corrected_baseline_training.csv", index=False)
    summary = metrics.groupby(["role", "model"], as_index=False).agg(
        mean_nrmse=("nrmse", "mean"), pooled_mse=("mse", "mean"), real_mae=("real_mae", "mean"),
        imaginary_mae=("imag_mae", "mean"), sign_accuracy=("sign_accuracy", "mean")
    )
    test_ridge = float(summary[(summary.role == "architecture_test") & (summary.model == "Ordinary ridge")].pooled_mse.iloc[0])
    summary["mse_skill_vs_ridge"] = np.where(summary.role == "architecture_test", 1.0 - summary.pooled_mse / test_ridge, np.nan)
    summary.to_csv(reports / "corrected_baseline_summary.csv", index=False)

    test_metrics = metrics[metrics.role == "architecture_test"]
    local = test_metrics[test_metrics.model == "Local-state PSD"]
    paired_rows = []
    for name, group in test_metrics.groupby("model"):
        if name == "Local-state PSD":
            continue
        paired = local.merge(group, on=["event_id", "route"], suffixes=("_local", "_other"), validate="one_to_one")
        differences = paired.assign(delta=paired.nrmse_local - paired.nrmse_other).groupby("event_id").delta.mean().to_numpy()
        paired_rows.append({"local_state_minus": name, **_event_bootstrap(differences, int(amendment["bootstrap"]["seed"]), int(amendment["bootstrap"]["replicates"]))})
    pd.DataFrame(paired_rows).to_csv(reports / "corrected_baseline_paired_bootstrap.csv", index=False)
    ridge_rows = test_metrics[test_metrics.model == "Ordinary ridge"]
    learned_vs_ridge = []
    for name in ["Linear PSD decoder", "CNN PSD", "BiGRU PSD", "State-space PSD", "Local-state PSD"]:
        group = test_metrics[test_metrics.model == name]
        paired = group.merge(
            ridge_rows,
            on=["event_id", "route"],
            suffixes=("_model", "_ridge"),
            validate="one_to_one",
        )
        differences = paired.assign(delta=paired.nrmse_model - paired.nrmse_ridge).groupby("event_id").delta.mean().to_numpy()
        learned_vs_ridge.append(
            {"model": name, "baseline": "Ordinary ridge", **_event_bootstrap(
                differences,
                int(amendment["bootstrap"]["seed"]) + len(learned_vs_ridge),
                int(amendment["bootstrap"]["replicates"]),
            )}
        )
    pd.DataFrame(learned_vs_ridge).to_csv(reports / "corrected_learned_vs_ridge_bootstrap.csv", index=False)
    receipt = {
        "amendment_sha256": sha256(amendment_path),
        "q_bins": q_bins,
        "coordinate_mode": "channel_lag FFT-centred",
        "selected_shrinkage_development_only": selected_shrinkage,
        "selected_rank_development_only": selected_rank,
        "architecture_test_summary": summary[summary.role == "architecture_test"].to_dict("records"),
    }
    path = reports / "corrected_baseline_receipt.json"
    write_json(path, receipt)
    return path


def _load_npz_target(dataset, indices: np.ndarray, key: str) -> np.ndarray:
    rows = []
    for sample in indices:
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            values = loaded[key]
        rows.append(np.stack([values.real, values.imag], axis=-1).astype(np.float32))
    return np.stack(rows)


def run_information_ceiling(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models, _ = _audit_paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    test = np.flatnonzero(roles == "architecture_test")
    mask = reliable_mask(cfg)
    first = _load_npz_target(dataset, test, "target_gamma_first")
    second = _load_npz_target(dataset, test, "target_gamma_second")
    taper0 = _load_npz_target(dataset, test, "target_gamma_taper0")
    taper1 = _load_npz_target(dataset, test, "target_gamma_taper1")
    local = np.load(models / "corrected_gat_bissm_psd_architecture_test_predictions.npy")
    ridge = np.load(models / "baseline_ordinary_ridge_architecture_test.npy")
    comparisons = {
        "split-half reproducibility": (first, second),
        "independent-taper reproducibility": (taper0, taper1),
        "local-state model vs full target": (local, dataset.targets[test]),
        "ordinary ridge vs full target": (ridge, dataset.targets[test]),
    }
    rows = []
    metrics_by_name = {}
    for name, (left, right) in comparisons.items():
        metrics = _rich_sample_metrics(left, right, mask)
        metrics_by_name[name] = metrics
        for row, sample in enumerate(test):
            rows.append(
                {
                    "comparison": name,
                    "event_id": str(dataset.event_ids[sample]),
                    "route": str(dataset.routes[sample]),
                    **{metric: float(values[row]) for metric, values in metrics.items()},
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(reports / "information_ceiling_event_route_metrics.csv", index=False)
    split_mse = metrics_by_name["split-half reproducibility"]["mse"]
    taper_mse = metrics_by_name["independent-taper reproducibility"]["mse"]
    ceiling_mse = 0.5 * (split_mse + taper_mse)
    skill_rows = []
    for name in ["local-state model vs full target", "ordinary ridge vs full target"]:
        model_mse = metrics_by_name[name]["mse"]
        target_power = np.asarray([
            float(np.mean(np.abs(_complex(dataset.targets[test][row])) ** 2)) for row in range(len(test))
        ])
        latent_variance = np.maximum(target_power - 0.5 * ceiling_mse, np.finfo(float).eps)
        for row, sample in enumerate(test):
            skill_rows.append(
                {
                    "model": name.replace(" vs full target", ""),
                    "event_id": str(dataset.event_ids[sample]),
                    "route": str(dataset.routes[sample]),
                    "empirical_reproducibility_mse": float(ceiling_mse[row]),
                    "model_mse": float(model_mse[row]),
                    "ceiling_normalized_skill": float(1.0 - model_mse[row] / max(ceiling_mse[row], np.finfo(float).eps)),
                    "heuristic_explainable_fraction": float(1.0 - max(model_mse[row] - 0.5 * ceiling_mse[row], 0.0) / latent_variance[row]),
                }
            )
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(reports / "information_ceiling_normalized_skill.csv", index=False)
    summary = {
        "amendment_sha256": sha256(amendment_path),
        "comparison_means": table.groupby("comparison").mean(numeric_only=True).reset_index().to_dict("records"),
        "normalized_skill_means": skill.groupby("model").mean(numeric_only=True).reset_index().to_dict("records"),
        "interpretation": "Split-half and independent-taper disagreement quantify finite-window reproducibility, not error to a noise-free latent truth.",
    }
    path = reports / "information_ceiling_summary.json"
    write_json(path, summary)
    return path


def _interpolate_complex(known_lags: np.ndarray, values: np.ndarray, requested_lags: np.ndarray, method: str) -> np.ndarray:
    if method == "linear_complex":
        return np.interp(requested_lags, known_lags, values.real) + 1j * np.interp(requested_lags, known_lags, values.imag)
    if method == "cubic_spline":
        return CubicSpline(known_lags, values.real, extrapolate=True)(requested_lags) + 1j * CubicSpline(
            known_lags, values.imag, extrapolate=True
        )(requested_lags)
    if method == "exponential_phase":
        magnitude = np.maximum(np.abs(values), 1e-4)
        amplitude_fit = np.polyfit(known_lags, np.log(magnitude), 1)
        phase_fit = np.polyfit(known_lags, np.unwrap(np.angle(values)), 1)
        return np.exp(np.polyval(amplitude_fit, requested_lags) + 1j * np.polyval(phase_fit, requested_lags))
    raise ValueError(method)


def _ridge_interpolation_prediction(
    normalized: np.ndarray,
    dataset,
    development: np.ndarray,
    test: np.ndarray,
    train_lag_indices: list[int],
    test_lag_indices: list[int],
    method: str,
    alpha: float,
) -> np.ndarray:
    train_targets = dataset.targets[development][:, :, :, train_lag_indices, :]
    x_train = normalized[development].reshape(-1, normalized.shape[-1])
    y_train = train_targets.reshape(-1, int(np.prod(train_targets.shape[2:])))
    model = Ridge(alpha=float(alpha)).fit(x_train, y_train)
    predicted_known = model.predict(normalized[test].reshape(-1, normalized.shape[-1])).reshape(
        len(test), dataset.targets.shape[1], dataset.targets.shape[2], len(train_lag_indices), 2
    )
    known_complex = _complex(predicted_known)
    output = np.zeros_like(dataset.targets[test])
    known_lags = np.asarray(dataset.lags)[train_lag_indices]
    requested = np.asarray(dataset.lags)[test_lag_indices]
    for sample in range(len(test)):
        for block in range(dataset.targets.shape[1]):
            for band in range(dataset.targets.shape[2]):
                fitted = _interpolate_complex(known_lags, known_complex[sample, block, band], requested, method)
                output[sample, block, band, test_lag_indices, 0] = fitted.real
                output[sample, block, band, test_lag_indices, 1] = fitted.imag
    return output.astype(np.float32)


def _matrix_metrics_from_probabilities(
    probabilities: np.ndarray,
    dataset,
    indices: np.ndarray,
    q_bins: int,
    model: str,
    pattern: str,
) -> list[dict]:
    rows = []
    q = _fft_q(q_bins)
    for row, sample in enumerate(indices):
        values = []
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            target = loaded["target_anchor"]
            positions = loaded["anchor_local"]
        for block in range(target.shape[0]):
            for band in range(target.shape[1]):
                predicted = spectral_mixture_operator(probabilities[row, block, band], positions, q)
                truth = normalize_coherency(target[block, band])
                off = ~np.eye(len(truth), dtype=bool)
                frobenius = np.linalg.norm((predicted - truth)[off]) / max(np.linalg.norm(truth[off]), np.finfo(float).eps)
                _, pv = np.linalg.eigh(predicted)
                _, tv = np.linalg.eigh(truth)
                angle = np.degrees(np.arccos(np.clip(abs(np.vdot(pv[:, -1], tv[:, -1])), 0.0, 1.0)))
                values.append([frobenius, angle, abs(effective_rank(predicted) - effective_rank(truth))])
        values = np.asarray(values)
        rows.append(
            {
                "model": model,
                "pattern": pattern,
                "event_id": str(dataset.event_ids[sample]),
                "route": str(dataset.routes[sample]),
                "matrix_frobenius_nrmse": float(values[:, 0].mean()),
                "dominant_eigenspace_angle_deg": float(values[:, 1].mean()),
                "effective_rank_absolute_error": float(values[:, 2].mean()),
            }
        )
    return rows


def run_held_separation_audit(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models, _ = _audit_paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    normalized, _, _ = _scaler(dataset, roles)
    mask = reliable_mask(cfg)
    development = np.flatnonzero(roles == "model_development")
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    q_bins = int(amendment["withheld_separation"]["final_q_bins"])
    gauges = _gauge_lengths(dataset)
    rows = []
    matrix_rows = []
    training_rows = []
    for pattern, definition in amendment["withheld_separation"]["patterns"].items():
        train_lags = list(map(int, definition["train_lag_indices"]))
        held_lags = list(map(int, definition["test_lag_indices"]))
        values = _mask_lag_features(normalized, held_lags)
        train_mask = _held_lag_mask(mask, train_lags)
        held_mask = _held_lag_mask(mask, held_lags)
        for family in amendment["withheld_separation"]["models"]:
            predictions, probabilities, training = _save_model_predictions(
                str(family), values, dataset, development, calibration, test, train_mask, cfg, amendment, models, q_bins, f"held_{pattern}"
            )
            label = "Linear PSD decoder" if family == "linear_psd" else "Local-state PSD"
            rows.extend(_metric_rows(label, pattern, predictions["architecture_test"], dataset, test, held_mask))
            matrix_rows.extend(_matrix_metrics_from_probabilities(probabilities["architecture_test"], dataset, test, q_bins, label, pattern))
            training_rows.extend([{**record, "pattern": pattern} for record in training])
        comparator_predictions = {
            "Persistence": _context_predictions(dataset, test),
            "Climatology": _climatology_prediction(dataset, roles, test, gauges),
        }
        for method in amendment["withheld_separation"]["interpolation"]:
            comparator_predictions[f"Ridge + {method}"] = _ridge_interpolation_prediction(
                normalized, dataset, development, test, train_lags, held_lags, str(method),
                float(amendment["classical_baselines"]["ridge_alpha"]),
            )
        for label, prediction in comparator_predictions.items():
            rows.extend(_metric_rows(label, pattern, prediction, dataset, test, held_mask))
            safe = label.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
            np.save(models / f"held_{pattern}_{safe}_architecture_test_predictions.npy", prediction)
        print(f"completed held-separation pattern: {pattern}", flush=True)
    metrics = pd.DataFrame(rows).rename(columns={"role": "pattern"})
    matrices = pd.DataFrame(matrix_rows)
    metrics.to_csv(reports / "held_separation_event_route_metrics.csv", index=False)
    matrices.to_csv(reports / "held_separation_matrix_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(reports / "held_separation_training.csv", index=False)
    summary = metrics.groupby(["pattern", "model"], as_index=False).agg(
        nrmse=("nrmse", "mean"), sign_accuracy=("sign_accuracy", "mean"), magnitude_mae=("magnitude_mae", "mean"), phase_mae=("phase_mae", "mean")
    )
    summary.to_csv(reports / "held_separation_summary.csv", index=False)
    path = reports / "held_separation_receipt.json"
    write_json(
        path,
        {
            "amendment_sha256": sha256(amendment_path),
            "q_bins": q_bins,
            "test_results": summary.to_dict("records"),
            "claim_rule": "No continuous-operator claim is made unless a learned model materially exceeds ordinary interpolation and persistence across patterns.",
        },
    )
    return path


def _missing_indices(dataset, sample: int, scenario: str, replicate: int) -> np.ndarray:
    blocks = dataset.features.shape[1]
    rng = np.random.default_rng(20260906 + 1009 * replicate + 17 * sample + int(hashlib.sha256(scenario.encode()).hexdigest()[:6], 16))
    if scenario.startswith("random_"):
        count = int(scenario.split("_")[1])
        return np.sort(rng.choice(blocks, size=min(count, blocks), replace=False))
    if scenario.startswith("contiguous_"):
        count = int(scenario.split("_")[1])
        start = int(rng.integers(0, blocks - count + 1))
        return np.arange(start, start + count)
    if scenario == "route_end":
        return np.arange(max(0, blocks - 2), blocks)
    if scenario == "alternating":
        return np.arange(0, blocks, 2)
    with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
        if scenario == "highest_noise":
            noise = np.real(np.diagonal(loaded["noise_anchor"], axis1=-2, axis2=-1)).mean(axis=(1, 2))
            return np.asarray([int(np.argmax(noise))])
        if scenario == "lowest_snr":
            snr = loaded["context_noise_power_ratio"].mean(axis=1)
            return np.asarray([int(np.argmin(snr))])
        if scenario == "natural_quality_mask":
            coverage = loaded["pick_coverage"]
            selected = np.flatnonzero(coverage < 0.5)
            return selected[: max(1, min(2, len(selected)))] if len(selected) else np.asarray([], dtype=int)
    raise ValueError(scenario)


def _masked_values(values: np.ndarray, dataset, indices: np.ndarray, scenario: str, replicate: int, mask_feature: bool) -> np.ndarray:
    output = values.copy()
    for row, sample in enumerate(indices):
        missing = _missing_indices(dataset, int(sample), scenario, replicate)
        if not len(missing):
            continue
        if mask_feature:
            output[row, missing, :-1] = 0.0
            output[row, missing, -1] = 0.0
        else:
            output[row, missing] = 0.0
    return output


def _load_model_members(
    family: str,
    features: int,
    dataset,
    amendment: dict,
    models: Path,
    prefix: str,
    q_bins: int,
) -> list[nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = []
    for seed in amendment["training"]["final_seeds"]:
        model = _new_model(family, features, dataset, amendment, q_bins).to(device)
        state = torch.load(models / f"{prefix}_{family}_q{q_bins}_seed{seed}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        members.append(model)
    return members


def _ensemble_predict(members: list[nn.Module], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(values.astype(np.float32)).to(next(members[0].parameters()).device)
    predictions, probabilities = [], []
    indices = np.arange(len(values))
    for model in members:
        prediction, probability = _predict(model, x, indices)
        predictions.append(prediction)
        probabilities.append(probability)
    return np.mean(predictions, axis=0), np.mean(probabilities, axis=0)


def run_missingness_stress(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models, _ = _audit_paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    normalized, _, _ = _scaler(dataset, roles)
    observed = np.concatenate([normalized, np.ones((*normalized.shape[:-1], 1), dtype=np.float32)], axis=-1)
    mask = reliable_mask(cfg)
    development = np.flatnonzero(roles == "model_development")
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    q_bins = int(amendment["decoder_audit"]["final_q_bins"])
    training_rows = []
    members: dict[str, list[nn.Module]] = {}
    labels = {
        "linear_psd": "Mask-aware linear PSD",
        "conv1d_psd": "Mask-aware CNN PSD",
        "bigru_psd": "Mask-aware BiGRU PSD",
        "bissm_psd": "Mask-aware state-space PSD",
        "gat_bissm_psd": "Mask-aware local-state PSD",
    }
    for family in amendment["missingness"]["families"]:
        _, _, training = _save_model_predictions(
            str(family), observed, dataset, development, calibration, test, mask, cfg, amendment, models, q_bins, "missing", augment=True
        )
        training_rows.extend(training)
        members[labels[str(family)]] = _load_model_members(str(family), observed.shape[-1], dataset, amendment, models, "missing", q_bins)
        print(f"completed mask-aware training: {family}", flush=True)
    members["Original local-state PSD"] = _load_model_members(
        "gat_bissm_psd", normalized.shape[-1], dataset, amendment, models, "corrected", q_bins
    )
    pd.DataFrame(training_rows).to_csv(reports / "missingness_training.csv", index=False)

    base_scores = {}
    clean_rows = []
    for label, model_members in members.items():
        values = observed[test] if label.startswith("Mask-aware") else normalized[test]
        prediction, _ = _ensemble_predict(model_members, values)
        base_scores[label] = _sample_nrmse(prediction, dataset.targets[test], mask)
        event_scores = pd.DataFrame(
            {
                "event_id": dataset.event_ids[test].astype(str),
                "route": dataset.routes[test].astype(str),
                "nrmse": base_scores[label],
            }
        )
        complete = event_scores.groupby("event_id").nrmse.mean().to_numpy()
        clean_rows.append(
            {
                "model": label,
                **_event_bootstrap(complete, int(amendment["bootstrap"]["seed"]), int(amendment["bootstrap"]["replicates"])),
                "terra_mean": float(event_scores[event_scores.route == "TERRA"].nrmse.mean()),
                "kkfls_mean": float(event_scores[event_scores.route == "KKFL-S"].nrmse.mean()),
            }
        )
    clean_table = pd.DataFrame(clean_rows)
    clean_table.to_csv(reports / "missingness_clean_performance.csv", index=False)
    scenarios = []
    for level in amendment["missingness"]["random_levels"]:
        scenarios.extend([(f"random_{level}", replicate) for replicate in range(int(amendment["missingness"]["random_masks_per_level"]))])
    scenarios.extend((name, replicate) for name in ["contiguous_1", "contiguous_2"] for replicate in range(32))
    scenarios.extend((name, 0) for name in ["route_end", "alternating", "highest_noise", "lowest_snr", "natural_quality_mask"])
    rows = []
    raw_path = reports / "missingness_raw_predictions.h5"
    with h5py.File(raw_path, "w") as raw:
        raw.create_dataset("scenario", data=np.asarray([name.encode() for name, _ in scenarios]))
        raw.create_dataset("replicate", data=np.asarray([replicate for _, replicate in scenarios], dtype=np.int32))
        for label in members:
            key = label.lower().replace(" ", "_").replace("-", "_")
            raw.create_dataset(
                key,
                shape=(len(scenarios), len(test), *dataset.targets.shape[1:]),
                dtype="f4",
                chunks=(1, len(test), *dataset.targets.shape[1:]),
                compression="gzip",
                compression_opts=4,
            )
        for scenario_index, (scenario, replicate) in enumerate(scenarios):
            for label, model_members in members.items():
                mask_feature = label.startswith("Mask-aware")
                base_values = observed[test] if mask_feature else normalized[test]
                values = _masked_values(base_values, dataset, test, scenario, replicate, mask_feature)
                prediction, _ = _ensemble_predict(model_members, values)
                key = label.lower().replace(" ", "_").replace("-", "_")
                raw[key][scenario_index] = prediction
                scores = _sample_nrmse(prediction, dataset.targets[test], mask)
                for row, sample in enumerate(test):
                    rows.append(
                        {
                            "scenario": scenario,
                            "replicate": int(replicate),
                            "model": label,
                            "event_id": str(dataset.event_ids[sample]),
                            "route": str(dataset.routes[sample]),
                            "nrmse": float(scores[row]),
                            "degradation": float(scores[row] - base_scores[label][row]),
                        }
                    )
            if (scenario_index + 1) % 50 == 0:
                print(f"missingness scenarios {scenario_index + 1}/{len(scenarios)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_parquet(reports / "missingness_stress_event_route.parquet", index=False)
    summary_rows = []
    seed = int(amendment["bootstrap"]["seed"])
    replicates = int(amendment["bootstrap"]["replicates"])
    for (scenario, model), group in table.groupby(["scenario", "model"]):
        complete = group.groupby(["replicate", "event_id"], as_index=False).nrmse.mean().groupby("event_id").nrmse.mean().to_numpy()
        bootstrap = _event_bootstrap(complete, seed, replicates)
        summary_rows.append(
            {
                "scenario": scenario,
                "model": model,
                **bootstrap,
                "worst_case_nrmse": float(group.nrmse.max()),
                "median_degradation": float(group.degradation.median()),
                "terra_mean": float(group[group.route == "TERRA"].nrmse.mean()),
                "kkfls_mean": float(group[group.route == "KKFL-S"].nrmse.mean()),
            }
        )
        seed += 1
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(reports / "missingness_stress_summary.csv", index=False)
    path = reports / "missingness_stress_receipt.json"
    write_json(
        path,
        {
            "amendment_sha256": sha256(amendment_path),
            "random_masks_per_level": int(amendment["missingness"]["random_masks_per_level"]),
            "models": list(members),
            "clean_performance": clean_table.to_dict("records"),
            "raw_predictions_sha256": sha256(raw_path),
            "summary": summary.to_dict("records"),
        },
    )
    return path


def _finite_conformal_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    rank = min(len(values), int(math.ceil((len(values) + 1) * float(coverage))))
    return float(values[max(rank - 1, 0)])


def run_uncertainty_audit(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models, _ = _audit_paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    mask = reliable_mask(cfg)
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    q_bins = int(amendment["decoder_audit"]["final_q_bins"])
    normalized, _, _ = _scaler(dataset, roles)
    members = _load_model_members("gat_bissm_psd", normalized.shape[-1], dataset, amendment, models, "corrected", q_bins)

    role_values = {}
    for role, indices in [("calibration", calibration), ("architecture_test", test)]:
        x = torch.from_numpy(normalized.astype(np.float32)).to(next(members[0].parameters()).device)
        member_predictions = [_predict(model, x, indices)[0] for model in members]
        ensemble = np.mean(member_predictions, axis=0)
        error = _sample_nrmse(ensemble, dataset.targets[indices], mask)
        spread = np.mean(
            np.stack([_sample_nrmse(prediction, ensemble, mask) for prediction in member_predictions]), axis=0
        )
        role_values[role] = pd.DataFrame(
            {
                "event_id": dataset.event_ids[indices].astype(str),
                "route": dataset.routes[indices].astype(str),
                "error_nrmse": error,
                "ensemble_spread": spread,
            }
        )
    calibration_table = role_values["calibration"]
    test_table = role_values["architecture_test"]
    coverage_rows = []
    for coverage in amendment["uncertainty"]["nominal_coverages"]:
        pooled_scores = calibration_table.groupby("event_id").error_nrmse.mean().to_numpy()
        maximum_scores = calibration_table.groupby("event_id").error_nrmse.max().to_numpy()
        rules = {
            "pooled_grouped_conformal": {"ALL": _finite_conformal_quantile(pooled_scores, coverage)},
            "maximum_route_grouped_conformal": {"ALL": _finite_conformal_quantile(maximum_scores, coverage)},
            "route_conditional_conformal": {
                route: _finite_conformal_quantile(group.error_nrmse.to_numpy(), coverage)
                for route, group in calibration_table.groupby("route")
            },
        }
        for method, quantiles in rules.items():
            bounds = np.asarray([quantiles.get(route, quantiles.get("ALL")) for route in test_table.route])
            covered = test_table.error_nrmse.to_numpy() <= bounds
            for route in ["ALL", *sorted(test_table.route.unique())]:
                selected = np.ones(len(test_table), dtype=bool) if route == "ALL" else test_table.route.to_numpy() == route
                coverage_rows.append(
                    {
                        "method": method,
                        "nominal_coverage": float(coverage),
                        "route": route,
                        "observed_coverage": float(np.mean(covered[selected])),
                        "mean_upper_bound": float(np.mean(bounds[selected])),
                        "absolute_calibration_error": float(abs(np.mean(covered[selected]) - coverage)),
                    }
                )
    coverage_table = pd.DataFrame(coverage_rows)
    coverage_table.to_csv(reports / "uncertainty_conformal_coverage.csv", index=False)
    missing_shift_rows = []
    missing_path = reports / "missingness_stress_event_route.parquet"
    if missing_path.exists():
        stress = pd.read_parquet(missing_path)
        observed = np.concatenate([normalized, np.ones((*normalized.shape[:-1], 1), dtype=np.float32)], axis=-1)
        calibration_errors = {"Original local-state PSD": calibration_table}
        missing_members = _load_model_members(
            "gat_bissm_psd", observed.shape[-1], dataset, amendment, models, "missing", q_bins
        )
        missing_prediction, _ = _ensemble_predict(missing_members, observed[calibration])
        calibration_errors["Mask-aware local-state PSD"] = pd.DataFrame(
            {
                "event_id": dataset.event_ids[calibration].astype(str),
                "route": dataset.routes[calibration].astype(str),
                "error_nrmse": _sample_nrmse(missing_prediction, dataset.targets[calibration], mask),
            }
        )
        for model, calibration_errors_table in calibration_errors.items():
            model_stress = stress[stress.model == model]
            grouped_calibration = calibration_errors_table.groupby("event_id").error_nrmse.max().to_numpy()
            for coverage in amendment["uncertainty"]["nominal_coverages"]:
                bound = _finite_conformal_quantile(grouped_calibration, float(coverage))
                for scenario, group in model_stress.groupby("scenario"):
                    missing_shift_rows.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "nominal_coverage": float(coverage),
                            "upper_bound": bound,
                            "observed_coverage": float(np.mean(group.nrmse.to_numpy() <= bound)),
                            "mean_nrmse": float(group.nrmse.mean()),
                        }
                    )
    missing_shift = pd.DataFrame(missing_shift_rows)
    missing_shift.to_csv(reports / "uncertainty_missingness_shift.csv", index=False)
    diagnostic_rows = []
    for role, table in role_values.items():
        correlation = spearmanr(table.ensemble_spread, table.error_nrmse).statistic
        for retained in [1.0, 0.8, 0.6, 0.4]:
            count = max(1, int(math.ceil(len(table) * retained)))
            selected = table.nsmallest(count, "ensemble_spread")
            diagnostic_rows.append(
                {
                    "role": role,
                    "retained_fraction": retained,
                    "mean_error": float(selected.error_nrmse.mean()),
                    "spread_error_spearman": float(correlation),
                }
            )
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(reports / "uncertainty_selective_risk.csv", index=False)
    path = reports / "uncertainty_audit_summary.json"
    write_json(
        path,
        {
            "amendment_sha256": sha256(amendment_path),
            "coverage": coverage_table.to_dict("records"),
            "selective_risk": diagnostics.to_dict("records"),
            "missingness_shift": missing_shift.to_dict("records"),
            "claim_rule": "A useful uncertainty claim requires near-nominal grouped coverage and decreasing error under selective retention without test tuning.",
        },
    )
    return path


def _low_rank_matrix(matrix: np.ndarray, rank: int) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(matrix)
    keep = np.argsort(values)[-int(rank):]
    values_out = np.zeros_like(values)
    values_out[keep] = np.maximum(values[keep], 0.0)
    return (vectors * values_out) @ vectors.conj().T


def _downstream_grid(
    dataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    q_bins: int,
    loading: float,
    predicted_shrinkage: float,
    classical_shrinkage: float,
    low_rank: int,
    role: str,
) -> pd.DataFrame:
    q = _fft_q(q_bins)
    rows = []
    for row, sample in enumerate(indices):
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            target, noise, context, positions = (
                loaded["target_anchor"], loaded["noise_anchor"], loaded["context_anchor"], loaded["anchor_local"]
            )
        for block in range(target.shape[0]):
            for band in range(target.shape[1]):
                rt, rn, rc = target[block, band], noise[block, band], context[block, band]
                if not (rt.shape == rn.shape == rc.shape == (32, 32)):
                    raise ValueError("Downstream comparators require identical 32-channel matrices")
                if np.asarray(positions).shape[0] != 32:
                    raise ValueError("Downstream coordinate vector must contain the same 32 channels")
                diagonal = np.diag(np.maximum(np.real(np.diag(rc)), np.finfo(float).eps))
                correlation = spectral_mixture_operator(probabilities[row, block, band], positions, q)
                scale = np.sqrt(np.diag(diagonal)[:, None] * np.diag(diagonal)[None, :])
                predicted = scale * correlation
                predicted = (1.0 - predicted_shrinkage) * predicted + predicted_shrinkage * diagonal
                classical = (1.0 - classical_shrinkage) * rc + classical_shrinkage * diagonal
                lowrank = _low_rank_matrix(rc, low_rank) + 1e-8 * np.eye(len(rc))
                candidates = {
                    "Diagonal": diagonal,
                    "Early persistence": rc,
                    "Classical shrinkage": classical,
                    "Low-rank persistence": lowrank,
                    "Predicted covariance": predicted,
                    "Oracle later-S covariance": rt,
                }
                for method, signal in candidates.items():
                    weight = _generalized_weight(signal, rn, loading)
                    rows.append(
                        {
                            "role": role,
                            "event_id": str(dataset.event_ids[sample]),
                            "route": str(dataset.routes[sample]),
                            "block": block,
                            "band": band,
                            "channels": 32,
                            "method": method,
                            "snr_star_db": float(10.0 * np.log10(_ratio(rt, rn, weight))),
                        }
                    )
    return pd.DataFrame(rows)


def run_downstream_oracle_audit(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models, _ = _audit_paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    q_bins = int(amendment["decoder_audit"]["final_q_bins"])
    p_cal = np.load(models / "corrected_gat_bissm_psd_calibration_probabilities.npy")
    p_test = np.load(models / "corrected_gat_bissm_psd_architecture_test_probabilities.npy")
    spec = amendment["downstream_oracle"]
    defaults = {
        "loading": 0.001,
        "predicted_shrinkage": 0.75,
        "classical_shrinkage": 0.5,
        "rank": 8,
    }
    selection_rows = []

    def score_candidate(stage: str, loading_value: float, predicted_value: float, classical_value: float, rank_value: int) -> dict:
        grid = _downstream_grid(
            dataset, calibration, p_cal, q_bins, loading_value, predicted_value, classical_value, rank_value, "calibration"
        )
        event = grid.groupby(["event_id", "route", "method"], as_index=False).snr_star_db.mean()
        wide = event.pivot(index=["event_id", "route"], columns="method", values="snr_star_db")
        return {
            "stage": stage,
            "loading": loading_value,
            "predicted_shrinkage": predicted_value,
            "classical_shrinkage": classical_value,
            "rank": rank_value,
            "diagonal_mean_db": float(wide["Diagonal"].mean()),
            "predicted_minus_diagonal_db": float((wide["Predicted covariance"] - wide["Diagonal"]).mean()),
            "classical_minus_diagonal_db": float((wide["Classical shrinkage"] - wide["Diagonal"]).mean()),
            "low_rank_minus_diagonal_db": float((wide["Low-rank persistence"] - wide["Diagonal"]).mean()),
        }

    for value in spec["loading_candidates"]:
        selection_rows.append(score_candidate("loading", float(value), defaults["predicted_shrinkage"], defaults["classical_shrinkage"], defaults["rank"]))
    loading = float(max((row for row in selection_rows if row["stage"] == "loading"), key=lambda row: row["diagonal_mean_db"])["loading"])
    for value in spec["predicted_shrinkage_candidates"]:
        selection_rows.append(score_candidate("predicted_shrinkage", loading, float(value), defaults["classical_shrinkage"], defaults["rank"]))
    predicted_shrinkage = float(max((row for row in selection_rows if row["stage"] == "predicted_shrinkage"), key=lambda row: row["predicted_minus_diagonal_db"])["predicted_shrinkage"])
    for value in spec["classical_shrinkage_candidates"]:
        selection_rows.append(score_candidate("classical_shrinkage", loading, predicted_shrinkage, float(value), defaults["rank"]))
    classical_shrinkage = float(max((row for row in selection_rows if row["stage"] == "classical_shrinkage"), key=lambda row: row["classical_minus_diagonal_db"])["classical_shrinkage"])
    for value in spec["low_rank_candidates"]:
        selection_rows.append(score_candidate("low_rank", loading, predicted_shrinkage, classical_shrinkage, int(value)))
    rank = int(max((row for row in selection_rows if row["stage"] == "low_rank"), key=lambda row: row["low_rank_minus_diagonal_db"])["rank"])
    pd.DataFrame(selection_rows).to_csv(reports / "downstream_oracle_calibration_grid.csv", index=False)
    grid = _downstream_grid(
        dataset, test, p_test, q_bins, loading, predicted_shrinkage, classical_shrinkage, rank, "architecture_test"
    )
    grid.to_parquet(reports / "downstream_oracle_test_grid.parquet", index=False)
    event = grid.groupby(["event_id", "route", "method"], as_index=False).snr_star_db.mean()
    event.to_csv(reports / "downstream_oracle_event_route.csv", index=False)
    wide = event.pivot(index=["event_id", "route"], columns="method", values="snr_star_db").reset_index()
    comparisons = []
    seed = int(amendment["bootstrap"]["seed"])
    for method in ["Early persistence", "Classical shrinkage", "Low-rank persistence", "Predicted covariance", "Oracle later-S covariance"]:
        values = wide.assign(delta=wide[method] - wide["Diagonal"]).groupby("event_id").delta.mean().to_numpy()
        comparisons.append({"method": method, **_event_bootstrap(values, seed, int(amendment["bootstrap"]["replicates"]))})
        seed += 1
    result = pd.DataFrame(comparisons)
    result.to_csv(reports / "downstream_oracle_bootstrap.csv", index=False)
    path = reports / "downstream_oracle_summary.json"
    write_json(
        path,
        {
            "amendment_sha256": sha256(amendment_path),
            "locked_on_calibration": {"loading": loading, "predicted_shrinkage": predicted_shrinkage, "classical_shrinkage": classical_shrinkage, "rank": rank},
            "architecture_test_comparisons": comparisons,
            "interpretation_rule": "Oracle-minus-diagonal measures task headroom; learned-minus-diagonal measures usable recovered covariance under the identical processor.",
        },
    )
    return path


def run_decoder_grid_performance(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models, _ = _audit_paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    normalized, _, _ = _scaler(dataset, roles)
    development = np.flatnonzero(roles == "model_development")
    test = np.flatnonzero(roles == "architecture_test")
    mask = reliable_mask(cfg)
    rows = []
    training_rows = []
    seed = int(amendment["training"]["final_seeds"][0])
    for q_bins in amendment["decoder_audit"]["diagnostic_q_bins"]:
        model, record = _fit_locked_model(
            "linear_psd", normalized, dataset, development, mask, amendment, seed, int(q_bins), Path(cfg["_root"])
        )
        prediction, _ = _predict(
            model,
            torch.from_numpy(normalized.astype(np.float32)).to(next(model.parameters()).device),
            test,
        )
        metrics = _rich_sample_metrics(prediction, dataset.targets[test], mask)
        rows.append(
            {
                "q_bins": int(q_bins),
                "pattern": "all_observed_lags",
                "mean_nrmse": float(metrics["nrmse"].mean()),
                "mean_sign_accuracy": float(metrics["sign_accuracy"].mean()),
            }
        )
        checkpoint = models / f"grid_linear_psd_q{q_bins}_seed{seed}.pt"
        torch.save(model.state_dict(), checkpoint)
        np.save(models / f"grid_linear_psd_q{q_bins}_architecture_test_predictions.npy", prediction)
        write_json(
            models / f"grid_linear_psd_q{q_bins}_seed{seed}_history.json",
            {"selection": record.get("history", []), "refit_losses": record.get("refit_losses", [])},
        )
        record["checkpoint_sha256"] = sha256(checkpoint)
        training_rows.append(record)
        for pattern in ["alternating_a", "alternating_b"]:
            definition = amendment["withheld_separation"]["patterns"][pattern]
            train_lags = list(map(int, definition["train_lag_indices"]))
            held_lags = list(map(int, definition["test_lag_indices"]))
            values = _mask_lag_features(normalized, held_lags)
            train_mask = _held_lag_mask(mask, train_lags)
            held_mask = _held_lag_mask(mask, held_lags)
            held_model, held_record = _fit_locked_model(
                "linear_psd", values, dataset, development, train_mask, amendment, seed, int(q_bins), Path(cfg["_root"])
            )
            held_prediction, _ = _predict(
                held_model,
                torch.from_numpy(values.astype(np.float32)).to(next(held_model.parameters()).device),
                test,
            )
            held_metrics = _rich_sample_metrics(held_prediction, dataset.targets[test], held_mask)
            rows.append(
                {
                    "q_bins": int(q_bins),
                    "pattern": pattern,
                    "mean_nrmse": float(held_metrics["nrmse"].mean()),
                    "mean_sign_accuracy": float(held_metrics["sign_accuracy"].mean()),
                }
            )
            held_checkpoint = models / f"grid_{pattern}_linear_psd_q{q_bins}_seed{seed}.pt"
            torch.save(held_model.state_dict(), held_checkpoint)
            np.save(models / f"grid_{pattern}_linear_psd_q{q_bins}_architecture_test_predictions.npy", held_prediction)
            write_json(
                models / f"grid_{pattern}_linear_psd_q{q_bins}_seed{seed}_history.json",
                {"selection": held_record.get("history", []), "refit_losses": held_record.get("refit_losses", [])},
            )
            held_record["checkpoint_sha256"] = sha256(held_checkpoint)
            training_rows.append({**held_record, "pattern": pattern})
        print(f"completed decoder grid diagnostic: {q_bins}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(reports / "decoder_grid_performance.csv", index=False)
    pd.DataFrame(training_rows).drop(columns=["history", "refit_losses"], errors="ignore").to_csv(
        reports / "decoder_grid_training.csv", index=False
    )
    path = reports / "decoder_grid_performance_receipt.json"
    write_json(
        path,
        {
            "amendment_sha256": sha256(amendment_path),
            "selection_statement": "257 bins were fixed by the no-alias inequality before these performance outcomes.",
            "results": table.to_dict("records"),
        },
    )
    return path


def run_provenance_audit(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    reports, _, _ = _audit_paths(cfg, amendment)
    root = Path(cfg["_root"])
    rows = []
    for directory in ["src", "configs", "scripts", "tests", "manuscript/cageo_submission"]:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "output" not in path.parts and path.suffix not in {".pyc"}:
                rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(reports / "source_tree_manifest.csv", index=False)
    tree_hash = hashlib.sha256("\n".join(f"{row['sha256']}  {row['path']}" for row in rows).encode()).hexdigest()
    source_manifest = (root / cfg["study"]["source_manifest"]).resolve()
    source_split = (root / cfg["study"]["source_split_table"]).resolve()
    base, _ = _load_amendment(cfg)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    split = pd.read_csv(root / "reports" / "hybrid_search" / "architecture_test_split.csv", dtype={"event_id": str})
    source_groups = dict(zip(split.event_id.astype(str), split.source_group_id.astype(str), strict=True))
    input_rows = []
    for index, path_item in enumerate(dataset.paths):
        with np.load(path_item, allow_pickle=False) as loaded:
            input_rows.append(
                {
                    "event_id": str(dataset.event_ids[index]),
                    "source_group_id": source_groups.get(str(dataset.event_ids[index]), "unmapped"),
                    "route": str(dataset.routes[index]),
                    "role": str(roles[index]),
                    "gauge_length_m": float(loaded["gauge_length_m"]),
                    "operator_path": Path(path_item).relative_to(root).as_posix(),
                    "operator_sha256": sha256(path_item),
                }
            )
    input_table = pd.DataFrame(input_rows)
    input_table.to_parquet(reports / "experiment_input_manifest.parquet", index=False)
    package_inventory = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": dict(sorted(package_inventory.items(), key=lambda item: item[0].lower())),
    }
    write_json(reports / "runtime_environment.json", environment)
    artifact_rows = []
    artifact_roots = [
        reports,
        root / amendment["outputs"]["model_directory"],
        root / amendment["outputs"]["figure_directory"],
    ]
    excluded = {"provenance_receipt.json", "artifact_manifest.csv"}
    for artifact_root in artifact_roots:
        if not artifact_root.exists():
            continue
        for artifact in sorted(artifact_root.rglob("*")):
            if artifact.is_file() and artifact.name not in excluded:
                artifact_rows.append(
                    {
                        "path": artifact.relative_to(root).as_posix(),
                        "bytes": artifact.stat().st_size,
                        "sha256": sha256(artifact),
                    }
                )
    artifact_manifest = pd.DataFrame(artifact_rows)
    artifact_manifest.to_csv(reports / "artifact_manifest.csv", index=False)
    receipt = {
        "amendment_sha256": sha256(amendment_path),
        "git_commit": None,
        "git_limitation": "This workspace is not a Git worktree; a deterministic source-tree hash is recorded instead.",
        "source_tree_sha256": tree_hash,
        "source_manifest_sha256": sha256(source_manifest),
        "source_split_sha256": sha256(source_split),
        "experiment_input_manifest_sha256": sha256(reports / "experiment_input_manifest.parquet"),
        "artifact_manifest_sha256": sha256(reports / "artifact_manifest.csv"),
        "artifact_count": len(artifact_rows),
        "configuration_sha256": sha256(root / "configs" / "protocol_v1.0.yaml"),
        "random_seeds": amendment["training"]["final_seeds"],
        "bootstrap_seed": amendment["bootstrap"]["seed"],
    }
    path = reports / "provenance_receipt.json"
    write_json(path, receipt)
    return path


def record_external_validation_audit(cfg: dict) -> Path:
    amendment, amendment_path = _load_audit_amendment(cfg)
    reports, _, _ = _audit_paths(cfg, amendment)
    record = {
        "checked_utc": "2026-08-31T04:55:00Z",
        "amendment_sha256": sha256(amendment_path),
        "cook_inlet": {
            "project_page": "https://fiberlab.uw.edu/projects/alaska-cook-inlet/",
            "archive": "https://dasway.ess.washington.edu/gci/index.html",
            "official_index_accessible": True,
            "listed_collection_end": "2024-11-21",
            "anonymous_2024_waveform_object_status_last_checked": 403,
            "status": "metadata and event reports are public; inspected 2024 HDF5 objects are not anonymously readable",
        },
        "independent_candidates": [
            {
                "name": "OOI RCA multiplexed DAS 2024",
                "doi": "10.58046/4WEF-A282",
                "access": "public FTP, approximately 4 TB",
                "sampling": "200 Hz, approximately 10.2 m spacing, approximately 40.8 m gauge length",
                "compatibility": "independent submarine cable and interrogator, but not a drop-in match to the 25 Hz Cook Inlet event products",
            },
            {
                "name": "OOI RCA community DAS 2021",
                "access": "public OOI archive",
                "compatibility": "independent submarine acquisition; a new event catalog, window definition and frozen transfer protocol are required",
            },
        ],
        "outcome": "No external waveform outcome is included. A transfer evaluation must be frozen before downloading and inspecting a compatible event cohort.",
    }
    path = reports / "external_validation_access_audit.json"
    write_json(path, record)
    return path


def run_methodological_audit(cfg: dict) -> Path:
    amendment, _ = _load_audit_amendment(cfg)
    reports, _, _ = _audit_paths(cfg, amendment)
    outputs = {
        "decoder_coordinates": str(run_decoder_coordinate_audit(cfg)),
        "decoder_grid_performance": str(run_decoder_grid_performance(cfg)),
        "corrected_baselines": str(run_corrected_baselines(cfg)),
        "information_ceiling": str(run_information_ceiling(cfg)),
        "held_separation": str(run_held_separation_audit(cfg)),
        "missingness": str(run_missingness_stress(cfg)),
        "uncertainty": str(run_uncertainty_audit(cfg)),
        "downstream_oracle": str(run_downstream_oracle_audit(cfg)),
        "provenance": str(run_provenance_audit(cfg)),
        "external_validation": str(record_external_validation_audit(cfg)),
    }
    path = reports / "methodological_audit_summary.json"
    write_json(path, outputs)
    return path
