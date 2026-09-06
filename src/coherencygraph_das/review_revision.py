from __future__ import annotations

import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import linalg
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn

from .config import resolve, sha256, write_json
from .data import load_operator_dataset, reliable_mask
from .hybrid_search import _dataset_and_split, _load_amendment, _load_extension
from .models import ModernSpectralOperator
from .spectral import (
    cross_spectral_matrix,
    effective_rank,
    multitaper_fourier,
    normalize_coherency,
    robust_linear_pick_model,
    spectral_mixture_operator,
)


RAW_DATASET = "Acquisition/Raw[0]/RawData"


def _load_review_amendment(cfg: dict) -> tuple[dict, Path]:
    path = Path(cfg["_root"]) / "configs" / "protocol_amendment_04_deep_review.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def _load_fair_covariance_amendment(cfg: dict) -> tuple[dict, Path]:
    path = Path(cfg["_root"]) / "configs" / "protocol_amendment_05_fair_covariance_comparator.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def _paths(cfg: dict, amendment: dict) -> tuple[Path, Path]:
    reports = Path(cfg["_root"]) / amendment["outputs"]["directory"]
    models = Path(cfg["_root"]) / amendment["outputs"]["model_directory"]
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return reports, models


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model(
    family: str,
    features: int,
    dataset,
    cfg: dict,
    amendment: dict,
    q_bins: int | None = None,
) -> nn.Module:
    spec = amendment["neural"]
    return ModernSpectralOperator(
        features=features,
        hidden_dim=int(spec["hidden_dim"]),
        layers=int(spec["layers"]),
        dropout=float(spec["dropout"]),
        bands=dataset.targets.shape[2],
        q_bins=int(q_bins or cfg["spectral"]["spatial_wavenumber_bins"]),
        lags=list(map(int, cfg["spectral"]["channel_lags"])),
        family=family,
        heads=int(spec["attention_heads"]),
    )


def _complex(values: np.ndarray) -> np.ndarray:
    return values[..., 0] + 1j * values[..., 1]


def _sample_nrmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred = _complex(prediction)
    truth = _complex(target)
    keep = np.broadcast_to(mask[None, None, :, :], truth.shape)
    output = []
    for p, t, selected in zip(pred, truth, keep, strict=True):
        numerator = np.sqrt(np.mean(np.abs(p[selected] - t[selected]) ** 2))
        denominator = np.sqrt(np.mean(np.abs(t[selected]) ** 2))
        output.append(float(numerator / max(denominator, np.finfo(float).eps)))
    return np.asarray(output)


def _sample_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    error = _complex(prediction) - _complex(target)
    keep = np.broadcast_to(mask[None, None, :, :], error.shape)
    return np.asarray([float(np.mean(np.abs(row[selected]) ** 2)) for row, selected in zip(error, keep, strict=True)])


def _batches(
    indices: np.ndarray,
    event_ids: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    events = np.unique(event_ids[indices])
    rng.shuffle(events)
    return [
        indices[np.isin(event_ids[indices], events[start : start + size])]
        for start in range(0, len(events), size)
    ]


def _mask_blocks(
    values: np.ndarray,
    missing_blocks: int,
    seed: int,
    has_mask_feature: bool,
) -> np.ndarray:
    output = values.copy()
    rng = np.random.default_rng(seed)
    for sample in range(len(output)):
        selected = rng.choice(output.shape[1], size=min(missing_blocks, output.shape[1]), replace=False)
        if has_mask_feature:
            output[sample, selected, :-1] = 0.0
            output[sample, selected, -1] = 0.0
        else:
            output[sample, selected] = 0.0
    return output


def _training_augmentation(
    batch: torch.Tensor,
    rng: np.random.Generator,
    probability: float,
    maximum_missing: int,
) -> torch.Tensor:
    if probability <= 0 or maximum_missing <= 0:
        return batch
    augmented = batch.clone()
    for sample in range(len(augmented)):
        if rng.random() >= probability:
            continue
        count = int(rng.integers(1, maximum_missing + 1))
        selected = rng.choice(augmented.shape[1], size=count, replace=False)
        augmented[sample, selected, :-1] = 0.0
        augmented[sample, selected, -1] = 0.0
    return augmented


def _predict(model: nn.Module, values: torch.Tensor, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            prediction, probability = model(values[indices[start : start + 64]])
            predictions.append(prediction.cpu().numpy())
            probabilities.append(probability.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities)


def _fit_torch(
    family: str,
    values: np.ndarray,
    dataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    loss_mask: np.ndarray,
    cfg: dict,
    amendment: dict,
    seed: int,
    q_bins: int | None = None,
    smoothness_weight: float = 0.0,
    block_dropout_probability: float = 0.0,
    maximum_missing_blocks: int = 0,
) -> tuple[nn.Module, dict]:
    _seed(seed)
    spec = amendment["neural"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(values.astype(np.float32)).to(device)
    y = torch.from_numpy(dataset.targets).to(device)
    mask = torch.from_numpy(loss_mask).to(device)
    model = _model(family, values.shape[-1], dataset, cfg, amendment, q_bins=q_bins).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    best_state = None
    best_score = np.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(int(spec["train_epochs"])):
        model.train()
        for batch_indices in _batches(
            train_indices, dataset.event_ids, int(spec["batch_events"]), rng
        ):
            optimizer.zero_grad(set_to_none=True)
            batch_x = x[batch_indices]
            if values.shape[-1] == dataset.features.shape[-1] + 1:
                batch_x = _training_augmentation(
                    batch_x,
                    rng,
                    block_dropout_probability,
                    maximum_missing_blocks,
                )
            prediction, probability = model(batch_x)
            expanded = mask[None, None, :, :, None].expand_as(prediction)
            loss = ((prediction - y[batch_indices]) ** 2)[expanded].mean()
            if smoothness_weight:
                curvature = probability[..., 2:] - 2.0 * probability[..., 1:-1] + probability[..., :-2]
                loss = loss + float(smoothness_weight) * torch.mean(curvature**2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        prediction, _ = _predict(model, x, validation_indices)
        score = float(_sample_nrmse(prediction, dataset.targets[validation_indices], loss_mask).mean())
        if score < best_score - 1e-5:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= int(spec["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    return model, {
        "family": family,
        "seed": int(seed),
        "q_bins": int(q_bins or cfg["spectral"]["spatial_wavenumber_bins"]),
        "smoothness_weight": float(smoothness_weight),
        "best_epoch": int(best_epoch),
        "validation_nrmse": float(best_score),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": float(time.perf_counter() - started),
    }


def _scaler(dataset, roles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    development = roles == "model_development"
    mean = dataset.features[development].mean(axis=(0, 1), keepdims=True)
    std = dataset.features[development].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    return ((dataset.features - mean) / std).astype(np.float32), mean, std


def _context_predictions(dataset, indices: np.ndarray) -> np.ndarray:
    rows = []
    for sample in indices:
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            gamma = loaded["context_gamma"]
        rows.append(np.stack([gamma.real, gamma.imag], axis=-1).astype(np.float32))
    return np.stack(rows)


def _gauge_lengths(dataset) -> np.ndarray:
    values = []
    for path in dataset.paths:
        with np.load(path, allow_pickle=False) as loaded:
            values.append(float(loaded["gauge_length_m"]))
    return np.asarray(values)


def _climatology_prediction(
    dataset,
    roles: np.ndarray,
    indices: np.ndarray,
    gauges: np.ndarray,
) -> np.ndarray:
    development = np.flatnonzero(roles == "model_development")
    output = []
    for sample in indices:
        selected = development[
            (dataset.routes[development] == dataset.routes[sample])
            & np.isclose(gauges[development], gauges[sample])
        ]
        if not len(selected):
            selected = development[dataset.routes[development] == dataset.routes[sample]]
        output.append(dataset.targets[selected].mean(axis=0))
    return np.stack(output).astype(np.float32)


def _flatten_rows(values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.reshape(-1, values.shape[-1]), targets.reshape(-1, int(np.prod(targets.shape[2:])))


def _restore_rows(prediction: np.ndarray, samples: int, blocks: int, target_shape: tuple[int, ...]) -> np.ndarray:
    return prediction.reshape(samples, blocks, *target_shape)


def _event_bootstrap(values: np.ndarray, seed: int, replicates: int) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {
        "events": int(len(values)),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def _prediction_rows(
    name: str,
    prediction: np.ndarray,
    dataset,
    indices: np.ndarray,
    mask: np.ndarray,
    role: str,
) -> list[dict]:
    nrmse = _sample_nrmse(prediction, dataset.targets[indices], mask)
    mse = _sample_mse(prediction, dataset.targets[indices], mask)
    return [
        {
            "role": role,
            "model": name,
            "event_id": str(dataset.event_ids[sample]),
            "route": str(dataset.routes[sample]),
            "nrmse": float(nrmse[row]),
            "mse": float(mse[row]),
        }
        for row, sample in enumerate(indices)
    ]


def run_baseline_benchmark(cfg: dict) -> Path:
    amendment, amendment_path = _load_review_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models = _paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    normalized, mean, std = _scaler(dataset, roles)
    np.savez(models / "review_feature_scaler.npz", mean=mean, std=std)
    mask = reliable_mask(cfg)
    development = np.flatnonzero(roles == "model_development")
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    gauges = _gauge_lengths(dataset)
    target_shape = dataset.targets.shape[2:]
    rows: list[dict] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}

    for role, indices in [("calibration", calibration), ("architecture_test", test)]:
        predictions[(role, "Climatology")] = _climatology_prediction(dataset, roles, indices, gauges)
        predictions[(role, "Persistence")] = _context_predictions(dataset, indices)

    x_train, y_train = _flatten_rows(normalized[development], dataset.targets[development])
    x_calibration, _ = _flatten_rows(normalized[calibration], dataset.targets[calibration])
    x_test, _ = _flatten_rows(normalized[test], dataset.targets[test])
    ridge_search = []
    for alpha in amendment["baseline_benchmark"]["ridge_alpha_grid"]:
        model = Ridge(alpha=float(alpha)).fit(x_train, y_train)
        prediction = _restore_rows(
            model.predict(x_calibration), len(calibration), dataset.targets.shape[1], target_shape
        ).astype(np.float32)
        score = float(_sample_nrmse(prediction, dataset.targets[calibration], mask).mean())
        ridge_search.append({"alpha": float(alpha), "calibration_nrmse": score})
    ridge_choice = min(ridge_search, key=lambda row: row["calibration_nrmse"])
    ridge = Ridge(alpha=ridge_choice["alpha"]).fit(x_train, y_train)
    predictions[("calibration", "Ridge")] = _restore_rows(
        ridge.predict(x_calibration), len(calibration), dataset.targets.shape[1], target_shape
    ).astype(np.float32)
    predictions[("architecture_test", "Ridge")] = _restore_rows(
        ridge.predict(x_test), len(test), dataset.targets.shape[1], target_shape
    ).astype(np.float32)

    tree_search = []
    for candidate in amendment["baseline_benchmark"]["extra_trees_candidates"]:
        model = ExtraTreesRegressor(
            n_estimators=int(amendment["baseline_benchmark"]["extra_trees_estimators"]),
            min_samples_leaf=int(candidate["min_samples_leaf"]),
            max_features=float(candidate["max_features"]),
            random_state=int(amendment["neural"]["search_seed"]),
            n_jobs=-1,
        ).fit(x_train, y_train)
        prediction = _restore_rows(
            model.predict(x_calibration), len(calibration), dataset.targets.shape[1], target_shape
        ).astype(np.float32)
        score = float(_sample_nrmse(prediction, dataset.targets[calibration], mask).mean())
        tree_search.append({**candidate, "calibration_nrmse": score})
    tree_choice = min(tree_search, key=lambda row: row["calibration_nrmse"])
    trees = ExtraTreesRegressor(
        n_estimators=int(amendment["baseline_benchmark"]["extra_trees_estimators"]),
        min_samples_leaf=int(tree_choice["min_samples_leaf"]),
        max_features=float(tree_choice["max_features"]),
        random_state=int(amendment["neural"]["search_seed"]),
        n_jobs=-1,
    ).fit(x_train, y_train)
    predictions[("calibration", "Extra Trees")] = _restore_rows(
        trees.predict(x_calibration), len(calibration), dataset.targets.shape[1], target_shape
    ).astype(np.float32)
    predictions[("architecture_test", "Extra Trees")] = _restore_rows(
        trees.predict(x_test), len(test), dataset.targets.shape[1], target_shape
    ).astype(np.float32)

    for family, label in [("conv1d_psd", "1-D CNN PSD"), ("bigru_psd", "BiGRU PSD")]:
        member_store = {"calibration": [], "architecture_test": []}
        for seed in amendment["neural"]["final_seeds"]:
            model, record = _fit_torch(
                family,
                normalized,
                dataset,
                development,
                calibration,
                mask,
                cfg,
                amendment,
                int(seed),
            )
            checkpoint = models / f"{family}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            device = next(model.parameters()).device
            x = torch.from_numpy(normalized).to(device)
            for role, indices in [("calibration", calibration), ("architecture_test", test)]:
                prediction, _ = _predict(model, x, indices)
                member_store[role].append(prediction)
            record.update({"stage": "simple_sequence_baseline", "checkpoint_sha256": sha256(checkpoint)})
            rows.append(record)
        for role in member_store:
            predictions[(role, label)] = np.mean(member_store[role], axis=0)

    metric_rows: list[dict] = []
    for (role, name), prediction in predictions.items():
        indices = calibration if role == "calibration" else test
        metric_rows.extend(_prediction_rows(name, prediction, dataset, indices, mask, role))
        np.save(models / f"{name.lower().replace(' ', '_').replace('-', '')}_{role}.npy", prediction)

    # Existing equally trained neural results remain comparators, not the headline model.
    hybrid_models = Path(cfg["_root"]) / "models" / "hybrid_search"
    existing_names = {
        "gat_bissm_psd": "Local-state hybrid",
        "bissm_psd": "State-space only",
        "gatv2_psd": "GATv2 only",
        "chain_psd": "Chain graph baseline",
    }
    for family, name in existing_names.items():
        prediction = np.load(hybrid_models / f"{family}_architecture_test_predictions.npy")
        metric_rows.extend(
            _prediction_rows(name, prediction, dataset, test, mask, "architecture_test")
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(reports / "baseline_event_route_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(reports / "baseline_neural_training_runs.csv", index=False)
    pd.DataFrame(ridge_search).to_csv(reports / "ridge_calibration_search.csv", index=False)
    pd.DataFrame(tree_search).to_csv(reports / "extra_trees_calibration_search.csv", index=False)

    calibration_summary = (
        metrics[metrics.role == "calibration"].groupby("model", as_index=False)
        .agg(mean_nrmse=("nrmse", "mean"), mean_mse=("mse", "mean"))
        .sort_values("mean_nrmse")
    )
    nonneural = ["Climatology", "Persistence", "Ridge", "Extra Trees"]
    best_simple = str(calibration_summary[calibration_summary.model.isin(nonneural)].iloc[0].model)
    test_simple = metrics[(metrics.role == "architecture_test") & (metrics.model == best_simple)]
    test_summary = (
        metrics[metrics.role == "architecture_test"].groupby("model", as_index=False)
        .agg(mean_nrmse=("nrmse", "mean"), mean_mse=("mse", "mean"), records=("nrmse", "size"))
        .sort_values("mean_nrmse")
    )
    baseline_mse = float(test_simple.mse.mean())
    test_summary["skill_vs_best_simple"] = np.where(
        np.isfinite(test_summary.mean_mse),
        1.0 - test_summary.mean_mse / max(baseline_mse, np.finfo(float).eps),
        np.nan,
    )
    test_summary.to_csv(reports / "baseline_test_summary.csv", index=False)
    calibration_summary.to_csv(reports / "baseline_calibration_summary.csv", index=False)

    bootstrap_rows = []
    seed = int(amendment["bootstrap"]["seed"])
    best_rows = metrics[(metrics.role == "architecture_test") & (metrics.model == best_simple)]
    for model_name, model_rows in metrics[metrics.role == "architecture_test"].groupby("model"):
        if model_name == best_simple:
            continue
        paired = model_rows.merge(
            best_rows[["event_id", "route", "nrmse"]],
            on=["event_id", "route"],
            suffixes=("_model", "_baseline"),
            validate="one_to_one",
        )
        delta = paired.assign(delta=paired.nrmse_model - paired.nrmse_baseline).groupby("event_id").delta.mean().to_numpy()
        bootstrap_rows.append({"model": model_name, "baseline": best_simple, **_event_bootstrap(delta, seed, int(amendment["bootstrap"]["replicates"]))})
        seed += 1
    pd.DataFrame(bootstrap_rows).to_csv(reports / "baseline_paired_bootstrap.csv", index=False)
    receipt = {
        "amendment_sha256": sha256(amendment_path),
        "architecture_test_is_retrospective": True,
        "best_simple_baseline_selected_on_calibration": best_simple,
        "ridge_choice": ridge_choice,
        "extra_trees_choice": tree_choice,
        "test_summary": test_summary.to_dict("records"),
    }
    path = reports / "baseline_benchmark_summary.json"
    write_json(path, receipt)
    return path


def _mask_lag_features(values: np.ndarray, lag_indices: list[int]) -> np.ndarray:
    output = values.copy()
    lags_per_band = 8
    for start in [0, 32, 64, 96]:
        columns = [start + band * lags_per_band + lag for band in range(4) for lag in lag_indices]
        output[:, :, columns] = 0.0
    return output


def _held_lag_mask(base_mask: np.ndarray, indices: list[int]) -> np.ndarray:
    output = np.zeros_like(base_mask)
    output[:, indices] = base_mask[:, indices]
    return output


def _matrix_validation_rows(
    probabilities: np.ndarray,
    dataset,
    sample_indices: np.ndarray,
    q_bins: int,
    pattern: str,
    held_lags: list[int],
) -> list[dict]:
    q = np.linspace(-np.pi, np.pi, q_bins, endpoint=False)
    rows = []
    for row, sample in enumerate(sample_indices):
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            target_anchor = loaded["target_anchor"]
        metrics = []
        for block in range(target_anchor.shape[0]):
            for band in range(target_anchor.shape[1]):
                prediction = spectral_mixture_operator(
                    probabilities[row, block, band], dataset.anchors, q
                )
                target = normalize_coherency(target_anchor[block, band])
                off = ~np.eye(len(target), dtype=bool)
                frobenius = float(
                    np.linalg.norm((prediction - target)[off])
                    / max(np.linalg.norm(target[off]), np.finfo(float).eps)
                )
                pred_values, pred_vectors = np.linalg.eigh(prediction)
                true_values, true_vectors = np.linalg.eigh(target)
                cosine = float(abs(np.vdot(pred_vectors[:, -1], true_vectors[:, -1])))
                angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))
                metrics.append(
                    (
                        frobenius,
                        angle,
                        abs(effective_rank(prediction) - effective_rank(target)),
                    )
                )
        values = np.asarray(metrics)
        rows.append(
            {
                "pattern": pattern,
                "event_id": str(dataset.event_ids[sample]),
                "route": str(dataset.routes[sample]),
                "held_lags": ",".join(map(str, held_lags)),
                "off_diagonal_frobenius_nrmse": float(values[:, 0].mean()),
                "dominant_eigenspace_angle_deg": float(values[:, 1].mean()),
                "effective_rank_absolute_error": float(values[:, 2].mean()),
            }
        )
    return rows


def _persistence_matrix_rows(dataset, sample_indices: np.ndarray) -> list[dict]:
    rows = []
    for sample in sample_indices:
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            context = loaded["context_anchor"]
            target = loaded["target_anchor"]
        values = []
        for block in range(target.shape[0]):
            for band in range(target.shape[1]):
                prediction_matrix = normalize_coherency(context[block, band])
                target_matrix = normalize_coherency(target[block, band])
                off = ~np.eye(len(target_matrix), dtype=bool)
                frobenius = float(
                    np.linalg.norm((prediction_matrix - target_matrix)[off])
                    / max(np.linalg.norm(target_matrix[off]), np.finfo(float).eps)
                )
                _, pred_vectors = np.linalg.eigh(prediction_matrix)
                _, true_vectors = np.linalg.eigh(target_matrix)
                cosine = float(abs(np.vdot(pred_vectors[:, -1], true_vectors[:, -1])))
                values.append(
                    (
                        frobenius,
                        float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))),
                        abs(effective_rank(prediction_matrix) - effective_rank(target_matrix)),
                    )
                )
        values = np.asarray(values)
        rows.append(
            {
                "pattern": "early-window persistence",
                "event_id": str(dataset.event_ids[sample]),
                "route": str(dataset.routes[sample]),
                "held_lags": "all",
                "off_diagonal_frobenius_nrmse": float(values[:, 0].mean()),
                "dominant_eigenspace_angle_deg": float(values[:, 1].mean()),
                "effective_rank_absolute_error": float(values[:, 2].mean()),
            }
        )
    return rows


def run_withheld_separation(cfg: dict) -> Path:
    amendment, amendment_path = _load_review_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models = _paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    normalized, _, _ = _scaler(dataset, roles)
    base_mask = reliable_mask(cfg)
    development = np.flatnonzero(roles == "model_development")
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    specification = amendment["withheld_separation"]
    search_rows = []
    candidate_models: dict[tuple[str, int, float], nn.Module] = {}
    for pattern, definition in specification["patterns"].items():
        train_lags = list(map(int, definition["train_lag_indices"]))
        test_lags = list(map(int, definition["test_lag_indices"]))
        values = _mask_lag_features(normalized, test_lags)
        train_mask = _held_lag_mask(base_mask, train_lags)
        held_mask = _held_lag_mask(base_mask, test_lags)
        for q_bins in specification["q_bins"]:
            for smoothness in specification["smoothness_weights"]:
                model, record = _fit_torch(
                    str(specification["family"]),
                    values,
                    dataset,
                    development,
                    calibration,
                    train_mask,
                    cfg,
                    amendment,
                    int(amendment["neural"]["search_seed"]),
                    q_bins=int(q_bins),
                    smoothness_weight=float(smoothness),
                )
                device = next(model.parameters()).device
                prediction, _ = _predict(model, torch.from_numpy(values).to(device), calibration)
                held_score = float(_sample_nrmse(prediction, dataset.targets[calibration], held_mask).mean())
                record.update(
                    {
                        "pattern": pattern,
                        "train_lag_indices": ",".join(map(str, train_lags)),
                        "test_lag_indices": ",".join(map(str, test_lags)),
                        "calibration_held_lag_nrmse": held_score,
                    }
                )
                search_rows.append(record)
                candidate_models[(pattern, int(q_bins), float(smoothness))] = model
                print(
                    f"held-lag {pattern} q={q_bins} smooth={smoothness}: {held_score:.4f}",
                    flush=True,
                )
    search = pd.DataFrame(search_rows)
    search.to_csv(reports / "withheld_separation_calibration_search.csv", index=False)
    selection = (
        search.groupby(["q_bins", "smoothness_weight"], as_index=False)
        .calibration_held_lag_nrmse.mean()
        .sort_values("calibration_held_lag_nrmse")
        .iloc[0]
    )
    selected_q = int(selection.q_bins)
    selected_smoothness = float(selection.smoothness_weight)
    freeze = {
        "amendment_sha256": sha256(amendment_path),
        "created_before_withheld_test_metrics": True,
        "selected_q_bins": selected_q,
        "selected_smoothness_weight": selected_smoothness,
        "selection_endpoint": specification["selection_endpoint"],
        "architecture_test_status": amendment["data_boundary"]["architecture_test_status"],
    }
    write_json(reports / "withheld_separation_selection_freeze.json", freeze)

    event_rows = []
    matrix_rows = []
    for pattern, definition in specification["patterns"].items():
        train_lags = list(map(int, definition["train_lag_indices"]))
        test_lags = list(map(int, definition["test_lag_indices"]))
        values = _mask_lag_features(normalized, test_lags)
        train_mask = _held_lag_mask(base_mask, train_lags)
        held_mask = _held_lag_mask(base_mask, test_lags)
        member_predictions, member_probabilities = [], []
        for seed in specification["final_seeds"]:
            model, record = _fit_torch(
                str(specification["family"]),
                values,
                dataset,
                development,
                calibration,
                train_mask,
                cfg,
                amendment,
                int(seed),
                q_bins=selected_q,
                smoothness_weight=selected_smoothness,
            )
            checkpoint = models / f"withheld_{pattern}_q{selected_q}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            device = next(model.parameters()).device
            prediction, probability = _predict(model, torch.from_numpy(values).to(device), test)
            member_predictions.append(prediction)
            member_probabilities.append(probability)
        prediction = np.mean(member_predictions, axis=0)
        probabilities = np.mean(member_probabilities, axis=0)
        nrmse = _sample_nrmse(prediction, dataset.targets[test], held_mask)
        pred_complex = _complex(prediction)
        target_complex = _complex(dataset.targets[test])
        keep = np.broadcast_to(held_mask[None, None], pred_complex.shape)
        for row, sample in enumerate(test):
            sign = np.sign(pred_complex[row].real[keep[row]]) == np.sign(target_complex[row].real[keep[row]])
            event_rows.append(
                {
                    "pattern": pattern,
                    "event_id": str(dataset.event_ids[sample]),
                    "route": str(dataset.routes[sample]),
                    "held_lag_nrmse": float(nrmse[row]),
                    "held_lag_real_sign_accuracy": float(np.mean(sign)),
                }
            )
        matrix_rows.extend(
            _matrix_validation_rows(probabilities, dataset, test, selected_q, pattern, test_lags)
        )
    matrix_rows.extend(_persistence_matrix_rows(dataset, test))
    events = pd.DataFrame(event_rows)
    matrices = pd.DataFrame(matrix_rows)
    events.to_csv(reports / "withheld_separation_event_metrics.csv", index=False)
    matrices.to_csv(reports / "withheld_separation_matrix_metrics.csv", index=False)
    summary = {
        **freeze,
        "held_lag_event_mean_nrmse": float(events.held_lag_nrmse.mean()),
        "held_lag_real_sign_accuracy": float(events.held_lag_real_sign_accuracy.mean()),
        "matrix_metrics": matrices.groupby("pattern").mean(numeric_only=True).reset_index().to_dict("records"),
    }
    path = reports / "withheld_separation_summary.json"
    write_json(path, summary)
    return path


def _read_window(dataset: h5py.Dataset, start: float, stop: float, fs: float, columns: slice) -> np.ndarray:
    first = max(0, int(round(start * fs)))
    last = min(dataset.shape[0], int(round(stop * fs)))
    if last - first < 16:
        raise ValueError("requested rank-control window is outside the record")
    return np.asarray(dataset[first:last, columns], dtype=np.float32)


def _matched_rank_route(row: pd.Series, route: str, cfg: dict, amendment: dict) -> list[dict]:
    path = Path(row["terra_path"] if route == "TERRA" else row["kkfls_path"])
    fs = float(row["terra_sample_rate_hz"] if route == "TERRA" else row["kkfls_sample_rate_hz"])
    spacing = float(row["terra_channel_spacing_m"] if route == "TERRA" else row["kkfls_channel_spacing_m"])
    picks = pd.read_csv(row["picks_path"])
    pick_column = "TERRA_s_sec" if route == "TERRA" else "KKFLS_s_sec"
    pick_channels = picks["channel"].to_numpy(float)
    pick_times = pd.to_numeric(picks[pick_column], errors="coerce").to_numpy(float)
    route_reference = float(np.nanmedian(pick_times))
    starts = list(map(int, cfg["spectral"]["block_starts"]))
    block_channels = int(cfg["spectral"]["block_channels"])
    anchors = np.rint(
        np.linspace(0, block_channels - 1, int(cfg["spectral"]["anchor_channels"]))
    ).astype(int)
    specification = amendment["matched_rank"]
    low, high = map(float, specification["frequency_range_hz"])
    width = float(specification["equal_width_hz"])
    bands = [(edge, edge + width) for edge in np.arange(low, high, width)]
    rows = []
    with h5py.File(path, "r") as handle:
        raw = handle[RAW_DATASET]
        for block_index, block_start in enumerate(starts):
            s_reference, slope, _ = robust_linear_pick_model(
                pick_channels, pick_times, block_start, block_channels
            )
            if not np.isfinite(s_reference):
                s_reference, slope = route_reference, 0.0
            centre = block_start + 0.5 * (block_channels - 1)
            delay = slope * (np.arange(block_start, block_start + block_channels) - centre)
            window = _read_window(
                raw,
                s_reference + float(cfg["windows"]["target_relative_to_s_sec"][0]),
                s_reference + float(cfg["windows"]["target_relative_to_s_sec"][1]),
                fs,
                slice(block_start, block_start + block_channels),
            )
            transformed, frequencies = multitaper_fourier(
                window,
                fs,
                cfg["spectral"]["dpss_time_bandwidth"],
                int(specification["tapers"]),
                delay,
            )
            for band_low, band_high in bands:
                candidates = np.flatnonzero((frequencies >= band_low) & (frequencies < band_high))
                if len(candidates) < int(specification["fourier_bins_per_band"]):
                    continue
                centre_hz = 0.5 * (band_low + band_high)
                order = np.argsort(np.abs(frequencies[candidates] - centre_hz))
                frequency_indices = candidates[order[: int(specification["fourier_bins_per_band"])]]
                snapshots = transformed[:, frequency_indices][:, :, anchors].reshape(-1, len(anchors))
                matrix = normalize_coherency(
                    cross_spectral_matrix(snapshots, float(cfg["spectral"]["diagonal_loading"]))
                )
                rows.append(
                    {
                        "event_id": str(row["event_id"]),
                        "route": route,
                        "block": block_index,
                        "band_low_hz": float(band_low),
                        "band_high_hz": float(band_high),
                        "band_center_hz": float(centre_hz),
                        "fourier_bins": int(len(frequency_indices)),
                        "snapshots": int(snapshots.shape[0]),
                        "effective_rank": float(effective_rank(matrix)),
                        "spacing_m": spacing,
                    }
                )
    return rows


def run_matched_rank(cfg: dict) -> Path:
    amendment, _ = _load_review_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, _ = _paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    test_events = set(dataset.event_ids[roles == "architecture_test"].astype(str))
    cohort = pd.read_parquet(resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet")
    cohort["event_id"] = cohort["event_id"].astype(str)
    selected = cohort[cohort.event_id.isin(test_events)].copy()
    rows = []
    for completed, (_, row) in enumerate(selected.iterrows(), 1):
        for route in ["TERRA", "KKFL-S"]:
            rows.extend(_matched_rank_route(row, route, cfg, amendment))
        print(f"matched-rank {completed}/{len(selected)} earthquakes", flush=True)
    matched = pd.DataFrame(rows)
    matched.to_csv(reports / "matched_bandwidth_effective_rank.csv", index=False)

    octave_rows = []
    test_indices = np.flatnonzero(roles == "architecture_test")
    for sample in test_indices:
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            rank = loaded["target_effective_rank"]
        for block in range(rank.shape[0]):
            for band, bounds in enumerate(dataset.bands):
                octave_rows.append(
                    {
                        "event_id": str(dataset.event_ids[sample]),
                        "route": str(dataset.routes[sample]),
                        "block": block,
                        "band_low_hz": float(bounds[0]),
                        "band_high_hz": float(bounds[1]),
                        "band_center_hz": float(np.sqrt(bounds[0] * bounds[1])),
                        "effective_rank": float(rank[block, band]),
                    }
                )
    octave = pd.DataFrame(octave_rows)
    octave.to_csv(reports / "octave_band_effective_rank.csv", index=False)

    def contrasts(table: pd.DataFrame) -> pd.DataFrame:
        low = table[table.band_center_hz < 2.0].groupby(["event_id", "route"]).effective_rank.mean()
        high = table[table.band_center_hz >= 4.0].groupby(["event_id", "route"]).effective_rank.mean()
        result = pd.concat([low.rename("low"), high.rename("high")], axis=1).dropna().reset_index()
        result["high_minus_low_rank"] = result.high - result.low
        return result

    matched_contrast = contrasts(matched)
    octave_contrast = contrasts(octave)
    matched_contrast.to_csv(reports / "matched_rank_event_contrasts.csv", index=False)
    octave_contrast.to_csv(reports / "octave_rank_event_contrasts.csv", index=False)
    seed = int(amendment["bootstrap"]["seed"])
    matched_event = matched_contrast.groupby("event_id").high_minus_low_rank.mean().to_numpy()
    octave_event = octave_contrast.groupby("event_id").high_minus_low_rank.mean().to_numpy()
    summary = {
        "matched_bandwidth": _event_bootstrap(matched_event, seed, int(amendment["bootstrap"]["replicates"])),
        "octave_band": _event_bootstrap(octave_event, seed + 1, int(amendment["bootstrap"]["replicates"])),
        "matched_frequency_bands": int(matched[["band_low_hz", "band_high_hz"]].drop_duplicates().shape[0]),
        "matched_snapshots_per_estimate": sorted(matched.snapshots.unique().astype(int).tolist()),
        "interpretation_boundary": "The equal-width, equal-Fourier-bin result replaces the octave-band rank interpretation.",
    }
    path = reports / "matched_rank_summary.json"
    write_json(path, summary)
    return path


def _load_member_model(
    checkpoint: Path,
    family: str,
    feature_count: int,
    dataset,
    cfg: dict,
    amendment: dict,
    device: torch.device,
) -> nn.Module:
    model = _model(family, feature_count, dataset, cfg, amendment).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model


def _ensemble_event_uncertainty(
    members: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = members.mean(axis=0)
    error = _sample_nrmse(mean, target, mask)
    complex_members = members[..., 0] + 1j * members[..., 1]
    variance = np.mean(np.abs(complex_members - complex_members.mean(axis=0, keepdims=True)) ** 2, axis=0)
    keep = np.broadcast_to(mask[None, None, :, :], variance.shape)
    spread = np.asarray(
        [float(np.sqrt(np.mean(row[selected]))) for row, selected in zip(variance, keep, strict=True)]
    )
    return error, spread


def run_missingness_and_uncertainty(cfg: dict) -> Path:
    amendment, amendment_path = _load_review_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models = _paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    normalized, _, _ = _scaler(dataset, roles)
    observed = np.ones((*normalized.shape[:2], 1), dtype=np.float32)
    masked_values = np.concatenate([normalized, observed], axis=-1)
    mask = reliable_mask(cfg)
    development = np.flatnonzero(roles == "model_development")
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    specification = amendment["missingness"]
    candidate_rows = []
    for candidate in specification["candidates"]:
        model, record = _fit_torch(
            str(specification["family"]),
            masked_values,
            dataset,
            development,
            calibration,
            mask,
            cfg,
            amendment,
            int(amendment["neural"]["search_seed"]),
            block_dropout_probability=float(candidate["dropout_probability"]),
            maximum_missing_blocks=int(candidate["maximum_missing_blocks"]),
        )
        device = next(model.parameters()).device
        scores = []
        for missing in [0, 1, 2]:
            scenario = _mask_blocks(
                masked_values,
                missing,
                20261000 + missing,
                has_mask_feature=True,
            )
            prediction, _ = _predict(model, torch.from_numpy(scenario).to(device), calibration)
            scores.append(float(_sample_nrmse(prediction, dataset.targets[calibration], mask).mean()))
        candidate_rows.append(
            {
                **candidate,
                "calibration_full_nrmse": scores[0],
                "calibration_one_missing_nrmse": scores[1],
                "calibration_two_missing_nrmse": scores[2],
                "selection_score": float(np.mean(scores)),
                "parameters": record["parameters"],
            }
        )
        print(
            f"mask-aware p={candidate['dropout_probability']} max={candidate['maximum_missing_blocks']}: "
            f"{np.mean(scores):.4f}",
            flush=True,
        )
    candidates = pd.DataFrame(candidate_rows).sort_values("selection_score")
    candidates.to_csv(reports / "missingness_calibration_search.csv", index=False)
    selected = candidates.iloc[0]
    freeze = {
        "amendment_sha256": sha256(amendment_path),
        "created_before_missingness_test_metrics": True,
        "dropout_probability": float(selected.dropout_probability),
        "maximum_missing_blocks": int(selected.maximum_missing_blocks),
        "selection_score": float(selected.selection_score),
        "architecture_test_status": amendment["data_boundary"]["architecture_test_status"],
    }
    write_json(reports / "missingness_selection_freeze.json", freeze)

    scenario_members: dict[tuple[str, int], list[np.ndarray]] = {
        (role, missing): []
        for role in ["calibration", "architecture_test"]
        for missing in map(int, specification["test_missing_blocks"])
    }
    scenario_probabilities: dict[tuple[str, int], list[np.ndarray]] = {
        key: [] for key in scenario_members
    }
    training_rows = []
    for seed in specification["final_seeds"]:
        model, record = _fit_torch(
            str(specification["family"]),
            masked_values,
            dataset,
            development,
            calibration,
            mask,
            cfg,
            amendment,
            int(seed),
            block_dropout_probability=float(selected.dropout_probability),
            maximum_missing_blocks=int(selected.maximum_missing_blocks),
        )
        checkpoint = models / f"mask_aware_{specification['family']}_seed{seed}.pt"
        torch.save(model.state_dict(), checkpoint)
        record["checkpoint_sha256"] = sha256(checkpoint)
        training_rows.append(record)
        device = next(model.parameters()).device
        for role, indices in [("calibration", calibration), ("architecture_test", test)]:
            for missing in map(int, specification["test_missing_blocks"]):
                scenario = _mask_blocks(
                    masked_values,
                    missing,
                    20261100 + missing,
                    has_mask_feature=True,
                )
                prediction, probability = _predict(model, torch.from_numpy(scenario).to(device), indices)
                scenario_members[(role, missing)].append(prediction)
                scenario_probabilities[(role, missing)].append(probability)
    pd.DataFrame(training_rows).to_csv(reports / "missingness_final_training_runs.csv", index=False)

    # Equal-seed comparison with the original hybrid under the identical missing-block patterns.
    extension, _ = _load_extension(cfg)
    original_models = Path(cfg["_root"]) / "models" / "hybrid_search"
    original_members: dict[int, list[np.ndarray]] = {
        missing: [] for missing in map(int, specification["test_missing_blocks"])
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in specification["final_seeds"]:
        original = _load_member_model(
            original_models / f"gat_bissm_psd_seed{seed}.pt",
            "gat_bissm_psd",
            normalized.shape[-1],
            dataset,
            cfg,
            amendment,
            device,
        )
        for missing in map(int, specification["test_missing_blocks"]):
            scenario = _mask_blocks(normalized, missing, 20261100 + missing, False)
            prediction, _ = _predict(original, torch.from_numpy(scenario).to(device), test)
            original_members[missing].append(prediction)

    stress_rows = []
    uncertainty_rows = []
    for role, indices in [("calibration", calibration), ("architecture_test", test)]:
        for missing in map(int, specification["test_missing_blocks"]):
            members = np.stack(scenario_members[(role, missing)])
            probabilities = np.mean(scenario_probabilities[(role, missing)], axis=0)
            mean_prediction = members.mean(axis=0)
            errors, spreads = _ensemble_event_uncertainty(
                members, dataset.targets[indices], mask
            )
            np.save(models / f"mask_aware_{role}_missing{missing}_members.npy", members.astype(np.float32))
            np.save(models / f"mask_aware_{role}_missing{missing}_probabilities.npy", probabilities.astype(np.float32))
            for row, sample in enumerate(indices):
                uncertainty_rows.append(
                    {
                        "role": role,
                        "missing_blocks": missing,
                        "event_id": str(dataset.event_ids[sample]),
                        "route": str(dataset.routes[sample]),
                        "nrmse": float(errors[row]),
                        "ensemble_spread": float(spreads[row]),
                    }
                )
            stress_rows.append(
                {
                    "role": role,
                    "model": "Mask-aware hybrid",
                    "missing_blocks": missing,
                    "mean_nrmse": float(errors.mean()),
                }
            )
            if role == "architecture_test":
                original = np.stack(original_members[missing])
                original_error, _ = _ensemble_event_uncertainty(
                    original, dataset.targets[indices], mask
                )
                stress_rows.append(
                    {
                        "role": role,
                        "model": "Original hybrid",
                        "missing_blocks": missing,
                        "mean_nrmse": float(original_error.mean()),
                    }
                )
    stress = pd.DataFrame(stress_rows).drop_duplicates()
    uncertainty = pd.DataFrame(uncertainty_rows)
    stress.to_csv(reports / "missingness_test_comparison.csv", index=False)
    uncertainty.to_csv(reports / "uncertainty_event_route_metrics.csv", index=False)

    calibration_full = uncertainty[(uncertainty.role == "calibration") & (uncertainty.missing_blocks == 0)]
    test_full = uncertainty[(uncertainty.role == "architecture_test") & (uncertainty.missing_blocks == 0)]
    ratios = calibration_full.nrmse.to_numpy() / np.maximum(
        calibration_full.ensemble_spread.to_numpy(), 1e-8
    )
    conformal_rows = []
    for coverage in amendment["uncertainty"]["conformal_coverages"]:
        n = len(ratios)
        quantile = min(1.0, math.ceil((n + 1) * float(coverage)) / n)
        multiplier = float(np.quantile(ratios, quantile, method="higher"))
        covered = test_full.nrmse.to_numpy() <= multiplier * test_full.ensemble_spread.to_numpy()
        conformal_rows.append(
            {
                "nominal_coverage": float(coverage),
                "calibration_multiplier": multiplier,
                "test_coverage": float(np.mean(covered)),
                "test_records": int(len(covered)),
            }
        )
    pd.DataFrame(conformal_rows).to_csv(reports / "conformal_coverage.csv", index=False)

    selective_rows = []
    for role, part in uncertainty.groupby("role"):
        for missing, scenario in part.groupby("missing_blocks"):
            ordered = scenario.sort_values("ensemble_spread")
            for coverage in amendment["uncertainty"]["selective_coverages"]:
                retained = ordered.iloc[: max(1, int(math.ceil(float(coverage) * len(ordered))))]
                selective_rows.append(
                    {
                        "role": role,
                        "missing_blocks": int(missing),
                        "coverage": float(coverage),
                        "retained_nrmse": float(retained.nrmse.mean()),
                    }
                )
    pd.DataFrame(selective_rows).to_csv(reports / "selective_risk.csv", index=False)
    route_rows = []
    for route, part in test_full.groupby("route"):
        route_rows.append(
            {
                "route": route,
                "records": int(len(part)),
                "spearman_spread_error": float(spearmanr(part.ensemble_spread, part.nrmse).statistic),
                "mean_nrmse": float(part.nrmse.mean()),
                "mean_spread": float(part.ensemble_spread.mean()),
            }
        )
    pd.DataFrame(route_rows).to_csv(reports / "route_uncertainty_diagnostics.csv", index=False)
    overall_rho = float(spearmanr(test_full.ensemble_spread, test_full.nrmse).statistic)
    summary = {
        **freeze,
        "test_full_nrmse": float(test_full.nrmse.mean()),
        "test_spread_error_spearman": overall_rho,
        "conformal": conformal_rows,
        "missingness_comparison": stress[stress.role == "architecture_test"].to_dict("records"),
        "route_shift_boundary": "Only two routes share one interrogator; route-stratified calibration is descriptive, not independent OOD validation.",
    }
    path = reports / "missingness_uncertainty_summary.json"
    write_json(path, summary)
    return path


def _ratio(target: np.ndarray, noise: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.complex128)
    numerator = float(np.real(np.conj(w) @ target @ w))
    denominator = float(np.real(np.conj(w) @ noise @ w))
    return max(numerator, np.finfo(float).eps) / max(denominator, np.finfo(float).eps)


def _generalized_weight(signal: np.ndarray, noise: np.ndarray, loading: float) -> np.ndarray:
    signal = 0.5 * (signal + signal.conj().T)
    noise = 0.5 * (noise + noise.conj().T)
    scale = float(np.real(np.trace(noise)) / len(noise))
    loaded = noise + float(loading) * max(scale, np.finfo(float).eps) * np.eye(len(noise))
    _, vectors = linalg.eigh(signal, loaded, check_finite=False)
    weights = vectors[:, -1]
    return weights / max(np.linalg.norm(weights), np.finfo(float).eps)


def _processing_rows(
    dataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
    q_bins: int,
    shrinkage: float,
    loading: float,
    role: str,
) -> list[dict]:
    q = np.linspace(-np.pi, np.pi, q_bins, endpoint=False)
    rows = []
    for row, sample in enumerate(indices):
        with np.load(dataset.paths[sample], allow_pickle=False) as loaded:
            target = loaded["target_anchor"]
            noise = loaded["noise_anchor"]
            context = loaded["context_anchor"]
            anchors = loaded["anchor_local"]
        for block in range(target.shape[0]):
            for band in range(target.shape[1]):
                rt, rn, rc = target[block, band], noise[block, band], context[block, band]
                correlation = spectral_mixture_operator(probabilities[row, block, band], anchors, q)
                diagonal = np.maximum(np.real(np.diag(rc)), np.finfo(float).eps)
                predicted = np.sqrt(diagonal[:, None] * diagonal[None, :]) * correlation
                predicted = (1.0 - shrinkage) * predicted + shrinkage * np.diag(np.diag(predicted))
                predicted_weight = _generalized_weight(predicted, rn, loading)
                context_weight = _generalized_weight(rc, rn, loading)
                oracle_weight = _generalized_weight(rt, rn, loading)
                equal = np.ones(len(anchors), dtype=float) / np.sqrt(len(anchors))
                proxy = np.real(np.diag(rc)) / np.maximum(np.real(np.diag(rn)), np.finfo(float).eps)
                selected = np.argsort(proxy)[-8:]
                snr = np.zeros(len(anchors), dtype=float)
                snr[selected] = 1.0 / np.sqrt(len(selected))
                weights = {
                    "Predicted covariance GEV": predicted_weight,
                    "Early-context GEV": context_weight,
                    "Early-power ranked 8": snr,
                    "Equal all channels": equal,
                    "Target-window oracle": oracle_weight,
                }
                for method, vector in weights.items():
                    rows.append(
                        {
                            "role": role,
                            "event_id": str(dataset.event_ids[sample]),
                            "route": str(dataset.routes[sample]),
                            "block": block,
                            "band": band,
                            "method": method,
                            "shrinkage": float(shrinkage),
                            "loading": float(loading),
                            "snr_star_db": float(10.0 * np.log10(_ratio(rt, rn, vector))),
                        }
                    )
    return rows


def run_covariance_downstream(cfg: dict) -> Path:
    amendment, amendment_path = _load_review_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models = _paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    probabilities_calibration = np.load(models / "mask_aware_calibration_missing0_probabilities.npy")
    probabilities_test = np.load(models / "mask_aware_architecture_test_missing0_probabilities.npy")
    q_bins = int(cfg["spectral"]["spatial_wavenumber_bins"])
    search_rows = []
    search_summary = []
    for shrinkage in amendment["downstream"]["shrinkage_grid"]:
        for loading in amendment["downstream"]["diagonal_loading_grid"]:
            rows = _processing_rows(
                dataset,
                calibration,
                probabilities_calibration,
                q_bins,
                float(shrinkage),
                float(loading),
                "calibration",
            )
            table = pd.DataFrame(rows)
            search_rows.extend(rows)
            event = table.groupby(["event_id", "route", "method"], as_index=False).snr_star_db.mean()
            wide = event.pivot(index=["event_id", "route"], columns="method", values="snr_star_db")
            delta = wide["Predicted covariance GEV"] - wide["Early-power ranked 8"]
            search_summary.append(
                {
                    "shrinkage": float(shrinkage),
                    "loading": float(loading),
                    "calibration_predicted_minus_power_db": float(delta.mean()),
                }
            )
    pd.DataFrame(search_rows).to_parquet(reports / "covariance_processing_calibration_grid.parquet", index=False)
    search = pd.DataFrame(search_summary).sort_values("calibration_predicted_minus_power_db", ascending=False)
    search.to_csv(reports / "covariance_processing_calibration_search.csv", index=False)
    selected = search.iloc[0]
    freeze = {
        "amendment_sha256": sha256(amendment_path),
        "created_before_covariance_test_metrics": True,
        "selected_shrinkage": float(selected.shrinkage),
        "selected_loading": float(selected.loading),
        "calibration_predicted_minus_power_db": float(selected.calibration_predicted_minus_power_db),
        "architecture_test_status": amendment["data_boundary"]["architecture_test_status"],
    }
    write_json(reports / "covariance_processing_selection_freeze.json", freeze)
    test_rows = _processing_rows(
        dataset,
        test,
        probabilities_test,
        q_bins,
        float(selected.shrinkage),
        float(selected.loading),
        "architecture_test",
    )
    table = pd.DataFrame(test_rows)
    table.to_parquet(reports / "covariance_processing_test_grid.parquet", index=False)
    event = table.groupby(["event_id", "route", "method"], as_index=False).snr_star_db.mean()
    event.to_csv(reports / "covariance_processing_event_metrics.csv", index=False)
    wide = event.pivot(index=["event_id", "route"], columns="method", values="snr_star_db").reset_index()
    comparison_rows = []
    seed = int(amendment["bootstrap"]["seed"])
    for comparator in ["Early-power ranked 8", "Early-context GEV", "Equal all channels"]:
        differences = wide.assign(
            delta=wide["Predicted covariance GEV"] - wide[comparator]
        ).groupby("event_id").delta.mean().to_numpy()
        comparison_rows.append(
            {
                "comparator": comparator,
                **_event_bootstrap(differences, seed, int(amendment["bootstrap"]["replicates"])),
            }
        )
        seed += 1
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(reports / "covariance_processing_bootstrap.csv", index=False)
    summary = {
        **freeze,
        "test_method_means_db": event.groupby("method").snr_star_db.mean().to_dict(),
        "paired_comparisons": comparison_rows,
        "claim_boundary": "This retrospective covariance-sensitive test is not an operational early-warning evaluation.",
    }
    path = reports / "covariance_processing_summary.json"
    write_json(path, summary)
    return path


def run_fair_covariance_audit(cfg: dict) -> Path:
    amendment, _ = _load_review_amendment(cfg)
    correction, correction_path = _load_fair_covariance_amendment(cfg)
    base, _ = _load_amendment(cfg)
    reports, models = _paths(cfg, amendment)
    dataset, roles, _ = _dataset_and_split(cfg, base)
    calibration = np.flatnonzero(roles == "calibration")
    test = np.flatnonzero(roles == "architecture_test")
    probabilities_calibration = np.load(models / "mask_aware_calibration_missing0_probabilities.npy")
    probabilities_test = np.load(models / "mask_aware_architecture_test_missing0_probabilities.npy")
    q_bins = int(cfg["spectral"]["spatial_wavenumber_bins"])

    calibration_grid = pd.read_parquet(reports / "covariance_processing_calibration_grid.parquet")
    predicted = calibration_grid[calibration_grid.method == "Predicted covariance GEV"]
    event = predicted.groupby(
        ["shrinkage", "loading", "event_id", "route"], as_index=False
    ).snr_star_db.mean()
    diagonal = event[event.shrinkage == float(correction["comparison"]["diagonal_reference_shrinkage"])][
        ["loading", "event_id", "route", "snr_star_db"]
    ].rename(columns={"snr_star_db": "diagonal_snr_star_db"})
    comparison = event[event.shrinkage.isin(correction["comparison"]["predicted_candidates_shrinkage"])].merge(
        diagonal, on=["loading", "event_id", "route"], validate="many_to_one"
    )
    comparison["predicted_minus_diagonal_db"] = comparison.snr_star_db - comparison.diagonal_snr_star_db
    search = (
        comparison.groupby(["shrinkage", "loading"], as_index=False)
        .predicted_minus_diagonal_db.mean()
        .sort_values("predicted_minus_diagonal_db", ascending=False)
    )
    search.to_csv(reports / "fair_covariance_calibration_search.csv", index=False)
    selected = search.iloc[0]
    freeze = {
        "amendment_sha256": sha256(correction_path),
        "post_hoc_correction": True,
        "created_before_fair_covariance_test_metrics": True,
        "selected_predicted_shrinkage": float(selected.shrinkage),
        "selected_loading": float(selected.loading),
        "diagonal_reference_shrinkage": float(correction["comparison"]["diagonal_reference_shrinkage"]),
        "calibration_predicted_minus_diagonal_db": float(selected.predicted_minus_diagonal_db),
    }
    write_json(reports / "fair_covariance_selection_freeze.json", freeze)
    predicted_rows = _processing_rows(
        dataset,
        test,
        probabilities_test,
        q_bins,
        float(selected.shrinkage),
        float(selected.loading),
        "architecture_test",
    )
    diagonal_rows = _processing_rows(
        dataset,
        test,
        probabilities_test,
        q_bins,
        float(correction["comparison"]["diagonal_reference_shrinkage"]),
        float(selected.loading),
        "architecture_test",
    )
    pred_table = pd.DataFrame(predicted_rows)
    diag_table = pd.DataFrame(diagonal_rows)
    keys = ["event_id", "route", "block", "band", "method"]
    pred_method = pred_table[pred_table.method == "Predicted covariance GEV"]
    diag_method = diag_table[diag_table.method == "Predicted covariance GEV"]
    paired = pred_method.merge(
        diag_method[keys + ["snr_star_db"]].rename(columns={"snr_star_db": "diagonal_snr_star_db"}),
        on=keys,
        validate="one_to_one",
    )
    paired["predicted_minus_diagonal_db"] = paired.snr_star_db - paired.diagonal_snr_star_db
    paired.to_parquet(reports / "fair_covariance_test_grid.parquet", index=False)
    event_test = paired.groupby(["event_id", "route"], as_index=False).predicted_minus_diagonal_db.mean()
    event_test.to_csv(reports / "fair_covariance_event_metrics.csv", index=False)
    complete_event = event_test.groupby("event_id").predicted_minus_diagonal_db.mean().to_numpy()
    bootstrap = _event_bootstrap(
        complete_event,
        int(correction["bootstrap"]["seed"]),
        int(correction["bootstrap"]["replicates"]),
    )
    summary = {
        **freeze,
        "architecture_test": bootstrap,
        "off_diagonal_prediction_adds_value": bool(bootstrap["ci_low"] > 0),
        "interpretation": "The diagonal-only reference uses the same 32 channels, noise matrix, loading and eigensolver.",
    }
    path = reports / "fair_covariance_summary.json"
    write_json(path, summary)
    return path


def record_2024_access_audit(cfg: dict) -> Path:
    amendment, _ = _load_review_amendment(cfg)
    reports, _ = _paths(cfg, amendment)
    record = {
        "checked_utc": "2026-08-30T15:40:54Z",
        "official_index": "https://dasway.ess.washington.edu/gci/index.html",
        "official_index_status": 200,
        "daily_index": "https://dasway.ess.washington.edu/gci/events/2024-05-28/index.html",
        "daily_index_status": 200,
        "event_id": "11846071",
        "terra_hdf5_status": 403,
        "kkfln_hdf5_status": 403,
        "quakeml_status": 200,
        "kkfls_listed": False,
        "conclusion": "No untouched 2024 waveform evaluation can be executed from anonymous public object reads.",
    }
    path = reports / "official_2024_access_audit.json"
    write_json(path, record)
    return path


def run_review_revision(cfg: dict) -> Path:
    amendment, _ = _load_review_amendment(cfg)
    reports, _ = _paths(cfg, amendment)
    outputs = {
        "access_audit": str(record_2024_access_audit(cfg)),
        "baseline_benchmark": str(run_baseline_benchmark(cfg)),
        "withheld_separation": str(run_withheld_separation(cfg)),
        "matched_rank": str(run_matched_rank(cfg)),
        "missingness_uncertainty": str(run_missingness_and_uncertainty(cfg)),
        "covariance_downstream": str(run_covariance_downstream(cfg)),
        "fair_covariance_audit": str(run_fair_covariance_audit(cfg)),
    }
    path = reports / "deep_review_revision_summary.json"
    write_json(path, outputs)
    return path
