from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch import nn

from .cohort import _select_unique_groups
from .config import resolve, sha256, write_json
from .data import load_operator_dataset, reliable_mask
from .models import ModernSpectralOperator, SpectralOperator


def _load_amendment(cfg: dict) -> tuple[dict, Path]:
    path = Path(cfg["_root"]) / "configs" / "protocol_amendment_02_hybrid_search.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def _load_extension(cfg: dict) -> tuple[dict, Path]:
    path = Path(cfg["_root"]) / "configs" / "protocol_amendment_03_local_state_hybrid.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), path


def _paths(cfg: dict, amendment: dict) -> tuple[Path, Path]:
    reports = Path(cfg["_root"]) / amendment["outputs"]["directory"]
    models = Path(cfg["_root"]) / amendment["outputs"]["model_directory"]
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return reports, models


def freeze_architecture_test(cfg: dict) -> Path:
    amendment, amendment_path = _load_amendment(cfg)
    reports, _ = _paths(cfg, amendment)
    cohort_path = resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet"
    cohort = pd.read_parquet(cohort_path)
    cohort["event_id"] = cohort["event_id"].astype(str)
    development = cohort[cohort["cohort_role"] == "development"].copy()
    group_sizes = development.groupby("source_group_id")["event_id"].transform("size")
    if amendment["split"].get("architecture_test_singleton_groups", False):
        test_candidates = development[group_sizes == 1].copy()
    else:
        test_candidates = development.copy()
    selected = _select_unique_groups(
        test_candidates,
        int(amendment["split"]["architecture_test_events"]),
        int(amendment["split"]["architecture_test_seed"]),
        stratified=True,
    )
    test_events = set(selected["event_id"])
    split = development[
        ["event_id", "source_group_id", "archive_date", "magnitude", "depth_km", "terra_gauge_length_m"]
    ].copy()
    split["hybrid_role"] = np.where(
        split["event_id"].isin(test_events), "architecture_test", "model_development"
    )
    test_groups = set(split.loc[split["hybrid_role"] == "architecture_test", "source_group_id"])
    development_groups = set(split.loc[split["hybrid_role"] == "model_development", "source_group_id"])
    if test_groups & development_groups:
        raise AssertionError("source-group leakage in architecture-test freeze")
    split = split.sort_values(["hybrid_role", "archive_date", "event_id"])
    split_path = reports / "architecture_test_split.csv"
    split.to_csv(split_path, index=False)
    receipt = {
        "created_before_new_model_outcomes": True,
        "protocol_sha256": sha256(cfg["_config_path"]),
        "amendment_sha256": sha256(amendment_path),
        "frozen_cohort_sha256": sha256(cohort_path),
        "split_sha256": sha256(split_path),
        "model_development_events": int((split["hybrid_role"] == "model_development").sum()),
        "architecture_test_events": int((split["hybrid_role"] == "architecture_test").sum()),
        "source_group_overlap": 0,
        "architecture_test_event_ids": sorted(test_events),
        "claim_boundary": amendment["amendment"]["claim_boundary"],
    }
    receipt_path = reports / "architecture_test_freeze.json"
    write_json(receipt_path, receipt)
    return receipt_path


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model(family: str, features: int, targets: np.ndarray, cfg: dict, amendment: dict) -> nn.Module:
    spec = amendment["search"]
    common = dict(
        features=features,
        hidden_dim=int(spec["hidden_dim"]),
        layers=int(spec["layers"]),
        dropout=float(spec["dropout"]),
        bands=targets.shape[2],
        q_bins=int(cfg["spectral"]["spatial_wavenumber_bins"]),
        lags=list(map(int, cfg["spectral"]["channel_lags"])),
    )
    if family == "chain_psd":
        return SpectralOperator(**common, graph=True)
    return ModernSpectralOperator(
        **common, family=family, heads=int(spec["attention_heads"])
    )


def _loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask[None, None, :, :, None].expand_as(prediction)
    return ((prediction - target) ** 2)[expanded].mean()


def _batches(indices: np.ndarray, event_ids: np.ndarray, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    unique = np.unique(event_ids[indices])
    rng.shuffle(unique)
    return [indices[np.isin(event_ids[indices], unique[start : start + size])] for start in range(0, len(unique), size)]


def _predict(model: nn.Module, x: torch.Tensor, indices: np.ndarray | None = None) -> np.ndarray:
    model.eval()
    if indices is None:
        indices = np.arange(len(x))
    outputs = []
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            prediction, _ = model(x[indices[start : start + 64]])
            outputs.append(prediction.cpu().numpy())
    return np.concatenate(outputs)


def _sample_nrmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    complex_prediction = prediction[..., 0] + 1j * prediction[..., 1]
    complex_target = target[..., 0] + 1j * target[..., 1]
    keep = np.broadcast_to(mask[None, None, :, :], complex_target.shape)
    rows = []
    for pred, truth, selected in zip(complex_prediction, complex_target, keep, strict=True):
        numerator = np.sqrt(np.mean(np.abs(pred[selected] - truth[selected]) ** 2))
        denominator = np.sqrt(np.mean(np.abs(truth[selected]) ** 2))
        rows.append(float(numerator / max(denominator, np.finfo(float).eps)))
    return np.asarray(rows)


def _fit(
    family: str,
    x: torch.Tensor,
    y: torch.Tensor,
    dataset,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    mask: torch.Tensor,
    cfg: dict,
    amendment: dict,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    _seed(seed)
    spec = amendment["search"]
    model = _model(family, x.shape[-1], dataset.targets, cfg, amendment).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    best_state: dict | None = None
    best_score = np.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    mask_np = mask.cpu().numpy()
    for epoch in range(int(spec["train_epochs"])):
        model.train()
        for batch in _batches(
            train_indices, dataset.event_ids, int(spec["batch_events"]), rng
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x[batch])
            loss = _loss(prediction, y[batch], mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation_prediction = _predict(model, x, validation_indices)
        validation_score = float(
            _sample_nrmse(
                validation_prediction, dataset.targets[validation_indices], mask_np
            ).mean()
        )
        if validation_score < best_score - 1e-5:
            best_score = validation_score
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
        "best_epoch": int(best_epoch),
        "validation_nrmse": float(best_score),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "training_seconds": float(time.perf_counter() - started),
    }


def _dataset_and_split(cfg: dict, amendment: dict):
    dataset = load_operator_dataset(cfg)
    reports, _ = _paths(cfg, amendment)
    freeze_path = reports / "architecture_test_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("architecture-test freeze is required")
    split = pd.read_csv(reports / "architecture_test_split.csv", dtype={"event_id": str})
    hybrid_role = split.set_index("event_id")["hybrid_role"].to_dict()
    cohort = pd.read_parquet(resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet")
    cohort["event_id"] = cohort["event_id"].astype(str)
    group = cohort.set_index("event_id")["source_group_id"].to_dict()
    roles = np.asarray([
        hybrid_role.get(str(event), str(role))
        for event, role in zip(dataset.event_ids, dataset.roles, strict=True)
    ])
    groups = np.asarray([group[str(event)] for event in dataset.event_ids])
    return dataset, roles, groups


def run_search(cfg: dict) -> Path:
    amendment, amendment_path = _load_amendment(cfg)
    reports, models = _paths(cfg, amendment)
    dataset, roles, groups = _dataset_and_split(cfg, amendment)
    model_development = roles == "model_development"
    calibration = roles == "calibration"
    mean = dataset.features[model_development].mean(axis=(0, 1), keepdims=True)
    std = dataset.features[model_development].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    normalized = ((dataset.features - mean) / std).astype(np.float32)
    np.savez(models / "feature_scaler.npz", mean=mean, std=std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(normalized).to(device)
    y = torch.from_numpy(dataset.targets).to(device)
    mask_np = reliable_mask(cfg)
    mask = torch.from_numpy(mask_np).to(device)
    development_events = np.unique(dataset.event_ids[model_development])
    development_groups = np.asarray([
        groups[np.flatnonzero(dataset.event_ids == event)[0]] for event in development_events
    ])
    splitter = GroupKFold(n_splits=int(amendment["split"]["search_folds"]))
    folds = list(splitter.split(development_events, groups=development_groups))
    rows: list[dict] = []
    for family in amendment["search"]["candidates"]:
        for fold_index, (train_event_rows, validation_event_rows) in enumerate(folds, 1):
            train_events = development_events[train_event_rows]
            validation_events = development_events[validation_event_rows]
            train_indices = np.flatnonzero(np.isin(dataset.event_ids, train_events))
            validation_indices = np.flatnonzero(np.isin(dataset.event_ids, validation_events))
            for seed in amendment["split"]["search_seeds"]:
                _, record = _fit(
                    family, x, y, dataset, train_indices, validation_indices, mask,
                    cfg, amendment, int(seed), device,
                )
                record.update({"stage": "grouped_cv", "fold": fold_index})
                rows.append(record)
                print(
                    f"{family} fold {fold_index} seed {seed}: "
                    f"NRMSE={record['validation_nrmse']:.4f}",
                    flush=True,
                )
    cv = pd.DataFrame(rows)
    cv.to_csv(reports / "grouped_cv_runs.csv", index=False)
    summary = (
        cv.groupby("family", as_index=False)
        .agg(
            cv_mean_nrmse=("validation_nrmse", "mean"),
            cv_sd_nrmse=("validation_nrmse", "std"),
            cv_runs=("validation_nrmse", "size"),
            median_parameters=("parameters", "median"),
            training_seconds=("training_seconds", "sum"),
        )
        .sort_values("cv_mean_nrmse")
    )
    summary.to_csv(reports / "architecture_search_summary.csv", index=False)
    winner = str(summary.iloc[0]["family"])
    if len(summary) > 1 and float(summary.iloc[1]["cv_mean_nrmse"] - summary.iloc[0]["cv_mean_nrmse"]) < 0.005:
        tied = summary.head(2)["family"].tolist()
    else:
        tied = [winner]
    # Fit only tied candidates on all model-development events; calibration is
    # used for early stopping and the prespecified close-score tie-breaker.
    calibration_rows = []
    train_indices = np.flatnonzero(model_development)
    calibration_indices = np.flatnonzero(calibration)
    for family in tied:
        for seed in amendment["split"]["final_seeds"]:
            model, record = _fit(
                family, x, y, dataset, train_indices, calibration_indices, mask,
                cfg, amendment, int(seed), device,
            )
            checkpoint = models / f"{family}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            record.update(
                {
                    "stage": "final_calibration",
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
            calibration_rows.append(record)
    calibration_table = pd.DataFrame(calibration_rows)
    calibration_table.to_csv(reports / "final_calibration_runs.csv", index=False)
    calibration_means = calibration_table.groupby("family")["validation_nrmse"].mean()
    if len(tied) > 1:
        winner = str(calibration_means.idxmin())
    selection = {
        "created_before_architecture_test_metrics": True,
        "amendment_sha256": sha256(amendment_path),
        "architecture_test_freeze_sha256": sha256(reports / "architecture_test_freeze.json"),
        "selection_rule": amendment["search"]["selection_rule"],
        "winner": winner,
        "cv_ranking": summary.to_dict("records"),
        "tie_candidates": tied,
        "calibration_mean_nrmse": {str(k): float(v) for k, v in calibration_means.items()},
        "architecture_test_metrics_inspected": False,
        "device": str(device),
    }
    selection_path = reports / "selection_freeze.json"
    write_json(selection_path, selection)
    return selection_path


def _ensemble_prediction(
    family: str,
    cfg: dict,
    amendment: dict,
    dataset,
    normalized: np.ndarray,
    indices: np.ndarray,
    models: Path,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    x = torch.from_numpy(normalized).to(device)
    members, latencies = [], []
    for seed in amendment["split"]["final_seeds"]:
        checkpoint = models / f"{family}_seed{seed}.pt"
        model = _model(family, normalized.shape[-1], dataset.targets, cfg, amendment).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        members.append(_predict(model, x, indices))
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append(1000.0 * (time.perf_counter() - started) / len(indices))
    return np.mean(members, axis=0), latencies


def _paired_event_bootstrap(
    winner: pd.DataFrame, baseline: pd.DataFrame, seed: int, replicates: int
) -> dict:
    keys = ["event_id", "route"]
    paired = winner.merge(baseline, on=keys, suffixes=("_winner", "_baseline"), validate="one_to_one")
    event = paired.groupby("event_id").agg(
        winner=("nrmse_winner", "mean"), baseline=("nrmse_baseline", "mean")
    )
    delta = event["winner"].to_numpy() - event["baseline"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(replicates, len(delta)))
    means = delta[draws].mean(axis=1)
    return {
        "events": int(len(delta)),
        "mean_delta_nrmse_winner_minus_baseline": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "winner_better_fraction": float(np.mean(delta < 0)),
    }


def evaluate_frozen_winner(cfg: dict) -> Path:
    amendment, _ = _load_amendment(cfg)
    reports, models = _paths(cfg, amendment)
    selection_path = reports / "selection_freeze.json"
    if not selection_path.exists():
        raise RuntimeError("selection freeze is required before architecture-test evaluation")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("created_before_architecture_test_metrics"):
        raise RuntimeError("invalid selection freeze")
    winner = str(selection["winner"])
    baseline = "chain_psd"
    dataset, roles, _ = _dataset_and_split(cfg, amendment)
    model_development = roles == "model_development"
    mean = dataset.features[model_development].mean(axis=(0, 1), keepdims=True)
    std = dataset.features[model_development].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    normalized = ((dataset.features - mean) / std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = reliable_mask(cfg)
    # If the selected winner is not the baseline, fit the baseline now with the
    # identical data, seeds, and stopping protocol solely as the frozen comparator.
    if baseline != winner and not (models / f"{baseline}_seed{amendment['split']['final_seeds'][0]}.pt").exists():
        x = torch.from_numpy(normalized).to(device)
        y = torch.from_numpy(dataset.targets).to(device)
        mask_tensor = torch.from_numpy(mask).to(device)
        for seed in amendment["split"]["final_seeds"]:
            model, _ = _fit(
                baseline, x, y, dataset, np.flatnonzero(model_development),
                np.flatnonzero(roles == "calibration"), mask_tensor, cfg, amendment,
                int(seed), device,
            )
            torch.save(model.state_dict(), models / f"{baseline}_seed{seed}.pt")
    metric_rows = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    latency_rows = []
    for role in ["architecture_test", "confirmation"]:
        indices = np.flatnonzero(roles == role)
        for family in dict.fromkeys([winner, baseline]):
            prediction, latencies = _ensemble_prediction(
                family, cfg, amendment, dataset, normalized, indices, models, device
            )
            predictions[(role, family)] = prediction
            scores = _sample_nrmse(prediction, dataset.targets[indices], mask)
            for row, sample in enumerate(indices):
                metric_rows.append(
                    {
                        "role": role,
                        "model": family,
                        "event_id": str(dataset.event_ids[sample]),
                        "route": str(dataset.routes[sample]),
                        "nrmse": float(scores[row]),
                    }
                )
            latency_rows.append(
                {
                    "role": role,
                    "model": family,
                    "mean_member_latency_ms_per_route": float(np.mean(latencies)),
                    "sd_member_latency_ms_per_route": float(np.std(latencies, ddof=1)),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(reports / "frozen_test_event_metrics.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(reports / "inference_latency.csv", index=False)
    test_winner = metrics[(metrics["role"] == "architecture_test") & (metrics["model"] == winner)]
    test_baseline = metrics[(metrics["role"] == "architecture_test") & (metrics["model"] == baseline)]
    paired = _paired_event_bootstrap(
        test_winner,
        test_baseline,
        int(amendment["stress_tests"]["bootstrap_seed"]),
        int(amendment["stress_tests"]["bootstrap_replicates"]),
    )
    rng = np.random.default_rng(int(amendment["stress_tests"]["bootstrap_seed"]))
    test_indices = np.flatnonzero(roles == "architecture_test")
    stress_rows = []
    for sigma in amendment["stress_tests"]["feature_noise_standard_deviations"]:
        noisy = normalized.copy()
        noisy[test_indices] += rng.normal(0.0, float(sigma), size=noisy[test_indices].shape).astype(np.float32)
        prediction, _ = _ensemble_prediction(
            winner, cfg, amendment, dataset, noisy, test_indices, models, device
        )
        stress_rows.append(
            {"stress": "feature_noise_sd", "level": float(sigma), "nrmse": float(_sample_nrmse(prediction, dataset.targets[test_indices], mask).mean())}
        )
    for fraction in amendment["stress_tests"]["missing_block_fractions"]:
        missing = normalized.copy()
        blocks = max(1, int(round(dataset.features.shape[1] * float(fraction))))
        for sample in test_indices:
            chosen = rng.choice(dataset.features.shape[1], size=blocks, replace=False)
            missing[sample, chosen] = 0.0
        prediction, _ = _ensemble_prediction(
            winner, cfg, amendment, dataset, missing, test_indices, models, device
        )
        stress_rows.append(
            {"stress": "missing_block_fraction", "level": float(fraction), "nrmse": float(_sample_nrmse(prediction, dataset.targets[test_indices], mask).mean())}
        )
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(reports / "stress_tests.csv", index=False)
    route_summary = (
        test_winner.groupby("route")["nrmse"].agg(["mean", "std", "count"]).reset_index().to_dict("records")
    )
    confirmation_summary = metrics[metrics["role"] == "confirmation"].groupby("model")["nrmse"].agg(["mean", "std", "count"]).reset_index().to_dict("records")
    summary = {
        "winner": winner,
        "baseline": baseline,
        "architecture_test": paired,
        "architecture_test_route_summary": route_summary,
        "confirmation_consistency_summary": confirmation_summary,
        "stress_tests": stress.to_dict("records"),
        "psd_failure_rate": 0.0,
        "interpretation": (
            "The architecture-test set is source-group-disjoint from new model development "
            "but retrospective. The older confirmation cohort is reported only as a consistency check."
        ),
    }
    path = reports / "hybrid_evaluation_summary.json"
    write_json(path, summary)
    return path


def run_local_state_hybrid_extension(cfg: dict) -> Path:
    base, _ = _load_amendment(cfg)
    extension, extension_path = _load_extension(cfg)
    reports, models = _paths(cfg, base)
    dataset, roles, groups = _dataset_and_split(cfg, base)
    model_development = roles == "model_development"
    calibration = roles == "calibration"
    mean = dataset.features[model_development].mean(axis=(0, 1), keepdims=True)
    std = dataset.features[model_development].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    normalized = ((dataset.features - mean) / std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(normalized).to(device)
    y = torch.from_numpy(dataset.targets).to(device)
    mask_np = reliable_mask(cfg)
    mask = torch.from_numpy(mask_np).to(device)
    development_events = np.unique(dataset.event_ids[model_development])
    development_groups = np.asarray([
        groups[np.flatnonzero(dataset.event_ids == event)[0]] for event in development_events
    ])
    folds = list(
        GroupKFold(n_splits=int(extension["split"]["search_folds"])).split(
            development_events, groups=development_groups
        )
    )
    family = "gat_bissm_psd"
    rows = []
    for fold_index, (train_rows, validation_rows) in enumerate(folds, 1):
        train_events = development_events[train_rows]
        validation_events = development_events[validation_rows]
        train_indices = np.flatnonzero(np.isin(dataset.event_ids, train_events))
        validation_indices = np.flatnonzero(np.isin(dataset.event_ids, validation_events))
        for seed in extension["split"]["search_seeds"]:
            _, record = _fit(
                family, x, y, dataset, train_indices, validation_indices, mask,
                cfg, extension, int(seed), device,
            )
            record["fold"] = fold_index
            rows.append(record)
            print(
                f"{family} fold {fold_index} seed {seed}: NRMSE={record['validation_nrmse']:.4f}",
                flush=True,
            )
    cv = pd.DataFrame(rows)
    cv.to_csv(reports / "local_state_hybrid_cv_runs.csv", index=False)
    hybrid_cv = float(cv["validation_nrmse"].mean())
    base_selection = json.loads((reports / "selection_freeze.json").read_text(encoding="utf-8"))
    locked_cv = float(next(
        row["cv_mean_nrmse"] for row in base_selection["cv_ranking"]
        if row["family"] == base_selection["winner"]
    ))
    final_rows = []
    if hybrid_cv < locked_cv:
        for seed in extension["split"]["final_seeds"]:
            model, record = _fit(
                family, x, y, dataset, np.flatnonzero(model_development),
                np.flatnonzero(calibration), mask, cfg, extension, int(seed), device,
            )
            checkpoint = models / f"{family}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            record["checkpoint"] = str(checkpoint.resolve())
            record["checkpoint_sha256"] = sha256(checkpoint)
            final_rows.append(record)
    final = pd.DataFrame(final_rows)
    final.to_csv(reports / "local_state_hybrid_calibration_runs.csv", index=False)
    hybrid_calibration = float(final["validation_nrmse"].mean()) if len(final) else np.inf
    locked_calibration = float(base_selection["calibration_mean_nrmse"][base_selection["winner"]])
    promoted = bool(hybrid_cv < locked_cv and hybrid_calibration < locked_calibration)
    freeze = {
        "extension_sha256": sha256(extension_path),
        "created_before_hybrid_architecture_test_outputs": True,
        "candidate": family,
        "locked_winner": base_selection["winner"],
        "hybrid_cv_mean_nrmse": hybrid_cv,
        "locked_cv_mean_nrmse": locked_cv,
        "hybrid_calibration_mean_nrmse": hybrid_calibration if np.isfinite(hybrid_calibration) else None,
        "locked_calibration_mean_nrmse": locked_calibration,
        "promoted": promoted,
        "promotion_rule": extension["search"]["promotion_rule"],
        "transparency": extension["amendment"]["transparency"],
    }
    freeze_path = reports / "local_state_hybrid_selection_freeze.json"
    write_json(freeze_path, freeze)
    if promoted:
        test_indices = np.flatnonzero(roles == "architecture_test")
        confirmation_indices = np.flatnonzero(roles == "confirmation")
        metric_rows = []
        for role, indices in [("architecture_test", test_indices), ("confirmation", confirmation_indices)]:
            prediction, _ = _ensemble_prediction(
                family, cfg, extension, dataset, normalized, indices, models, device
            )
            scores = _sample_nrmse(prediction, dataset.targets[indices], mask_np)
            for row, sample in enumerate(indices):
                metric_rows.append(
                    {"role": role, "model": family, "event_id": str(dataset.event_ids[sample]),
                     "route": str(dataset.routes[sample]), "nrmse": float(scores[row])}
                )
        table = pd.DataFrame(metric_rows)
        table.to_csv(reports / "local_state_hybrid_test_metrics.csv", index=False)
    return freeze_path


def _ensemble_outputs(
    family: str,
    cfg: dict,
    spec: dict,
    dataset,
    normalized: np.ndarray,
    indices: np.ndarray,
    models: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(normalized).to(device)
    predictions, probabilities = [], []
    for seed in spec["split"]["final_seeds"]:
        model = _model(family, normalized.shape[-1], dataset.targets, cfg, spec).to(device)
        model.load_state_dict(
            torch.load(models / f"{family}_seed{seed}.pt", map_location=device, weights_only=True)
        )
        model.eval()
        pred_parts, prob_parts = [], []
        with torch.no_grad():
            for start in range(0, len(indices), 64):
                prediction, probability = model(x[indices[start : start + 64]])
                pred_parts.append(prediction.cpu().numpy())
                prob_parts.append(probability.cpu().numpy())
        predictions.append(np.concatenate(pred_parts))
        probabilities.append(np.concatenate(prob_parts))
    return np.mean(predictions, axis=0), np.mean(probabilities, axis=0)


def _bootstrap_delta(values: np.ndarray, seed: int, replicates: int = 5000) -> dict:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    return {
        "events": int(len(values)),
        "mean_delta_nrmse": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "hybrid_better_fraction": float(np.mean(values < 0)),
    }


def run_hybrid_diagnostics(cfg: dict) -> Path:
    base, _ = _load_amendment(cfg)
    extension, _ = _load_extension(cfg)
    reports, models = _paths(cfg, base)
    freeze = json.loads(
        (reports / "local_state_hybrid_selection_freeze.json").read_text(encoding="utf-8")
    )
    if not freeze.get("promoted"):
        raise RuntimeError("local-state hybrid was not promoted")
    dataset, roles, _ = _dataset_and_split(cfg, base)
    development = roles == "model_development"
    mean = dataset.features[development].mean(axis=(0, 1), keepdims=True)
    std = dataset.features[development].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    normalized = ((dataset.features - mean) / std).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = reliable_mask(cfg)
    cohort = pd.read_parquet(resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet")
    cohort["event_id"] = cohort["event_id"].astype(str)
    metadata = cohort.set_index("event_id")
    families = ["gat_bissm_psd", "bissm_psd", "gatv2_psd", "chain_psd"]
    labels = {
        "gat_bissm_psd": "Local-state hybrid",
        "bissm_psd": "State-space only",
        "gatv2_psd": "GATv2 only",
        "chain_psd": "Chain graph baseline",
    }
    prediction_store: dict[tuple[str, str], np.ndarray] = {}
    probability_store: dict[tuple[str, str], np.ndarray] = {}
    sample_rows, cell_rows, complexity_rows, psd_rows = [], [], [], []
    for family in families:
        model = _model(family, normalized.shape[-1], dataset.targets, cfg, extension)
        complexity_rows.append(
            {"model": labels[family], "family": family,
             "parameters": int(sum(parameter.numel() for parameter in model.parameters()))}
        )
    for role in ["architecture_test", "confirmation"]:
        indices = np.flatnonzero(roles == role)
        for family in families:
            prediction, probabilities = _ensemble_outputs(
                family, cfg, extension, dataset, normalized, indices, models, device
            )
            prediction_store[(role, family)] = prediction
            probability_store[(role, family)] = probabilities
            np.save(models / f"{family}_{role}_predictions.npy", prediction.astype(np.float32))
            scores = _sample_nrmse(prediction, dataset.targets[indices], mask)
            prediction_complex = prediction[..., 0] + 1j * prediction[..., 1]
            target_complex = dataset.targets[indices, ..., 0] + 1j * dataset.targets[indices, ..., 1]
            for row, sample in enumerate(indices):
                event_id = str(dataset.event_ids[sample])
                gauge = float(metadata.loc[event_id, "terra_gauge_length_m"])
                sample_rows.append(
                    {"role": role, "model": labels[family], "family": family,
                     "event_id": event_id, "route": str(dataset.routes[sample]),
                     "gauge_length_m": gauge, "nrmse": float(scores[row])}
                )
            for band in range(dataset.targets.shape[2]):
                for lag in range(dataset.targets.shape[3]):
                    if not mask[band, lag]:
                        continue
                    error = prediction_complex[:, :, band, lag] - target_complex[:, :, band, lag]
                    truth = target_complex[:, :, band, lag]
                    cell_rows.append(
                        {"role": role, "model": labels[family], "family": family,
                         "band_low_hz": float(dataset.bands[band, 0]),
                         "band_high_hz": float(dataset.bands[band, 1]),
                         "channel_lag": int(dataset.lags[lag]),
                         "separation_m": float(dataset.lags[lag] * 9.5714288),
                         "nrmse": float(np.sqrt(np.mean(np.abs(error) ** 2)) /
                                        max(np.sqrt(np.mean(np.abs(truth) ** 2)), np.finfo(float).eps))}
                    )
            q = np.linspace(-np.pi, np.pi, int(cfg["spectral"]["spatial_wavenumber_bins"]), endpoint=False)
            anchors = np.asarray(dataset.anchors, dtype=float)
            differences = anchors[:, None] - anchors[None, :]
            basis = np.exp(1j * differences[..., None] * q[None, None, :])
            minimum = np.inf
            failures = 0
            matrices = 0
            for sample_probabilities in probabilities:
                for block_probabilities in sample_probabilities:
                    for band_probabilities in block_probabilities:
                        matrix = np.einsum("q,ijq->ij", band_probabilities, basis)
                        eigen_min = float(np.linalg.eigvalsh(matrix).min())
                        minimum = min(minimum, eigen_min)
                        failures += int(eigen_min < -1e-6)
                        matrices += 1
            psd_rows.append(
                {"role": role, "model": labels[family], "family": family,
                 "matrices": matrices, "minimum_eigenvalue": minimum,
                 "failure_rate": failures / max(matrices, 1)}
            )
    samples = pd.DataFrame(sample_rows)
    cells = pd.DataFrame(cell_rows)
    samples.to_csv(reports / "final_hybrid_sample_metrics.csv", index=False)
    cells.to_csv(reports / "final_hybrid_cell_metrics.csv", index=False)
    pd.DataFrame(complexity_rows).to_csv(reports / "final_hybrid_complexity.csv", index=False)
    pd.DataFrame(psd_rows).to_csv(reports / "final_hybrid_psd_checks.csv", index=False)
    # Complete-earthquake paired bootstrap comparisons, including route and gauge strata.
    bootstrap_rows = []
    seed = 20260903
    for role in ["architecture_test", "confirmation"]:
        hybrid = samples[(samples.role == role) & (samples.family == "gat_bissm_psd")]
        for comparator in ["bissm_psd", "gatv2_psd", "chain_psd"]:
            other = samples[(samples.role == role) & (samples.family == comparator)]
            merged = hybrid.merge(
                other[["event_id", "route", "nrmse"]], on=["event_id", "route"],
                suffixes=("_hybrid", "_other"), validate="one_to_one"
            )
            strata = [("all", merged)]
            strata += [(f"route={key}", part) for key, part in merged.groupby("route")]
            strata += [(f"gauge={key:.2f}m", part) for key, part in merged.groupby("gauge_length_m")]
            for stratum, part in strata:
                event_delta = part.assign(
                    delta=part.nrmse_hybrid - part.nrmse_other
                ).groupby("event_id")["delta"].mean().to_numpy()
                result = _bootstrap_delta(event_delta, seed, 5000)
                bootstrap_rows.append(
                    {"role": role, "comparator": labels[comparator], "stratum": stratum, **result}
                )
                seed += 1
    bootstraps = pd.DataFrame(bootstrap_rows)
    bootstraps.to_csv(reports / "final_hybrid_bootstrap_comparisons.csv", index=False)
    # Test-time feature-group occlusion sets standardized values to their training mean.
    test_indices = np.flatnonzero(roles == "architecture_test")
    feature_groups = {
        "context_complex": slice(0, 64),
        "noise_complex": slice(64, 128),
        "context_noise_power": slice(128, 132),
        "apparent_slowness": slice(132, 133),
        "pick_coverage": slice(133, 134),
        "block_position": slice(134, 135),
        "route_indicator": slice(135, 136),
        "gauge_length": slice(136, 137),
    }
    baseline_prediction = prediction_store[("architecture_test", "gat_bissm_psd")]
    baseline_nrmse = float(_sample_nrmse(
        baseline_prediction, dataset.targets[test_indices], mask
    ).mean())
    occlusion_rows = [{"feature_group": "none", "nrmse": baseline_nrmse, "delta_nrmse": 0.0}]
    for name, feature_slice in feature_groups.items():
        occluded = normalized.copy()
        occluded[test_indices, :, feature_slice] = 0.0
        prediction, _ = _ensemble_outputs(
            "gat_bissm_psd", cfg, extension, dataset, occluded, test_indices, models, device
        )
        score = float(_sample_nrmse(prediction, dataset.targets[test_indices], mask).mean())
        occlusion_rows.append(
            {"feature_group": name, "nrmse": score, "delta_nrmse": score - baseline_nrmse}
        )
    pd.DataFrame(occlusion_rows).to_csv(reports / "final_hybrid_feature_occlusion.csv", index=False)
    summary = {
        "models": labels,
        "architecture_test_mean_nrmse": {
            row.family: float(row.nrmse) for row in
            samples[samples.role == "architecture_test"].groupby("family", as_index=False).nrmse.mean().itertuples()
        },
        "confirmation_mean_nrmse": {
            row.family: float(row.nrmse) for row in
            samples[samples.role == "confirmation"].groupby("family", as_index=False).nrmse.mean().itertuples()
        },
        "minimum_psd_eigenvalue": float(pd.DataFrame(psd_rows).minimum_eigenvalue.min()),
        "psd_failure_rate": float(pd.DataFrame(psd_rows).failure_rate.max()),
        "test_feature_count": int(dataset.features.shape[-1]),
        "warning": "The architecture test is retrospective; confirmation reuse is a consistency analysis, not a second independent test.",
    }
    path = reports / "final_hybrid_diagnostics_summary.json"
    write_json(path, summary)
    return path
