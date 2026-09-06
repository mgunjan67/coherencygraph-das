from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn

from .config import resolve, sha256, write_json
from .data import load_operator_dataset, reliable_mask, standardize_features
from .models import GraphAutoencoder, SpectralOperator, UnconstrainedOperator, masked_reconstruction_loss


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _event_batches(event_ids: np.ndarray, batch_events: int, rng: np.random.Generator) -> list[np.ndarray]:
    unique = np.unique(event_ids)
    rng.shuffle(unique)
    return [np.flatnonzero(np.isin(event_ids, unique[i : i + batch_events])) for i in range(0, len(unique), batch_events)]


def _loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask[None, None, :, :, None].expand_as(prediction)
    return ((prediction - target) ** 2)[expanded].mean()


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, indices: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        prediction, _ = model(x[indices])
        return float(torch.sqrt(_loss(prediction, y[indices], mask)).cpu())


def _pretrain(
    features: torch.Tensor,
    train_indices: np.ndarray,
    cfg: dict,
    device: torch.device,
    seed: int,
) -> dict:
    _seed(seed)
    spec = cfg["model"]
    model = GraphAutoencoder(
        features.shape[-1], spec["hidden_dim"], spec["graph_layers"], spec["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(seed + 900)
    model.train()
    for _ in range(int(spec["pretrain_epochs"])):
        original = features[train_indices]
        mask = torch.rand(original.shape, generator=generator, device=device) < 0.25
        corrupted = original.masked_fill(mask, 0.0)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(corrupted)
        loss = masked_reconstruction_loss(prediction, original, mask)
        loss.backward()
        optimizer.step()
    return deepcopy(model.backbone.state_dict())


def _fit_neural(
    family: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    event_ids: np.ndarray,
    train_indices: np.ndarray,
    calibration_indices: np.ndarray,
    mask: torch.Tensor,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    _seed(seed)
    spec = cfg["model"]
    common = dict(
        features=features.shape[-1], hidden_dim=spec["hidden_dim"], layers=spec["graph_layers"],
        dropout=spec["dropout"], bands=targets.shape[2],
    )
    if family in {"graph_psd", "graph_psd_no_pretrain", "graph_psd_no_slowness"}:
        model = SpectralOperator(
            **common, q_bins=cfg["spectral"]["spatial_wavenumber_bins"],
            lags=cfg["spectral"]["channel_lags"], graph=True,
        )
    elif family == "nograph_psd":
        model = SpectralOperator(
            **common, q_bins=cfg["spectral"]["spatial_wavenumber_bins"],
            lags=cfg["spectral"]["channel_lags"], graph=False,
        )
    elif family == "graph_unconstrained":
        model = UnconstrainedOperator(**common, lags=targets.shape[3])
    else:
        raise ValueError(family)
    model = model.to(device)
    if family in {"graph_psd", "graph_psd_no_slowness"}:
        pretrained = _pretrain(features, train_indices, cfg, device, seed)
        model.backbone.load_state_dict(pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    best_state, best_score, best_epoch, stale = None, np.inf, -1, 0
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(int(spec["train_epochs"])):
        model.train()
        losses = []
        for batch in _event_batches(event_ids[train_indices], int(spec["batch_events"]), rng):
            idx = train_indices[batch]
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(features[idx])
            loss = _loss(prediction, targets[idx], mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = _evaluate(model, features, targets, mask, calibration_indices)
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "calibration_rmse": score})
        if score < best_score - 1e-5:
            best_score, best_epoch = score, epoch + 1
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(spec["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("no model checkpoint selected")
    model.load_state_dict(best_state)
    return model, {"family": family, "seed": seed, "best_epoch": best_epoch, "calibration_rmse": best_score, "history": history}


def _predict(model: nn.Module, features: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    outputs, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(features), 64):
            prediction, p = model(features[start : start + 64])
            outputs.append(prediction.cpu().numpy())
            if p is not None:
                probabilities.append(p.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(probabilities) if probabilities else None


def _classical_predictions(features: np.ndarray, targets: np.ndarray, train: np.ndarray) -> dict[str, np.ndarray]:
    x = features.reshape(-1, features.shape[-1])
    y = targets.reshape(-1, np.prod(targets.shape[2:]))
    train_rows = np.repeat(train, features.shape[1])
    ridge = Ridge(alpha=10.0).fit(x[train_rows], y[train_rows])
    trees = ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=4, max_features=0.7, n_jobs=-1, random_state=20260830
    ).fit(x[train_rows], y[train_rows])
    shape = targets.shape
    return {
        "ridge": ridge.predict(x).reshape(shape).astype(np.float32),
        "extra_trees": trees.predict(x).reshape(shape).astype(np.float32),
    }


def _metadata_predictions(dataset, cfg: dict, targets: np.ndarray, train: np.ndarray) -> np.ndarray:
    cohort = pd.read_parquet(resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet")
    cohort["event_id"] = cohort["event_id"].astype(str)
    meta = cohort.set_index("event_id")
    sample_features = []
    for event_id, route in zip(dataset.event_ids, dataset.routes, strict=True):
        row = meta.loc[str(event_id)]
        bearing = np.deg2rad(float(row["source_bearing_deg"]))
        gauge = float(row["terra_gauge_length_m"])
        base = np.array(
            [float(row["magnitude"]), float(row["depth_km"]) / 100.0, np.sin(bearing), np.cos(bearing),
             1.0 if route == "TERRA" else 0.0, gauge / 24.0],
            dtype=np.float32,
        )
        blocks = []
        for block in range(targets.shape[1]):
            blocks.append(np.concatenate([base, [block / max(1, targets.shape[1] - 1)]]))
        sample_features.append(np.stack(blocks))
    x = np.stack(sample_features)
    flat_x = x.reshape(-1, x.shape[-1])
    flat_y = targets.reshape(-1, np.prod(targets.shape[2:]))
    train_rows = np.repeat(train, targets.shape[1])
    model = Ridge(alpha=10.0).fit(flat_x[train_rows], flat_y[train_rows])
    return model.predict(flat_x).reshape(targets.shape).astype(np.float32)


def train_models(cfg: dict) -> Path:
    dataset = load_operator_dataset(cfg)
    normalized, mean, std = standardize_features(dataset)
    mask_np = reliable_mask(cfg)
    train = dataset.roles == "development"
    calibration = dataset.roles == "calibration"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(normalized).to(device)
    y = torch.from_numpy(dataset.targets).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    outdir = resolve(cfg, cfg["paths"]["models"])
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "feature_scaler.npz", mean=mean, std=std)
    ensemble_predictions, ensemble_probabilities, checkpoints = [], [], []
    training_records = []
    for seed in cfg["model"]["ensemble_seeds"]:
        model, record = _fit_neural(
            "graph_psd", x, y, dataset.event_ids, np.flatnonzero(train), np.flatnonzero(calibration),
            mask, cfg, int(seed), device,
        )
        checkpoint = outdir / f"graph_psd_seed{seed}.pt"
        torch.save(model.state_dict(), checkpoint)
        prediction, probabilities = _predict(model, x)
        ensemble_predictions.append(prediction)
        ensemble_probabilities.append(probabilities)
        record["checkpoint"] = str(checkpoint.resolve())
        record["checkpoint_sha256"] = sha256(checkpoint)
        training_records.append(record)
        checkpoints.append(record["checkpoint_sha256"])
        print(f"trained graph_psd seed {seed}: calibration RMSE={record['calibration_rmse']:.4f}", flush=True)
    for family in ["nograph_psd", "graph_unconstrained"]:
        family_predictions = []
        for seed in cfg["model"]["ensemble_seeds"][:3]:
            model, record = _fit_neural(
                family, x, y, dataset.event_ids, np.flatnonzero(train), np.flatnonzero(calibration),
                mask, cfg, int(seed), device,
            )
            checkpoint = outdir / f"{family}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            prediction, _ = _predict(model, x)
            family_predictions.append(prediction)
            record["checkpoint"] = str(checkpoint.resolve())
            record["checkpoint_sha256"] = sha256(checkpoint)
            training_records.append(record)
            print(f"trained {family} seed {seed}: calibration RMSE={record['calibration_rmse']:.4f}", flush=True)
        np.save(outdir / f"{family}_predictions.npy", np.mean(family_predictions, axis=0).astype(np.float32))
    for family in ["graph_psd_no_pretrain", "graph_psd_no_slowness"]:
        family_predictions = []
        family_x = x
        if family == "graph_psd_no_slowness":
            family_x = x.clone()
            family_x[..., 132] = 0.0
        for seed in cfg["model"]["ensemble_seeds"][:3]:
            model, record = _fit_neural(
                family, family_x, y, dataset.event_ids,
                np.flatnonzero(train), np.flatnonzero(calibration),
                mask, cfg, int(seed), device,
            )
            checkpoint = outdir / f"{family}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            prediction, _ = _predict(model, family_x)
            family_predictions.append(prediction)
            record["checkpoint"] = str(checkpoint.resolve())
            record["checkpoint_sha256"] = sha256(checkpoint)
            training_records.append(record)
        np.save(outdir / f"{family}_predictions.npy", np.mean(family_predictions, axis=0).astype(np.float32))
        print(f"trained {family} ablation", flush=True)
    for source_route, target_route in [("TERRA", "KKFL-S"), ("KKFL-S", "TERRA")]:
        family_predictions = []
        source_train = train & (dataset.routes == source_route)
        source_calibration = calibration & (dataset.routes == source_route)
        for seed in cfg["model"]["ensemble_seeds"][:3]:
            model, record = _fit_neural(
                "graph_psd", x, y, dataset.event_ids,
                np.flatnonzero(source_train), np.flatnonzero(source_calibration),
                mask, cfg, int(seed), device,
            )
            checkpoint = outdir / f"route_transfer_{source_route.replace('-', '')}_to_{target_route.replace('-', '')}_seed{seed}.pt"
            torch.save(model.state_dict(), checkpoint)
            prediction, _ = _predict(model, x)
            family_predictions.append(prediction)
            record["family"] = f"route_transfer_{source_route}_to_{target_route}"
            record["checkpoint"] = str(checkpoint.resolve())
            record["checkpoint_sha256"] = sha256(checkpoint)
            training_records.append(record)
        np.save(
            outdir / f"route_transfer_{source_route.replace('-', '')}_to_{target_route.replace('-', '')}_predictions.npy",
            np.mean(family_predictions, axis=0).astype(np.float32),
        )
        print(f"trained strict route transfer {source_route} -> {target_route}", flush=True)
    predictions = np.stack(ensemble_predictions)
    probabilities = np.stack(ensemble_probabilities)
    np.save(outdir / "graph_psd_predictions.npy", predictions.astype(np.float32))
    np.save(outdir / "graph_psd_probabilities.npy", probabilities.astype(np.float32))
    classical = _classical_predictions(normalized, dataset.targets, train)
    for name, prediction in classical.items():
        np.save(outdir / f"{name}_predictions.npy", prediction)
    np.save(outdir / "metadata_ridge_predictions.npy", _metadata_predictions(dataset, cfg, dataset.targets, train))
    global_mean = dataset.targets[train].mean(axis=(0, 1), keepdims=True)
    np.save(outdir / "global_mean_predictions.npy", np.broadcast_to(global_mean, dataset.targets.shape).astype(np.float32))
    context = normalized * 0  # placeholder is overwritten below using raw context features
    persistence = np.empty_like(dataset.targets)
    for i, path in enumerate(dataset.paths):
        with np.load(path, allow_pickle=False) as loaded:
            gamma = loaded["context_gamma"]
        persistence[i] = np.stack([gamma.real, gamma.imag], axis=-1)
    np.save(outdir / "persistence_predictions.npy", persistence.astype(np.float32))
    pd.DataFrame(
        [{k: v for k, v in record.items() if k != "history"} for record in training_records]
    ).to_csv(outdir / "training_summary.csv", index=False)
    write_json(outdir / "training_history.json", training_records)
    freeze = {
        "protocol_sha256": sha256(cfg["_config_path"]),
        "device": str(device),
        "ensemble_checkpoint_sha256": checkpoints,
        "ensemble_seeds": list(map(int, cfg["model"]["ensemble_seeds"])),
        "reliable_mask": mask_np.astype(int).tolist(),
        "confirmation_predictions_generated": True,
        "confirmation_metrics_inspected": False,
        "note": "Predictions were generated after model/calibration freeze; confirmation metrics were not computed during training.",
    }
    write_json(outdir / "model_freeze.json", freeze)
    return outdir / "model_freeze.json"
