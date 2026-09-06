from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve


@dataclass
class OperatorDataset:
    features: np.ndarray
    targets: np.ndarray
    event_ids: np.ndarray
    routes: np.ndarray
    roles: np.ndarray
    paths: np.ndarray
    lags: np.ndarray
    bands: np.ndarray
    anchors: np.ndarray


def _features(data: dict[str, np.ndarray], route: str) -> np.ndarray:
    context = data["context_gamma"]
    noise = data["noise_gamma"]
    blocks = context.shape[0]
    components = [
        context.real.reshape(blocks, -1),
        context.imag.reshape(blocks, -1),
        noise.real.reshape(blocks, -1),
        noise.imag.reshape(blocks, -1),
        np.log1p(np.maximum(data["context_noise_power_ratio"], 0.0)),
        data["apparent_slowness_sec_m"][:, None] * 1.0e4,
        data["pick_coverage"][:, None],
        np.linspace(-1.0, 1.0, blocks)[:, None],
        np.full((blocks, 1), 1.0 if route == "TERRA" else 0.0),
        np.full((blocks, 1), float(data["gauge_length_m"]) / 24.0),
    ]
    return np.concatenate(components, axis=1).astype(np.float32)


def load_operator_dataset(cfg: dict) -> OperatorDataset:
    index = pd.read_parquet(resolve(cfg, cfg["paths"]["labels"]) / "operator_index.parquet")
    index = index[index["success"]].sort_values(["event_id", "route"]).reset_index(drop=True)
    features, targets, events, routes, roles, paths = [], [], [], [], [], []
    lags = bands = anchors = None
    for row in index.itertuples(index=False):
        # Portable releases keep the immutable cache beside the index. Never
        # require the original investigator's Windows drive or working folder.
        filename = str(row.path).replace('\\', '/').rsplit('/', 1)[-1]
        local_path = resolve(cfg, cfg['paths']['labels']) / filename
        source_path = local_path if local_path.exists() else Path(row.path)
        with np.load(source_path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        features.append(_features(data, row.route))
        target = np.stack([data["target_gamma"].real, data["target_gamma"].imag], axis=-1)
        targets.append(target.astype(np.float32))
        events.append(str(row.event_id))
        routes.append(row.route)
        roles.append(row.cohort_role)
        paths.append(str(source_path))
        if lags is None:
            lags = data["lags"]
            bands = data["bands_hz"]
            anchors = data["anchor_local"]
    return OperatorDataset(
        features=np.stack(features),
        targets=np.stack(targets),
        event_ids=np.asarray(events),
        routes=np.asarray(routes),
        roles=np.asarray(roles),
        paths=np.asarray(paths),
        lags=np.asarray(lags),
        bands=np.asarray(bands),
        anchors=np.asarray(anchors),
    )


def standardize_features(dataset: OperatorDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = dataset.roles == "development"
    mean = dataset.features[train].mean(axis=(0, 1), keepdims=True)
    std = dataset.features[train].std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)
    return ((dataset.features - mean) / std).astype(np.float32), mean, std


def reliable_mask(cfg: dict) -> np.ndarray:
    table = pd.read_csv(resolve(cfg, cfg["paths"]["results"]) / "target_reliability.csv")
    bands = cfg["spectral"]["bands_hz"]
    lags = cfg["spectral"]["channel_lags"]
    mask = np.zeros((len(bands), len(lags)), dtype=bool)
    for row in table.itertuples(index=False):
        band_index = next(
            i for i, band in enumerate(bands)
            if float(band[0]) == float(row.band_low_hz) and float(band[1]) == float(row.band_high_hz)
        )
        lag_index = lags.index(int(row.channel_lag))
        mask[band_index, lag_index] = bool(row.reliable)
    return mask
