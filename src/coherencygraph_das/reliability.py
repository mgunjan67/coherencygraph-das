from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve, write_json
from .spectral import lag_coherency, spectral_mixture_operator


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 8 or np.std(left[keep]) <= 1e-12 or np.std(right[keep]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left[keep], right[keep])[0, 1])


def _complex_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(
        np.concatenate([left.real.ravel(), left.imag.ravel()]),
        np.concatenate([right.real.ravel(), right.imag.ravel()]),
    )


def _load_role(index: pd.DataFrame, role: str) -> list[dict[str, np.ndarray]]:
    output = []
    for path in index.loc[(index["cohort_role"] == role) & index["success"], "path"]:
        with np.load(path, allow_pickle=False) as data:
            output.append({key: data[key] for key in data.files})
    return output


def synthetic_identifiability(seed: int = 20260830, replicates: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    positions = np.arange(96)
    lags = np.array([3, 5, 8, 13, 21, 34, 55, 89])
    errors = []
    psd_failures = 0
    for _ in range(replicates):
        weights = rng.dirichlet(np.ones(17) * 0.5)
        q = np.linspace(-np.pi, np.pi, len(weights), endpoint=False)
        gamma = spectral_mixture_operator(weights, positions, q)
        eigenvalues, eigenvectors = np.linalg.eigh(gamma + 0.15 * np.eye(len(positions)))
        factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        samples = (
            rng.normal(size=(96, 96)) + 1j * rng.normal(size=(96, 96))
        ) / np.sqrt(2)
        samples = samples @ factor.T
        estimated = lag_coherency(samples, lags)
        truth = np.array([np.mean(np.diag(gamma, k=int(lag))) for lag in lags])
        errors.append(float(np.sqrt(np.mean(np.abs(estimated - truth) ** 2))))
        if np.linalg.eigvalsh(gamma).min() < -1e-9:
            psd_failures += 1
    return {
        "replicates": replicates,
        "median_complex_rmse": float(np.median(errors)),
        "p95_complex_rmse": float(np.quantile(errors, 0.95)),
        "decoder_psd_failure_rate": float(psd_failures / replicates),
    }


def flat_frequency_false_positive(seed: int = 20260831, experiments: int = 500) -> dict:
    """Check that equal latent coherency across bands does not create a central-band maximum."""
    rng = np.random.default_rng(seed)
    false_positives = 0
    contrasts = []
    events = 30
    for _ in range(experiments):
        latent = rng.uniform(-0.15, 0.35, size=(events, 8))
        estimates = latent[:, None, :] + rng.normal(0.0, 0.06, size=(events, 4, 8))
        event_contrast = estimates[:, 1:3].mean(axis=(1, 2)) - estimates[:, [0, 3]].mean(axis=(1, 2))
        mean = float(event_contrast.mean())
        standard_error = float(event_contrast.std(ddof=1) / np.sqrt(events))
        lower = mean - 1.96 * standard_error
        false_positives += int(lower > 0)
        contrasts.append(mean)
    return {
        "experiments": experiments,
        "events_per_experiment": events,
        "false_positive_rate": float(false_positives / experiments),
        "mean_recovered_contrast": float(np.mean(contrasts)),
    }


def run_reliability(cfg: dict) -> Path:
    index = pd.read_parquet(resolve(cfg, cfg["paths"]["labels"]) / "operator_index.parquet")
    records = _load_role(index, "development")
    if not records:
        raise RuntimeError("no development operators")
    first = np.stack([x["target_gamma_first"] for x in records])
    second = np.stack([x["target_gamma_second"] for x in records])
    taper0 = np.stack([x["target_gamma_taper0"] for x in records])
    taper1 = np.stack([x["target_gamma_taper1"] for x in records])
    bands = records[0]["bands_hz"]
    lags = records[0]["lags"]
    rows = []
    for band_index, band in enumerate(bands):
        for lag_index, lag in enumerate(lags):
            a = first[:, :, band_index, lag_index].ravel()
            b = second[:, :, band_index, lag_index].ravel()
            c = taper0[:, :, band_index, lag_index].ravel()
            d = taper1[:, :, band_index, lag_index].ravel()
            rows.append(
                {
                    "band_low_hz": float(band[0]),
                    "band_high_hz": float(band[1]),
                    "channel_lag": int(lag),
                    "split_real_r": _pearson(a.real, b.real),
                    "split_imag_r": _pearson(a.imag, b.imag),
                    "split_complex_r": _complex_correlation(a, b),
                    "taper_real_r": _pearson(c.real, d.real),
                    "taper_imag_r": _pearson(c.imag, d.imag),
                    "taper_complex_r": _complex_correlation(c, d),
                    "split_complex_rmse": float(np.sqrt(np.nanmean(np.abs(a - b) ** 2))),
                    "taper_complex_rmse": float(np.sqrt(np.nanmean(np.abs(c - d) ** 2))),
                }
            )
    table = pd.DataFrame(rows)
    floor = float(cfg["model"]["reliability_floor"])
    table["reliable"] = (
        (table["split_complex_r"] >= floor) & (table["taper_complex_r"] >= floor)
    )
    reliable_fraction = float(table["reliable"].mean())
    target_mode = "complex_lag_operator" if reliable_fraction >= 0.5 else "stable_lag_subset"
    outdir = resolve(cfg, cfg["paths"]["results"])
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "target_reliability.csv"
    table.to_csv(path, index=False)
    synthetic = synthetic_identifiability()
    flat_null = flat_frequency_false_positive()
    summary = {
        "development_event_route_records": len(records),
        "reliability_floor": floor,
        "reliable_band_lag_fraction": reliable_fraction,
        "median_split_complex_r": float(table["split_complex_r"].median()),
        "median_taper_complex_r": float(table["taper_complex_r"].median()),
        "target_mode": target_mode,
        "reliable_cells": table.loc[table["reliable"], ["band_low_hz", "band_high_hz", "channel_lag"]].to_dict("records"),
        "synthetic_identifiability": synthetic,
        "flat_frequency_null": flat_null,
    }
    write_json(outdir / "reliability_summary.json", summary)
    return path
