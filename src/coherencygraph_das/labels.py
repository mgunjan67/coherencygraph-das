from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import resolve, write_json
from .spectral import (
    band_snapshots,
    cross_spectral_matrix,
    effective_rank,
    lag_coherency,
    multitaper_fourier,
    normalize_coherency,
    robust_linear_pick_model,
)


RAW_DATASET = "Acquisition/Raw[0]/RawData"


def _read_window(dataset: h5py.Dataset, start: float, stop: float, fs: float, columns: slice) -> np.ndarray:
    first = max(0, int(round(start * fs)))
    last = min(dataset.shape[0], int(round(stop * fs)))
    if last - first < 16:
        raise ValueError(f"window [{start}, {stop}] is outside the record")
    return np.asarray(dataset[first:last, columns], dtype=np.float32)


def _route_pick_column(route: str) -> str:
    return "TERRA_s_sec" if route == "TERRA" else "KKFLS_s_sec"


def _process_route(row: dict, route: str, cfg: dict, output: Path) -> dict:
    spectral = cfg["spectral"]
    windows = cfg["windows"]
    path = Path(row["terra_path"] if route == "TERRA" else row["kkfls_path"])
    picks = pd.read_csv(row["picks_path"])
    pick_channel = picks["channel"].to_numpy(float)
    pick_time = pd.to_numeric(picks[_route_pick_column(route)], errors="coerce").to_numpy(float)
    route_s_reference = float(np.nanmedian(pick_time))
    if not np.isfinite(route_s_reference):
        raise ValueError("route has no finite S reference")
    fs = float(row["terra_sample_rate_hz"] if route == "TERRA" else row["kkfls_sample_rate_hz"])
    gauge = float(row["terra_gauge_length_m"] if route == "TERRA" else row["kkfls_gauge_length_m"])
    starts = list(map(int, spectral["block_starts"]))
    block_channels = int(spectral["block_channels"])
    bands = [tuple(map(float, x)) for x in spectral["bands_hz"]]
    lags = list(map(int, spectral["channel_lags"]))
    anchors_local = np.rint(
        np.linspace(0, block_channels - 1, int(spectral["anchor_channels"]))
    ).astype(int)
    shape_bl = (len(starts), len(bands), len(lags))
    context_gamma = np.full(shape_bl, np.nan + 1j * np.nan, np.complex64)
    noise_gamma = np.full_like(context_gamma, np.nan + 1j * np.nan)
    target_gamma = np.full_like(context_gamma, np.nan + 1j * np.nan)
    target_first = np.full_like(context_gamma, np.nan + 1j * np.nan)
    target_second = np.full_like(context_gamma, np.nan + 1j * np.nan)
    target_taper0 = np.full_like(context_gamma, np.nan + 1j * np.nan)
    target_taper1 = np.full_like(context_gamma, np.nan + 1j * np.nan)
    anchor_shape = (len(starts), len(bands), len(anchors_local), len(anchors_local))
    target_anchor = np.full(anchor_shape, np.nan + 1j * np.nan, np.complex64)
    noise_anchor = np.full_like(target_anchor, np.nan + 1j * np.nan)
    context_anchor = np.full_like(target_anchor, np.nan + 1j * np.nan)
    power_ratio = np.full((len(starts), len(bands)), np.nan, np.float32)
    target_effective_rank = np.full((len(starts), len(bands)), np.nan, np.float32)
    slowness = np.full(len(starts), np.nan, np.float32)
    pick_coverage = np.zeros(len(starts), np.float32)
    finite_fraction = 1.0

    with h5py.File(path, "r") as handle:
        dataset = handle[RAW_DATASET]
        for block_index, block_start in enumerate(starts):
            column_slice = slice(block_start, block_start + block_channels)
            s_reference, slope_channel, coverage = robust_linear_pick_model(
                pick_channel, pick_time, block_start, block_channels
            )
            if not np.isfinite(s_reference):
                # The event passed the frozen route-level pick-coverage gate. A
                # missing terminal-block fit is therefore represented explicitly
                # with zero local coverage and the route-median causal trigger,
                # rather than silently discarding the complete earthquake.
                s_reference = route_s_reference
                slope_channel = 0.0
                coverage = 0.0
            pick_coverage[block_index] = coverage
            spacing = float(
                row["terra_channel_spacing_m"] if route == "TERRA" else row["kkfls_channel_spacing_m"]
            )
            slowness[block_index] = slope_channel / spacing
            centre = block_start + 0.5 * (block_channels - 1)
            channel_delay = slope_channel * (
                np.arange(block_start, block_start + block_channels) - centre
            )
            noise = _read_window(
                dataset, float(windows["noise_sec"][0]), float(windows["noise_sec"][1]), fs, column_slice
            )
            context = _read_window(
                dataset,
                s_reference + float(windows["context_relative_to_s_sec"][0]),
                s_reference + float(windows["context_relative_to_s_sec"][1]),
                fs,
                column_slice,
            )
            target_start = s_reference + float(windows["target_relative_to_s_sec"][0])
            target_stop = s_reference + float(windows["target_relative_to_s_sec"][1])
            target = _read_window(dataset, target_start, target_stop, fs, column_slice)
            finite_fraction = min(
                finite_fraction,
                float(np.isfinite(noise).mean()),
                float(np.isfinite(context).mean()),
                float(np.isfinite(target).mean()),
            )
            f_noise, hz_noise = multitaper_fourier(
                noise, fs, spectral["dpss_time_bandwidth"], spectral["dpss_tapers"]
            )
            f_context, hz_context = multitaper_fourier(
                context,
                fs,
                spectral["dpss_time_bandwidth"],
                spectral["dpss_tapers"],
                channel_delay,
            )
            f_target, hz_target = multitaper_fourier(
                target,
                fs,
                spectral["dpss_time_bandwidth"],
                spectral["dpss_tapers"],
                channel_delay,
            )
            half = target.shape[0] // 2
            f_first, hz_first = multitaper_fourier(
                target[:half], fs, spectral["dpss_time_bandwidth"], spectral["dpss_tapers"], channel_delay
            )
            f_second, hz_second = multitaper_fourier(
                target[half:], fs, spectral["dpss_time_bandwidth"], spectral["dpss_tapers"], channel_delay
            )
            for band_index, band in enumerate(bands):
                snapshots_n = band_snapshots(f_noise, hz_noise, band)
                snapshots_c = band_snapshots(f_context, hz_context, band)
                snapshots_t = band_snapshots(f_target, hz_target, band)
                context_gamma[block_index, band_index] = lag_coherency(snapshots_c, lags)
                noise_gamma[block_index, band_index] = lag_coherency(snapshots_n, lags)
                target_gamma[block_index, band_index] = lag_coherency(snapshots_t, lags)
                target_first[block_index, band_index] = lag_coherency(
                    band_snapshots(f_first, hz_first, band), lags
                )
                target_second[block_index, band_index] = lag_coherency(
                    band_snapshots(f_second, hz_second, band), lags
                )
                target_taper0[block_index, band_index] = lag_coherency(
                    band_snapshots(f_target, hz_target, band, np.array([0])), lags
                )
                target_taper1[block_index, band_index] = lag_coherency(
                    band_snapshots(f_target, hz_target, band, np.array([1])), lags
                )
                xa_n = snapshots_n[:, anchors_local]
                xa_c = snapshots_c[:, anchors_local]
                xa_t = snapshots_t[:, anchors_local]
                rn = cross_spectral_matrix(xa_n, spectral["diagonal_loading"])
                rc = cross_spectral_matrix(xa_c, spectral["diagonal_loading"])
                rt = cross_spectral_matrix(xa_t, spectral["diagonal_loading"])
                noise_anchor[block_index, band_index] = rn
                context_anchor[block_index, band_index] = rc
                target_anchor[block_index, band_index] = rt
                power_ratio[block_index, band_index] = float(
                    np.real(np.trace(rc)) / max(np.real(np.trace(rn)), np.finfo(float).eps)
                )
                target_effective_rank[block_index, band_index] = effective_rank(
                    normalize_coherency(rt)
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        event_id=np.asarray(row["event_id"]),
        route=np.asarray(route),
        cohort_role=np.asarray(row["cohort_role"]),
        gauge_length_m=np.float32(gauge),
        block_starts=np.asarray(starts, np.int32),
        anchor_local=np.asarray(anchors_local, np.int32),
        lags=np.asarray(lags, np.int32),
        bands_hz=np.asarray(bands, np.float32),
        context_gamma=context_gamma,
        noise_gamma=noise_gamma,
        target_gamma=target_gamma,
        target_gamma_first=target_first,
        target_gamma_second=target_second,
        target_gamma_taper0=target_taper0,
        target_gamma_taper1=target_taper1,
        context_anchor=context_anchor,
        noise_anchor=noise_anchor,
        target_anchor=target_anchor,
        context_noise_power_ratio=power_ratio,
        target_effective_rank=target_effective_rank,
        apparent_slowness_sec_m=slowness,
        pick_coverage=pick_coverage,
    )
    return {
        "event_id": str(row["event_id"]),
        "route": route,
        "cohort_role": row["cohort_role"],
        "path": str(output.resolve()),
        "finite_fraction": finite_fraction,
        "gauge_length_m": gauge,
        "success": True,
        "error": "",
    }


def _process_event(row: dict, cfg: dict, outdir: str) -> list[dict]:
    records = []
    for route in ("TERRA", "KKFL-S"):
        output = Path(outdir) / f"{row['event_id']}_{route.replace('-', '')}.npz"
        try:
            records.append(_process_route(row, route, cfg, output))
        except Exception as exc:  # fail is recorded by complete event-route
            records.append(
                {
                    "event_id": str(row["event_id"]),
                    "route": route,
                    "cohort_role": row["cohort_role"],
                    "path": str(output.resolve()),
                    "finite_fraction": np.nan,
                    "gauge_length_m": float(row["terra_gauge_length_m"]),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return records


def build_labels(cfg: dict, workers: int = 4) -> Path:
    cohort = pd.read_parquet(resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet")
    cohort = cohort[
        cohort["cohort_role"].isin(["development", "calibration", "confirmation"])
    ].copy()
    outdir = resolve(cfg, cfg["paths"]["labels"])
    outdir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_event, row._asdict(), cfg, str(outdir))
            for row in cohort.itertuples(index=False)
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            if completed % 10 == 0 or completed == len(futures):
                print(f"labelled {completed}/{len(futures)} earthquakes", flush=True)
    index = pd.DataFrame(records).sort_values(["cohort_role", "event_id", "route"])
    index_path = outdir / "operator_index.parquet"
    index.to_parquet(index_path, index=False)
    index.to_csv(outdir / "operator_index.csv", index=False)
    paired_success = index.groupby("event_id")["success"].all()
    summary = {
        "requested_events": int(len(cohort)),
        "route_records": int(len(index)),
        "successful_route_records": int(index["success"].sum()),
        "paired_successful_events": int(paired_success.sum()),
        "role_successful_events": {
            role: int(
                paired_success.loc[
                    cohort.loc[cohort["cohort_role"] == role, "event_id"].astype(str)
                ].sum()
            )
            for role in ["development", "calibration", "confirmation"]
        },
        "errors": index.loc[~index["success"], ["event_id", "route", "error"]].to_dict("records"),
    }
    write_json(outdir / "label_summary.json", summary)
    return index_path
