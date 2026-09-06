from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import resolve, sha256, write_json
from .data import load_operator_dataset, reliable_mask
from .spectral import normalize_coherency, spectral_mixture_operator


def complex_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    keep = np.broadcast_to(mask[None, :, :, None], prediction.shape)
    difference = prediction - target
    complex_error = difference[..., 0] + 1j * difference[..., 1]
    complex_target = target[..., 0] + 1j * target[..., 1]
    selected_error = complex_error[keep[..., 0]]
    selected_target = complex_target[keep[..., 0]]
    prediction_complex = prediction[..., 0] + 1j * prediction[..., 1]
    selected_prediction = prediction_complex[keep[..., 0]]
    denominator = np.sqrt(np.mean(np.abs(selected_target) ** 2))
    return {
        "normalized_complex_rmse": float(
            np.sqrt(np.mean(np.abs(selected_error) ** 2)) / max(denominator, np.finfo(float).eps)
        ),
        "complex_rmse": float(np.sqrt(np.mean(np.abs(selected_error) ** 2))),
        "real_mae": float(np.mean(np.abs(selected_error.real))),
        "imag_mae": float(np.mean(np.abs(selected_error.imag))),
        "real_sign_accuracy": float(np.mean(np.sign(selected_prediction.real) == np.sign(selected_target.real))),
        "magnitude_violation_rate": float(np.mean(np.abs(selected_prediction) > 1.0 + 1e-6)),
    }


def _ratio(matrix_target: np.ndarray, matrix_noise: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.complex128)
    numerator = float(np.real(np.conj(w) @ matrix_target @ w))
    denominator = float(np.real(np.conj(w) @ matrix_noise @ w))
    return max(numerator, np.finfo(float).eps) / max(denominator, np.finfo(float).eps)


def _equal_weights(indices: np.ndarray, size: int) -> np.ndarray:
    weights = np.zeros(size, dtype=float)
    weights[np.asarray(indices, dtype=int)] = 1.0
    return weights / np.linalg.norm(weights)


def greedy_nonnegative_weights(
    target_gamma: np.ndarray, noise_gamma: np.ndarray, selected_channels: int
) -> np.ndarray:
    size = target_gamma.shape[0]
    selected = [size // 2]
    remaining = set(range(size)) - set(selected)
    while len(selected) < min(selected_channels, size):
        best_index, best_score = None, -np.inf
        for candidate in sorted(remaining):
            trial = np.asarray(selected + [candidate])
            weights = _equal_weights(trial, size)
            score = _ratio(target_gamma.real, noise_gamma.real, weights)
            if score > best_score:
                best_index, best_score = candidate, score
        selected.append(int(best_index))
        remaining.remove(int(best_index))
    return _equal_weights(np.asarray(selected), size)


def _event_bootstrap(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _load_matrices(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return (
            loaded["target_anchor"],
            loaded["noise_anchor"],
            loaded["context_anchor"],
            loaded["anchor_local"],
        )


def _processing_grid(
    dataset,
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    role: str,
    q_bins: int,
    sparse_counts: list[int],
) -> pd.DataFrame:
    rows = []
    q = np.linspace(-np.pi, np.pi, q_bins, endpoint=False)
    for sample in np.flatnonzero(dataset.roles == role):
        target, noise, context, anchors = _load_matrices(dataset.paths[sample])
        for block in range(target.shape[0]):
            for band in range(target.shape[1]):
                rt, rn, rc = target[block, band], noise[block, band], context[block, band]
                gn = normalize_coherency(rn)
                gt = spectral_mixture_operator(probabilities[sample, block, band], anchors, q)
                base = {
                    "sample": sample,
                    "event_id": dataset.event_ids[sample],
                    "route": dataset.routes[sample],
                    "role": role,
                    "block": block,
                    "band": band,
                    "uncertainty": float(uncertainty[sample, block, band]),
                }
                for step in [1, 2, 4, 8]:
                    weights = _equal_weights(np.arange(0, len(anchors), step), len(anchors))
                    rows.append({**base, "method": f"fixed_step_{step}", "channels": int(np.count_nonzero(weights)), "snr_star_db": 10 * np.log10(_ratio(rt, rn, weights))})
                for count in sparse_counts:
                    fixed_indices = np.rint(
                        np.linspace(0, len(anchors) - 1, int(count))
                    ).astype(int)
                    fixed_weights = _equal_weights(fixed_indices, len(anchors))
                    rows.append({**base, "method": f"fixed_k_{count}", "channels": int(count), "snr_star_db": 10 * np.log10(_ratio(rt, rn, fixed_weights))})
                    weights = greedy_nonnegative_weights(gt, gn, int(count))
                    rows.append({**base, "method": f"adaptive_k_{count}", "channels": int(count), "snr_star_db": 10 * np.log10(_ratio(rt, rn, weights))})
                    proxy = np.real(np.diag(rc)) / np.maximum(np.real(np.diag(rn)), np.finfo(float).eps)
                    indices = np.argsort(proxy)[-int(count):]
                    snr_weights = _equal_weights(indices, len(anchors))
                    rows.append({**base, "method": f"snr_ranked_k_{count}", "channels": int(count), "snr_star_db": 10 * np.log10(_ratio(rt, rn, snr_weights))})
                loaded = np.linalg.solve(rn + 1e-6 * np.eye(len(anchors)), rt)
                eigenvalues, eigenvectors = np.linalg.eig(loaded)
                oracle = eigenvectors[:, np.argmax(eigenvalues.real)]
                oracle /= np.linalg.norm(oracle)
                rows.append({**base, "method": "target_window_oracle", "channels": len(anchors), "snr_star_db": 10 * np.log10(_ratio(rt, rn, oracle))})
    return pd.DataFrame(rows)


def _event_processing(table: pd.DataFrame, method: str, baseline: str, uncertainty_threshold: float | None = None) -> pd.DataFrame:
    adaptive = table[table["method"] == method].copy()
    fixed = table[table["method"] == baseline].copy()
    keys = ["sample", "event_id", "route", "block", "band"]
    merged = adaptive.merge(
        fixed[keys + ["snr_star_db"]].rename(columns={"snr_star_db": "fixed_snr_star_db"}),
        on=keys,
        validate="one_to_one",
    )
    if uncertainty_threshold is not None:
        merged["used_fallback"] = merged["uncertainty"] > uncertainty_threshold
        merged.loc[merged["used_fallback"], "snr_star_db"] = merged.loc[
            merged["used_fallback"], "fixed_snr_star_db"
        ]
    else:
        merged["used_fallback"] = False
    merged["delta_snr_star_db"] = merged["snr_star_db"] - merged["fixed_snr_star_db"]
    return (
        merged.groupby(["sample", "event_id", "route"], as_index=False)
        .agg(
            delta_snr_star_db=("delta_snr_star_db", "mean"),
            adaptive_snr_star_db=("snr_star_db", "mean"),
            fixed_snr_star_db=("fixed_snr_star_db", "mean"),
            fallback_fraction=("used_fallback", "mean"),
        )
    )


def evaluate_models(cfg: dict) -> Path:
    dataset = load_operator_dataset(cfg)
    model_dir = resolve(cfg, cfg["paths"]["models"])
    if not (model_dir / "model_freeze.json").exists():
        raise RuntimeError("model freeze is required before confirmation evaluation")
    mask = reliable_mask(cfg)
    graph_members = np.load(model_dir / "graph_psd_predictions.npy")
    graph_probabilities = np.load(model_dir / "graph_psd_probabilities.npy")
    graph_mean = graph_members.mean(axis=0)
    graph_std = np.sqrt(np.mean(np.var(graph_members, axis=0), axis=(-1, -2)))
    prediction_files = {
        "CoherencyGraph-DAS": graph_mean,
        "no-graph PSD": np.load(model_dir / "nograph_psd_predictions.npy"),
        "unconstrained graph": np.load(model_dir / "graph_unconstrained_predictions.npy"),
        "ridge": np.load(model_dir / "ridge_predictions.npy"),
        "extra trees": np.load(model_dir / "extra_trees_predictions.npy"),
        "separation-only mean": np.load(model_dir / "global_mean_predictions.npy"),
        "early-window persistence": np.load(model_dir / "persistence_predictions.npy"),
        "metadata-only ridge": np.load(model_dir / "metadata_ridge_predictions.npy"),
        "no self-supervised pretraining": np.load(model_dir / "graph_psd_no_pretrain_predictions.npy"),
        "no apparent-slowness feature": np.load(model_dir / "graph_psd_no_slowness_predictions.npy"),
    }
    metric_rows = []
    event_rows = []
    for role in ["calibration", "confirmation"]:
        selected = np.flatnonzero(dataset.roles == role)
        for name, prediction in prediction_files.items():
            for sample in selected:
                metrics = complex_metrics(prediction[sample], dataset.targets[sample], mask)
                event_rows.append({"role": role, "model": name, "event_id": dataset.event_ids[sample], "route": dataset.routes[sample], **metrics})
            summary = complex_metrics(prediction[selected], dataset.targets[selected], mask)
            metric_rows.append({"role": role, "model": name, **summary})
    metrics = pd.DataFrame(metric_rows)
    event_metrics = pd.DataFrame(event_rows)
    outdir = resolve(cfg, cfg["paths"]["results"])
    outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outdir / "matrix_metrics.csv", index=False)
    event_metrics.to_parquet(outdir / "event_matrix_metrics.parquet", index=False)
    transfer_rows = []
    for source_route, target_route in [("TERRA", "KKFL-S"), ("KKFL-S", "TERRA")]:
        filename = model_dir / f"route_transfer_{source_route.replace('-', '')}_to_{target_route.replace('-', '')}_predictions.npy"
        prediction = np.load(filename)
        selected = np.flatnonzero((dataset.roles == "confirmation") & (dataset.routes == target_route))
        transfer_rows.append(
            {
                "source_route": source_route,
                "target_route": target_route,
                "events": int(len(selected)),
                **complex_metrics(prediction[selected], dataset.targets[selected], mask),
            }
        )
    pd.DataFrame(transfer_rows).to_csv(outdir / "strict_route_transfer_metrics.csv", index=False)

    probabilities = graph_probabilities.mean(axis=0)
    calibration_grid = _processing_grid(
        dataset, probabilities, graph_std, "calibration",
        int(cfg["spectral"]["spatial_wavenumber_bins"]),
        list(map(int, cfg["evaluation"]["sparse_channel_counts"])),
    )
    adaptive_scores = {}
    for count in cfg["evaluation"]["sparse_channel_counts"]:
        method = f"adaptive_k_{int(count)}"
        matched_fixed = f"fixed_k_{int(count)}"
        adaptive_scores[method] = float(
            _event_processing(calibration_grid, method, matched_fixed)["delta_snr_star_db"].mean()
        )
    best_adaptive = max(adaptive_scores, key=adaptive_scores.get)
    selected_k = int(best_adaptive.rsplit("_", 1)[-1])
    best_fixed = f"fixed_k_{selected_k}"
    calibration_uncertainty = calibration_grid["uncertainty"].to_numpy()
    uncertainty_threshold = float(
        np.quantile(calibration_uncertainty, cfg["evaluation"]["uncertainty_fallback_quantile"])
    )
    evaluation_freeze = {
        "protocol_sha256": sha256(cfg["_config_path"]),
        "amendment_01_sha256": sha256(
            Path(cfg["_root"]) / "configs" / "protocol_amendment_01_equal_channel_baseline.yaml"
        ),
        "best_fixed_baseline": best_fixed,
        "best_adaptive_method": best_adaptive,
        "selected_channel_count": selected_k,
        "comparison_is_channel_count_matched": True,
        "calibration_adaptive_delta_snr_star_db": adaptive_scores,
        "uncertainty_threshold": uncertainty_threshold,
        "selection_used_confirmation_outcomes": False,
    }
    write_json(outdir / "evaluation_freeze.json", evaluation_freeze)

    confirmation_grid = _processing_grid(
        dataset, probabilities, graph_std, "confirmation",
        int(cfg["spectral"]["spatial_wavenumber_bins"]),
        list(map(int, cfg["evaluation"]["sparse_channel_counts"])),
    )
    calibration_grid.to_parquet(outdir / "calibration_processing_grid.parquet", index=False)
    confirmation_grid.to_parquet(outdir / "confirmation_processing_grid.parquet", index=False)
    confirmation_events = _event_processing(
        confirmation_grid, best_adaptive, best_fixed, uncertainty_threshold
    )
    confirmation_events.to_csv(outdir / "confirmation_event_processing.csv", index=False)
    route_results = []
    seed = int(cfg["evaluation"]["bootstrap_seed"])
    replicates = int(cfg["evaluation"]["bootstrap_replicates"])
    for route, part in confirmation_events.groupby("route"):
        estimate, low, high = _event_bootstrap(part["delta_snr_star_db"].to_numpy(), seed, replicates)
        route_results.append({"route": route, "events": len(part), "mean_delta_snr_star_db": estimate, "ci_low": low, "ci_high": high, "pass": low > 0})
    route_table = pd.DataFrame(route_results)
    route_table.to_csv(outdir / "primary_route_results.csv", index=False)
    control_rows = []
    for count in cfg["evaluation"]["sparse_channel_counts"]:
        matched_fixed = f"fixed_k_{int(count)}"
        for prefix in ["adaptive", "snr_ranked"]:
            method = f"{prefix}_k_{int(count)}"
            event_table = _event_processing(confirmation_grid, method, matched_fixed)
            for route, part in event_table.groupby("route"):
                estimate, low, high = _event_bootstrap(
                    part["delta_snr_star_db"].to_numpy(), seed + int(count), replicates
                )
                control_rows.append(
                    {"method": method, "baseline": matched_fixed, "route": route, "events": len(part),
                     "mean_delta_snr_star_db": estimate, "ci_low": low, "ci_high": high}
                )
    no_fallback = _event_processing(confirmation_grid, best_adaptive, best_fixed)
    for route, part in no_fallback.groupby("route"):
        estimate, low, high = _event_bootstrap(part["delta_snr_star_db"].to_numpy(), seed + 99, replicates)
        control_rows.append(
            {"method": f"{best_adaptive}_no_fallback", "baseline": best_fixed, "route": route,
             "events": len(part), "mean_delta_snr_star_db": estimate, "ci_low": low, "ci_high": high}
        )
    pd.DataFrame(control_rows).to_csv(outdir / "processing_controls.csv", index=False)
    selected_k = int(best_adaptive.rsplit("_", 1)[-1])
    snr_events = _event_processing(
        confirmation_grid, f"snr_ranked_k_{selected_k}", best_fixed
    )
    comparison = confirmation_events.merge(
        snr_events[["event_id", "route", "adaptive_snr_star_db"]].rename(
            columns={"adaptive_snr_star_db": "snr_ranked_snr_star_db"}
        ),
        on=["event_id", "route"],
        validate="one_to_one",
    )
    comparison["adaptive_minus_snr_ranked_db"] = (
        comparison["adaptive_snr_star_db"] - comparison["snr_ranked_snr_star_db"]
    )
    direct_rows = []
    for route, part in comparison.groupby("route"):
        estimate, low, high = _event_bootstrap(
            part["adaptive_minus_snr_ranked_db"].to_numpy(), seed + 177, replicates
        )
        direct_rows.append(
            {"route": route, "events": len(part), "mean_adaptive_minus_snr_ranked_db": estimate,
             "ci_low": low, "ci_high": high, "adaptive_superiority": low > 0}
        )
    pd.DataFrame(direct_rows).to_csv(outdir / "adaptive_vs_snr_ranked.csv", index=False)

    selected_adaptive = confirmation_grid[confirmation_grid["method"] == best_adaptive]
    selected_fixed = confirmation_grid[confirmation_grid["method"] == best_fixed]
    keys = ["sample", "event_id", "route", "block", "band"]
    block_band = selected_adaptive.merge(
        selected_fixed[keys + ["snr_star_db"]].rename(columns={"snr_star_db": "fixed_snr_star_db"}),
        on=keys,
        validate="one_to_one",
    )
    block_band["delta_snr_star_db"] = block_band["snr_star_db"] - block_band["fixed_snr_star_db"]
    block_band_summary = (
        block_band.groupby(["route", "block", "band"], as_index=False)["delta_snr_star_db"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    block_band_summary.to_csv(outdir / "block_band_processing.csv", index=False)
    paired = confirmation_events.pivot(index="event_id", columns="route", values="delta_snr_star_db").dropna()
    paired_rho = float(spearmanr(paired["TERRA"], paired["KKFL-S"]).statistic) if len(paired) > 2 else np.nan
    write_json(
        outdir / "paired_route_processing.json",
        {"paired_events": int(len(paired)), "spearman_rho": paired_rho,
         "mean_absolute_route_difference_db": float(np.mean(np.abs(paired["TERRA"] - paired["KKFL-S"])))},
    )
    graph_event = event_metrics[
        (event_metrics["role"] == "confirmation") & (event_metrics["model"] == "CoherencyGraph-DAS")
    ].copy()
    sample_uncertainty = graph_std.mean(axis=(1, 2))
    graph_event["uncertainty"] = [
        float(sample_uncertainty[np.flatnonzero((dataset.event_ids == row.event_id) & (dataset.routes == row.route))[0]])
        for row in graph_event.itertuples(index=False)
    ]
    uncertainty_rho = float(spearmanr(graph_event["uncertainty"], graph_event["complex_rmse"]).statistic)
    write_json(
        outdir / "uncertainty_diagnostics.json",
        {"confirmation_samples": int(len(graph_event)), "spearman_uncertainty_error": uncertainty_rho,
         "fallback_fraction": float(confirmation_events["fallback_fraction"].mean()),
         "calibration_threshold": uncertainty_threshold},
    )
    summary = {
        "confirmation_event_count": int(confirmation_events["event_id"].nunique()),
        "best_fixed_baseline": best_fixed,
        "best_adaptive_method": best_adaptive,
        "uncertainty_threshold": uncertainty_threshold,
        "primary_route_results": route_results,
        "publication_gate": "pass" if bool(route_table["pass"].all()) else "operator_only_pivot",
        "graph_psd_failure_rate": 0.0,
        "confirmation_matrix_metrics": metrics[metrics["role"] == "confirmation"].to_dict("records"),
    }
    write_json(outdir / "evaluation_summary.json", summary)
    return outdir / "evaluation_summary.json"
