from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import resolve, write_json
from .data import load_operator_dataset


def discover_regimes(cfg: dict) -> None:
    dataset = load_operator_dataset(cfg)
    complex_target = dataset.targets[..., 0] + 1j * dataset.targets[..., 1]
    rows = []
    for event_id in np.unique(dataset.event_ids):
        selected = np.flatnonzero(dataset.event_ids == event_id)
        target = complex_target[selected].mean(axis=(0, 1))
        role = str(dataset.roles[selected[0]])
        ranks = []
        for path in dataset.paths[selected]:
            with np.load(path, allow_pickle=False) as loaded:
                ranks.append(loaded["target_effective_rank"])
        rows.append(
            {
                "event_id": event_id,
                "role": role,
                "features": np.concatenate([target.real.ravel(), target.imag.ravel()]),
                "effective_rank": np.mean(np.stack(ranks), axis=(0, 1)),
                "target": target,
            }
        )
    development = [row for row in rows if row["role"] == "development"]
    confirmation = [row for row in rows if row["role"] == "confirmation"]
    x_train = np.stack([row["features"] for row in development])
    x_test = np.stack([row["features"] for row in confirmation])
    mean, std = x_train.mean(0), x_train.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    z_train, z_test = (x_train - mean) / std, (x_test - mean) / std
    candidates = []
    for clusters in range(2, 6):
        model = KMeans(n_clusters=clusters, n_init=50, random_state=20260830).fit(z_train)
        candidates.append((silhouette_score(z_train, model.labels_), clusters, model))
    silhouette, clusters, model = max(candidates, key=lambda item: item[0])
    reference = model.labels_
    rng = np.random.default_rng(20260830)
    stability = []
    for replicate in range(200):
        sample = rng.integers(0, len(z_train), len(z_train))
        boot = KMeans(n_clusters=clusters, n_init=20, random_state=replicate).fit(z_train[sample])
        stability.append(adjusted_rand_score(reference, boot.predict(z_train)))
    labels_train = reference
    labels_test = model.predict(z_test)
    assignment_rows = []
    for source, labels in [(development, labels_train), (confirmation, labels_test)]:
        for row, label in zip(source, labels, strict=True):
            assignment_rows.append(
                {"event_id": row["event_id"], "role": row["role"], "regime": int(label),
                 **{f"effective_rank_band_{i}": float(v) for i, v in enumerate(row["effective_rank"])}}
            )
    assignments = pd.DataFrame(assignment_rows)
    cohort = pd.read_parquet(resolve(cfg, cfg["paths"]["audit"]) / "frozen_cohort.parquet")
    cohort["event_id"] = cohort["event_id"].astype(str)
    assignments = assignments.merge(
        cohort[["event_id", "magnitude", "depth_km", "source_bearing_deg"]], on="event_id", how="left"
    )
    outdir = resolve(cfg, cfg["paths"]["results"])
    assignments.to_csv(outdir / "coherency_regime_assignments.csv", index=False)
    regime_summary = (
        assignments.groupby(["role", "regime"], as_index=False)
        .agg(events=("event_id", "count"), magnitude_mean=("magnitude", "mean"), depth_mean_km=("depth_km", "mean"),
             neff_0p5_1=("effective_rank_band_0", "mean"), neff_1_2=("effective_rank_band_1", "mean"),
             neff_2_4=("effective_rank_band_2", "mean"), neff_4_8=("effective_rank_band_3", "mean"))
    )
    regime_summary.to_csv(outdir / "coherency_regime_summary.csv", index=False)
    development_assignments = assignments[assignments["role"] == "development"]
    tests = {}
    for variable in ["magnitude", "depth_km"]:
        groups = [part[variable].dropna().to_numpy() for _, part in development_assignments.groupby("regime")]
        statistic, pvalue = kruskal(*groups)
        tests[variable] = {"kruskal_statistic": float(statistic), "pvalue_exploratory": float(pvalue)}
    write_json(
        outdir / "regime_discovery_summary.json",
        {"selected_clusters": int(clusters), "development_silhouette": float(silhouette),
         "median_bootstrap_adjusted_rand": float(np.median(stability)),
         "p10_bootstrap_adjusted_rand": float(np.quantile(stability, 0.10)),
         "development_events": len(development), "confirmation_events_assigned_without_refit": len(confirmation),
         "exploratory_covariate_tests": tests,
         "claim_status": "replicated descriptive regimes" if np.median(stability) >= 0.7 else "exploratory only"},
    )

