from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import resolve, sha256, write_json


def _rank(event_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{event_id}".encode()).hexdigest()


def _stratum(frame: pd.DataFrame) -> pd.Series:
    depth = pd.cut(
        frame["depth_km"], [-np.inf, 75.0, 110.0, np.inf],
        labels=["shallow", "intermediate", "deep"], right=False,
    ).astype(str)
    magnitude = np.where(frame["magnitude"] < 2.5, "low", "high")
    return depth + "_" + magnitude


def _apportion(counts: pd.Series, total: int) -> dict[str, int]:
    raw = counts / counts.sum() * total
    base = np.floor(raw).astype(int)
    remaining = total - int(base.sum())
    order = (raw - base).sort_values(ascending=False).index.tolist()
    for key in order[:remaining]:
        base.loc[key] += 1
    return {str(k): int(min(base.loc[k], counts.loc[k])) for k in counts.index}


def _select_unique_groups(
    candidates: pd.DataFrame,
    n: int,
    seed: int,
    excluded_groups: Iterable[str] = (),
    stratified: bool = True,
) -> pd.DataFrame:
    frame = candidates[~candidates["source_group_id"].isin(set(excluded_groups))].copy()
    frame["selection_rank"] = frame["event_id"].map(lambda x: _rank(str(x), seed))
    frame = frame.sort_values(["source_group_id", "selection_rank"])
    frame = frame.drop_duplicates("source_group_id", keep="first")
    if len(frame) < n:
        raise RuntimeError(f"only {len(frame)} source-disjoint candidates for requested {n}")
    frame["selection_stratum"] = _stratum(frame)
    if not stratified:
        return frame.sort_values("selection_rank").head(n).copy()
    quotas = _apportion(frame["selection_stratum"].value_counts(), n)
    pieces = []
    for key, quota in quotas.items():
        pieces.append(
            frame[frame["selection_stratum"] == key]
            .sort_values("selection_rank")
            .head(quota)
        )
    selected = pd.concat(pieces, ignore_index=True)
    if len(selected) < n:
        reserves = frame[~frame["event_id"].isin(selected["event_id"])].sort_values(
            "selection_rank"
        )
        selected = pd.concat([selected, reserves.head(n - len(selected))], ignore_index=True)
    return selected.sort_values(["archive_date", "event_id"]).reset_index(drop=True)


def _source_path(cfg: dict, key: str) -> Path:
    """Use the byte-identical archived metadata in portable release copies."""
    original = resolve(cfg, cfg["study"][key])
    archived = resolve(cfg, "data/provenance/" + original.name)
    return archived if archived.exists() else original


def build_frozen_cohort(cfg: dict) -> pd.DataFrame:
    manifest_path = _source_path(cfg, "source_manifest")
    split_path = _source_path(cfg, "source_split_table")
    manifest = pd.read_parquet(manifest_path)
    groups = pd.read_csv(split_path, dtype={"event_id": str})[
        ["event_id", "source_group_id"]
    ]
    manifest["event_id"] = manifest["event_id"].astype(str)
    manifest["archive_date"] = pd.to_datetime(manifest["archive_date"])
    frame = manifest.merge(groups, on="event_id", how="left", validate="one_to_one")
    q = cfg["quality_control"]
    frame["qc_pass"] = (
        frame["local_files_complete"].fillna(False)
        & frame["terra_hdf5_ok"].fillna(False)
        & frame["kkfls_hdf5_ok"].fillna(False)
        & (frame["terra_sample_rate_hz"] == q["sample_rate_hz"])
        & (frame["kkfls_sample_rate_hz"] == q["sample_rate_hz"])
        & (frame["terra_s_coverage"] >= q["minimum_s_pick_coverage"])
        & (frame["kkfls_s_coverage"] >= q["minimum_s_pick_coverage"])
        & frame["terra_s_median_sec"].between(
            q["minimum_s_time_sec"], q["maximum_s_time_sec"]
        )
        & frame["kkfls_s_median_sec"].between(
            q["minimum_s_time_sec"], q["maximum_s_time_sec"]
        )
    )
    frame["cohort_role"] = "excluded_qc"
    eligible = frame[frame["qc_pass"]].copy()
    split = cfg["split"]
    seed = int(split["seed"])
    confirm_period = pd.Period(split["confirmation_month"], freq="M")
    confirm_candidates = eligible[
        eligible["archive_date"].dt.to_period("M") == confirm_period
    ]
    confirmation = _select_unique_groups(
        confirm_candidates,
        int(split["confirmation_events"]),
        seed,
        stratified=True,
    )
    confirmation_groups = set(confirmation["source_group_id"])
    calibration_candidates = eligible[
        eligible["archive_date"].between(
            pd.Timestamp(split["calibration_start"]),
            pd.Timestamp(split["calibration_end"]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
        )
    ]
    calibration = _select_unique_groups(
        calibration_candidates,
        int(split["calibration_events"]),
        seed + 1,
        excluded_groups=confirmation_groups,
        stratified=True,
    )
    calibration_groups = set(calibration["source_group_id"])
    train = eligible[
        (eligible["archive_date"] < pd.Timestamp("2023-12-01"))
        & ~eligible["source_group_id"].isin(confirmation_groups | calibration_groups)
    ].copy()
    frame.loc[frame["event_id"].isin(train["event_id"]), "cohort_role"] = "development"
    frame.loc[
        frame["event_id"].isin(calibration["event_id"]), "cohort_role"
    ] = "calibration"
    frame.loc[
        frame["event_id"].isin(confirmation["event_id"]), "cohort_role"
    ] = "confirmation"
    frame.loc[
        frame["qc_pass"] & frame["cohort_role"].eq("excluded_qc"), "cohort_role"
    ] = "excluded_group_or_time"
    selected = frame[frame["cohort_role"].isin(["development", "calibration", "confirmation"])].copy()
    role_groups = {
        role: set(selected.loc[selected["cohort_role"] == role, "source_group_id"])
        for role in ["development", "calibration", "confirmation"]
    }
    for left, right in [("development", "calibration"), ("development", "confirmation"), ("calibration", "confirmation")]:
        if role_groups[left] & role_groups[right]:
            raise AssertionError(f"source-group leakage between {left} and {right}")
    return frame.sort_values(["archive_date", "event_id"]).reset_index(drop=True)


def run_audit(cfg: dict) -> Path:
    frame = build_frozen_cohort(cfg)
    outdir = resolve(cfg, cfg["paths"]["audit"])
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_out = outdir / "frozen_cohort.parquet"
    frame.to_parquet(manifest_out, index=False)
    frame.to_csv(outdir / "frozen_cohort.csv", index=False)
    selected = frame[frame["cohort_role"].isin(["development", "calibration", "confirmation"])]
    summary = {
        "protocol_sha256": sha256(cfg["_config_path"]),
        "source_manifest_sha256": sha256(_source_path(cfg, "source_manifest")),
        "total_manifest_events": int(len(frame)),
        "qc_pass_events": int(frame["qc_pass"].sum()),
        "selected_events": int(len(selected)),
        "role_counts": {k: int(v) for k, v in selected["cohort_role"].value_counts().items()},
        "role_source_groups": {
            role: int(part["source_group_id"].nunique())
            for role, part in selected.groupby("cohort_role")
        },
        "date_ranges": {
            role: [str(part["archive_date"].min().date()), str(part["archive_date"].max().date())]
            for role, part in selected.groupby("cohort_role")
        },
        "gauge_counts": {
            role: {str(k): int(v) for k, v in part["terra_gauge_length_m"].value_counts().items()}
            for role, part in selected.groupby("cohort_role")
        },
        "public_2024_waveform_status": "HTTP 403 AccessDenied verified 2026-08-30",
        "claim_boundary": "chronological late-2023 confirmation; no 2024 replication claim",
    }
    write_json(outdir / "audit_summary.json", summary)
    return manifest_out
