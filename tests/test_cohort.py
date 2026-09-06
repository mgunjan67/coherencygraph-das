from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.cohort import build_frozen_cohort
from coherencygraph_das.config import load_config


def test_frozen_cohort_is_group_disjoint_and_sized() -> None:
    cfg = load_config(ROOT / "configs" / "protocol_v1.0.yaml")
    frame = build_frozen_cohort(cfg)
    selected = frame[frame.cohort_role.isin(["development", "calibration", "confirmation"])]
    assert (selected.cohort_role == "confirmation").sum() == 30
    assert (selected.cohort_role == "calibration").sum() == 24
    groups = {
        role: set(selected.loc[selected.cohort_role == role, "source_group_id"])
        for role in ["development", "calibration", "confirmation"]
    }
    assert not groups["development"] & groups["calibration"]
    assert not groups["development"] & groups["confirmation"]
    assert not groups["calibration"] & groups["confirmation"]

