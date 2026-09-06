from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.methodological_audit import (
    _generalized_weight,
    _nearest_psd_toeplitz,
    _ratio,
)
from coherencygraph_das.models import spectral_basis


def test_legacy_65_bin_grid_has_documented_antiperiodic_alias() -> None:
    q_bins = 65
    cosine, sine, _, _ = spectral_basis(
        q_bins, [24, 89], mode="legacy_channel_lag"
    )
    phase = np.asarray(cosine + 1j * sine)
    assert np.max(np.abs(phase[1] + phase[0])) < 1e-5


def test_nearest_psd_toeplitz_is_admissible_at_supervised_lags() -> None:
    lags = np.asarray([3, 5, 8, 13, 21, 34, 55, 89])
    rng = np.random.default_rng(7)
    values = 0.4 * (rng.normal(size=len(lags)) + 1j * rng.normal(size=len(lags)))
    fitted = _nearest_psd_toeplitz(values, lags, iterations=8)
    assert np.isfinite(fitted).all()
    assert np.max(np.abs(fitted)) <= 1.0 + 1e-6


def test_source_group_roles_do_not_overlap() -> None:
    split = pd.read_csv(ROOT / "reports" / "hybrid_search" / "architecture_test_split.csv")
    groups = {
        role: set(group.source_group_id)
        for role, group in split.groupby("hybrid_role")
    }
    roles = list(groups)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1 :]:
            assert groups[left].isdisjoint(groups[right])


def test_calibration_and_architecture_test_events_do_not_overlap() -> None:
    split = pd.read_csv(ROOT / "reports" / "hybrid_search" / "architecture_test_split.csv", dtype={"event_id": str})
    calibration = set(split.loc[split.hybrid_role == "calibration", "event_id"])
    test = set(split.loc[split.hybrid_role == "architecture_test", "event_id"])
    assert calibration.isdisjoint(test)


def test_declared_channel_lags_map_to_declared_metric_separations() -> None:
    lags = np.asarray([3, 5, 8, 13, 21, 34, 55, 89], dtype=float)
    metres = lags * 9.5714288
    expected = np.asarray([28.7, 47.9, 76.6, 124.4, 201.0, 325.4, 526.4, 851.9])
    assert np.allclose(metres, expected, atol=0.06)


def test_oracle_generalized_weight_is_not_worse_than_diagonal_weight() -> None:
    rng = np.random.default_rng(11)
    factor = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    target = factor @ factor.conj().T + 0.1 * np.eye(8)
    noise = np.diag(np.linspace(0.8, 1.2, 8))
    oracle = _generalized_weight(target, noise, 0.001)
    diagonal = _generalized_weight(np.diag(np.diag(target)), noise, 0.001)
    assert _ratio(target, noise, oracle) >= _ratio(target, noise, diagonal) - 1e-9


def test_downstream_saved_comparators_use_identical_32_channel_support() -> None:
    path = ROOT / "reports" / "methodological_audit" / "downstream_oracle_test_grid.parquet"
    table = pd.read_parquet(path)
    assert set(table.channels) == {32}
    counts = table.groupby("method").size()
    assert counts.nunique() == 1
