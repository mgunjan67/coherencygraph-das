from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.review_revision import (
    _generalized_weight,
    _held_lag_mask,
    _mask_blocks,
    _mask_lag_features,
)


def test_withheld_lag_features_are_removed_from_every_context_window() -> None:
    values = np.ones((2, 8, 137), dtype=float)
    masked = _mask_lag_features(values, [1, 6])
    expected = {
        start + band * 8 + lag
        for start in (0, 32, 64, 96)
        for band in range(4)
        for lag in (1, 6)
    }
    for column in range(137):
        if column in expected:
            assert np.all(masked[:, :, column] == 0)
        else:
            assert np.all(masked[:, :, column] == 1)


def test_held_lag_mask_keeps_only_declared_separations() -> None:
    base = np.ones((4, 8), dtype=bool)
    held = _held_lag_mask(base, [0, 2, 7])
    assert held.sum() == 12
    assert np.all(held[:, [0, 2, 7]])
    assert not np.any(held[:, [1, 3, 4, 5, 6]])


def test_missing_block_mask_zeros_data_and_marks_observation_state() -> None:
    values = np.ones((3, 8, 6), dtype=float)
    masked = _mask_blocks(values, missing_blocks=2, seed=17, has_mask_feature=True)
    for sample in masked:
        missing = np.flatnonzero(sample[:, -1] == 0)
        assert len(missing) == 2
        assert np.all(sample[missing] == 0)
        assert np.all(sample[np.setdiff1d(np.arange(8), missing), -1] == 1)


def test_generalized_weight_is_finite_and_unit_norm() -> None:
    signal = np.array([[2.0, 0.4], [0.4, 1.0]], dtype=np.complex128)
    noise = np.array([[1.0, 0.1], [0.1, 0.8]], dtype=np.complex128)
    weight = _generalized_weight(signal, noise, loading=1e-3)
    assert np.isfinite(weight).all()
    assert np.isclose(np.linalg.norm(weight), 1.0)
