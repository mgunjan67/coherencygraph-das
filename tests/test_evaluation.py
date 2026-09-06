from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.evaluation import complex_metrics, greedy_nonnegative_weights


def test_complex_metrics_are_zero_for_identity_prediction() -> None:
    target = np.zeros((8, 4, 8, 2), dtype=float)
    mask = np.ones((4, 8), dtype=bool)
    metrics = complex_metrics(target.copy(), target, mask)
    assert metrics["complex_rmse"] == 0.0


def test_greedy_weights_are_nonnegative_and_sparse() -> None:
    target = np.eye(8) + 0.2 * (np.ones((8, 8)) - np.eye(8))
    noise = np.eye(8)
    weights = greedy_nonnegative_weights(target, noise, 4)
    assert np.all(weights >= 0)
    assert np.count_nonzero(weights) == 4
    assert np.isclose(np.linalg.norm(weights), 1.0)

