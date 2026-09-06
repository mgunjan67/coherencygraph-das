from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.reliability import flat_frequency_false_positive, synthetic_identifiability


def test_synthetic_operator_recovery_and_psd() -> None:
    result = synthetic_identifiability(seed=3, replicates=20)
    assert result["decoder_psd_failure_rate"] == 0.0
    assert result["median_complex_rmse"] < 0.25


def test_flat_frequency_null_controls_false_positives() -> None:
    result = flat_frequency_false_positive(seed=9, experiments=200)
    assert result["false_positive_rate"] <= 0.075
