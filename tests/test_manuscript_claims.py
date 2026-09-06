"""Check revised generated claims, without requiring superseded claims in prose."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript/cageo_submission"
pytestmark = pytest.mark.skipif(not (PAPER / 'main.tex').exists(),
    reason='Private manuscript-source tests are not part of the public code/data release')
AUDIT = ROOT / "reports/methodological_audit"
REVISION = ROOT / "reports/critical_review"


def macros() -> dict[str, str]:
    text = (PAPER / "generated/numbers.tex").read_text()
    text += (PAPER / "generated/additional_numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}", text))


def test_revised_primary_values_are_generated_from_matched_results() -> None:
    text = (PAPER / "main.tex").read_text(encoding="utf8")
    text += (PAPER / "submission_revision_body.tex").read_text(encoding="utf8")
    values = macros()
    scores = pd.read_csv(REVISION / "matched_summary.csv").set_index("model")
    mapping = {"BlockRidge": "block_ridge", "FullRidge": "full_ridge",
               "FullMlpDirect": "full_mlp_direct", "FullMlpPsd": "full_mlp_psd",
               "StateDirect": "state_direct", "StatePsd": "state_psd"}
    for macro, model in mapping.items():
        assert values[macro] == f"{scores.loc[model, 'mean']:.4f}"
        # The historical full comparison remains generated and archived;
        # only the central historical pair must stay in the revised prose.
        if macro in {"StatePsd", "BlockRidge"}:
            assert "\\" + macro + "{}" in text
    summary = json.loads((REVISION / "manuscript_summary.json").read_text())
    for stem, method in [("LearnedDb", "Learned revised recurrence"),
                         ("ClassicalDb", "Classical shrinkage"),
                         ("OracleDb", "Oracle same estimate")]:
        row = next(r for r in summary["downstream"] if r["method"] == method
                   and r["reference"] == "Diagonal" and r["frame"] == "corrected_frame"
                   and r["evaluation"] == "full_window_in_sample")
        for suffix, column in [("", "mean"), ("Low", "low"), ("High", "high")]:
            assert values[stem + suffix] == f"{row[column]:.4f}"
    assert int(values["AmbiguitySigns"]) == summary["ambiguity_sign_identified"]
    assert int(values["AmbiguityCases"]) == summary["ambiguity_cases"]


def test_historical_endpoints_remain_explicitly_separate() -> None:
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf8")
    historical = (PAPER / "generated/s_historical.tex").read_text()
    source = pd.read_csv(AUDIT / "corrected_baseline_summary.csv")
    source = source[source.role == "architecture_test"].set_index("model")
    for model in ["State-space PSD", "Local-state PSD", "Ordinary ridge"]:
        assert f"{source.loc[model, 'mean_nrmse']:.6f}" in historical
    assert "not the revised all-cell comparison" in supplement
    held = pd.read_csv(AUDIT / "held_separation_summary.csv")
    table = (PAPER / "generated/s_held_lags.tex").read_text()
    for row in held[held.model == "Ridge + linear_complex"].itertuples():
        assert f"{row.nrmse:.3f}" in table
    assert "Ridge interpolation is a more relevant control" in supplement


def test_authorship_task_and_public_release_claims_remain_bounded() -> None:
    text = (PAPER / "main.tex").read_text(encoding="utf8")
    text += (PAPER / "submission_revision_body.tex").read_text(encoding="utf8")
    assert "mgunjan67@gmail.com" in text
    assert "d6622300231@g.siit.tu.ac.th" in text
    assert "vijay" not in text.lower() and "ghimire" not in text.lower()
    assert "Independent Researcher" in text and "Kathmandu" in text
    assert "not causal forecasting" in text
    assert "earthquake forecasting and operational early warning are not established" in text.lower()
    assert "not a temporal forecasting model" in text
    assert "https://github.com/mgunjan67/coherencygraph-das" in text
    import json
    receipt = json.loads((PAPER.parents[1] / "reports/critical_review/public_repository_verification.json").read_text())
    assert receipt["verified_public_browsable_source"] is True
    assert receipt["anonymous_clone_tests_passed"] == 11
    assert "PUBLIC-REPOSITORY-URL" not in text
    assert "independent human replication" in text
    assert "8" in macros()["AmbiguitySigns"]
