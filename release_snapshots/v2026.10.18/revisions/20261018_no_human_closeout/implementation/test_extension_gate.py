import json
from pathlib import Path
import pytest

from run_lcmv_ricker_extension import load_gate,sha,BLOCKS,HERE


def base():
    return dict(status="FROZEN_DEVELOPMENT_COMPARATOR",blocks=list(BLOCKS),
        assessment_96h_disabled=True,model_retraining=False,
        model_inference_or_ridge_projection=False,primary_cohort_or_values_changed=False,
        expected_new_rows=86400,execution_source_sha256={
            name:sha(HERE/name) for name in ("run_lcmv_ricker_extension.py","derivative_lcmv.py")})


@pytest.mark.parametrize("kind",["proposal","hash","source","scope","assessment","count"])
def test_gate_rejects_unapproved_or_modified_design_before_waveform_read(tmp_path,kind):
    p = base()
    if kind == "proposal": p["status"] = "PROPOSAL_ONLY"
    if kind == "source": p["execution_source_sha256"]["derivative_lcmv.py"] = "0"*64
    if kind == "scope": p["blocks"] = p["blocks"][:-1]
    if kind == "assessment": p["assessment_96h_disabled"] = False
    if kind == "count": p["expected_new_rows"] = 42
    path = tmp_path/"protocol.json"
    path.write_text(json.dumps(p),encoding="utf-8")
    expected = "0"*64 if kind == "hash" else sha(path)
    with pytest.raises(PermissionError):
        load_gate(path,expected)


def test_exact_explicit_bound_gate_accepts_without_reading_field_arrays(tmp_path):
    path = tmp_path/"protocol.json"
    path.write_text(json.dumps(base()),encoding="utf-8")
    assert load_gate(path,sha(path))["expected_new_rows"] == 86400
