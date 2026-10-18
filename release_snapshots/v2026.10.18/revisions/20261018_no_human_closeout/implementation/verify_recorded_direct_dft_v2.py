"""Transparent precision-semantics repair of the frozen independent verifier.

The full DFT, direct solves, case selection and tolerance are unchanged. No
scientific output or sealed source is overwritten. This wrapper replaces only
the independently coded matrix constructor's pre-arithmetic dtype conversion.
It still imports no scientific filter, scoring or LCMV helper.
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from execution_guard import sha

spec = importlib.util.spec_from_file_location("frozen_independent_dft_v1", HERE / "verify_recorded_direct_dft.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)

def construct_matrix_f64(context, kernel, method):
    # Match the scientific program's declared precision BEFORE multiplication,
    # square root, diagonal extraction or shrinkage, not after those operations.
    context = np.asarray(context, dtype=np.complex128)
    if kernel is not None:
        kernel = np.asarray(kernel, dtype=np.complex128)
    powers = np.diagonal(context, axis1=-2, axis2=-1).real
    diagonal = powers[:, :, None] * np.eye(32)
    if method in ("neural", "ridge"):
        return .25 * kernel * np.sqrt(powers[:, :, None] * powers[:, None, :]) + .75 * diagonal
    fraction = .75 if method == "context_0.75" else .25
    return fraction * context + (1 - fraction) * diagonal

def main():
    frozen.construct_matrix = construct_matrix_f64
    frozen.main()
    path = HERE / "recorded_component_results/DIRECT_DFT_VERIFICATION.json"
    receipt = json.loads(path.read_text())
    assert receipt["tolerance"] == 2e-8 and receipt["cases_replayed"] == 384
    receipt.update(
        verifier_version="v2_pre_arithmetic_complex128_semantics",
        frozen_v1_verifier_sha256=sha(HERE / "verify_recorded_direct_dft.py"),
        verifier_sha256=sha(__file__),
        retained_v1_failure_sha256=sha(HERE / "DIRECT_DFT_V1_FAILURE.json"),
        fix_scope="Only pre-arithmetic context/kernel dtype conversion. FullDFT, directsolve, all384cases/12metrics and2e-8tolerance unchanged. Saved field results untouched.",
        failure_history_retained=True,
    )
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("V2 verification receipt identifies wrapper and retained v1 precision failure", flush=True)

if __name__ == "__main__":
    main()
