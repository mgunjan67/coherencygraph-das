"""Bind the implemented fixed Ricker comparator to root's predeclared scope.

Metadata only; does not execute scientific data and never overwrites a protocol.
Run after all implementation edits, then include its output in root scope seal.
"""
from pathlib import Path
import json
from execution_guard import sha

HERE = Path(__file__).resolve().parent


def main():
    parent_path = HERE.parent/"protocol.json"
    parent = json.loads(parent_path.read_text())
    if (parent["experiment_a"]["expected_added_rows"]!=86400
            or parent["assessment_96h_disabled"] is not True):
        raise PermissionError("Unexpected parent scope")
    target = HERE/"ricker_protocol.json"
    if target.exists():
        raise FileExistsError("Do not overwrite a frozen scientific protocol")
    proposal = json.loads((HERE/"SAVED_COMPONENT_COMPARATOR_PROTOCOL_PROPOSAL.json").read_text())
    proposal.update(status="FROZEN_DEVELOPMENT_COMPARATOR",
        parent_protocol=str(parent_path),parent_protocol_sha256=sha(parent_path),
        scope_seal_required=True,execution_source_sha256={name:sha(HERE/name) for name in (
            "run_lcmv_ricker_extension.py","derivative_lcmv.py")})
    proposal["execution_budget_proposal"]["new_scratch_bytes"] = 1073741824
    target.write_text(json.dumps(proposal,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(path=str(target),sha256=sha(target),scientific_run=False)))


if __name__=="__main__":
    main()
