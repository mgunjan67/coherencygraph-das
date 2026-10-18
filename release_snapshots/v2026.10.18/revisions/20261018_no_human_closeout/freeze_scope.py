"""Seal inputs and predeclared scope; never alter a protected baseline."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "revisions/20261013_research_reassessment"
OLD = ROOT / "revisions/20261010_forecast_processing"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    scope = json.loads((HERE / "protocol.json").read_text())
    assert scope["assessment_96h_disabled"] and scope["new_training"] is False
    paths = [HERE / "protocol.json", Path(__file__)]
    paths += sorted((HERE / "implementation").glob("*.py"))
    paths += [HERE / "implementation/ricker_protocol.json"]
    paths += sorted((HERE / "evidence_audit").glob("*.py"))
    paths += [HERE / "evidence_audit/source_spec.json", HERE / "evidence_audit/source_seal.json"]
    paths += sorted((HERE / "evidence_audit").glob("*protocol*.json"))
    paths += sorted((HERE / "evidence_audit").glob("*steering*.npz"))
    for block in scope["experiment_b"]["backgrounds"]:
        paths += [OLD / "development_outcomes" / block / name
                  for name in ("adapter_arrays.npz", "cases.parquet", "receipt.json")]
    paths += [OLD / name for name in ("frequency_filter.py", "forecast_adapter.py", "processing_core.py", "DEVELOPMENT_PROTOCOL.json")]
    hashes = {str(p): sha(p) for p in paths}
    old_manifest = json.loads((BASE / "REASSESSMENT_ASSET_MANIFEST.json").read_text())
    protected = {str(BASE / row["path"]): row["sha256"] for row in old_manifest["files"]}
    # This manifest omits itself, whose seal is retained explicitly too.
    protected[str(BASE / "REASSESSMENT_ASSET_MANIFEST.json")] = sha(BASE / "REASSESSMENT_ASSET_MANIFEST.json")
    for path, expected in protected.items():
        assert sha(path) == expected, path
    dest = HERE / "scope_seal.json"
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=sha(HERE / "protocol.json"),
                  source_hashes=hashes, protected_baseline_sha256=protected,
                  all_new_field_outputs_uninspected=True,
                  heldout_accessed=False, assessment_96h_disabled=True)
    if dest.exists():
        previous = json.loads(dest.read_text())
        assert previous["source_hashes"] == hashes
        assert previous["protected_baseline_sha256"] == protected
        print("Existing scope and baseline seals verified")
    else:
        dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("Sealed", len(hashes), "inputs and", len(protected), "protected baseline assets")

if __name__ == "__main__":
    main()
