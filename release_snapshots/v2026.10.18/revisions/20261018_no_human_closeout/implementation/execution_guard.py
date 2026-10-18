"""Exact local scope/process guards; no data or model loading."""
import hashlib
import json
import os
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(2**20),b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_owned_watchdog():
    expected = os.environ.get("CLOSEOUT_WATCHDOG_PID")
    if expected is None:
        raise PermissionError("Field scoring requires the approved external owned-process watchdog")
    # Windows venv redirector may introduce one intermediate Python parent.
    if int(expected) not in [p.pid for p in psutil.Process().parents()[:2]]:
        raise PermissionError("Exact watchdog ancestor mismatch")


def verify_scope(protocol_path,protocol_sha256):
    protocol_path = Path(protocol_path).resolve()
    if protocol_path != (SCOPE/"protocol.json").resolve() or sha(protocol_path)!=protocol_sha256:
        raise PermissionError("Exact local root protocol/hash required")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("assessment_96h_disabled") is not True or protocol.get("new_training") is not False
            or protocol.get("new_architecture_or_tuning_search") is not False):
        raise PermissionError("Root scope cannot unlock training, tuning or96hour assessment")
    seal = json.loads((SCOPE/"scope_seal.json").read_text(encoding="utf-8"))
    if seal.get("protocol_sha256")!=protocol_sha256 or seal.get("assessment_96h_disabled") is not True:
        raise PermissionError("Root scope seal mismatch")
    for path,expected in seal["source_hashes"].items():
        if sha(path)!=expected:
            raise RuntimeError(f"Sealed source/input changed: {path}")
    return protocol,seal
