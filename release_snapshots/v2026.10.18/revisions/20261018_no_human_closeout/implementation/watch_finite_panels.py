"""Owned-process watchdog: two fixed sequential panels, total cap1800seconds."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

from execution_guard import sha,verify_scope

HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def scratch_bytes():
    paths = [HERE/"lcmv_ricker_results",HERE/"recorded_component_results"]
    total = sum(p.stat().st_size for folder in paths if folder.exists()
                for p in folder.rglob("*") if p.is_file())
    total += sum(p.stat().st_size for p in HERE.glob("execution_*.log"))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel",choices=("ricker","recorded"),required=True)
    parser.add_argument("--parent-protocol",type=Path,required=True)
    parser.add_argument("--parent-sha256",required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    protocol,seal = verify_scope(args.parent_protocol,args.parent_sha256)
    caps = protocol["resource_caps"]
    if caps["scientific_execution_wall_seconds"]!=1800:
        raise PermissionError("This watchdog implements the exact1800second combined cap only")
    receipt_path = HERE/f"watchdog_{args.panel}_receipt.json"
    log_path = HERE/f"execution_{args.panel}.log"
    if receipt_path.exists() or log_path.exists():
        raise FileExistsError("Preserve previous watchdog attempt; no unlogged retry")
    previous_seconds = 0.0
    for panel in ("ricker","recorded"):
        path = HERE/f"watchdog_{panel}_receipt.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["status"]!="RESOURCE_WATCHDOG_PASS":
                raise RuntimeError("Prior scientific panel failed; no automatic rescue/continuation")
            previous_seconds += previous["elapsed_seconds"]
    remaining = min(900.0,1800.0-previous_seconds)
    if remaining<=0:
        raise RuntimeError("No fixed execution time remains")
    if args.panel=="ricker":
        child_protocol = HERE/"ricker_protocol.json"
        command = [sys.executable,str(HERE/"run_lcmv_ricker_extension.py"),
                   "--protocol",str(child_protocol),"--protocol-sha256",sha(child_protocol)]
    else:
        command = [sys.executable,str(HERE/"run_recorded_extension.py"),
                   "--parent-protocol",str(args.parent_protocol),"--parent-sha256",args.parent_sha256]
    environment = dict(os.environ,CLOSEOUT_WATCHDOG_PID=str(os.getpid()))
    peak_rss,peak_scratch,reason = 0,0,None
    with log_path.open("x",encoding="utf-8") as log:
        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
        owned_root = psutil.Process(child.pid)
        while child.poll() is None:
            rss = psutil.Process().memory_info().rss
            try:
                owned = [owned_root]+owned_root.children(recursive=True)
            except psutil.NoSuchProcess:
                owned = []
            for item in owned:
                try:
                    rss += item.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            used = scratch_bytes()
            peak_rss,peak_scratch = max(peak_rss,rss),max(peak_scratch,used)
            if time.perf_counter()-start>=remaining:
                reason = "panel_or_combined_wall_cap"
            elif rss>=caps["peak_process_rss_bytes"]:
                reason = "owned_process_tree_rss_cap"
            elif used>=caps["new_scientific_scratch_bytes"]:
                reason = "combined_new_scientific_scratch_cap"
            if reason:
                try:
                    for item in reversed(owned_root.children(recursive=True)):
                        try:item.kill()
                        except psutil.NoSuchProcess:pass
                    owned_root.kill()
                except psutil.NoSuchProcess:
                    pass
                break
            time.sleep(.2)
        returncode = child.wait(timeout=15)
    receipt = dict(status="RESOURCE_WATCHDOG_PASS" if reason is None and returncode==0 else "STOPPED_OR_FAILED",
        panel=args.panel,returncode=returncode,stopped_reason=reason,
        elapsed_seconds=time.perf_counter()-start,previous_panel_seconds=previous_seconds,
        panel_remaining_cap_seconds=remaining,combined_cap_seconds=1800,
        sampled_combined_peak_rss_bytes=peak_rss,combined_new_scientific_scratch_peak_bytes=peak_scratch,
        sampling_interval_seconds=.2,worker_pid=child.pid,watchdog_pid=os.getpid(),
        protocol_sha256=args.parent_sha256,scope_seal_sha256=sha(SCOPE/"scope_seal.json"),
        watchdog_sha256=sha(__file__),log_sha256=sha(log_path),
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(receipt_path,receipt)
    print(json.dumps(receipt),flush=True)
    if returncode!=0 or reason:
        raise SystemExit(1)


if __name__=="__main__":
    main()
