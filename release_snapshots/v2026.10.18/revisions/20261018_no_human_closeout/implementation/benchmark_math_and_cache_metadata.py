"""Bounded synthetic CPU benchmark and cached-array HEADER inventory only.

No field array payload is loaded and no field outcome is produced. Existing
archive SHA256 verification can be requested separately at the final run gate.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import psutil

from derivative_lcmv import derivative_frequency_weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT/"revisions/20261010_forecast_processing"
sys.path.insert(0, str(OLD))
from frequency_filter import frequency_weights


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(20261018)
    z = rng.normal(size=(4,32,32)) + 1j*rng.normal(size=(4,32,32))
    matrices = z @ z.conj().swapaxes(-1,-2)/32 + .3*np.eye(32)
    delays = .004*(np.arange(32)-15.5)
    timing = {}
    for name, function in (("archived_mvdr", frequency_weights),
                           ("derivative_lcmv", derivative_frequency_weights)):
        function(matrices,175,delays)  # one declared warm-up, not outcome selection
        durations = []
        for _ in range(20):
            tic = time.perf_counter()
            function(matrices,175,delays)
            durations.append(time.perf_counter()-tic)
        timing[name] = dict(repeats=20,mean_seconds=float(np.mean(durations)),
            median_seconds=float(np.median(durations)),
            maximum_seconds=float(np.max(durations)),samples=175,channels=32,bands=4)
    inventory = []
    for day in range(4,8):
        for hour in ("00","12"):
            block = f"2024-08-{day:02d}_{hour}"
            archive = OLD/"development_outcomes"/block/"adapter_arrays.npz"
            receipt_path = archive.parent/"receipt.json"
            receipt = json.loads(receipt_path.read_text())
            members = []
            with zipfile.ZipFile(archive) as file:
                for entry in file.infolist():
                    with file.open(entry) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1,0):
                            shape,fortran,dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version in ((2,0),(3,0)):
                            shape,fortran,dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise ValueError(f"Unsupported NPY header: {version}")
                    members.append(dict(name=entry.filename,shape=shape,dtype=str(dtype),
                        decoded_member_bytes=entry.file_size,compressed_bytes=entry.compress_size,
                        header_only_no_array_payload_loaded=True))
            inventory.append(dict(block=block,path=str(archive),bytes=archive.stat().st_size,
                historical_sha256=receipt["adapter_arrays_sha256"],
                receipt_sha256=sha(receipt_path),archive_hash_recomputed=False,members=members))
    per_call = timing["derivative_lcmv"]["mean_seconds"]
    result = dict(status="SYNTHETIC_MATH_BENCHMARK_AND_HEADER_INVENTORY_ONLY",
        no_field_payload_loaded=True, no_models_loaded=True,no_waveform_experiment_run=True,
        source_sha256={p.name:sha(p) for p in (Path(__file__),HERE/"derivative_lcmv.py")},
        python=sys.version,numpy=np.__version__,thread_count_environment=2,
        process_rss_at_finish_bytes=psutil.Process().memory_info().rss,
        elapsed_seconds=time.perf_counter()-start,timing=timing,inventory=inventory,
        archive_bytes_total=sum(x["bytes"] for x in inventory),
        hypothetical_full_old_design_new_baseline=dict(
            independent_dates=4,recording_hours=8,cutoffs_per_hour=30,apertures=8,
            assumed_slopes=3,wavelets=3,response_conditions=5,new_metric_rows=86400,
            filter_weight_calls=5760,
            synthetic_measured_weight_only_seconds=5760*per_call,
            excluded_costs=["archive decompression","source-component construction",
                "component FFT/metrics","table/figure writing","independent verification"],
            execution_authorized_by_this_file=False))
    (HERE/"math_benchmark_and_cache_inventory.json").write_text(
        json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(status=result["status"],timing=timing,
        archive_bytes_total=result["archive_bytes_total"],elapsed_seconds=result["elapsed_seconds"])))


if __name__ == "__main__":
    main()
