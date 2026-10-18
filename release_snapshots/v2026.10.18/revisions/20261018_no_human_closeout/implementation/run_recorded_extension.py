"""Gated controlled recorded-wavefield transplantation, not phase validation."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil

from derivative_lcmv import derivative_frequency_weights
from recorded_component_metrics import evaluate_recorded_component
from execution_guard import sha,require_owned_watchdog,verify_scope

HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent
ROOT = HERE.parents[2]
OLD = ROOT/"revisions/20261010_forecast_processing"
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(ROOT/"src"))
from forecast_adapter import context_matrix,forecast_matrix,STARTS
from frequency_filter import frequency_weights,apply_filter

BLOCKS = tuple(f"2024-08-{day:02d}_{hour}" for day in range(4,8) for hour in ("00","12"))
SOURCES = ("kef0131","kef0140")
METHODS = ("neural","ridge","conventional","context_0.25","context_0.75","lcmv_context_0.25")
KEYS = ["source_event","block","cutoff","aperture","method"]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def load_sources(protocol_sha256):
    folder = SCOPE/"evidence_audit/decoded_sources"
    receipt_path = folder/"source_extraction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status")!="COMPLETE_BOUNDED_SOURCE_EXTRACTION_ONLY"
            or receipt.get("parent_protocol_sha256")!=protocol_sha256
            or tuple(r["event_id"] for r in receipt["sources"])!=SOURCES):
        raise PermissionError("Exact complete metadata-frozen source extraction required")
    arrays = []
    for row in receipt["sources"]:
        path = folder/(row["event_id"]+"_source_component.npz")
        if Path(row["output"]).resolve()!=path.resolve() or sha(path)!=row["output_sha256"]:
            raise RuntimeError("Exact source component file/hash mismatch")
        with np.load(path,allow_pickle=False) as source:
            component = source["component"]
            if (component.shape!=(175,381) or not np.isfinite(component).all()
                    or not np.array_equal(source["roi_recorded_ids"],np.arange(4000,5901,5))
                    or not np.array_equal(source["relative_time_seconds"],np.arange(175)/25)):
                raise RuntimeError("Source component sample/channel support mismatch")
        import hashlib
        if hashlib.sha256(component.tobytes(order="C")).hexdigest()!=row["component_sha256"]:
            raise RuntimeError("Decoded component value hash mismatch")
        arrays.append(component)
    return np.asarray(arrays),receipt,sha(receipt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-protocol",type=Path,required=True)
    parser.add_argument("--parent-sha256",required=True)
    args = parser.parse_args()
    require_owned_watchdog()
    started = time.perf_counter()
    protocol,seal = verify_scope(args.parent_protocol,args.parent_sha256)
    p = protocol["experiment_b"]
    if (tuple(p["source_events"])!=SOURCES or tuple(p["backgrounds"])!=BLOCKS
            or tuple(p["methods"])!=METHODS or p["source_scale"]!=1
            or p["expected_rows"]!=23040):
        raise PermissionError("This fixed runner cannot execute a different source/method/design")
    components,source_receipt,source_receipt_sha = load_sources(args.parent_sha256)
    with np.load(SCOPE/"evidence_audit/source_catalog_steering.npz",allow_pickle=False) as bank:
        delays = bank["delays_seconds"]
        if (delays.shape!=(2,8,32) or not np.isfinite(delays).all()
                or not np.array_equal(bank["aperture_starts"],STARTS)
                or not np.array_equal(bank["recorded_roi_ids"],np.arange(4000,5901,5))):
            raise RuntimeError("Frozen geometry/nominal steering mismatch")
    dest = HERE/"recorded_component_results"
    if dest.exists():
        raise FileExistsError("Preserve any prior/partial experiment; no unlogged rerun")
    dest.mkdir()
    all_rows,completed,inputs = [],[],{}
    maximum_linearity_error,peak_rss = 0.0,0
    caps = protocol["resource_caps"]
    try:
        for block in BLOCKS:
            folder = OLD/"development_outcomes"/block
            receipt = json.loads((folder/"receipt.json").read_text())
            archive_path = folder/"adapter_arrays.npz"
            if sha(archive_path)!=receipt["adapter_arrays_sha256"]:
                raise RuntimeError("Archived adapter array hash mismatch")
            inputs[str(archive_path)] = receipt["adapter_arrays_sha256"]
            with np.load(archive_path,allow_pickle=False) as archive:
                arrays = {name:archive[name] for name in archive.files}
            if (arrays["context"].shape!=(30,8,4,32,32)
                    or arrays["background"].shape!=(30,175,381)
                    or not np.array_equal(arrays["starts"],STARTS)
                    or not np.array_equal(arrays["cutoffs"],np.arange(120,3601,120))):
                raise RuntimeError("Exact archived background design changed")
            rows = []
            for ci,cutoff in enumerate(arrays["cutoffs"]):
                peak_rss = max(peak_rss,psutil.Process().memory_info().rss)
                if time.perf_counter()-started>900 or peak_rss>caps["peak_process_rss_bytes"]:
                    raise RuntimeError("Per-panel bounded execution cap reached")
                for aperture,left in enumerate(STARTS):
                    ctx = arrays["context"][ci,aperture]
                    background = arrays["background"][ci,:,left:left+32]
                    matrices = dict(neural=forecast_matrix(arrays["neural_kernel"][ci,aperture],ctx),
                        ridge=forecast_matrix(arrays["ridge_kernel"][ci,aperture],ctx),
                        conventional=ctx,
                        **{"context_0.25":context_matrix(ctx,.25),"context_0.75":context_matrix(ctx,.75)})
                    for si,source_id in enumerate(SOURCES):
                        component = components[si,:,left:left+32]
                        nominal = delays[si,aperture]
                        weights = {name:frequency_weights(matrix,175,nominal,conventional=name=="conventional")
                                   for name,matrix in matrices.items()}
                        weights["lcmv_context_0.25"] = derivative_frequency_weights(matrices["context_0.25"],175,nominal)
                        reference = weights["conventional"]
                        for method in METHODS:
                            w = weights[method]
                            metrics = evaluate_recorded_component(background,component,w,reference)
                            # Everycase gets its own numerical linearity check;
                            # no clean-source inference follows from linearity.
                            mixed = apply_filter(background+component,w)
                            separate = apply_filter(background,w)+apply_filter(component,w)
                            scale = max(float(np.max(np.abs(mixed))),float(np.max(np.abs(separate))),1e-15)
                            error = float(np.max(np.abs(mixed-separate))/scale)
                            if not np.isfinite(error) or error>1e-10:
                                raise RuntimeError("All-case separated-component linearity check failed")
                            maximum_linearity_error = max(maximum_linearity_error,error)
                            rows.append(dict(source_event=source_id,source_recording_date="2024-08-02",
                                block=block,background_date=block[:10],cutoff=int(cutoff),aperture=aperture,
                                method=method,linearity_scaled_error=error,**metrics))
            frame = pd.DataFrame(rows)
            if len(frame)!=2880 or frame.duplicated(KEYS).any():
                raise RuntimeError("Complete recorded-component panel case count failed")
            frame.to_parquet(dest/f"{block}_cases.parquet",index=False)
            write(dest/f"{block}_receipt.json",dict(status="PASS_RECORDED_COMPONENT_BLOCK",rows=len(frame),
                case_sha256=sha(dest/f"{block}_cases.parquet"),elapsed_total_seconds=time.perf_counter()-started))
            all_rows.append(frame)
            completed.append(block)
            print(block,len(frame),f"{time.perf_counter()-started:.2f}s",flush=True)
        combined = pd.concat(all_rows,ignore_index=True)
        if len(combined)!=23040 or combined.duplicated(KEYS).any():
            raise RuntimeError("Full recorded-component panel incomplete")
        combined.to_parquet(dest/"all_cases.parquet",index=False)
        write(dest/"receipt.json",dict(status="PASS_CONTROLLED_RECORDED_COMPONENT_DEVELOPMENT_NOT_PHASE_VALIDATION",
            rows=len(combined),source_events=list(SOURCES),background_blocks=list(BLOCKS),methods=list(METHODS),
            maximum_linearity_scaled_error=maximum_linearity_error,linearity_checks=len(combined),
            elapsed_seconds=time.perf_counter()-started,sampled_worker_peak_rss_bytes=peak_rss,
            input_sha256=inputs,source_extraction_receipt_sha256=source_receipt_sha,
            parent_protocol_sha256=args.parent_sha256,scope_seal_sha256=sha(SCOPE/"scope_seal.json"),
            case_sha256=sha(dest/"all_cases.parquet"),assessment_96h_disabled=True,
            new_human_labels=False,no_training_or_model_inference=True,
            reference="Conventional steered sum of same known added recorded component, not clean truth",
            source_purity="Unseparated recorded source-window ambient/instrument response retained",
            evidence_unit="Two source earthquakes sharing one source day, crossed with four exposed background dates"))
    except Exception as error:
        write(dest/"FAILED_RECEIPT.json",dict(status="STOPPED_PRESERVE_PARTIAL_NO_SUBSTITUTION",
            error=repr(error),completed_blocks=completed,elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    main()
