"""ONE fixed new comparator on saved exposed arrays; gated, no training/inference.

This file never modifies the original777,600 case rows. The parent must freeze
the exact protocol and supply its SHA256 explicitly before any field read.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import psutil

from derivative_lcmv import derivative_frequency_weights
from execution_guard import require_owned_watchdog,verify_scope

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT/"revisions/20261010_forecast_processing"
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(ROOT/"src"))
from forecast_adapter import context_matrix, forecast_matrix
from frequency_filter import (frequency_weights,ricker_wavelet,inject_component,
                               evaluate_components)

BLOCKS = tuple(f"2024-08-{day:02d}_{hour}" for day in range(4,8) for hour in ("00","12"))
METHOD = "derivative_lcmv_context_0.25"
CONDITIONS = (("matched",0,0),("delay_minus",-.001,0),("delay_plus",.001,0),
              ("amplitude_minus",0,-.2),("amplitude_plus",0,.2))
KEYS = ["block","cutoff","aperture","delay_step","wavelet_hz","condition","method"]
METRICS = ["background_output_power","signal_output_power","input_snr_db",
           "output_snr_db","snr_gain_db","signal_relative_mse","signal_power_ratio_db"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def load_gate(protocol_path,expected_sha):
    if sha(protocol_path) != expected_sha:
        raise PermissionError("Protocol SHA256 differs from explicit frozen execution binding")
    p = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN_DEVELOPMENT_COMPARATOR":
        raise PermissionError("Protocol is only a proposal; root freeze is required")
    if (tuple(p.get("blocks",())) != BLOCKS or p.get("assessment_96h_disabled") is not True
            or p.get("model_retraining") is not False
            or p.get("model_inference_or_ridge_projection") is not False
            or p.get("primary_cohort_or_values_changed") is not False
            or p.get("expected_new_rows") != 86400):
        raise PermissionError("Frozen gate does not authorize this exact bounded design")
    # The algorithm/design values are fixed in code, not taken from an editable
    # arbitraryconfig; source SHA is part of the explicit freeze record.
    source = p.get("execution_source_sha256",{})
    for path in (Path(__file__),HERE/"derivative_lcmv.py"):
        if source.get(path.name) != sha(path):
            raise PermissionError(f"Execution source not frozen: {path.name}")
    return p


def compare_fixture(old_cases,arrays,block):
    """First fixed cutoff/aperture, all405 old design/method rows, every block."""
    context = arrays["context"][0,0]
    background = arrays["background"][0,:,int(arrays["starts"][0]):int(arrays["starts"][0])+32]
    matrices = {f"context_{f:g}":context_matrix(context,f) for f in (0,.1,.25,.5,.75,1)}
    matrices.update(neural=forecast_matrix(arrays["neural_kernel"][0,0],context),
        ridge=forecast_matrix(arrays["ridge_kernel"][0,0],context),conventional=context)
    coordinate = np.arange(32)-15.5
    amplitude = np.sqrt(np.trace(context,axis1=-2,axis2=-1).real.mean()/32)
    rows = []
    for slope in (-.004,0,.004):
        weights = {m:frequency_weights(r,175,slope*coordinate,conventional=m=="conventional")
                   for m,r in matrices.items()}
        for wavelet_hz in (1,2,4):
            wavelet = amplitude*ricker_wavelet(central_frequency=wavelet_hz)
            for condition,delta,gradient in CONDITIONS:
                desired = inject_component(wavelet,(slope+delta)*coordinate,1+gradient*coordinate/15.5)
                for method,w in weights.items():
                    rows.append(dict(block=block,cutoff=int(arrays["cutoffs"][0]),aperture=0,
                        delay_step=slope,wavelet_hz=wavelet_hz,condition=condition,method=method,
                        **evaluate_components(background,desired,wavelet,w)))
    actual = pd.DataFrame(rows)
    reference = old_cases[(old_cases.cutoff==int(arrays["cutoffs"][0]))&(old_cases.aperture==0)]
    paired = actual.merge(reference,on=KEYS,suffixes=("_actual","_old"),validate="one_to_one")
    if len(paired)!=405 or len(reference)!=405:
        raise RuntimeError("Exact unchanged405-row fixture completeness failed")
    errors = {}
    for metric in METRICS:
        x,y = paired[metric+"_actual"].to_numpy(),paired[metric+"_old"].to_numpy()
        error = float(np.max(np.abs(x-y)/np.maximum(1.0,np.abs(y))))
        errors[metric] = error
        if not np.isfinite(error) or error>1e-9:
            raise RuntimeError(f"Old fixture mismatch: {block}/{metric}/{error}")
    return dict(rows=405,maximum_scaled_errors=errors,tolerance=1e-9)


def aggregate_tables(old,new):
    combined = pd.concat([old,new],ignore_index=True)
    if len(combined)!=864000 or combined.duplicated(KEYS).any():
        raise RuntimeError("Expanded all-method design completeness failed")
    by_date = combined.groupby(["date","condition","method"],sort=True).agg(
        rows=("snr_gain_db","size"),snr_gain_mean=("snr_gain_db","mean"),
        relative_mse_mean=("signal_relative_mse","mean"),
        snr_gain_q05=("snr_gain_db",lambda x:x.quantile(.05,interpolation="linear")),
        relative_mse_q95=("signal_relative_mse",lambda x:x.quantile(.95,interpolation="linear"))).reset_index()
    if len(by_date)!=200 or not (by_date.rows==4320).all():
        raise RuntimeError("Expanded date/condition/method row counts failed")
    return by_date


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--protocol-sha256",required=True)
    args = parser.parse_args(argv)
    require_owned_watchdog()
    protocol = load_gate(args.protocol,args.protocol_sha256)
    verify_scope(protocol["parent_protocol"],protocol["parent_protocol_sha256"])
    dest = HERE/"lcmv_ricker_results"
    if dest.exists():
        raise FileExistsError("Do not overwrite prior experiment; root must prepare a new documented revision")
    dest.mkdir()
    start = time.perf_counter()
    old_frames,new_frames,fixtures,completed = [],[],{},[]
    inputs = {}
    peak_rss = 0
    cap = protocol["execution_budget_proposal"]
    try:
        for block in BLOCKS:
            folder = OLD/"development_outcomes"/block
            receipt = json.loads((folder/"receipt.json").read_text())
            for filename,key in (("adapter_arrays.npz","adapter_arrays_sha256"),("cases.parquet","cases_sha256")):
                actual_sha = sha(folder/filename)
                if actual_sha != receipt[key]:
                    raise RuntimeError(f"Archived source hash changed: {block}/{filename}")
                inputs[str(folder/filename)] = actual_sha
            old = pd.read_parquet(folder/"cases.parquet")
            if len(old)!=97200 or old.duplicated(KEYS).any():
                raise RuntimeError(f"Archived case completeness failed: {block}")
            old_frames.append(old)
            # One block at a time; these exact saved arrays avoid costly model/
            # projection reruns and preserve the historical processing inputs.
            with np.load(folder/"adapter_arrays.npz",allow_pickle=False) as archive:
                arrays = {name:archive[name] for name in archive.files}
            if (arrays["context"].shape!=(30,8,4,32,32)
                    or arrays["background"].shape!=(30,175,381)
                    or not np.array_equal(arrays["cutoffs"],np.arange(120,3601,120))):
                raise RuntimeError("Exact archived array shape/cutoffs changed")
            fixtures[block] = compare_fixture(old,arrays,block)
            rows = []
            coordinate = np.arange(32)-15.5
            for ci,cutoff in enumerate(arrays["cutoffs"]):
                elapsed = time.perf_counter()-start
                peak_rss = max(peak_rss,psutil.Process().memory_info().rss)
                if elapsed>cap["wall_seconds"] or peak_rss>cap["process_tree_rss_bytes"]:
                    raise RuntimeError("Declared bounded execution cap reached")
                for aperture,left in enumerate(arrays["starts"]):
                    context = arrays["context"][ci,aperture]
                    background = arrays["background"][ci,:,left:left+32]
                    covariance = context_matrix(context,.25)
                    amplitude = np.sqrt(np.trace(context,axis1=-2,axis2=-1).real.mean()/32)
                    for slope in (-.004,0,.004):
                        weights = derivative_frequency_weights(covariance,175,slope*coordinate)
                        for wavelet_hz in (1,2,4):
                            wavelet = amplitude*ricker_wavelet(central_frequency=wavelet_hz)
                            for condition,delta,gradient in CONDITIONS:
                                desired = inject_component(wavelet,(slope+delta)*coordinate,
                                                           1+gradient*coordinate/15.5)
                                rows.append(dict(block=block,date=block[:10],cutoff=int(cutoff),
                                    aperture=aperture,delay_step=slope,wavelet_hz=wavelet_hz,
                                    condition=condition,method=METHOD,
                                    **evaluate_components(background,desired,wavelet,weights)))
            frame = pd.DataFrame(rows)
            if len(frame)!=10800 or frame.duplicated(KEYS).any():
                raise RuntimeError("New comparator design completeness failed")
            new_frames.append(frame)
            frame.to_parquet(dest/f"{block}_new_cases.parquet",index=False)
            write_json(dest/f"{block}_receipt.json",dict(status="PASS_EXPOSED_DEVELOPMENT_COMPARATOR",
                rows=len(frame),elapsed_total_seconds=time.perf_counter()-start,
                fixture=fixtures[block],case_sha256=sha(dest/f"{block}_new_cases.parquet")))
            completed.append(block)
            print(block,len(frame),f"{time.perf_counter()-start:.2f}s",flush=True)
        old,new = pd.concat(old_frames,ignore_index=True),pd.concat(new_frames,ignore_index=True)
        aggregate_tables(old,new).to_csv(dest/"expanded_date_condition_table.csv",index=False)
        new.to_parquet(dest/"all_new_cases.parquet",index=False)
        write_json(dest/"receipt.json",dict(status="PASS_EXPOSED_ONE_COMPARATOR_NOT_CONFIRMATORY",
            rows_new=len(new),rows_original_unchanged=len(old),inputs_sha256=inputs,
            fixture_reproductions=fixtures,elapsed_seconds=time.perf_counter()-start,
            sampled_worker_peak_rss_bytes=peak_rss,protocol_sha256=args.protocol_sha256,
            script_sha256=sha(__file__),helper_sha256=sha(HERE/"derivative_lcmv.py"),
            assessment_96h_disabled=True,manuscript_primary_values_changed=False,
            inputs_modified=False,no_models_or_training=True))
    except Exception as error:
        write_json(dest/"FAILED_RECEIPT.json",dict(status="STOPPED_NO_CASE_EXCLUSION_OR_RESCUE",
            error=repr(error),elapsed_seconds=time.perf_counter()-start,
            complete_blocks=completed,inputs_sha256=inputs))
        raise


if __name__ == "__main__":
    main()
