"""Independent full-DFT/direct-solve replay, not importing scoring/filter helpers.

Deterministic384 cases: two sources, every background block, first/last cutoffs,
first/last apertures, all six methods. Replays ALL12 saved numeric metrics.
"""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from execution_guard import sha,verify_scope

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT/"revisions/20261010_forecast_processing/development_outcomes"
SOURCES = ("kef0131","kef0140")
METHODS = ("neural","ridge","conventional","context_0.25","context_0.75","lcmv_context_0.25")
N = 175
FREQUENCY = np.fft.fftfreq(N,1/25)
DFT = np.exp(-2j*np.pi*np.outer(np.arange(N),np.arange(N))/N)
ACTIVE = (np.abs(FREQUENCY)>=.5)&(np.abs(FREQUENCY)<8)
BANDS = ((.5,1),(1,2),(2,4),(4,8))


def construct_matrix(context,kernel,method):
    powers = np.diagonal(context,axis1=-2,axis2=-1).real
    diagonal = powers[:,:,None]*np.eye(32)
    if method in ("neural","ridge"):
        return .25*kernel*np.sqrt(powers[:,:,None]*powers[:,None,:])+.75*diagonal
    fraction = .75 if method=="context_0.75" else .25
    return fraction*context+(1-fraction)*diagonal


def direct_weights(matrix,delays,method):
    w = np.zeros((N,32),complex)
    q = (np.arange(32)-15.5)/15.5
    for k in np.flatnonzero(ACTIVE):
        f = FREQUENCY[k]
        a = np.exp(-2j*np.pi*f*delays)
        if method=="conventional":
            w[k] = a/32
            continue
        band = next(i for i,(lo,hi) in enumerate(BANDS) if lo<=abs(f)<hi)
        r = np.asarray(matrix[band],complex)
        if f<0:r=r.conj()
        r = r+.001*np.trace(r).real/32*np.eye(32)
        if method=="lcmv_context_0.25":
            c = np.column_stack((a,q*a))
            z = np.linalg.solve(r,c)
            w[k] = z@np.linalg.solve(c.conj().T@z,[1,0])
        else:
            z = np.linalg.solve(r,a)
            w[k] = z/np.vdot(a,z).real
    return w


def inverse(spectrum):
    result = DFT.conj().T@spectrum/N
    if np.max(np.abs(result.imag))>1e-9*max(np.max(np.abs(result.real)),1e-15):
        raise RuntimeError("Direct full-DFT conjugate symmetry failed")
    return result.real


def direct_metrics(background,component,w,reference):
    x,y = DFT@background,DFT@component
    x[~ACTIVE]=0;y[~ACTIVE]=0
    band_n,band_s = inverse(x),inverse(y)
    out_n = inverse(np.sum(x*w.conj(),axis=1))
    out_s = inverse(np.sum(y*w.conj(),axis=1))
    ref_s = inverse(np.sum(y*reference.conj(),axis=1))
    ps,pn,pref,pis,pin = map(float,(np.mean(out_s**2),np.mean(out_n**2),np.mean(ref_s**2),
                                  np.mean(band_s**2),np.mean(band_n**2)))
    error = float(np.mean((out_s-ref_s)**2))
    return dict(recorded_component_output_power=ps,background_output_power=pn,
        conventional_reference_output_power=pref,recorded_component_input_power=pis,
        background_input_power=pin,recorded_component_relative_mse=error/pref,
        output_error_power=error,reference_to_input_power_ratio=pref/pis,
        component_input_ratio_db=float(10*np.log10(pis/pin)),
        component_output_ratio_db=float(10*np.log10(ps/pn)),
        component_ratio_gain_db=float(10*np.log10(ps/pn)-10*np.log10(pis/pin)),
        component_power_relative_reference_db=float(10*np.log10(ps/pref)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent-sha256",required=True)
    args=parser.parse_args()
    start=time.perf_counter()
    verify_scope(HERE.parent/"protocol.json",args.parent_sha256)
    dest=HERE/"recorded_component_results"
    result_path=dest/"DIRECT_DFT_VERIFICATION.json"
    if result_path.exists():raise FileExistsError("Preserve previous independent numerical evidence")
    receipt=json.loads((dest/"receipt.json").read_text())
    if sha(dest/"all_cases.parquet")!=receipt["case_sha256"]:
        raise RuntimeError("Savedcase hash mismatch")
    cases=pd.read_parquet(dest/"all_cases.parquet")
    if len(cases)!=23040 or cases.linearity_scaled_error.max()>1e-10:
        raise RuntimeError("All-case count/linearity failure")
    sources=[]
    for event in SOURCES:
        with np.load(HERE.parent/f"evidence_audit/decoded_sources/{event}_source_component.npz") as archive:
            sources.append(archive["component"])
    with np.load(HERE.parent/"evidence_audit/source_catalog_steering.npz") as archive:
        delays=archive["delays_seconds"]
    checked,max_errors=0,{}
    for block in receipt["background_blocks"]:
        with np.load(OLD/block/"adapter_arrays.npz") as archive:
            values={name:archive[name] for name in archive.files}
        for ci in (0,29):
            for aperture in (0,7):
                left=int(values["starts"][aperture]);ctx=values["context"][ci,aperture]
                background=values["background"][ci,:,left:left+32]
                for si,source_id in enumerate(SOURCES):
                    component=sources[si][:,left:left+32]
                    reference=direct_weights(ctx,delays[si,aperture],"conventional")
                    for method in METHODS:
                        kernel=values[method+"_kernel"][ci,aperture] if method in ("neural","ridge") else None
                        matrix=construct_matrix(ctx,kernel,method)
                        weights=direct_weights(matrix,delays[si,aperture],method)
                        metrics=direct_metrics(background,component,weights,reference)
                        selected=cases[(cases.source_event==source_id)&(cases.block==block)
                            &(cases.cutoff==int(values["cutoffs"][ci]))&(cases.aperture==aperture)&(cases.method==method)]
                        if len(selected)!=1:raise RuntimeError("Replaycase not unique")
                        for name,value in metrics.items():
                            old=float(selected.iloc[0][name])
                            error=abs(value-old)/max(1.0,abs(old))
                            max_errors[name]=max(max_errors.get(name,0.0),error)
                            if not np.isfinite(error) or error>2e-8:
                                raise RuntimeError(f"Independent directDFT/solve mismatch:{source_id}/{block}/{ci}/{aperture}/{method}/{name}/{error}")
                        checked+=1
    if checked!=384:raise RuntimeError("Deterministic independent replay count failed")
    result=dict(status="PASS_INDEPENDENT_CODE_DIRECT_DFT_NOT_HUMAN_REVIEW",cases_replayed=checked,
        metrics_per_case=12,numeric_comparisons=checked*12,maximum_scaled_errors=max_errors,tolerance=2e-8,
        all_case_linearity_checks=23040,all_case_linearity_maximum=float(cases.linearity_scaled_error.max()),
        elapsed_seconds=time.perf_counter()-start,verifier_sha256=sha(__file__),
        case_sha256=receipt["case_sha256"],parent_protocol_sha256=args.parent_sha256,
        independence="No scoring/filter/LCMVhelper import; explicit full175pointDFT and direct32channelmatrixsolve; separate code, not human replication")
    result_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
