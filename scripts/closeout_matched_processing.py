"""Bounded G2 closeout: validate projections and evaluate saved full-context ridge.

Never trains models or overwrites historical evidence.  Historical accepted
processing scores are retained; only failed projections are repaired, using
the existing geometry_fit objective and deterministic fallback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from coherencygraph_das import submission_revision as sr
from coherencygraph_das import final_revision as historical
from coherencygraph_das.critical_revision import bundle, boot, complex_values, upper_kernel_matrix
from coherencygraph_das.review_revision import _generalized_weight, _ratio

OUT = ROOT / "reports/submission_closeout/processing"
OLD = ROOT / "reports/submission_revision"
KINDS = [(tag, "block ridge") for tag in ("block_sparse", "block_dense", "local_sparse", "local_dense")] + [(tag, "full-context ridge") for tag in ("local_sparse", "local_dense")]
KEY = ["event_id", "route", "block", "band", "support", "model", "seed", "endpoint"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")


def acceptance(p, A, z):
    """Same strict first-order gap as the original solver; explicit simplex QA."""
    p = np.asarray(p)
    gradient = A.T @ (A @ p - z)
    gap = float(p @ gradient - gradient.min())
    unit_error = float(abs(p.sum() - 1))
    minimum = float(p.min())
    return dict(accepted=bool(np.isfinite(p).all() and np.isfinite(gap) and gap < 1e-8 and unit_error <= 1e-10 and minimum >= -1e-12),
                first_order_gap=gap, simplex_sum_error=unit_error, minimum_mass=minimum,
                squared_objective=float(.5 * np.sum((A @ p-z)**2)), moment_residual=float(abs(A @ p-z).max()))


def prediction_path(support, model):
    if support == "block_sparse": return historical.MOD / "primary_block_ridge_predictions.npy"
    suffix = "block_ridge" if model == "block ridge" else "full_ridge"
    return sr.MOD / f"{support}_{suffix}_predictions.npy"


def score_projection(p, record, block, band, support, model):
    rc, rt, rn = (record[k+"_matrix"][block,band] for k in ("context", "target", "noise"))
    D = np.diag(np.maximum(rc.diagonal().real, 1e-15))
    kernel = upper_kernel_matrix(p, record["positions"])
    scale = np.sqrt(D.diagonal()[:,None] * D.diagonal()[None,:])
    covariance = .25 * kernel * scale + .75 * D
    w = _generalized_weight(covariance, rn, .0001)
    observed = rt / np.sqrt(rt.diagonal().real[:,None] * rt.diagonal().real[None,:])
    off = ~np.eye(len(D), dtype=bool)
    mse = float(np.mean(abs(kernel[off]-observed[off])**2))
    return [dict(block=block, band=band, support=support, model=model, seed=0, endpoint=endpoint,
                 ratio_db=float(10 * np.log10(_ratio(record[name+"_matrix"][block,band], rn, w))),
                 covariance_entry_mse=mse)
            for endpoint,name in (("full","target"),("half0","first"),("half1","second"))]


def record_worker(job):
    i, event, route, cache, old_rows = job
    with threadpool_limits(limits=1):
        record = dict(np.load(cache))
        assert np.array_equal(record["positions"], np.arange(484,516))
        predictions = {key: np.load(prediction_path(*key), mmap_mode="r") for key in KINDS}
        lags = {"block_sparse": historical.LAGS, "block_dense": np.arange(1,32),
                "local_sparse": np.array(sr.CFG["experiments"]["local_sparse_lags"]), "local_dense": np.arange(1,32)}
        diagnostics, changed, replay, probabilities = [], [], [], []
        saved = {(int(r["block"]),int(r["band"]),r["support"]):r for r in old_rows}
        for block,band in np.ndindex(8,4):
            for support,model in KINDS:
                values = complex_values(predictions[support,model][i,block,band])
                p,A,z,q,fit = historical.fit_grid(values, lags[support])
                before = acceptance(p,A,z)
                is_old = model == "block ridge"
                prior = saved[block,band,support] if is_old else None
                if is_old:
                    assert bool(prior["fit_converged"]) == before["accepted"], (event,route,block,band,support,prior, before)
                    assert abs(prior["fit_gap"]-fit["fit_gap"]) < 1e-10
                    replay += [dict(event_id=event, route=route, **r) for r in score_projection(p,record,block,band,support,model)]
                repaired = not before["accepted"]
                after_fit = fit
                if repaired:
                    p,A,z,q,after_fit = sr.geometry_fit(values,lags[support])
                after = acceptance(p,A,z)
                if not after["accepted"]:
                    raise AssertionError(("projection not accepted",event,route,block,band,support,model,after))
                if after["squared_objective"] > before["squared_objective"] + 1e-12:
                    raise AssertionError("Repair increased unchanged least-squares objective")
                if repaired or not is_old:
                    changed += [dict(event_id=event, route=route, **r) for r in score_projection(p,record,block,band,support,model)]
                row = dict(event_id=event,route=route,block=block,band=band,support=support,model=model,
                           historical_case=is_old, repaired=repaired,
                           historical_fit_converged=prior["fit_converged"] if is_old else None,
                           historical_fit_gap=prior["fit_gap"] if is_old else None,
                           **{"before_"+k:v for k,v in before.items()}, **{"after_"+k:v for k,v in after.items()},
                           **{"solver_"+k:v for k,v in after_fit.items()})
                diagnostics.append(row); probabilities.append(p)
        stem = f"{event}_{route.replace('-','')}"
        pd.DataFrame(diagnostics).to_parquet(OUT/"records"/(stem+"_projections.parquet"),index=False)
        pd.DataFrame(changed).to_parquet(OUT/"records"/(stem+"_scores.parquet"),index=False)
        pd.DataFrame(replay).to_parquet(OUT/"records"/(stem+"_historical_replay.parquet"),index=False)
        np.savez_compressed(OUT/"records"/(stem+"_simplex.npz"), probability=np.array(probabilities),q=q)
        return stem, len(diagnostics), sum(r["repaired"] for r in diagnostics)


def aggregate(frame):
    """Average waveform cells/route and split halves inside seed, then seeds."""
    frame = frame.copy()
    frame["evaluation"] = np.where(frame.endpoint == "full", "full", "split_target")
    frame["method"] = frame.support + " / " + frame.model
    per_seed = frame.groupby(["evaluation","event_id","method","seed"],sort=True).ratio_db.mean().reset_index()
    per_event = per_seed.groupby(["evaluation","event_id","method"],sort=True).ratio_db.mean().reset_index()
    return per_seed, per_event


def comparison_frames(event, mapping):
    comparisons, deletion, checks = [], [], []
    for endpoint,group in event.groupby("evaluation"):
        pivot = group.pivot(index="event_id",columns="method",values="ratio_db")
        assert not pivot.isna().any().any()
        for method in pivot:
            for reference in pivot:
                v = pivot[method]-pivot[reference]
                labels = np.array([mapping[str(e)] for e in v.index])
                summary = boot(v,labels)
                diffmeans = float(pivot[method].mean()-pivot[reference].mean())
                assert abs(summary["mean"]-diffmeans) < 1e-12
                comparisons.append(dict(endpoint=endpoint,method=method,reference=reference,**summary))
                d=[]
                for component in sorted(set(labels)):
                    keep=labels!=component
                    d.append(dict(endpoint=endpoint,method=method,reference=reference,omitted_component=component,
                                  retained_events=int(keep.sum()),mean=float(v[keep].mean())))
                reconstructed=float(np.average([r["mean"] for r in d],weights=[r["retained_events"] for r in d]))
                assert abs(reconstructed-summary["mean"]) < 1e-12
                checks.append(dict(endpoint=endpoint,method=method,reference=reference,
                                   paired_mean_minus_difference_of_means=summary["mean"]-diffmeans,
                                   weighted_deletion_mean_minus_full_mean=reconstructed-summary["mean"]))
                deletion.extend(d)
    return pd.DataFrame(comparisons),pd.DataFrame(deletion),pd.DataFrame(checks)


def summarise():
    new = pd.concat([pd.read_parquet(p) for p in sorted((OUT/"records").glob("*_scores.parquet"))],ignore_index=True)
    diagnostics=pd.concat([pd.read_parquet(p) for p in sorted((OUT/"records").glob("*_projections.parquet"))],ignore_index=True)
    replay=pd.concat([pd.read_parquet(p) for p in sorted((OUT/"records").glob("*_historical_replay.parquet"))],ignore_index=True)
    old=pd.read_parquet(OLD/"geometry_matched_processing.parquet")
    for f in (old,new,replay,diagnostics): f["event_id"]=f.event_id.astype(str)
    r=old.merge(replay,on=KEY,suffixes=("_archived","_replayed"),validate="one_to_one")
    assert len(r)==6144*3
    r["ratio_db_change"]=r.ratio_db_replayed-r.ratio_db_archived
    r["mse_change"]=r.covariance_entry_mse_replayed-r.covariance_entry_mse_archived
    assert abs(r.ratio_db_change).max() < 1e-8
    assert abs(r.mse_change).max() < 1e-10
    r.to_parquet(OUT/"historical_score_replay.parquet",index=False)
    repairkeys = new.loc[new.model=="block ridge",KEY]
    merged=old.merge(repairkeys,on=KEY,how="left",indicator=True,validate="one_to_one")
    retained=merged.loc[merged._merge=="left_only",old.columns]
    final=pd.concat([retained,new],ignore_index=True).sort_values(KEY)
    assert len(final)==92160 and not final.duplicated(KEY).any()
    final.to_parquet(OUT/"geometry_matched_processing.parquet",index=False)
    diagnostics.to_csv(OUT/"projection_acceptance.csv",index=False)
    diagnostics.groupby(["support","model"]).agg(cases=("after_accepted","size"),historical=("historical_case","sum"),
        original_failures=("repaired","sum"),accepted=("after_accepted","sum"),max_before_gap=("before_first_order_gap","max"),
        max_after_gap=("after_first_order_gap","max"),max_simplex_sum_error=("after_simplex_sum_error","max"),
        min_mass=("after_minimum_mass","min")).reset_index().to_csv(OUT/"projection_acceptance_summary.csv",index=False)
    seed,event=aggregate(final)
    seed.to_csv(OUT/"processing_event_seed_scores.csv",index=False)
    event.to_csv(OUT/"processing_event_scores.csv",index=False)
    mapping=pd.read_parquet(ROOT/"reports/critical_review/event_metadata.parquet").set_index("event_id").sensitivity_component_25km.to_dict()
    comp,delete,checks=comparison_frames(event,mapping)
    comp.to_csv(OUT/"processing_all_comparisons.csv",index=False)
    delete.to_csv(OUT/"processing_component_deletion.csv",index=False)
    checks.to_csv(OUT/"aggregation_checks.csv",index=False)
    oldcomp=pd.read_csv(OLD/"processing_comparisons.csv")
    reconcile=oldcomp.merge(comp,on=["endpoint","method","reference"],suffixes=("_before","_after"),validate="one_to_one")
    for col in ("mean","low","high"): reconcile[col+"_change"]=reconcile[col+"_after"]-reconcile[col+"_before"]
    reconcile.to_csv(OUT/"numeric_reconciliation.csv",index=False)
    oldseed=pd.read_csv(OLD/"processing_event_seed_scores.csv",dtype={"event_id":str})
    s=oldseed.merge(seed,on=["evaluation","event_id","method","seed"],suffixes=("_before","_after"),validate="one_to_one")
    unchanged=s.loc[~s.method.str.endswith(" / block ridge")]
    assert np.array_equal(unchanged.ratio_db_before.to_numpy(),unchanged.ratio_db_after.to_numpy()) or np.max(abs(unchanged.ratio_db_before-unchanged.ratio_db_after))<1e-12
    rawchange=old.merge(final,on=KEY,suffixes=("_before","_after"),validate="one_to_one")
    rawchange["change_db"]=rawchange.ratio_db_after-rawchange.ratio_db_before
    rawchange.to_parquet(OUT/"processing_case_reconciliation.parquet",index=False)
    rec={"status":"PASS","historical_projections":6144,"historical_failed_repaired":int(diagnostics.loc[diagnostics.historical_case,"repaired"].sum()),
         "new_full_context_projections":3072,"new_full_context_initial_failures_repaired":int(diagnostics.loc[~diagnostics.historical_case,"repaired"].sum()),
         "accepted_final_projections":int(diagnostics.after_accepted.sum()),"projection_cases":len(diagnostics),
         "score_rows":len(final),"historical_score_replay_max_abs_db":float(abs(r.ratio_db_change).max()),
         "max_historical_case_correction_db":float(abs(rawchange.change_db).max()),
         "max_paired_mean_correction_db":float(abs(reconcile.mean_change).max()),
         "paired_aggregation_checks":len(checks),"preserved_nonridge_event_scores":len(unchanged),
         "main_endpoint":"full", "secondary_endpoint":"split_target",
         "interpretation":"Only failed numerical projections repaired; all former successful projections and all neural/reference scores retained. No training, target, cohort, hyperparameter or processing geometry changed."}
    write_json(OUT/"verification.json",rec)
    print(json.dumps(rec,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["run","summarise"]);parser.add_argument("--workers",type=int,default=8);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"records").mkdir(exist_ok=True)
    if args.stage=="summarise": return summarise()
    _,ds,roles,_=bundle()
    old=pd.read_csv(OLD/"processing_projection_checks.csv",dtype={"event_id":str})
    paths=[Path(__file__),sr.SPEC,OLD/"processing_projection_checks.csv",OLD/"geometry_matched_processing.parquet",OLD/"processing_comparisons.csv"]
    paths += [prediction_path(*k) for k in KINDS]
    jobs=[]
    for i in np.flatnonzero(roles=="architecture_test"):
        event,route=str(ds.event_ids[i]),str(ds.routes[i]);cache=sr.cache_path(ds,i);paths.append(cache)
        jobs.append((int(i),event,route,str(cache),old.loc[(old.event_id==event)&(old.route==route)].to_dict("records")))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    receipt=OUT/"specification.json"
    if receipt.exists(): assert json.loads(receipt.read_text())["input_sha256"]==hashes
    else: write_json(receipt,{"created_utc":datetime.now(timezone.utc).isoformat(),"input_sha256":hashes,
        "bounded_change":"Audit all historical ridge projections, repair only failed fits via unchanged geometry_fit; add two saved exact-local full-context ridge supports.",
        "simplex_tolerances":{"sum_error_max":1e-10,"minimum_mass_min":-1e-12,"first_order_gap_strict_max":1e-8},
        "processing":{"positions":[484,515],"cached_matrix_trace_loading":.001,"generalized_eigen_noise_loading":.0001,"covariance_weights":[.25,.75]},
        "aggregation":"Average routes, 8 blocks, 4 bands and (secondary endpoint) two halves within each event/seed; average three seed scores within earthquake; bootstrap complete 25-km components, 5000 draws, seed20260906.",
        "not_preregistration":True,"historical_test_exposure_preserved":True,"workers":args.workers,"threads_per_worker":1})
    start=time.perf_counter()
    pending=[j for j in jobs if not (OUT/"records"/(j[1]+"_"+j[2].replace('-','')+"_simplex.npz")).exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(record_worker,j) for j in pending]
        for n,future in enumerate(as_completed(futures),1):print(n,"/",len(pending),future.result(),flush=True)
    assert all(digest(ROOT/p)==v for p,v in hashes.items())
    write_json(OUT/"execution.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"wall_seconds":time.perf_counter()-start,"recomputed_routes":len(pending),"cached_routes":len(jobs)-len(pending),"workers":args.workers,"input_hashes_unchanged":True})
    summarise()


if __name__=="__main__":main()
