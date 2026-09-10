"""Independent saved-array G3 check; no application imports or model fitting."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/submission_closeout/prediction"
OLD=ROOT/"reports/submission_revision"
MODELS=ROOT/"models/submission_revision"
ENDPOINTS=("waveform_safe","block_dense","local_sparse","local_dense")
SEEDS=(19,43,71)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def complex_array(x):
    x=np.asarray(x)
    return x if np.iscomplexobj(x) else x[...,0]+1j*x[...,1]


def score(pred,target):
    p=complex_array(pred).reshape(-1);t=complex_array(target).reshape(-1)
    assert p.shape==t.shape and np.isfinite(p).all() and np.isfinite(t).all()
    mse=np.mean(abs(p-t)**2);energy=np.mean(abs(t)**2)
    return dict(mse=float(mse),target_energy=float(energy),
                nrmse=float(np.sqrt(mse)/max(np.sqrt(energy),np.finfo(float).eps)),cells=len(t))


def independent_bootstrap(values,labels):
    values=np.asarray(values,float);labels=np.asarray(labels)
    groups,inverse=np.unique(labels,return_inverse=True)
    totals=np.bincount(inverse,weights=values);counts=np.bincount(inverse)
    indices=np.random.default_rng(20260906).integers(0,len(groups),(5000,len(groups)))
    draws=totals[indices].sum(1)/counts[indices].sum(1)
    return np.r_[values.mean(),np.quantile(draws,[.025,.975])]


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    rolepath=ROOT/"reports/final_revision/event_roles.csv"
    roles=pd.read_csv(rolepath,dtype={"event_id":str})
    indexpath=ROOT/"data/processed/operators/operator_index.parquet"
    index=pd.read_parquet(indexpath);index=index[index.success].sort_values(["event_id","route"]).reset_index(drop=True)
    assert index.event_id.astype(str).tolist()==roles.event_id.tolist()
    assert index.route.tolist()==roles.route.tolist()
    selected=roles.index[roles.role=="architecture_test"].to_numpy()
    assert len(selected)==48 and roles.iloc[selected].event_id.nunique()==24
    old=pd.read_csv(OLD/"prediction_event_scores.csv",dtype={"event_id":str})
    saved_comparisons=pd.read_csv(OLD/"prediction_both_ridge_comparisons.csv")
    metapath=ROOT/"reports/critical_review/event_metadata.parquet"
    meta=pd.read_parquet(metapath);meta.event_id=meta.event_id.astype(str);meta=meta.set_index("event_id")
    targetpaths={"block_dense":ROOT/"models/final_revision/dense_targets.npy",
                 "local_dense":MODELS/"local_dense_targets.npy"}
    targets={k:np.load(v,mmap_mode="r") for k,v in targetpaths.items()}
    inputs=[Path(__file__),rolepath,indexpath,metapath,OLD/"prediction_event_scores.csv",OLD/"prediction_both_ridge_comparisons.csv",*targetpaths.values()]
    operators={}
    for i in selected:
        row=roles.iloc[i];filename=str(row.path).replace("\\","/").rsplit("/",1)[-1]
        path=ROOT/"data/processed/operators"/filename
        with np.load(path) as z:
            assert str(z["event_id"])==row.event_id and str(z["route"])==row.route
            operators[i]=z["target_gamma"].astype(np.complex64)
        inputs.append(path)
    for endpoint in ENDPOINTS:
        for name in ["block_ridge","full_ridge",*[f"state_psd_seed{s}" for s in SEEDS],"state_psd_seed_mean"]:
            inputs.append(MODELS/f"{endpoint}_{name}_predictions.npy")
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in inputs}
    receipt=OUT/"input_inventory.json"
    receipt.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=hashes,
        purpose="Preventive independent saved-array aggregation check, not training or new inference selection"),indent=2)+"\n")
    case_rows=[];ensemble_rows=[];summaries=[]
    for endpoint in ENDPOINTS:
        arrays={name:np.load(MODELS/f"{endpoint}_{name}_predictions.npy",mmap_mode="r")
                for name in ["block_ridge","full_ridge",*[f"state_psd_seed{s}" for s in SEEDS],"state_psd_seed_mean"]}
        recomputed=np.mean([arrays[f"state_psd_seed{s}"] for s in SEEDS],axis=0)
        assert np.array_equal(recomputed,arrays["state_psd_seed_mean"])
        max_complex_mean_error=0.
        for i in selected:
            row=roles.iloc[i]
            if endpoint=="waveform_safe":truth=operators[i]
            elif endpoint=="block_dense":truth=targets["block_dense"][i]
            elif endpoint=="local_sparse":truth=targets["local_dense"][i][:,:,np.array([3,5,8,13,21])-1,:]
            else:truth=targets["local_dense"][i]
            complex_mean=np.mean([complex_array(arrays[f"state_psd_seed{s}"][i]) for s in SEEDS],axis=0)
            component_mean=complex_array(recomputed[i])
            error=float(np.max(abs(complex_mean-component_mean)))
            max_complex_mean_error=max(max_complex_mean_error,error)
            np.testing.assert_allclose(complex_mean,component_mean,rtol=0,atol=2e-7)
            actual={name:score(p[i],truth) for name,p in arrays.items()}
            for name,result in actual.items():
                source=old.loc[(old.endpoint==endpoint)&(old.model==name)&(old.event_id==row.event_id)&(old.route==row.route)]
                assert len(source)==1
                source=source.iloc[0]
                for key in ("mse","target_energy","nrmse"):
                    np.testing.assert_allclose(result[key],float(source[key]),rtol=0,atol=1e-12)
                assert result["cells"]==source.cells
                case_rows.append(dict(endpoint=endpoint,event_id=row.event_id,route=row.route,model=name,
                    nrmse_saved=float(source.nrmse),nrmse_recomputed=result["nrmse"],difference=result["nrmse"]-float(source.nrmse),cells=result["cells"]))
            mean_seed=float(np.mean([actual[f"state_psd_seed{s}"]["nrmse"] for s in SEEDS]))
            ensemble=actual["state_psd_seed_mean"]["nrmse"]
            ensemble_rows.append(dict(endpoint=endpoint,event_id=row.event_id,route=row.route,
                ensemble_prediction_score=ensemble,mean_single_seed_score=mean_seed,
                ensemble_minus_mean_seed_score=ensemble-mean_seed))
        current=pd.DataFrame(ensemble_rows).query("endpoint==@endpoint")
        assert (abs(current.ensemble_minus_mean_seed_score)>1e-10).all()
        summaries.append(dict(endpoint=endpoint,route_cases=len(current),seed_prediction_mean_bitwise_equal=True,
            maximum_complex_mean_roundoff=max_complex_mean_error,
            ensemble_score=float(current.ensemble_prediction_score.mean()),
            mean_single_seed_score=float(current.mean_single_seed_score.mean()),
            ensemble_minus_mean_seed_score=float(current.ensemble_minus_mean_seed_score.mean()),
            all_route_scores_distinguish_ensemble_from_mean_seed_score=True))
    cases=pd.DataFrame(case_rows);cases.to_csv(OUT/"prediction_case_score_verification.csv",index=False)
    pd.DataFrame(ensemble_rows).to_csv(OUT/"ensemble_vs_mean_seed_scores.csv",index=False)
    pd.DataFrame(summaries).to_csv(OUT/"ensemble_summary.csv",index=False)
    comp=[]
    for endpoint,g in cases.groupby("endpoint"):
        event=g.groupby(["event_id","model"]).nrmse_recomputed.mean().unstack()
        labels=meta.loc[event.index,"sensitivity_component_25km"].to_numpy()
        assert len(event)==24 and len(set(labels))==11
        for reference in ("block_ridge","full_ridge"):
            values=event.state_psd_seed_mean-event[reference]
            result=independent_bootstrap(values,labels)
            source=saved_comparisons.loc[(saved_comparisons.endpoint==endpoint)&(saved_comparisons.reference==reference)].iloc[0]
            np.testing.assert_allclose(result,source[["mean","low","high"]].to_numpy(float),rtol=0,atol=1e-12)
            assert abs(result[0]-(event.state_psd_seed_mean.mean()-event[reference].mean()))<1e-12
            comp.append(dict(endpoint=endpoint,reference=reference,mean=result[0],low=result[1],high=result[2],
                ensemble_score=event.state_psd_seed_mean.mean(),reference_score=event[reference].mean(),
                events=len(event),components=len(set(labels)),maximum_saved_comparison_difference=float(np.max(abs(result-source[["mean","low","high"]].to_numpy(float))))))
    pd.DataFrame(comp).to_csv(OUT/"both_ridge_comparisons_verified.csv",index=False)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    status=dict(status="PASS",endpoints=4,seed_mean_arrays_bitwise_equal=4,route_model_scores_reproduced=len(cases),
        ensemble_route_cases=192,distinct_ensemble_vs_mean_seed_scores=192,
        named_ridge_comparisons_reproduced=len(comp),maximum_case_score_error=float(abs(cases.difference).max()),
        maximum_comparison_error=max(r["maximum_saved_comparison_difference"] for r in comp),
        bootstrap_replicates=5000,bootstrap_seed=20260906,source_component_count=11,events=24,
        input_hashes_unchanged=True,no_training_or_application_modules_imported=True,
        elapsed_seconds=time.perf_counter()-start,completed_utc=datetime.now(timezone.utc).isoformat())
    (OUT/"verification.json").write_text(json.dumps(status,indent=2)+"\n")
    print(json.dumps(status,indent=2))


if __name__=="__main__":main()
