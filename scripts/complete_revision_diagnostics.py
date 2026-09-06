"""Final diagnostic summaries. No model selection or test-set tuning."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from coherencygraph_das.critical_revision import (bundle,OUT,MODELS,Q,SEED,metrics,
    complex_values,conformal_quantile,coverage_interval,boot)

cfg,ds,roles,groups=bundle()
meta=pd.read_parquet(OUT/'event_metadata.parquet').set_index('event_id')
comp=meta.sensitivity_component_25km.to_dict()
seeds=[19,43,71]
p=np.stack([np.load(MODELS/f'matched_state_psd_seed{s}_all_predictions.npy') for s in seeds])
prediction=p.mean(0)
spread=np.sqrt(np.mean(np.var(p,axis=0),axis=(1,2,3,4)))
rows=[]
for i in range(len(ds.event_ids)):
    rows.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],role=roles[i],component=comp[ds.event_ids[i]],
        spread=float(spread[i]),quality=float(np.mean(np.load(ds.paths[i])['context_noise_power_ratio'])),
        **metrics(prediction[i],ds.targets[i])))
frame=pd.DataFrame(rows);frame.to_csv(OUT/'revised_uncertainty_scores.csv',index=False)
coverage=[]
for unit in ['earthquake','component']:
    key='event_id' if unit=='earthquake' else 'component'
    cal=frame[frame.role=='calibration'].groupby(key).nrmse.max()
    test=frame[frame.role=='architecture_test'].groupby(key).nrmse.max()
    for nominal in [.8,.9,.95]:
        threshold,k=conformal_quantile(cal,nominal)
        success=int((test<=threshold).sum());low,high=coverage_interval(success,len(test))
        coverage.append(dict(model='revised_state_psd',unit=unit,nominal=nominal,threshold=threshold,order=k,
            calibration_units=len(cal),test_units=len(test),successes=success,coverage=success/len(test),ci_low=low,ci_high=high))
pd.DataFrame(coverage).to_csv(OUT/'revised_joint_coverage.csv',index=False)
event=frame[frame.role=='architecture_test'].groupby('event_id')[['nrmse','spread','quality']].mean()
rng=np.random.default_rng(SEED);cs=np.array([comp[e] for e in event.index]);unique=np.unique(cs)
rank=[];risk=[]
for score in ['spread','quality']:
    observed=spearmanr(event[score],event.nrmse).statistic;draws=[]
    for _ in range(5000):
        ix=np.concatenate([np.flatnonzero(cs==c) for c in rng.choice(unique,len(unique))])
        value=spearmanr(event[score].to_numpy()[ix],event.nrmse.to_numpy()[ix]).statistic
        if np.isfinite(value):draws.append(value)
    low,high=np.quantile(draws,[.025,.975]);rank.append(dict(score=score,spearman=observed,low=low,high=high,valid_draws=len(draws)))
    ordered=event.sort_values(score,ascending=score=='spread')
    for fraction in [1.,.8,.6,.4]:
        n=max(1,int(np.ceil(fraction*len(event))))
        draws=[]
        for _ in range(5000):
            ix=np.concatenate([np.flatnonzero(cs==c) for c in rng.choice(unique,len(unique))])
            values=event.iloc[ix].sort_values(score,ascending=score=='spread')
            count=max(1,int(np.ceil(fraction*len(values))))
            draws.append(values.iloc[:count].nrmse.mean())
        low,high=np.quantile(draws,[.025,.975])
        risk.append(dict(score=score,fraction=fraction,retained=n,nrmse=ordered.iloc[:n].nrmse.mean(),low=low,high=high,
            bootstrap='source components, ranking repeated inside each draw'))
pd.DataFrame(rank).to_csv(OUT/'uncertainty_ranking.csv',index=False)
pd.DataFrame(risk).to_csv(OUT/'uncertainty_risk_coverage.csv',index=False)
# Seed agreement is not identifiability: report both spectral and unseen-lag spread.
probs=np.stack([np.load(MODELS/f'matched_state_psd_seed{s}_all_probabilities.npy') for s in seeds])
test=np.flatnonzero(roles=='architecture_test');sr=[]
for i in test:
    for lag in [1,2,4,10]:
        values=np.einsum('sxbq,q->sxb',probs[:,i],np.exp(1j*lag*Q))
        sr.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],lag=lag,
            mean_real_seed_range=float(np.ptp(values.real,axis=0).mean()),
            mean_spectral_l1=float(np.abs(probs[:,i]-probs[:,i].mean(0)).sum(-1).mean())))
pd.DataFrame(sr).to_csv(OUT/'seed_unseen_lag_spread.csv',index=False)
print(pd.DataFrame(coverage).to_string(index=False))
print(pd.DataFrame(rank).to_string(index=False))
