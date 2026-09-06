"""Amendment 07: explicit metrics, provenance, and identifiability diagnostics.

Historical outputs are read-only. Every new product is written to a new directory.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta, spearmanr

from .config import load_config, sha256, write_json
from .data import reliable_mask
from .hybrid_search import _dataset_and_split, _load_amendment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/critical_review"
MODELS = ROOT / "models/critical_review"
OLD = ROOT / "models/methodological_audit"
SEED = 20260906
LAGS = np.array([3, 5, 8, 13, 21, 34, 55, 89])
Q = np.sort(np.fft.fftfreq(257) * 2 * np.pi)


def bundle():
    cfg = load_config(ROOT / "configs/protocol_v1.0.yaml")
    amendment, _ = _load_amendment(cfg)
    ds, roles, groups = _dataset_and_split(cfg, amendment)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    return cfg, ds, roles, groups


def complex_values(a):
    a = np.asarray(a)
    return a if np.iscomplexobj(a) else a[..., 0] + 1j * a[..., 1]


def components(a):
    return np.stack([np.real(a), np.imag(a)], axis=-1).astype(np.float32)


def metrics(pred, truth, mask=None, blocks=None):
    """One event-route score; all nested cells have equal weight before RMS.

    MSE is |complex error|^2, not the mean of real and imaginary squares.
    Normalization is RMS target energy in exactly the same cells.
    """
    p, t = complex_values(pred), complex_values(truth)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} != {t.shape}")
    keep = np.ones(t.shape, bool)
    if mask is not None:
        keep &= np.broadcast_to(mask, t.shape)
    if blocks is not None:
        keep &= np.asarray(blocks, bool)[..., None, None]
    if not np.all(np.isfinite(p[keep])) or not np.all(np.isfinite(t[keep])):
        raise ValueError("Non-finite prediction/target; never silently omit cells")
    p, t = p[keep], t[keep]
    if not len(t):
        raise ValueError("Empty endpoint")
    err = p-t
    mse, energy = np.mean(abs(err)**2), np.mean(abs(t)**2)
    phase = abs(np.angle(p*np.conj(t)))
    phase_weight = abs(t)
    sign = abs(t.real) >= 0.02
    return dict(mse=float(mse), target_energy=float(energy),
        nrmse=float(np.sqrt(mse)/max(np.sqrt(energy), np.finfo(float).eps)),
        real_mae=float(np.mean(abs(err.real))), imag_mae=float(np.mean(abs(err.imag))),
        weighted_phase_mae=float(np.sum(phase*phase_weight)/max(phase_weight.sum(),1e-15)),
        sign_all=float(np.mean(np.sign(p.real)==np.sign(t.real))),
        sign_excluding_near_zero=float(np.mean(np.sign(p.real[sign])==np.sign(t.real[sign]))) if sign.any() else None,
        sign_scored_cells=int(sign.sum()), cells=int(len(t)))


def boot(values, groups=None, replicates=5000, seed=SEED):
    values=np.asarray(values, float)
    if groups is None: groups=np.arange(len(values))
    groups=np.asarray(groups)
    unique=np.unique(groups)
    totals=np.array([values[groups==g].sum() for g in unique])
    counts=np.array([np.sum(groups==g) for g in unique])
    rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(unique),(replicates,len(unique)))
    draws=totals[ix].sum(1)/counts[ix].sum(1)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(mean=float(values.mean()), low=float(lo), high=float(hi),
        events=len(values), groups=len(unique), median=float(np.median(values)),
        q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)),
        fraction_negative=float(np.mean(values<0)))


def reliability_for(ds, indices):
    arrays={k:[] for k in ['target_gamma_first','target_gamma_second','target_gamma_taper0','target_gamma_taper1']}
    for i in indices:
        with np.load(ds.paths[i]) as z:
            for k in arrays: arrays[k].append(z[k])
    arrays={k:np.stack(v) for k,v in arrays.items()}
    rows=[]
    for b,l in np.ndindex(4,8):
        corrs=[]
        for k1,k2 in [('target_gamma_first','target_gamma_second'),('target_gamma_taper0','target_gamma_taper1')]:
            a,c=arrays[k1][:,:,b,l].ravel(),arrays[k2][:,:,b,l].ravel()
            corrs.append(float(np.corrcoef(np.r_[a.real,a.imag],np.r_[c.real,c.imag])[0,1]))
        rows.append(dict(band=b,lag_index=l,lag=int(LAGS[l]),split_r=corrs[0],taper_r=corrs[1],reliable=min(corrs)>=.35))
    table=pd.DataFrame(rows)
    return table.reliable.to_numpy().reshape(4,8),table


def metadata(ds,roles,groups):
    cohort=pd.read_parquet(ROOT/'reports/audit/frozen_cohort.parquet')
    cohort.event_id=cohort.event_id.astype(str)
    role_map=dict(zip(ds.event_ids,roles))
    table=cohort[cohort.event_id.isin(ds.event_ids)].copy()
    table['analysis_role']=table.event_id.map(role_map)
    lat=np.radians(table.latitude_deg.to_numpy())
    lon=np.radians(table.longitude_deg.to_numpy())
    a=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    horizontal=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    distance=np.hypot(horizontal, table.depth_km.to_numpy()[:,None]-table.depth_km.to_numpy()[None,:])
    _,labs=connected_components(sparse.csr_matrix(distance<=25),directed=False)
    table['sensitivity_component_25km']=[f'component_{i:03d}' for i in labs]
    table.to_parquet(OUT/'event_metadata.parquet',index=False)
    table.to_csv(OUT/'event_metadata.csv',index=False)
    cross=[]
    for r1,r2 in itertools.combinations(sorted(table.analysis_role.unique()),2):
        d=distance[np.ix_(table.analysis_role.eq(r1),table.analysis_role.eq(r2))]
        cross.append(dict(role_1=r1,role_2=r2,minimum_hypocentral_km=float(d.min()),pairs_within_25km=int((d<=25).sum())))
    pd.DataFrame(cross).to_csv(OUT/'cross_role_distances.csv',index=False)
    summary=[]
    for role,g in table.groupby('analysis_role'):
        summary.append(dict(role=role,events=len(g),source_bins=g.source_group_id.nunique(),
            components_25km=g.sensitivity_component_25km.nunique(),
            magnitude_min=g.magnitude.min(),magnitude_median=g.magnitude.median(),magnitude_max=g.magnitude.max(),
            depth_min=g.depth_km.min(),depth_median=g.depth_km.median(),depth_max=g.depth_km.max(),
            date_min=str(g.archive_date.min()),date_max=str(g.archive_date.max()),
            gauge_small=int((g.terra_gauge_length_m<20).sum()),gauge_large=int((g.terra_gauge_length_m>=20).sum())))
    pd.DataFrame(summary).to_csv(OUT/'cohort_summary.csv',index=False)
    table.groupby(['analysis_role','source_group_id']).size().rename('events').reset_index().to_csv(OUT/'source_group_sizes.csv',index=False)
    return table


HISTORICAL={
 'Local-state PSD':'corrected_gat_bissm_psd',
 'State-space PSD':'corrected_bissm_psd',
 'BiGRU PSD':'corrected_bigru_psd',
 'CNN PSD':'corrected_conv1d_psd',
 'Affine-logit spectral':'corrected_linear_psd',
}


def historical_predictions(role):
    out={name:np.load(OLD/f'{prefix}_{role}_predictions.npy') for name,prefix in HISTORICAL.items()}
    for path in OLD.glob(f'baseline_*_{role}.npy'):
        name=path.name[len('baseline_'):-len(f'_{role}.npy')].replace('_',' ')
        out[name]=np.load(path)
    return out


def audit_baseline():
    cfg,ds,roles,groups=bundle()
    meta=metadata(ds,roles,groups)
    frozen=reliable_mask(cfg)
    dev=np.flatnonzero(roles=='model_development')
    revised,rt=reliability_for(ds,dev)
    rt['historical_reliable']=frozen.ravel()
    rt.to_csv(OUT/'reliability_mask_audit.csv',index=False)
    original_development=ds.roles=='development'
    prior_test=np.unique(ds.event_ids[original_development & (roles=='architecture_test')])
    write_json(OUT/'reliability_provenance.json',dict(
        historical_development_events=len(np.unique(ds.event_ids[original_development])),
        architecture_test_events_in_historical_mask=len(prior_test),event_ids=prior_test.tolist(),
        historical_cells=int(frozen.sum()),development_only_cells=int(revised.sum()),
        revised_primary='all 32 cells; reliable-subset analyses are secondary',
        implication='Historical target selection is not independent of the later architecture test.'))
    rows=[]; long=[]
    for role in ['calibration','architecture_test']:
        ix=np.flatnonzero(roles==role)
        for name,pred in historical_predictions(role).items():
            for n,i in enumerate(ix):
                for label,mask in [('historical_28',frozen),('development_only',revised),('all_32',np.ones((4,8),bool))]:
                    rows.append(dict(model=name,role=role,event_id=ds.event_ids[i],source_group_id=groups[i],route=ds.routes[i],endpoint=label,**metrics(pred[n],ds.targets[i],mask)))
                p,t=complex_values(pred[n]),complex_values(ds.targets[i])
                for block,b,l in np.ndindex(p.shape):
                    long.append(dict(model=name,seed='ensemble_or_classical',cohort=role,event_id=ds.event_ids[i],
                        source_group_id=groups[i],route=ds.routes[i],block=block,band=b,lag=int(LAGS[l]),
                        historical_reliable=bool(frozen[b,l]),development_reliable=bool(revised[b,l]),
                        missingness_mask='none',prediction_real=float(p[block,b,l].real),prediction_imag=float(p[block,b,l].imag),
                        target_real=float(t[block,b,l].real),target_imag=float(t[block,b,l].imag)))
    frame=pd.DataFrame(rows)
    frame.to_parquet(OUT/'historical_metrics.parquet',index=False)
    pd.DataFrame(long).to_parquet(OUT/'historical_predictions_long.parquet',index=False)
    summary=frame.groupby(['role','endpoint','model']).mean(numeric_only=True).reset_index()
    summary.to_csv(OUT/'historical_recomputed_summary.csv',index=False)
    old=pd.read_csv(ROOT/'reports/methodological_audit/corrected_baseline_summary.csv')
    name_map={'Affine-logit spectral':'Linear PSD decoder'}
    check=[]
    for row in summary[summary.endpoint=='historical_28'].itertuples():
        match=old[(old.role==row.role)&(old.model.str.lower()==name_map.get(row.model,row.model).lower())]
        if len(match):
            check.append(dict(model=row.model,role=row.role,old_nrmse=float(match.mean_nrmse.iloc[0]),new_nrmse=row.nrmse,
                absolute_difference=abs(float(match.mean_nrmse.iloc[0])-row.nrmse)))
    pd.DataFrame(check).to_csv(OUT/'historical_number_check.csv',index=False)
    if max(x['absolute_difference'] for x in check)>2e-6: raise AssertionError('Historical metric mismatch')
    repeat=pd.read_csv(ROOT/'reports/methodological_audit/information_ceiling_normalized_skill.csv')
    rep=[]
    for name,g in repeat.groupby('model'):
        rep.append(dict(model=name,model_mse=g.model_mse.mean(),repeat_mse=g.empirical_reproducibility_mse.mean(),
            ratio_of_means=1-g.model_mse.mean()/g.empirical_reproducibility_mse.mean(),
            mean_of_ratios=np.mean(1-g.model_mse/g.empirical_reproducibility_mse)))
    pd.DataFrame(rep).to_csv(OUT/'repeat_ratio_reconciliation.csv',index=False)
    manifest=[]
    for folder in ['configs','src','scripts','tests','manuscript/cageo_submission','models/methodological_audit']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in str(p) and p.suffix not in ['.aux','.log','.out','.blg']:
                manifest.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)))
    pd.DataFrame(manifest).to_csv(OUT/'baseline_input_manifest.csv',index=False)
    write_json(OUT/'baseline_receipt.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        baseline_preserved='revisions/20260906_pre_review',git_commit=None,
        numerical_comparisons=len(check),maximum_absolute_difference=max(x['absolute_difference'] for x in check),
        source_group_algorithm='0.2 degree latitude/longitude by 30 km depth hard bins, no temporal threshold',
        raw_available=int(sum(Path(x).exists() for x in meta.terra_path))+int(sum(Path(x).exists() for x in meta.kkfls_path))))
    print(json.dumps({'baseline_checks':len(check),'historical_mask_test_events':len(prior_test),'repeat':rep}),flush=True)


def spectrum_fit(values, lags=LAGS, max_iter=50000, tol=1e-8):
    """Projected-gradient minimization of ||A p-y||² on the probability simplex."""
    basis=np.exp(1j*np.asarray(lags)[:,None]*Q)
    A=np.r_[basis.real,basis.imag]
    y=np.r_[np.real(values),np.imag(values)]
    # DFT rows are orthogonal for distinct positive lags below M/2.
    lipschitz=float(np.linalg.norm(A@A.T,2))
    p=np.ones(len(Q))/len(Q)
    z=p.copy(); acceleration=1.
    for it in range(max_iter):
        v=z-A.T@(A@z-y)/lipschitz
        u=np.sort(v)[::-1]; css=np.cumsum(u)-1
        rho=np.flatnonzero(u-css/(np.arange(len(u))+1)>0)[-1]
        new=np.maximum(v-css[rho]/(rho+1),0)
        change=np.max(abs(new-p))
        grad=A.T@(A@new-y)
        # This Frank-Wolfe gap bounds objective suboptimality on the simplex.
        gap=float(new@grad-grad.min())
        acc=(1+np.sqrt(1+4*acceleration**2))/2
        z=new+(acceleration-1)/acc*(new-p)
        p,acceleration=new,acc
        if gap<tol: break
    residual=float(np.max(abs(A@p-y)))
    return p,dict(iterations=it+1,converged=gap<tol,duality_gap=gap,max_component_residual=residual,
                  observed_rmse=float(np.sqrt(np.mean(abs(basis@p-values)**2))))


def upper_kernel_matrix(p,positions):
    """Fit gamma at UPPER diagonals; C_ij = gamma(position_j-position_i)."""
    d=np.asarray(positions)[None,:]-np.asarray(positions)[:,None]
    return np.einsum('k,ijk->ij',p,np.exp(1j*d[...,None]*Q))


def feasible_bounds(values, unseen, slack=.01):
    p,fit=spectrum_fit(values)
    B=np.exp(1j*LAGS[:,None]*Q); A=np.r_[B.real,B.imag]; y=np.r_[values.real,values.imag]
    tolerance=fit['max_component_residual']+slack
    upper=np.r_[y+tolerance,-y+tolerance]
    rows=[]
    for lag in unseen:
        c=np.cos(lag*Q)
        low=linprog(c,A_ub=np.r_[A,-A],b_ub=upper,A_eq=np.ones((1,len(Q))),b_eq=[1],bounds=(0,None),method='highs')
        high=linprog(-c,A_ub=np.r_[A,-A],b_ub=upper,A_eq=np.ones((1,len(Q))),b_eq=[1],bounds=(0,None),method='highs')
        if not low.success or not high.success: raise RuntimeError('Feasible-range solver failed')
        rows.append(dict(lag=lag,lower=float(low.fun),upper=float(-high.fun),width=float(-high.fun-low.fun),
            sign_identified=bool(low.fun>0 or -high.fun<0),tolerance=tolerance,**fit))
    return rows


def audit_identifiability():
    _,ds,roles,_=bundle()
    B=np.exp(1j*LAGS[:,None]*Q)
    p_plus=(1+.9*np.cos(Q))/257; p_minus=(1-.9*np.cos(Q))/257
    example=[]
    for lag in range(33):
        example.append(dict(lag=lag,plus=float(np.real(np.exp(1j*lag*Q)@p_plus)),minus=float(np.real(np.exp(1j*lag*Q)@p_minus)),supervised=bool(lag in LAGS)))
    pd.DataFrame(example).to_csv(OUT/'synthetic_nonuniqueness.csv',index=False)
    rank=np.linalg.matrix_rank(np.r_[np.ones((1,257)),B.real,B.imag])
    pos=np.arange(32)
    write_json(OUT/'spectral_nonuniqueness.json',dict(rank=int(rank),nullity=257-int(rank),
        minimum_probability=float(min(p_plus.min(),p_minus.min())),
        supervised_difference=float(np.max(abs(B@(p_plus-p_minus)))),
        nearest_neighbour_plus=example[1]['plus'],nearest_neighbour_minus=example[1]['minus'],
        matrix_minimum_eigenvalue=float(min(np.linalg.eigvalsh(upper_kernel_matrix(p,pos)).min() for p in [p_plus,p_minus])),
        synthetic=True))
    ix=np.flatnonzero(roles=='architecture_test')
    rows=[]; fits=[]
    for n,i in enumerate(ix):
        targets=complex_values(ds.targets[i])
        for b in range(4):
            values=targets[:,b].mean(0)
            for row in feasible_bounds(values,[1,2,4,10]):
                rows.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],band=b,**row))
        for block,b in np.ndindex(8,4):
            _,fit=spectrum_fit(targets[block,b])
            fits.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],block=block,band=b,**fit))
        if n%12==0: print(f'identifiability {n+1}/{len(ix)}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'real_feasible_ranges.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'target_informed_spectral_fit.csv',index=False)
    # Validate a sign-certification/abstention rule on several known spectra.
    rng=np.random.default_rng(SEED); synt=[]
    for family in ['white','smooth','narrow','multimode']:
        for rep in range(20):
            if family=='white': p=np.ones(257)/257
            elif family=='smooth':
                centre=rng.uniform(-np.pi,np.pi); p=np.exp(1.5*np.cos(Q-centre)); p/=p.sum()
            elif family=='narrow':
                centre=rng.uniform(-np.pi,np.pi); p=np.exp(20*np.cos(Q-centre)); p/=p.sum()
            else:
                p=np.exp(12*np.cos(Q-rng.uniform(-np.pi,np.pi)))+.6*np.exp(8*np.cos(Q-rng.uniform(-np.pi,np.pi))); p/=p.sum()
            observed=B@p+rng.uniform(-.01,.01,8)+1j*rng.uniform(-.01,.01,8)
            for r in feasible_bounds(observed,[1,2,4,10]):
                truth=float(np.cos(r['lag']*Q)@p)
                synt.append(dict(family=family,replicate=rep,truth=truth,truth_covered=r['lower']-1e-7<=truth<=r['upper']+1e-7,
                    wrong_certified_sign=r['sign_identified'] and np.sign(truth)!=np.sign(r['lower']+r['upper']),**r))
    pd.DataFrame(synt).to_csv(OUT/'synthetic_abstention_validation.csv',index=False)


def coverage_interval(success,n):
    return (0. if success==0 else float(beta.ppf(.025,success,n-success+1)),
            1. if success==n else float(beta.ppf(.975,success+1,n-success)))


def conformal_quantile(values,nominal):
    a=np.sort(np.asarray(values,float)); k=math.ceil((len(a)+1)*nominal)
    return (float(a[k-1]) if k<=len(a) else float('inf')),k


def audit_uncertainty():
    _,ds,roles,groups=bundle()
    table=pd.read_parquet(OUT/'historical_metrics.parquet')
    rows=[]
    for endpoint in ['historical_28','all_32']:
        local=table[(table.model=='Local-state PSD')&(table.endpoint==endpoint)]
        cal=local[local.role=='calibration'].groupby('event_id').nrmse.max()
        test=local[local.role=='architecture_test']
        for nominal in [.8,.9,.95]:
            threshold,k=conformal_quantile(cal.to_numpy(),nominal)
            units={'joint_earthquake':test.groupby('event_id').nrmse.max()}
            units.update({route:g.set_index('event_id').nrmse for route,g in test.groupby('route')})
            for unit,errors in units.items():
                count=int((errors<=threshold).sum()); low,high=coverage_interval(count,len(errors))
                rows.append(dict(endpoint=endpoint,unit=unit,nominal=nominal,threshold=threshold,order=k,
                    successes=count,n=len(errors),coverage=count/len(errors),ci_low=low,ci_high=high))
    pd.DataFrame(rows).to_csv(OUT/'joint_conformal_coverage.csv',index=False)
    print(pd.DataFrame(rows).query("unit == 'joint_earthquake'").to_string(index=False),flush=True)


def main(stage):
    if stage=='baseline': audit_baseline()
    elif stage=='identifiability': audit_identifiability()
    elif stage=='uncertainty': audit_uncertainty()
    else: raise ValueError(stage)
