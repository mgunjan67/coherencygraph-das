"""Matched experiments for critical-review revision; never tunes on test scores."""
from __future__ import annotations
import hashlib
import itertools
import json
import time
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy import linalg
from scipy.stats import spearmanr

from .critical_revision import (ROOT,OUT,OLD,MODELS,LAGS,Q,SEED,bundle,metrics,components,complex_values,
    boot,spectrum_fit,upper_kernel_matrix,coverage_interval,conformal_quantile)
from .config import write_json,sha256
from .models import ModernBackbone
from .methodological_audit import _load_audit_amendment,_load_model_members,_ensemble_predict
from .review_revision import _generalized_weight,_ratio

torch.set_num_threads(4)


def protocol():
    return yaml.safe_load((ROOT/'configs/protocol_amendment_07_critical_review.yaml').read_text())


class MatchedModel(nn.Module):
    def __init__(self,features,kind):
        super().__init__(); self.kind=kind
        self.psd=kind.endswith('psd')
        if kind.startswith('full_mlp'):
            self.encoder=nn.Sequential(nn.Flatten(1),nn.Linear(features*8,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,128),nn.GELU())
            self.head=nn.Linear(128,8*4*(257 if self.psd else 16))
        else:
            self.encoder=ModernBackbone(features,96,2,.1,'bissm_psd')
            self.head=nn.Linear(96,4*(257 if self.psd else 16))
        basis=np.exp(1j*LAGS[:,None]*Q)
        self.register_buffer('basis_re',torch.tensor(basis.real,dtype=torch.float32))
        self.register_buffer('basis_im',torch.tensor(basis.imag,dtype=torch.float32))

    def forward(self,x):
        z=self.head(self.encoder(x))
        if self.psd:
            p=torch.softmax(z.reshape(-1,8,4,257),-1)
            y=torch.stack([torch.einsum('nxbq,lq->nxbl',p,self.basis_re),torch.einsum('nxbq,lq->nxbl',p,self.basis_im)],-1)
            return y,p
        return z.reshape(-1,8,4,8,2),None


def _scale(x,indices):
    mean=x[indices].mean((0,1),keepdims=True); std=x[indices].std((0,1),keepdims=True)
    std=np.where(std<1e-5,1,std)
    if x.shape[-1]==138:
        mean[...,137]=0.;std[...,137]=1.
    return ((x-mean)/std).astype('float32'),mean,std


def _score_array(p,t):
    # Last axis stores real and imaginary components; factors of two cancel.
    return np.sqrt(np.mean((p-t)**2,axis=(1,2,3,4))/np.maximum(np.mean(t**2,axis=(1,2,3,4)),1e-15))


def _partition(ds,indices,groups,seed):
    unique=sorted(set(groups[indices]),key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    validation=set(unique[:max(3,len(unique)//5)])
    val=indices[np.isin(groups[indices],list(validation))]
    train=indices[~np.isin(indices,val)]
    if not len(train) or not len(val): raise ValueError('Insufficient source groups for selection')
    return train,val


def _torch_fit(x,y,ds,indices,groups,kind,seed,tag,augment=False):
    path=MODELS/f'{tag}_{kind}_seed{seed}.pt'; receipt=path.with_suffix('.json')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if path.exists() and receipt.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=MatchedModel(x.shape[-1],kind).to(device);model.load_state_dict(saved['state']);model.eval()
        return model,saved['mean'].numpy(),saved['std'].numpy(),json.loads(receipt.read_text())
    config=protocol()['models']; begin=time.perf_counter()
    tr,va=_partition(ds,indices,groups,seed)
    normalized,inner_mean,inner_std=_scale(x,tr)
    target=torch.tensor(y,device=device); values=torch.tensor(normalized,device=device)

    def new():
        torch.manual_seed(seed);np.random.seed(seed)
        return MatchedModel(x.shape[-1],kind).to(device)

    def epoch(model,opt,train,rng):
        model.train(); events=np.unique(ds.event_ids[train]);rng.shuffle(events); loss_sum=[]
        for start in range(0,len(events),12):
            batch=train[np.isin(ds.event_ids[train],events[start:start+12])]
            bx=values[batch].clone()
            if augment:
                for n in range(len(bx)):
                    if rng.random()<.8:
                        miss=rng.choice(8,size=int(rng.integers(1,3)),replace=False)
                        bx[n,miss,:134]=0
                        if bx.shape[-1]>137: bx[n,miss,137]=0
            opt.zero_grad(set_to_none=True)
            pred,_=model(bx); loss=((pred-target[batch])**2).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
            loss_sum.append(float(loss.detach()))
        return float(np.mean(loss_sum))
    model=new(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);best=float('inf');best_epoch=1;stale=0;history=[]
    for e in range(config['maximum_epochs']):
        loss=epoch(model,opt,tr,rng);model.eval()
        with torch.no_grad(): pred,_=model(values[va])
        score=float(_score_array(pred.cpu().numpy(),y[va]).mean())
        history.append(dict(epoch=e+1,loss=loss,validation_nrmse=score))
        if score<best-1e-5: best=score;best_epoch=e+1;stale=0
        else: stale+=1
        if stale>=config['patience']:break
    normalized,mean,std=_scale(x,indices);values=torch.tensor(normalized,device=device)
    model=new();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed)
    losses=[epoch(model,opt,indices,rng) for _ in range(best_epoch)]
    torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},mean=torch.tensor(mean),std=torch.tensor(std)),path)
    model.to(device).eval()
    record=dict(kind=kind,seed=seed,selection_events=len(np.unique(ds.event_ids[tr])),
        validation_events=len(np.unique(ds.event_ids[va])),selected_epoch=best_epoch,validation_nrmse=best,
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.perf_counter()-begin,
        hidden=128 if kind.startswith('full_mlp') else 96,layers=2,dropout=.1,
        history=history,refit_losses=losses,scaler_fit='training fold for selection, all permitted development for refit',
        train_ids=sorted(set(ds.event_ids[indices])),validation_ids=sorted(set(ds.event_ids[va])),
        checkpoint_sha256=sha256(path))
    write_json(receipt,record)
    print(f'{tag} {kind} seed {seed}: {best_epoch} epochs, {record["training_seconds"]:.1f}s',flush=True)
    return model,mean,std,record


def predict_model(model,x,mean,std):
    device=next(model.parameters()).device
    values=((x-mean)/std).astype('float32');out=[];probs=[]
    with torch.no_grad():
        for start in range(0,len(values),48):
            p,q=model(torch.tensor(values[start:start+48],device=device))
            out.append(p.cpu().numpy())
            if q is not None:probs.append(q.cpu().numpy())
    return np.concatenate(out),np.concatenate(probs) if probs else None


def ridge_fit(x,y,ds,train,groups,kind,tag,context=None):
    cfg=protocol()['models']; folds=list(GroupKFold(4).split(train,groups=groups[train])); grid=[]
    target=y if kind!='residual_ridge' else y-context
    def design(values,response,indices):
        if kind=='block_ridge':return values[indices].reshape(-1,values.shape[-1]),response[indices].reshape(-1,64)
        return values[indices].reshape(len(indices),-1),response[indices].reshape(len(indices),-1)
    for alpha in cfg['ridge_alphas']:
        scores=[]
        for fold,(a,b) in enumerate(folds):
            tr,va=train[a],train[b];z,_,_=_scale(x,tr)
            tx,ty=design(z,target,tr); vx,_=design(z,target,va)
            fit=Ridge(alpha=alpha).fit(tx,ty)
            p=fit.predict(vx).reshape(y[va].shape)
            if kind=='residual_ridge': p+=context[va]
            scores.extend(_score_array(p,y[va]))
        grid.append(dict(kind=kind,alpha=alpha,development_group_cv_nrmse=float(np.mean(scores))))
    alpha=min(grid,key=lambda r:r['development_group_cv_nrmse'])['alpha']
    z,mean,std=_scale(x,train);tx,ty=design(z,target,train);ix=np.arange(len(y));vx,_=design(z,target,ix)
    start=time.perf_counter();fit=Ridge(alpha=alpha).fit(tx,ty);pred=fit.predict(vx).reshape(y.shape)
    if kind=='residual_ridge':pred+=context
    np.savez_compressed(MODELS/f'{tag}_{kind}_parameters.npz',coef=fit.coef_,intercept=fit.intercept_,mean=mean,std=std,alpha=alpha)
    record=dict(kind=kind,selected_alpha=alpha,parameters=fit.coef_.size+fit.intercept_.size,training_seconds=time.perf_counter()-start,grid=grid)
    write_json(MODELS/f'{tag}_{kind}_selection.json',record)
    return pred.astype('float32'),record


def metric_table(predictions,ds,roles,groups,targets,indices=None):
    if indices is None: indices=np.flatnonzero(np.isin(roles,['calibration','architecture_test']))
    rows=[]
    for name,pred in predictions.items():
        for i in indices:
            rows.append(dict(model=name,role=roles[i],event_id=ds.event_ids[i],source_group_id=groups[i],route=ds.routes[i],**metrics(pred[i],targets[i])))
    return pd.DataFrame(rows)


def run_training():
    _,ds,roles,groups=bundle();train=np.flatnonzero(roles=='model_development')
    x=ds.features;y=ds.targets
    context=np.stack([components(np.load(p)['context_gamma']) for p in ds.paths])
    predictions={}; records=[]
    for kind in ['block_ridge','full_ridge','residual_ridge']:
        predictions[kind],rec=ridge_fit(x,y,ds,train,groups,kind,'matched',context)
        np.save(MODELS/f'matched_{kind}_all_predictions.npy',predictions[kind]);records.append(rec)
        print(f'completed {kind} (development-selected alpha {rec["selected_alpha"]})',flush=True)
    for kind in ['full_mlp_direct','full_mlp_psd','state_direct','state_psd']:
        members=[];ps=[]
        for seed in [19,43,71]:
            model,mean,std,rec=_torch_fit(x,y,ds,train,groups,kind,seed,'matched')
            p,q=predict_model(model,x,mean,std);members.append(p);records.append(rec)
            np.save(MODELS/f'matched_{kind}_seed{seed}_all_predictions.npy',p)
            if q is not None:ps.append(q);np.save(MODELS/f'matched_{kind}_seed{seed}_all_probabilities.npy',q)
        predictions[kind]=np.mean(members,axis=0)
        np.save(MODELS/f'matched_{kind}_all_predictions.npy',predictions[kind])
        if ps:np.save(MODELS/f'matched_{kind}_all_probabilities.npy',np.mean(ps,axis=0))
    metric_table(predictions,ds,roles,groups,y).to_parquet(OUT/'matched_model_metrics.parquet',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['history','refit_losses','grid','train_ids','validation_ids']} for r in records]).to_csv(OUT/'matched_compute.csv',index=False)
    print('matched training complete',flush=True)


def run_sensitivity():
    _,ds,roles,groups=bundle();train=np.flatnonzero(roles=='model_development')
    indices=np.flatnonzero(np.isin(roles,['model_development','calibration','architecture_test']))
    output=[]
    for variant in ['equal_7s','global_cutoff','pool_before_normalize']:
        x=ds.features.copy(); y=ds.targets.copy()
        for i in indices:
            p=ROOT/'data/processed/critical_review'/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz'
            with np.load(p) as z:
                x[i]=z[f'{variant}_features']
                if variant=='pool_before_normalize':y[i]=z['pool_before_normalize_targets']
        predictions={}
        predictions['full_ridge'],_=ridge_fit(x,y,ds,train,groups,'full_ridge',variant)
        for kind in ['full_mlp_direct','state_psd']:
            model,mean,std,_=_torch_fit(x,y,ds,train,groups,kind,19,variant)
            predictions[kind],_=predict_model(model,x,mean,std)
        for name,p in predictions.items():np.save(MODELS/f'{variant}_{name}_all_predictions.npy',p)
        t=metric_table(predictions,ds,roles,groups,y);t['variant']=variant;t['neural_seeds']=1;output.append(t)
        pd.concat(output).to_parquet(OUT/'estimator_sensitivity_metrics.parquet',index=False)
        print(f'completed sensitivity {variant}',flush=True)


def run_internal():
    _,ds,roles,groups=bundle();meta=pd.read_parquet(OUT/'event_metadata.parquet')
    comp=meta.set_index('event_id').sensitivity_component_25km.to_dict();groups=np.array([comp[str(e)] for e in ds.event_ids])
    development=np.flatnonzero(roles=='model_development')
    folds=[]
    for f,(a,b) in enumerate(GroupKFold(4).split(development,groups=groups[development])):
        folds.append((f'group_fold_{f}',development[a],development[b]))
    dates=meta.set_index('event_id').archive_date.to_dict()
    group_latest={g:max(str(dates[str(ds.event_ids[i])]) for i in development if groups[i]==g) for g in set(groups[development])}
    ordered=sorted(group_latest,key=lambda g:group_latest[g]);held=set(ordered[-max(3,len(ordered)//4):])
    # Retain the original latest-group-date stress under its precise name.
    # A separate purged temporal split enforces chronological order on events.
    folds.append(('chronological',development[~np.isin(groups[development],list(held))],development[np.isin(groups[development],list(held))]))
    dev_events=meta[meta.event_id.isin(ds.event_ids[development])].sort_values(['archive_date','event_id'])
    cutoff=dev_events.archive_date.iloc[int(.8*len(dev_events))]
    future_events=set(dev_events.loc[dev_events.archive_date>=cutoff,'event_id'])
    temporal_test=development[np.isin(ds.event_ids[development],list(future_events))]
    purge=set(groups[temporal_test])
    temporal_train=development[(~np.isin(ds.event_ids[development],list(future_events)))&(~np.isin(groups[development],list(purge)))]
    write_json(OUT/'temporal_split_specification.json',dict(
        cutoff=str(cutoff),rule='80th percentile development event date; purge every training event connected within 25 km to any temporal test component',
        amendment_note='Corrects initial latest-group-date stress, which did not guarantee chronological event order. No model choice uses stress outcomes.',
        training_events=sorted(set(ds.event_ids[temporal_train])),test_events=sorted(set(ds.event_ids[temporal_test]))))
    if len(set(groups[temporal_train]))>=4:
        folds.append(('purged_temporal',temporal_train,temporal_test))
    records=[]
    for tag,tr,te in folds:
        predictions={}
        # Nested source-component CV selects ridge alpha; neural epoch selection
        # occurs on a further inner training-only component split.
        predictions['full_ridge'],_=ridge_fit(ds.features,ds.targets,ds,tr,groups,'full_ridge',tag)
        for kind in ['full_mlp_direct','state_psd']:
            model,mean,std,_=_torch_fit(ds.features,ds.targets,ds,tr,groups,kind,19,tag)
            predictions[kind],_=predict_model(model,ds.features,mean,std)
        table=metric_table(predictions,ds,roles,groups,ds.targets,te);table['internal_fold']=tag
        table['training_events']=len(set(ds.event_ids[tr]));records.append(table)
        for name,p in predictions.items():np.save(MODELS/f'{tag}_{name}_all_predictions.npy',p)
        pd.concat(records).to_parquet(OUT/'internal_validation.parquet',index=False)
        print(f'completed internal {tag}',flush=True)


def run_missingness():
    cfg,ds,roles,groups=bundle();tr=np.flatnonzero(roles=='model_development');te=np.flatnonzero(roles=='architecture_test')
    x=ds.features;mask_x=np.concatenate([x,np.ones((*x.shape[:2],1),'float32')],axis=-1)
    families={}
    for kind,augment,tag in [('state_psd',False,'indicator_only'),('full_mlp_direct',True,'mlp_imputation')]:
        members=[]
        for seed in [19,43,71]:
            model,mean,std,rec=_torch_fit(mask_x,ds.targets,ds,tr,groups,kind,seed,tag,augment)
            members.append((model,mean,std))
        families[tag]=members
    # Historical models use their archived scaler. Preserve static position,
    # route and gauge on the new missingness evaluation (the old script zeroed them).
    amendment,_=_load_audit_amendment(cfg)
    hist_scaler=np.load(OLD/'corrected_feature_scaler.npz')
    for label,prefix,mask in [('historical_unaugmented','corrected',False),('historical_mask_dropout','missing',True)]:
        mem=_load_model_members('gat_bissm_psd',137+int(mask),ds,amendment,OLD,prefix,257)
        families[label]=[(m,hist_scaler['mean'],hist_scaler['std']) for m in mem]
    records=[];mask_rows=[]
    for k in [1,2,4]:
        for mi,missing in enumerate(itertools.combinations(range(8),k)):
            flags=np.ones(8,bool);flags[list(missing)]=False
            mask_rows.append(dict(level=k,mask_id=mi,missing_blocks=','.join(map(str,missing))))
            for name,members in families.items():
                preds=[]
                for model,mean,std in members:
                    values=((x[te]-mean[...,:137])/std[...,:137]).astype('float32')
                    values[:,list(missing),:134]=0
                    if name!='historical_unaugmented':
                        indicator=np.ones((*values.shape[:2],1),'float32');indicator[:,list(missing)]=0
                        # Appended indicator remains 1 when observed: historical
                        # models were trained with that convention.
                        values=np.concatenate([values,indicator],-1)
                    with torch.no_grad():p,_=model(torch.tensor(values,device=next(model.parameters()).device))
                    preds.append(p.cpu().numpy())
                prediction=np.mean(preds,axis=0)
                for n,i in enumerate(te):
                    for endpoint,bl in [('all_blocks',np.ones(8,bool)),('missing_only',~flags),('retained_only',flags)]:
                        records.append(dict(model=name,event_id=ds.event_ids[i],route=ds.routes[i],source_group_id=groups[i],
                            missing_count=k,mask_id=mi,endpoint=endpoint,**metrics(prediction[n],ds.targets[i],blocks=bl)))
            if mi%14==0:print(f'missingness k={k} mask={mi}',flush=True)
        pd.DataFrame(records).to_parquet(OUT/'exhaustive_missingness.parquet',index=False)
    pd.DataFrame(mask_rows).to_csv(OUT/'unique_mask_inventory.csv',index=False)


def run_downstream():
    _,ds,roles,groups=bundle();te=np.flatnonzero(roles=='architecture_test')
    probability=np.load(OLD/'corrected_gat_bissm_psd_architecture_test_probabilities.npy')
    revised_probability=np.load(MODELS/'matched_state_psd_all_probabilities.npy')
    records=[];geometry=[]
    for n,i in enumerate(te):
        cached=ROOT/'data/processed/critical_review'/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz'
        with np.load(cached) as z:
            rt,rc,rn,h1,h2=[z[k] for k in ['target_anchor','context_anchor','noise_anchor_corrected','target_first_anchor','target_second_anchor']]
        with np.load(ds.paths[i]) as z:positions=z['anchor_local'];old_noise=z['noise_anchor']
        d=abs(positions[:,None]-positions[None,:]);off=~np.eye(32,dtype=bool)
        geometry.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],channels=','.join(map(str,positions)),
            aperture_channels=int(positions.max()-positions.min()),ordered_offdiagonal=off.sum(),
            supervised_entries=int(np.isin(d[off],LAGS).sum()),beyond_max_lag=int((d[off]>89).sum())))
        for b,f in np.ndindex(8,4):
            D=np.diag(np.maximum(np.diag(rc[b,f]).real,1e-15));scale=np.sqrt(np.diag(D)[:,None]*np.diag(D)[None,:])
            corrected=upper_kernel_matrix(probability[n,b,f],positions)*scale
            revised=upper_kernel_matrix(revised_probability[i,b,f],positions)*scale
            legacy=corrected.conj()
            classical=.25*rc[b,f]+.75*D
            for mode,noise in [('corrected_frame',rn[b,f]),('legacy_noise_frame',old_noise[b,f])]:
                methods={'Diagonal':D,'Classical shrinkage':classical,
                    'Learned corrected sign':.25*corrected+.75*D,
                    'Learned revised recurrence':.25*revised+.75*D,
                    'Learned legacy sign':.25*legacy+.75*D,'Oracle same estimate':rt[b,f]}
                for name,cov in methods.items():
                    w=_generalized_weight(cov,noise,.0001)
                    records.append(dict(event_id=ds.event_ids[i],source_group_id=groups[i],route=ds.routes[i],block=b,band=f,
                        frame=mode,evaluation='full_window_in_sample',method=name,
                        ratio_db=10*np.log10(_ratio(rt[b,f],noise,w)),
                        minimum_eigenvalue=float(np.linalg.eigvalsh(cov).min()),condition_number=float(np.linalg.cond(cov))))
            # Equal-duration halves provide a cross-fit stability diagnostic;
            # half-window nonstationarity and smoothing differ from full window.
            for fold,(fit,eval_) in enumerate([(h1[b,f],h2[b,f]),(h2[b,f],h1[b,f])]):
                methods={'Diagonal':D,'Classical shrinkage':classical,'Learned corrected sign':.25*corrected+.75*D,
                    'Learned revised recurrence':.25*revised+.75*D,
                    'Oracle other half':fit,'Oracle same half':eval_}
                for name,cov in methods.items():
                    w=_generalized_weight(cov,rn[b,f],.0001)
                    records.append(dict(event_id=ds.event_ids[i],source_group_id=groups[i],route=ds.routes[i],block=b,band=f,
                        frame='corrected_frame',evaluation=f'half_crossfit_{fold}',method=name,
                        ratio_db=10*np.log10(_ratio(eval_,rn[b,f],w))))
        if n%12==0:print(f'downstream {n+1}/{len(te)}',flush=True)
    pd.DataFrame(records).to_parquet(OUT/'downstream_revision.parquet',index=False)
    pd.DataFrame(geometry).to_csv(OUT/'actual_downstream_geometry.csv',index=False)


def run_projection_and_synthetics():
    from .spectral import multitaper_fourier,band_snapshots,lag_coherency
    from .critical_raw import pooled_coherency
    _,ds,roles,groups=bundle();indices=np.flatnonzero(np.isin(roles,['calibration','architecture_test']))
    records=[];diagnostics=[]
    for kind in ['full_ridge','full_mlp_direct','state_direct']:
        pred=np.load(MODELS/f'matched_{kind}_all_predictions.npy');projected=pred.copy()
        for n,i in enumerate(indices):
            for b,f in np.ndindex(8,4):
                values=complex_values(pred[i,b,f]);p,fit=spectrum_fit(values)
                fitted=np.exp(1j*LAGS[:,None]*Q)@p
                projected[i,b,f]=components(fitted)
                diagnostics.append(dict(model=kind,event_id=ds.event_ids[i],route=ds.routes[i],block=b,band=f,**fit))
        np.save(MODELS/f'matched_{kind}_simplex_all_predictions.npy',projected)
        records.extend(metric_table({kind+'_simplex':projected},ds,roles,groups,ds.targets).to_dict('records'))
        print('simplex projection '+kind,flush=True)
    pd.DataFrame(records).to_parquet(OUT/'simplex_projection_metrics.parquet',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'simplex_solver_diagnostics.csv',index=False)
    # Stationary real Gaussian time series have known spatial covariance.
    # Independent times intentionally isolate finite-window spectral estimation.
    rng=np.random.default_rng(SEED);nch=96;rho=.8
    covariance=rho**abs(np.arange(nch)[:,None]-np.arange(nch)[None,:])
    factor=np.linalg.cholesky(covariance);samples=[]
    for duration in [5.,7.,14.]:
        for rep in range(80):
            waveform=rng.normal(size=(int(25*duration),nch))@factor.T
            tr,hz=multitaper_fourier(waveform,25,2.5,3)
            for b,band in enumerate([[.5,1],[1,2],[2,4],[4,8]]):
                snaps=band_snapshots(tr,hz,band)
                for name,est in [('pair_normalized',lag_coherency(snaps,LAGS)),('pooled_power',pooled_coherency(snaps,LAGS))]:
                    for li,lag in enumerate(LAGS):
                        samples.append(dict(duration=duration,replicate=rep,band=b,estimator=name,lag=lag,
                            estimate_real=float(est[li].real),estimate_imag=float(est[li].imag),truth=rho**lag,
                            snapshots=len(snaps),half_bandwidth_hz=2.5/duration))
    table=pd.DataFrame(samples);table.to_parquet(OUT/'stationary_estimator_synthetics.parquet',index=False)
    table['sq_error']=(table.estimate_real-table.truth)**2+table.estimate_imag**2
    table['bias']=table.estimate_real-table.truth
    table.groupby(['duration','band','estimator']).agg(mean_bias=('bias','mean'),mse=('sq_error','mean')).to_csv(OUT/'stationary_estimator_summary.csv')
    historical=np.load(OLD/'baseline_low-rank_persistence_architecture_test.npy')
    persistence=np.load(OLD/'baseline_persistence_architecture_test.npy')
    write_json(OUT/'low_rank_diagnostic.json',dict(
        algorithm='interpolate sparse lags, Hermitian circulant embedding, retain largest 16 Fourier eigenvalues, rescale sum to period',
        no_small_diagonal_division=True,
        median_magnitude_low_rank=float(np.median(abs(complex_values(historical)))),
        median_magnitude_context=float(np.median(abs(complex_values(persistence)))),
        max_magnitude_low_rank=float(np.max(abs(complex_values(historical)))),
        interpretation='Concentrating the spectrum into 16 bins then restoring unit variance imposes strong correlations; not a numerical overflow or a validated denoising rule.'))


def main(stage):
    if stage=='train':run_training()
    elif stage=='sensitivity':run_sensitivity()
    elif stage=='internal':run_internal()
    elif stage=='missingness':run_missingness()
    elif stage=='downstream':run_downstream()
    elif stage=='projection':run_projection_and_synthetics()
    elif stage in ['summary','figures','release','verify']:
        from .critical_paper import main as paper_main
        paper_main(stage)
    else:raise ValueError(stage)
