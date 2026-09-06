"""Finite Amendment-08 experiments. Historical data/results are read-only."""
from pathlib import Path
import json, shutil, time, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from .critical_revision import ROOT, bundle, components, complex_values, metrics, boot, LAGS, Q, spectrum_fit, upper_kernel_matrix
from .critical_experiments import _scale, _score_array, MatchedModel, predict_model, _partition
from .config import sha256, write_json

OUT=ROOT/'reports/final_revision'; CACHE=ROOT/'data/processed/final_revision'; MOD=ROOT/'models/final_revision'
for p in (OUT,CACHE,MOD): p.mkdir(exist_ok=True,parents=True)
SPEC=ROOT/'configs/protocol_amendment_08_final_revision.yaml'
CFG=yaml.safe_load(SPEC.read_text())
DENSE=np.arange(1,32)
MEASURE=np.array(sorted(set(DENSE)|set(LAGS)))

def event_table(pred,truth,ds,roles,name,endpoint):
    return pd.DataFrame([dict(event_id=str(ds.event_ids[i]),route=str(ds.routes[i]),model=name,endpoint=endpoint,**metrics(pred[i],truth[i])) for i in np.flatnonzero(roles=='architecture_test')])

def paired(frame,reference,keys=('endpoint',)):
    meta=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet')
    mapping=meta.set_index('event_id').sensitivity_component_25km.to_dict()
    rows=[]
    for key,g in frame.groupby(list(keys)):
        if not isinstance(key,tuple): key=(key,)
        e=g.groupby(['event_id','model']).nrmse.mean().unstack()
        for model in e.columns:
            if reference not in e: continue
            values=e[model]-e[reference]
            rows.append(dict(zip(keys,key),model=model,reference=reference,score=float(e[model].mean()),**boot(values,[mapping[str(e)] for e in values.index])))
    return pd.DataFrame(rows)

def audit():
    cfg,ds,roles,groups=bundle()
    frozen=OUT/'event_roles.csv'
    frame=pd.DataFrame(dict(event_id=ds.event_ids,route=ds.routes,role=roles,source_bin=groups,path=ds.paths))
    content=frame.to_csv(index=False)
    if frozen.exists(): assert frozen.read_text()==content
    else: frozen.write_text(content)
    (OUT/'analysis_specification.sha256').write_text(sha256(SPEC)+'\n')
    snapshot=ROOT/'revisions/20260906_pre_final_revision'
    if not snapshot.exists():
        snapshot.mkdir(parents=True)
        shutil.copytree(ROOT/'manuscript/cageo_submission',snapshot/'manuscript')
        for name in ['FINAL_REVIEW_GATE_REPORT.md','AUTHOR_ACTIONS.md','CLAIMS_LEDGER.md']:
            shutil.copy2(ROOT/name,snapshot/name)
    inventory=[]
    for folder in ['src','scripts','configs','manuscript/cageo_submission','models/critical_review','reports/critical_review','data/processed/critical_review']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.log','.aux','.out']:
                inventory.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)))
    pd.DataFrame(inventory).to_csv(OUT/'asset_inventory.csv',index=False)
    meta=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet')
    pd.DataFrame([dict(event_id=r.event_id,kind=k,path=getattr(r,k),available=Path(getattr(r,k)).exists()) for r in meta.itertuples() for k in ['terra_path','kkfls_path','picks_path']]).to_csv(OUT/'raw_availability.csv',index=False)
    rows=[]
    for name in ['block_ridge','full_ridge','residual_ridge','full_mlp_direct','full_mlp_psd','state_direct','state_psd']:
        p=np.load(ROOT/f'models/critical_review/matched_{name}_all_predictions.npy')
        rows.append(event_table(p,ds.targets,ds,roles,name,'historical_all32'))
    scores=pd.concat(rows);scores.to_csv(OUT/'baseline_event_scores.csv',index=False)
    summary=paired(scores,'block_ridge');summary.to_csv(OUT/'baseline_reproduction.csv',index=False)
    prior=pd.read_csv(ROOT/'reports/critical_review/matched_summary.csv').set_index('model')['mean']
    summary['prior_score']=summary.model.map(prior)
    summary['difference']=summary.score-summary.prior_score
    summary[['model','score','prior_score','difference']].to_csv(OUT/'discrepancy_log.csv',index=False)
    write_json(OUT/'starting_point.json',dict(role_counts=frame.groupby('role').event_id.nunique().to_dict(),raw_records_available=int(sum(Path(p).exists() for p in meta.terra_path))+int(sum(Path(p).exists() for p in meta.kkfls_path)),requested_main8_pdf_found=False,source_used='current main.pdf and matching editable source; main(8).pdf not found in Downloads',spec_sha256=sha256(SPEC),manifest_sha256=sha256(frozen)))
    print(summary[['model','score','mean','low','high']].to_string(index=False),flush=True)

def ridge(x,y,ds,train,groups,kind,tag):
    path=MOD/f'{tag}_{kind}_predictions.npy'
    if path.exists(): return np.load(path)
    def design(z,ix):
        if kind=='block_ridge':return z[ix].reshape(-1,z.shape[-1]),y[ix].reshape(-1,np.prod(y.shape[2:]))
        return z[ix].reshape(len(ix),-1),y[ix].reshape(len(ix),-1)
    records=[]
    for alpha in CFG['baseline']['ridge_alphas']:
        scores=[]
        for a,b in GroupKFold(4).split(train,groups=groups[train]):
            tr,va=train[a],train[b]; z,_,_=_scale(x,tr)
            a1,a2=design(z,tr);b1,_=design(z,va)
            p=Ridge(alpha=alpha).fit(a1,a2).predict(b1).reshape(y[va].shape)
            scores.extend(_score_array(p,y[va]))
        records.append(dict(alpha=alpha,cv_nrmse=float(np.mean(scores))))
    alpha=min(records,key=lambda r:r['cv_nrmse'])['alpha']
    z,mean,std=_scale(x,train);a,b=design(z,train);fit=Ridge(alpha=alpha).fit(a,b)
    pred=fit.predict(design(z,np.arange(len(y)))[0]).reshape(y.shape).astype('float32')
    np.save(path,pred)
    np.savez_compressed(MOD/f'{tag}_{kind}_parameters.npz',coef=fit.coef_,intercept=fit.intercept_,mean=mean,std=std,alpha=alpha)
    write_json(MOD/f'{tag}_{kind}_selection.json',dict(alpha=alpha,grid=records,events=len(set(ds.event_ids[train])),training_rows=len(a),feature_dimension=a.shape[1],output_dimension=b.shape[1],parameters=fit.coef_.size+fit.intercept_.size,sharing='one common coefficient matrix across all blocks' if kind=='block_ridge' else 'one route-flattened model',folds='four development source-bin folds',scaling='per-feature moments pooled across training routes and blocks only'))
    print(tag,kind,'selected',alpha,flush=True)
    return pred

def baselines():
    _,ds,roles,groups=bundle();tr=np.flatnonzero(roles=='model_development');use=np.flatnonzero(roles!='prior_consistency')
    frames=[]
    for endpoint in CFG['baseline']['controls']:
        x=ds.features.copy();y=ds.targets.copy()
        if endpoint!='primary':
            for i in range(len(x)):
                f=ROOT/'data/processed/critical_review'/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz'
                if not f.exists():continue
                with np.load(f) as z:
                    x[i]=z[f'{endpoint}_features']
                    if endpoint=='pool_before_normalize':y[i]=z['pool_before_normalize_targets']
        for name in ['block_ridge','full_ridge']:
            p=ridge(x,y,ds,tr,groups,name,endpoint)
            frames.append(event_table(p,y,ds,roles,name,endpoint))
        context=(x[:,:,:32]+1j*x[:,:,32:64]).reshape(len(x),8,4,8)
        frames.append(event_table(components(context),y,ds,roles,'persistence',endpoint))
        names=['state_psd','state_direct','full_mlp_direct','full_mlp_psd'] if endpoint=='primary' else ['state_psd','full_mlp_direct']
        for name in names:
            tag='matched' if endpoint=='primary' else endpoint
            p=np.load(ROOT/f'models/critical_review/{tag}_{name}_all_predictions.npy')
            frames.append(event_table(p,y,ds,roles,name,endpoint))
    all_=pd.concat(frames);all_.to_csv(OUT/'corrected_event_scores.csv',index=False)
    paired(all_,'block_ridge').to_csv(OUT/'corrected_comparisons.csv',index=False)
    paired(all_,'full_ridge').to_csv(OUT/'secondary_full_ridge.csv',index=False)
    primary=all_.query('endpoint=="primary"')
    pd.concat([paired(primary.query('model in @names'),ref) for names,ref in [(['state_psd','state_direct'],'state_direct'),(['full_mlp_psd','full_mlp_direct'],'full_mlp_direct')]]).to_csv(OUT/'head_comparisons.csv',index=False)

def extract_one(row,route,cfg):
    from .labels import _read_window, RAW_DATASET
    from .spectral import robust_linear_pick_model,multitaper_fourier,band_snapshots,lag_coherency,cross_spectral_matrix
    import h5py
    path=CACHE/f'{row["event_id"]}_{route.replace("-","")}.npz'
    if path.exists():return str(path)
    prefix='terra' if route=='TERRA' else 'kkfls'
    archived_picks=ROOT/'data/provenance/picks'/f'{row["event_id"]}.csv'
    picks=pd.read_csv(archived_picks if archived_picks.exists() else row['picks_path'])
    ch=picks.channel.to_numpy(float);s=pd.to_numeric(picks['TERRA_s_sec' if route=='TERRA' else 'KKFLS_s_sec'],errors='coerce').to_numpy(float)
    gam={k:[] for k in ['context','target','first','second']};mats={k:[] for k in ['context','target','noise','first','second']}
    anchors=np.arange(484,516)
    with h5py.File(row[prefix+'_path'],'r') as h:
        raw=h[RAW_DATASET]
        for start in cfg['spectral']['block_starts']:
            ref,slope,_=robust_linear_pick_model(ch,s,start,1000)
            if not np.isfinite(ref): ref,slope=float(np.nanmedian(s)),0.
            delay=slope*(np.arange(1000)-499.5)
            w={k:_read_window(raw,a,b,25.,slice(start,start+1000)) for k,(a,b) in {'context':(ref-12,ref+2),'target':(ref+3,ref+10),'noise':(.5,5.5)}.items()}
            split=len(w['target'])//2;w['first']=w['target'][:split];w['second']=w['target'][split:]
            tr={k:multitaper_fourier(v,25.,2.5,3,delay) for k,v in w.items()}
            for k in mats:
                g=[];m=[]
                for band in cfg['spectral']['bands_hz']:
                    x=band_snapshots(*tr[k],band)
                    m.append(cross_spectral_matrix(x[:,anchors],.001))
                    if k in gam:g.append(lag_coherency(x,MEASURE))
                mats[k].append(m)
                if k in gam:gam[k].append(g)
    np.savez_compressed(path,measured_lags=MEASURE,positions=anchors,**{k+'_gamma':np.asarray(v) for k,v in gam.items()},**{k+'_matrix':np.asarray(v) for k,v in mats.items()})
    return str(path)

def raw():
    cfg,ds,roles,_=bundle();meta=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet');lookup=meta.set_index('event_id').to_dict('index')
    jobs=[(dict(event_id=str(ds.event_ids[i]),**lookup[str(ds.event_ids[i])]),str(ds.routes[i]),cfg) for i in range(len(ds.event_ids)) if roles[i] in ['model_development','calibration','architecture_test']]
    result=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(extract_one,*job) for job in jobs]
        for f in as_completed(futures):
            p=Path(f.result());result.append(dict(path=str(p),sha256=sha256(p)))
            if len(result)%12==0:print('new lag extraction',len(result),'/',len(jobs),flush=True)
    pd.DataFrame(result).to_csv(OUT/'expanded_raw_manifest.csv',index=False)

def main(stage):
    name={'bounds':'run_bounds','dense':'dense_training'}.get(stage,stage)
    globals()[name]()

def fit_grid(values,lags,grid=257):
    q=np.sort(np.fft.fftfreq(grid)*2*np.pi);B=np.exp(1j*np.asarray(lags)[:,None]*q)
    A=np.r_[B.real,B.imag];z=np.r_[values.real,values.imag]
    lip=np.linalg.norm(A@A.T,2);p=np.ones(grid)/grid;v=p.copy();t=1.
    for it in range(20000):
        u=v-A.T@(A@v-z)/lip;s=np.sort(u)[::-1];css=np.cumsum(s)-1
        rho=np.flatnonzero(s-css/np.arange(1,grid+1)>0)[-1]
        new=np.maximum(u-css[rho]/(rho+1),0)
        g=A.T@(A@new-z);gap=float(new@g-g.min())
        nt=(1+np.sqrt(1+4*t*t))/2;v=new+(t-1)/nt*(new-p);p,t=new,nt
        if gap<1e-8:break
    return p,A,z,q,dict(fit_gap=gap,fit_converged=gap<1e-8,fit_iterations=it+1,fit_residual=float(abs(A@p-z).max()))

def validated_min(c,H,b):
    """Dual-feasible outer lower bound, not a possibly infeasible primal optimum.

    For lambda<=0, nu=min(c-H.T lambda), r=c-H.T lambda-nu>=0.
    Any feasible simplex p has c'p >= b'lambda+nu. Accumulate the
    final products in long double and subtract a scale-dependent margin.
    """
    result=linprog(c,A_ub=H,b_ub=b,A_eq=np.ones((1,len(c))),b_eq=[1.],bounds=(0,None),method='highs',options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
    if not result.success:
        return dict(valid=False,status=int(result.status),reason=result.message,outer=None,primal=None,gap=None,primal_violation=None,dual_violation=None,rounding_margin=None)
    lam=np.minimum(np.asarray(result.ineqlin.marginals,np.longdouble),0)
    h=np.asarray(H,np.longdouble);cc=np.asarray(c,np.longdouble);bb=np.asarray(b,np.longdouble)
    reduced=cc-h.T@lam;nu=reduced.min()
    scale=1+np.sum(abs(bb*lam))+np.max(abs(h).T@abs(lam))
    margin=float(1e-10*scale)
    lower=float(bb@lam+nu)-margin
    dual_violation=float(max(0.,-(reduced-nu).min()))
    primal_violation=float(max(0.,(H@result.x-b).max(),abs(result.x.sum()-1),-result.x.min()))
    gap=float(result.fun-lower)
    valid=primal_violation<=1e-7 and dual_violation<=1e-10 and gap<=1e-6 and gap>=-1e-7
    return dict(valid=bool(valid),status=int(result.status),outer=lower,primal=float(result.fun),gap=gap,primal_violation=primal_violation,dual_violation=dual_violation,rounding_margin=margin)

def bounds(values,lags,unseen=(1,2,4,10),grid=257,offset=.01):
    p,A,z,q,fit=fit_grid(np.asarray(values),lags,grid)
    eps=fit['fit_residual']+offset;H=np.r_[A,-A];b=np.r_[z+eps,-z+eps];rows=[]
    for d in unseen:
        c=np.cos(d*q);lo=validated_min(c,H,b);hi=validated_min(-c,H,b)
        valid=lo['valid'] and hi['valid'];lower=lo['outer'];upper=-hi['outer'] if hi['outer'] is not None else None
        sign=1 if valid and lower>1e-7 else (-1 if valid and upper<-1e-7 else 0)
        rows.append(dict(lag=int(d),grid=grid,offset=offset,tolerance=eps,lower=lower,upper=upper,width=upper-lower if valid else None,valid=valid,sign=sign,identified=sign!=0,directly_measured=bool(d in lags),eta=1e-7,**fit,**{'low_'+k:v for k,v in lo.items()},**{'high_'+k:v for k,v in hi.items()}))
    return rows

def _bounds_record(i,event,route,targets,first,second):
    settings=[('historical',LAGS,m,s) for m,s in [(257,.005),(257,.01),(257,.03),(129,.01),(513,.01)]]
    settings += [(name,np.array(CFG['identifiability'][name+'_lags']),257,.01) for name in ['same_budget','augmented']]
    rows=[]
    for name,lags,grid,offset in settings:
        ids=[list(MEASURE).index(d) for d in lags]
        for band in range(4):
            y=targets[:,band,ids].mean(0)
            for r in bounds(y,lags,grid=grid,offset=offset):
                ix=list(MEASURE).index(r['lag']);truth=float(targets[:,band,ix].real.mean());a=float(first[:,band,ix].real.mean());b=float(second[:,band,ix].real.mean())
                rows.append(dict(event_id=event,route=route,band=band,design=name,input_source='measured_later_window_block_mean',estimated_additional_lag=truth,first_half_estimate=a,second_half_estimate=b,half_difference=abs(a-b),empirical_in_range=bool(r['valid'] and r['lower']<=truth<=r['upper']),empirical_sign_disagreement=bool(r['identified'] and np.sign(truth)!=r['sign']),**r))
    return rows

def run_bounds():
    _,ds,roles,_=bundle();rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[]
        for i in np.flatnonzero(roles=='architecture_test'):
            with np.load(CACHE/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz') as z:
                futures.append(pool.submit(_bounds_record,int(i),str(ds.event_ids[i]),str(ds.routes[i]),z['target_gamma'],z['first_gamma'],z['second_gamma']))
        for f in as_completed(futures):
            rows.extend(f.result());print('bound rows',len(rows),flush=True)
    table=pd.DataFrame(rows);table.to_parquet(OUT/'measurement_design_bounds.parquet',index=False)
    table.groupby(['design','grid','offset','directly_measured']).agg(cases=('identified','size'),identified=('identified','sum'),valid=('valid','sum'),mean_width=('width','mean'),max_fit_residual=('fit_residual','max'),empirical_agreement=('empirical_in_range','mean'),wrong_sign=('empirical_sign_disagreement','sum'),mean_half_difference=('half_difference','mean'),events=('event_id','nunique')).reset_index().to_csv(OUT/'measurement_design_summary.csv',index=False)
    table.groupby(['design','grid','offset','band','lag']).agg(cases=('identified','size'),identified=('identified','sum'),mean_width=('width','mean'),events=('event_id','nunique')).reset_index().to_csv(OUT/'bounds_by_band_lag.csv',index=False)

def synthetics():
    rng=np.random.default_rng(20260908);rows=[]
    for family in CFG['identifiability']['synthetic_conditions']:
        for rep in range(20):
            q=Q;centre=rng.uniform(-np.pi,np.pi)
            if family=='on_grid_white':p=np.ones(257)/257
            elif family=='on_grid_smooth':p=np.exp(1.5*np.cos(q-centre));p/=p.sum()
            elif family=='on_grid_narrow':p=np.exp(20*np.cos(q-centre));p/=p.sum()
            elif family=='on_grid_multimode':p=np.exp(12*np.cos(q-centre))+.6*np.exp(8*np.cos(q-rng.uniform(-np.pi,np.pi)));p/=p.sum()
            elif family=='off_grid':q=Q+np.pi/257;p=np.exp(30*np.cos(q-centre));p/=p.sum()
            else:
                # Spatially nonstationary unit-variance chirp covariance v v^H.
                # Lag-dependent pair averaging is not a common stationary law.
                x=np.arange(96);phase=centre*x+.007*x*x;v=np.exp(-1j*phase)
            all_lags=np.r_[LAGS,[1,2,4,10]]
            if family=='nonstationary':truth=np.array([np.mean(v[:-d]*v[d:].conj()) for d in all_lags])
            else:truth=np.exp(1j*all_lags[:,None]*q)@p
            y=truth[:8]+rng.uniform(-.01,.01,8)+1j*rng.uniform(-.01,.01,8)
            for j,r in enumerate(bounds(y,LAGS)):
                t=float(truth[8+j].real)
                rows.append(dict(family=family,replicate=rep,truth=t,covered=bool(r['valid'] and r['lower']<=t<=r['upper']),incorrect_certificate=bool(r['identified'] and r['sign']!=np.sign(t)),violated_assumption='none' if family.startswith('on_grid') else ('finite grid' if family=='off_grid' else 'stationary population'),**r))
    table=pd.DataFrame(rows);table.to_csv(OUT/'known_truth_validation.csv',index=False)
    table.groupby('family').agg(cases=('lag','size'),covered=('covered','sum'),identified=('identified','sum'),incorrect=('incorrect_certificate','sum'),valid=('valid','sum'),mean_width=('width','mean')).to_csv(OUT/'known_truth_summary.csv')

def dense_training():
    import torch
    from torch import nn
    torch.set_num_threads(4)
    _,ds,roles,groups=bundle();train=np.flatnonzero(roles=='model_development')
    y=np.zeros((len(ds.event_ids),8,4,31,2),dtype='float32')
    for i in range(len(y)):
        p=CACHE/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz'
        if p.exists():
            with np.load(p) as z:y[i]=components(z['target_gamma'][...,:31])
    np.save(MOD/'dense_targets.npy',y)
    frames=[]
    for name in ['block_ridge','full_ridge']:
        p=ridge(ds.features,y,ds,train,groups,name,'dense')
        frames.append(event_table(p,y,ds,roles,name,'dense31'))
    path=MOD/'dense_state_psd_seed19.pt'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def new():
        torch.manual_seed(19)
        m=MatchedModel(137,'state_psd')
        m.basis_re=torch.tensor(np.exp(1j*DENSE[:,None]*Q).real,dtype=torch.float32)
        m.basis_im=torch.tensor(np.exp(1j*DENSE[:,None]*Q).imag,dtype=torch.float32)
        return m.to(device)
    if path.exists():
        z=torch.load(path,map_location='cpu',weights_only=True);m=new();m.load_state_dict(z['state']);mean=z['mean'].numpy();std=z['std'].numpy()
    else:
        tr,va=_partition(ds,train,groups,19);x,_,_=_scale(ds.features,tr)
        values=torch.tensor(x,device=device);target=torch.tensor(y,device=device)
        def epoch(m,opt,ix,rng):
            m.train();events=np.unique(ds.event_ids[ix]);rng.shuffle(events);losses=[]
            for start in range(0,len(events),12):
                batch=ix[np.isin(ds.event_ids[ix],events[start:start+12])]
                opt.zero_grad(set_to_none=True);p,_=m(values[batch]);loss=((p-target[batch])**2).mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach()))
            return float(np.mean(losses))
        m=new();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(19);best=np.inf;best_epoch=1;stale=0;history=[]
        for e in range(110):
            loss=epoch(m,opt,tr,rng);m.eval()
            with torch.no_grad():p,_=m(values[va])
            score=float(_score_array(p.cpu().numpy(),y[va]).mean());history.append(dict(epoch=e+1,loss=loss,validation=score))
            if score<best-1e-5:best=score;best_epoch=e+1;stale=0
            else:stale+=1
            if stale>=16:break
        x,mean,std=_scale(ds.features,train);values=torch.tensor(x,device=device)
        m=new();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(19)
        for e in range(best_epoch):epoch(m,opt,train,rng)
        torch.save(dict(state={k:v.cpu() for k,v in m.state_dict().items()},mean=torch.tensor(mean),std=torch.tensor(std)),path)
        write_json(MOD/'dense_state_psd_selection.json',dict(seed=19,selected_epoch=best_epoch,validation_nrmse=best,parameters=sum(p.numel() for p in m.parameters()),history=history,learning_rate=.001,weight_decay=.0001,optimizer='AdamW',training_events=93,train_ids=sorted(set(ds.event_ids[train])),internal_validation_ids=sorted(set(ds.event_ids[va])),feature_design='unchanged historical 137 context features',output_lags=DENSE.tolist()))
    m.eval();p,q=predict_model(m,ds.features,mean,std)
    np.save(MOD/'dense_state_psd_predictions.npy',p);np.save(MOD/'dense_state_psd_probabilities.npy',q)
    frames.append(event_table(p,y,ds,roles,'state_psd','dense31'))
    table=pd.concat(frames);table.to_csv(OUT/'dense_prediction_scores.csv',index=False);paired(table,'block_ridge').to_csv(OUT/'dense_prediction_comparisons.csv',index=False)
    print('dense model complete',flush=True)

def processing():
    from .review_revision import _generalized_weight,_ratio
    _,ds,roles,_=bundle();ix=np.flatnonzero(roles=='architecture_test')
    hist=np.load(ROOT/'models/critical_review/matched_state_psd_all_probabilities.npy')
    seed=np.load(ROOT/'models/critical_review/matched_state_psd_seed19_all_probabilities.npy')
    dense=np.load(MOD/'dense_state_psd_probabilities.npy')
    rh=np.load(MOD/'primary_block_ridge_predictions.npy');rd=np.load(MOD/'dense_block_ridge_predictions.npy')
    rows=[];geo=[];fitrows=[]
    for n,i in enumerate(ix):
        small=np.load(CACHE/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz')
        large=np.load(ROOT/'data/processed/critical_review'/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz')
        for geometry,positions,rt,rc,rn,h1,h2 in [
            ('wide',np.load(ds.paths[i])['anchor_local'],large['target_anchor'],large['context_anchor'],large['noise_anchor_corrected'],large['target_first_anchor'],large['target_second_anchor']),
            ('small',small['positions'],small['target_matrix'],small['context_matrix'],small['noise_matrix'],small['first_matrix'],small['second_matrix'])]:
            d=abs(positions[:,None]-positions[None,:]);off=~np.eye(32,dtype=bool)
            geo.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],geometry=geometry,positions=','.join(map(str,positions)),aperture=int(d.max()),ordered_entries=int(off.sum()),historical_supervised=int(np.isin(d[off],LAGS).sum()),dense_supervised=int(np.isin(d[off],DENSE).sum()),beyond89=int((d[off]>89).sum()),beyond_period=bool(d.max()>257)))
            for b,f in np.ndindex(8,4):
                D=np.diag(np.maximum(rc[b,f].diagonal().real,1e-15));scale=np.sqrt(D.diagonal()[:,None]*D.diagonal()[None,:])
                kernels={'historical recurrence ensemble':upper_kernel_matrix(hist[i,b,f],positions),'historical recurrence seed19':upper_kernel_matrix(seed[i,b,f],positions)}
                if geometry=='small':
                    kernels['dense31 recurrence seed19']=upper_kernel_matrix(dense[i,b,f],positions)
                    for name,values,lags in [('historical block ridge + simplex',rh[i,b,f],LAGS),('dense31 block ridge + simplex',rd[i,b,f],DENSE)]:
                        prob,_,_,_,rec=fit_grid(complex_values(values),lags);kernels[name]=upper_kernel_matrix(prob,positions)
                        fitrows.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],block=b,band=f,model=name,**rec))
                methods={'Diagonal':D,'Classical shrinkage':.25*rc[b,f]+.75*D}
                methods.update({name:.25*ker*scale+.75*D for name,ker in kernels.items()})
                target_norm=rt[b,f]/np.sqrt(rt[b,f].diagonal().real[:,None]*rt[b,f].diagonal().real[None,:])
                for mode,target,oracle in [('full',rt[b,f],rt[b,f]),('half0',h1[b,f],h2[b,f]),('half1',h2[b,f],h1[b,f])]:
                    local=dict(methods);local['Target same estimate' if mode=='full' else 'Target opposite half']=oracle
                    for name,cov in local.items():
                        w=_generalized_weight(cov,rn[b,f],.0001)
                        err=float(np.mean(abs(kernels[name][off]-target_norm[off])**2)) if name in kernels else None
                        rows.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],block=b,band=f,geometry=geometry,evaluation=mode,model=name,ratio_db=float(10*np.log10(_ratio(target,rn[b,f],w))),covariance_entry_mse=err,minimum_eigenvalue=float(np.linalg.eigvalsh(cov).min()),condition=float(np.linalg.cond(cov))))
        print('processing',n+1,'/',len(ix),flush=True)
    frame=pd.DataFrame(rows);frame.to_parquet(OUT/'geometry_processing.parquet',index=False);pd.DataFrame(geo).to_csv(OUT/'geometry_inventory.csv',index=False);pd.DataFrame(fitrows).to_csv(OUT/'ridge_projection_diagnostics.csv',index=False)
    meta=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet').set_index('event_id').sensitivity_component_25km.to_dict();summary=[]
    frame['endpoint']=np.where(frame.evaluation=='full','full','opposite_half')
    for (geometry,endpoint),g in frame.groupby(['geometry','endpoint']):
        e=g.groupby(['event_id','model']).ratio_db.mean().unstack()
        for ref in ['Diagonal','Classical shrinkage','historical recurrence seed19']:
            for model in e:
                v=e[model]-e[ref];summary.append(dict(geometry=geometry,endpoint=endpoint,model=model,reference=ref,**boot(v,[meta[str(e)] for e in v.index])))
    pd.DataFrame(summary).to_csv(OUT/'geometry_processing_summary.csv',index=False)
    frame.groupby(['geometry','model']).agg(entry_mse=('covariance_entry_mse','mean'),min_eigenvalue=('minimum_eigenvalue','min'),max_condition=('condition','max')).to_csv(OUT/'covariance_diagnostics.csv')

def split_bounds():
    """Retrospective first-to-second-half diagnostic; no population-coverage claim."""
    _,ds,roles,_=bundle();rows=[]
    for i in np.flatnonzero(roles=='architecture_test'):
        with np.load(CACHE/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz') as z:
            for design,lags in [('historical',LAGS),('same_budget',np.array(CFG['identifiability']['same_budget_lags']))]:
                ids=[list(MEASURE).index(d) for d in lags]
                for band in range(4):
                    y=z['first_gamma'][:,band,ids].mean(0)
                    for r in bounds(y,lags,unseen=[4,10]):
                        t=float(z['second_gamma'][:,band,list(MEASURE).index(r['lag'])].real.mean())
                        rows.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],band=band,design=design,comparison_estimate=t,in_range=bool(r['valid'] and r['lower']<=t<=r['upper']),sign_disagreement=bool(r['identified'] and np.sign(t)!=r['sign']),**r))
    table=pd.DataFrame(rows);table.to_csv(OUT/'split_window_bounds.csv',index=False)
    table.groupby('design').agg(cases=('lag','size'),identified=('identified','sum'),in_range=('in_range','sum'),sign_disagreement=('sign_disagreement','sum'),width=('width','mean')).to_csv(OUT/'split_window_bounds_summary.csv')
    full=pd.read_parquet(OUT/'measurement_design_bounds.parquet')
    use=full[(full.grid==257)&np.isclose(full.offset,.01)&full.lag.isin([4,10])]
    meta=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet').set_index('event_id').sensitivity_component_25km.to_dict()
    summary=[]
    for quantity in ['identified','width']:
        e=use.groupby(['event_id','design'])[quantity].mean().unstack()
        for design in ['same_budget','augmented']:
            v=e[design]-e.historical;summary.append(dict(quantity=quantity,design=design,reference='historical',**boot(v,[meta[str(e)] for e in v.index])))
    pd.DataFrame(summary).to_csv(OUT/'paired_design_effects.csv',index=False)

def verify():
    import torch, scipy, sklearn, subprocess, sys
    _,ds,roles,_=bundle();rows=[]
    for i in range(len(ds.event_ids)):
        p=CACHE/f'{ds.event_ids[i]}_{ds.routes[i].replace("-","")}.npz'
        if p.exists():
            with np.load(p) as z:
                ids=[list(MEASURE).index(d) for d in LAGS]
                diff=float(np.max(abs(z['target_gamma'][...,ids]-complex_values(ds.targets[i]))))
                rows.append(dict(check='raw historical-lag recovery',event_id=ds.event_ids[i],route=ds.routes[i],difference=diff,passed=diff<1e-7))
    assert len(rows)==282 and all(r['passed'] for r in rows)
    dense=np.load(MOD/'dense_targets.npy');pred=np.load(MOD/'dense_state_psd_predictions.npy')
    model=MatchedModel(137,'state_psd');model.basis_re=torch.tensor(np.exp(1j*DENSE[:,None]*Q).real,dtype=torch.float32);model.basis_im=torch.tensor(np.exp(1j*DENSE[:,None]*Q).imag,dtype=torch.float32)
    chk=torch.load(MOD/'dense_state_psd_seed19.pt',map_location='cpu',weights_only=True);model.load_state_dict(chk['state']);model.eval()
    regenerated,_=predict_model(model,ds.features,chk['mean'].numpy(),chk['std'].numpy())
    delta=float(np.max(abs(regenerated-pred)));assert delta<2e-5
    rows.append(dict(check='dense checkpoint CPU regeneration',difference=delta,passed=True))
    saved=pd.read_csv(OUT/'dense_prediction_scores.csv').query('model=="state_psd"').set_index(['event_id','route'])
    for i in np.flatnonzero(roles=='architecture_test'):
        v=metrics(pred[i],dense[i])['nrmse'];prior=float(saved.loc[(int(ds.event_ids[i]),ds.routes[i]),'nrmse'])
        rows.append(dict(check='dense saved prediction score',event_id=ds.event_ids[i],route=ds.routes[i],difference=abs(v-prior),passed=abs(v-prior)<1e-10))
    table=pd.read_parquet(OUT/'measurement_design_bounds.parquet')
    assert table.valid.all() and table.shape[0]==5376
    assert table.low_gap.max()<1e-6 and table.high_gap.max()<1e-6
    p=json.loads((OUT/'starting_point.json').read_text());assert p['manifest_sha256']==sha256(OUT/'event_roles.csv') and p['spec_sha256']==sha256(SPEC)
    pd.DataFrame(rows).to_csv(OUT/'numerical_consistency_audit.csv',index=False)
    proc=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (OUT/'test_log.txt').write_text(proc.stdout);assert proc.returncode==0,proc.stdout
    write_json(OUT/'numerical_verification.json',dict(checks=len(rows),all_passed=all(r['passed'] for r in rows),validated_bound_cases=len(table),maximum_primal_violation=float(max(table.low_primal_violation.max(),table.high_primal_violation.max())),maximum_dual_gap=float(max(table.low_gap.max(),table.high_gap.max())),test_log=proc.stdout.strip(),scipy=scipy.__version__,torch=torch.__version__,sklearn=sklearn.__version__,public_write_performed=False))
    print('Final numerical verification passed',len(rows),'checks',flush=True)
