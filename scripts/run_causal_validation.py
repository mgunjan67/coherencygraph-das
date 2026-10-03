"""Resumable public-only replication, with explicit development/validation gates."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from coherencygraph_das.causal_validation import (
    sha256, source_components, candidate_order, extract_file, DENSE, SPARSE, Q,
    storage_guard, metadata, upper_kernel, hypocentral_distances,
)
OUT = ROOT/'revisions/20260922_unseen_validation'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str)+'\n', encoding='utf-8')


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT/'cohort_lock.json'
    if dest.exists():
        verify_lock()
        print('Existing locked cohort verified; not reselecting.', flush=True)
        return
    exposed = pd.read_parquet(ROOT/'data/provenance/paired_event_manifest.parquet')
    exposed['event_id'] = exposed.event_id.astype(str)
    # Cross-project exposure: original wavefield cohorts must also be excluded.
    other = ROOT.parent/'das-cook-inlet-wavefield-engineering/data/processed'
    sources = [other/'manifests/pilot_cohort.parquet', other/'publication/confirmation_manifest.parquet']
    for source in sources:
        frame = pd.read_parquet(source)
        if not set(frame.event_id.astype(str)) <= set(exposed.event_id):
            raise RuntimeError(f'Additional exposed IDs found in {source}; extend exclusion before locking')
    catalog = pd.read_parquet(OUT/'availability/live_catalog.parquet')
    pool, ordered = candidate_order(catalog, exposed)
    if len(ordered) < 30:
        raise RuntimeError('Fewer than 30 source-buffered candidates under the protocol')
    pool.to_parquet(OUT/'eligible_candidates.parquet', index=False)
    ordered.to_csv(OUT/'locked_candidate_order.csv', index=False)
    exposed.to_csv(OUT/'excluded_previously_used_events.csv', index=False)
    roles = pd.read_csv(ROOT/'reports/submission_revision/unchanged_event_roles.csv', dtype={'event_id': str})
    ids = set(roles.loc[roles.role.eq('model_development'), 'event_id'])
    dev = exposed[exposed.event_id.isin(ids)].sort_values('event_id').copy()
    if len(dev) != 93:
        raise RuntimeError('Development set differs from the original 93 events')
    dev['source_component_25km'] = source_components(dev, 25)
    if dev.source_component_25km.nunique() < 4:
        raise RuntimeError('Fewer than four development source components; declared grouped CV unavailable')
    rows = []
    for row in dev.to_dict('records'):
        for route, prefix in [('TERRA','terra'), ('KKFL-S','kkfls')]:
            path = Path(row[prefix+'_path'])
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(dict(event_id=row['event_id'], route=route, raw_path=str(path),
                             component=row['source_component_25km']))
    pd.DataFrame(rows).to_csv(OUT/'development_records.csv', index=False)
    files = ['PROTOCOL.md', 'locked_candidate_order.csv', 'excluded_previously_used_events.csv',
             'development_records.csv', 'availability/live_catalog.parquet']
    lock = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), eligible=len(pool),
                primary=min(30,len(ordered)), reserves=max(0,len(ordered)-30),
                development_events=93, development_components=int(dev.source_component_25km.nunique()),
                new_waveform_outcomes_inspected=False,
                hashes={p: sha256(OUT/p) for p in files})
    save_json(dest, lock)
    print(json.dumps(lock, indent=2), flush=True)


def verify_lock():
    lock = json.loads((OUT/'cohort_lock.json').read_text())
    for path, expected in lock['hashes'].items():
        if sha256(OUT/path) != expected:
            raise RuntimeError(f'Frozen cohort/protocol changed: {path}')
    return lock


def extract_worker(row, folder):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        path = folder/f"{row['event_id']}_{row['route'].replace('-','')}.npz"
        return dict(event_id=row['event_id'], route=row['route'], **extract_file(row['raw_path'],row['route'],path))


def extract_development(workers):
    verify_lock()
    table = pd.read_csv(OUT/'development_records.csv', dtype={'event_id':str})
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs = [pool.submit(extract_worker, row, OUT/'development/cache') for row in table.to_dict('records')]
        for n, future in enumerate(as_completed(fs), 1):
            results.append(future.result())
            if n % 4 == 0 or n == len(fs):
                print('Development extraction',n,'/',len(fs),flush=True)
                pd.DataFrame(results).to_csv(OUT/'development/extraction_receipt.csv',index=False)
    pd.DataFrame(results).to_csv(OUT/'development/extraction_receipt.csv',index=False)


def load_development():
    table = pd.read_csv(OUT/'development_records.csv', dtype={'event_id':str})
    x=[];y=[];ids=[];groups=[]
    for row in table.itertuples():
        with np.load(OUT/f'development/cache/{row.event_id}_{row.route.replace("-","")}.npz') as z:
            x.extend(z['features']); y.extend(z['targets'])
            ids.extend([row.event_id]*4); groups.extend([row.component]*4)
    return np.array(x),np.array(y),np.array(ids),np.array(groups)


def scale(x, indexes):
    mean=x[indexes].mean((0,1),keepdims=True)
    std=x[indexes].std((0,1),keepdims=True)
    std=np.where(std<1e-5,1,std)
    return ((x-mean)/std).astype('float32'),mean,std


def event_mse(p,y,ids):
    error=(p-y)**2
    if error.shape[-1]!=2:raise ValueError('Expected real/imaginary last dimension')
    error=error.sum(-1)
    per_row=np.mean(error,axis=tuple(range(1,error.ndim)))
    return float(pd.Series(per_row).groupby(ids).mean().mean())


def train():
    verify_lock()
    import torch
    from torch import nn
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from coherencygraph_das.causal_validation_model import CausalSpectralModel
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x,complex_y,ids,groups=load_development(); indexes=np.arange(len(x))
    folder=OUT/'development/models';folder.mkdir(parents=True,exist_ok=True)
    print('Training on',len(set(ids)),'development events; device',device,flush=True)
    all_receipts=[]
    for tag,lags in [('sparse',SPARSE),('dense',DENSE)]:
        cy=complex_y[...,lags-1]
        y=np.stack([cy.real,cy.imag],-1).astype('float32')
        ridge_path=folder/f'{tag}_full_ridge_f64.npz'
        if not ridge_path.exists():
            scores=[]
            for alpha in [.1,1,10,100,1000]:
                fold_scores=[];fold_sizes=[]
                for tr,va in GroupKFold(4).split(indexes,groups=groups):
                    z,_,_=scale(x,tr)
                    fit=Ridge(alpha=alpha,solver='svd').fit(z[tr].reshape(len(tr),-1).astype('float64'),y[tr].reshape(len(tr),-1).astype('float64'))
                    p=fit.predict(z[va].reshape(len(va),-1).astype('float64')).reshape(y[va].shape)
                    fold_scores.append(event_mse(p,y[va],ids[va]))
                    fold_sizes.append(len(set(ids[va])))
                scores.append(dict(alpha=alpha,mean_fold_event_mse=float(np.average(fold_scores,weights=fold_sizes))))
            alpha=min(scores,key=lambda r:r['mean_fold_event_mse'])['alpha']
            z,mean,std=scale(x,indexes)
            fit=Ridge(alpha=alpha,solver='svd').fit(z.reshape(len(z),-1).astype('float64'),y.reshape(len(y),-1).astype('float64'))
            np.savez_compressed(ridge_path,coef=fit.coef_,intercept=fit.intercept_,mean=mean,std=std,lags=lags,alpha=alpha)
            save_json(ridge_path.with_suffix('.json'),dict(tag=tag,alpha=alpha,grid=scores,training_events=sorted(set(ids)),
                       checkpoint_sha256=sha256(ridge_path),solver='float64 SVD',
                       numerical_note='Float32 normal-equation preflight showed ill-conditioning; fixed before new validation scoring.'))
            print('Ridge',tag,'alpha',alpha,flush=True)
        for seed in [19,43,71]:
            path=folder/f'{tag}_recurrence_seed{seed}.pt'
            if path.exists():
                if sha256(path) != json.loads(path.with_suffix('.json').read_text())['checkpoint_sha256']:
                    raise RuntimeError('Existing checkpoint hash mismatch')
                continue
            unique=sorted(set(groups),key=lambda g:hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
            held=set(unique[:max(3,int(np.ceil(.2*len(unique))))])
            va=indexes[np.isin(groups,list(held))];tr=indexes[~np.isin(groups,list(held))]
            if not len(tr) or not len(va):
                raise RuntimeError('Empty development source-group training/validation split')
            normalized,_,_=scale(x,tr)
            values=torch.tensor(normalized,device=device);targets=torch.tensor(y,device=device)
            def create():
                torch.manual_seed(seed);np.random.seed(seed)
                return CausalSpectralModel(lags).to(device)
            def one_epoch(model,opt,ix,rng):
                model.train();events=np.unique(ids[ix]);rng.shuffle(events);losses=[]
                for start in range(0,len(events),12):
                    batch=ix[np.isin(ids[ix],events[start:start+12])]
                    opt.zero_grad(set_to_none=True)
                    pred,_=model(values[batch]);loss=((pred-targets[batch])**2).sum(-1).mean()
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    losses.append(float(loss.detach()))
                return float(np.mean(losses))
            model=create();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed);best=float('inf');epoch_best=1;stale=0;history=[]
            for epoch in range(80):
                loss=one_epoch(model,opt,tr,rng);model.eval()
                with torch.no_grad(): p,_=model(values[va])
                score=event_mse(p.cpu().numpy(),y[va],ids[va])
                history.append(dict(epoch=epoch+1,training_loss=loss,validation_event_mse=score))
                if score < best-1e-8:
                    best=score;epoch_best=epoch+1;stale=0
                else:stale+=1
                if stale>=12:break
            z,mean,std=scale(x,indexes);values=torch.tensor(z,device=device)
            model=create();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed)
            refit=[one_epoch(model,opt,indexes,rng) for _ in range(epoch_best)]
            torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},
                            mean=torch.tensor(mean),std=torch.tensor(std),lags=torch.tensor(lags)),path)
            save_json(path.with_suffix('.json'),dict(tag=tag,seed=seed,selected_epoch=epoch_best,
                validation_event_mse=best,history=history,refit_loss=refit,
                training_events=sorted(set(ids)),internal_validation_events=sorted(set(ids[va])),
                checkpoint_sha256=sha256(path),device=str(device)))
            print('Recurrence',tag,'seed',seed,'selected epoch',epoch_best,flush=True)
    print('Development fitting complete. Fresh validation has not been scored.',flush=True)


def download():
    """Download locked candidates only. Never computes features or target scores."""
    verify_lock()
    import requests
    import h5py
    import threading
    from concurrent.futures import ThreadPoolExecutor
    candidates=pd.read_csv(OUT/'locked_candidate_order.csv',dtype={'event_id':str})
    folder=OUT/'raw';folder.mkdir(exist_ok=True)
    receipt_path=OUT/'download_receipts.json'
    receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    accepted=[]
    allocation_lock=threading.Lock()
    reserved={}

    def fetch(row,route,prefix):
        path=folder/row.event_id/(prefix.upper()+'.h5')
        url=getattr(row,prefix+'_url')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            with h5py.File(path,'r') as h:info=metadata(h)
            digest=sha256(path)
            prior=[r for r in receipts if r.get('raw_path')==str(path) and r.get('sha256')]
            if prior and prior[-1]['sha256']!=digest:
                raise RuntimeError('Downloaded raw checksum changed')
            return dict(event_id=row.event_id,route=route,raw_path=str(path),url=url,
                        bytes=path.stat().st_size,sha256=digest,status='downloaded',**info)
        last=None
        for attempt in range(5):
            try:
                head=requests.head(url,timeout=40,allow_redirects=True)
                if head.status_code in (403,404):
                    return dict(event_id=row.event_id,route=route,url=url,status=f'http_{head.status_code}')
                if head.status_code in (429,503):
                    wait=min(60,max(20*(attempt+1),float(head.headers.get('Retry-After','0')) if head.headers.get('Retry-After','0').isdigit() else 0))
                    print('Archive backoff',int(wait),'seconds for HTTP',head.status_code,flush=True)
                    time.sleep(wait)
                    continue
                head.raise_for_status()
                length=int(head.headers.get('Content-Length',0))
                if length<=0:raise RuntimeError('Missing Content-Length; bounded download refused')
                with allocation_lock:
                    used=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
                    # Atomic, conservative allocation accounts for all in-flight files.
                    storage_guard(folder,length+sum(reserved.values()),used)
                    reserved[str(path)]=length
                part=path.with_suffix('.h5.part');offset=part.stat().st_size if part.exists() else 0
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with requests.get(url,headers=headers,stream=True,timeout=(40,90)) as response:
                    if response.status_code in (403,404):
                        return dict(event_id=row.event_id,route=route,url=url,status=f'http_{response.status_code}')
                    if response.status_code in (429,503):
                        wait=min(60,20*(attempt+1))
                        print('Archive backoff',wait,'seconds for HTTP',response.status_code,flush=True)
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    if offset and response.status_code==206:
                        if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise RuntimeError('Invalid resume Content-Range')
                        mode='ab'
                    else:mode='wb'
                    with part.open(mode) as stream:
                        for chunk in response.iter_content(1024*1024):
                            if chunk:stream.write(chunk)
                if part.stat().st_size!=length:raise RuntimeError('Waveform size mismatch')
                with h5py.File(part,'r') as h:info=metadata(h)
                part.replace(path)
                return dict(event_id=row.event_id,route=route,raw_path=str(path),url=url,
                            bytes=length,sha256=sha256(path),status='downloaded',**info)
            except (requests.RequestException,OSError,ValueError) as exc:
                last=str(exc)
                time.sleep(min(60,10*(attempt+1)))
            finally:
                with allocation_lock:reserved.pop(str(path),None)
        return dict(event_id=row.event_id,route=route,url=url,status='temporarily_failed',error=last)

    # Transfer concurrency cannot change rank-based acceptance or reserve order.
    cursor=0
    while len(accepted)//2<30 and cursor<len(candidates):
        wanted=30-len(accepted)//2
        batch=candidates.iloc[cursor:cursor+wanted]
        cursor+=len(batch)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(fetch,row,route,prefix) for row in batch.itertuples()
                for route,prefix in [('TERRA','terra'),('KKFL-S','kkfls')]]
            for future in as_completed(fs):
                result=future.result()
                receipts=[r for r in receipts if (str(r['event_id']),r['route'])!=(result['event_id'],result['route'])]+[result]
                save_json(receipt_path,receipts)
                print('Raw file',result['event_id'],result['route'],result['status'],flush=True)
        if any(r['status']=='temporarily_failed' and str(r['event_id']) in set(batch.event_id) for r in receipts):
            raise RuntimeError('Transient archive failures: retain cohort and partials; resume download later, do not substitute reserves')
        for row in batch.itertuples():
            pair=[r for r in receipts if str(r['event_id'])==row.event_id]
            if len(pair)==2 and all(r['status']=='downloaded' for r in pair):accepted.extend(pair)
        print('Complete paired candidates',len(accepted)//2,'/ 30',flush=True)
    pd.DataFrame(accepted).to_csv(OUT/'validation_records.csv',index=False)
    print('Download stage complete:',len(accepted)//2,'paired events. No outcome analysis performed.',flush=True)


def seal():
    verify_lock()
    import xml.etree.ElementTree as ET
    tests=OUT/'causal_test_results.xml'
    suite=ET.parse(tests).getroot()[0]
    if int(suite.attrib['failures']) or int(suite.attrib['errors']):
        raise RuntimeError('Causal test gate did not pass')
    models=OUT/'development/models'
    expected=[models/f'{tag}_full_ridge_f64.npz' for tag in ['sparse','dense']]
    expected += [models/f'{tag}_recurrence_seed{seed}.pt' for tag in ['sparse','dense'] for seed in [19,43,71]]
    development=set(pd.read_csv(OUT/'development_records.csv',dtype={'event_id':str}).event_id)
    for path in expected:
        receipt=json.loads(path.with_suffix('.json').read_text())
        if sha256(path)!=receipt['checkpoint_sha256'] or set(receipt['training_events'])!=development:
            raise RuntimeError(f'Model development provenance failed: {path}')
    source=[Path(__file__),ROOT/'src/coherencygraph_das/causal_validation.py',
            ROOT/'src/coherencygraph_das/causal_validation_model.py',ROOT/'src/coherencygraph_das/models.py',
            ROOT/'src/coherencygraph_das/spectral.py',ROOT/'src/coherencygraph_das/covariance_numerics.py',
            ROOT/'src/coherencygraph_das/processor_audit.py',ROOT/'tests/test_causal_validation.py']
    inputs=sorted((OUT/'development/cache').glob('*.npz'))
    if len(inputs)!=186:raise RuntimeError('Incomplete development extraction')
    environment=OUT/'environment_lock.txt'
    if not environment.exists():
        import subprocess
        environment.write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    direct=OUT/'development/direct_array_verification.json'
    if not json.loads(direct.read_text())['passed']:
        raise RuntimeError('Independent direct-array extraction check failed')
    items=expected+source+inputs+[OUT/'cohort_lock.json',tests,environment,direct]
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in items}
    dest=OUT/'pre_outcome_seal.json'
    if dest.exists():
        if json.loads(dest.read_text())['hashes']!=hashes:
            raise RuntimeError('Pre-outcome seal differs: do not silently change models or code')
    else:
        save_json(dest,dict(sealed_utc=pd.Timestamp.now(tz='UTC').isoformat(),hashes=hashes,
                           model_count=len(expected),development_records=len(inputs),
                           validation_scores_inspected=False))
    print('Pre-outcome seal verified; eight model assets and all development inputs frozen.',flush=True)


def verify_seal():
    verify_lock()
    value=json.loads((OUT/'pre_outcome_seal.json').read_text())
    for path,expected in value['hashes'].items():
        if sha256(ROOT/path)!=expected:raise RuntimeError(f'Pre-outcome seal changed: {path}')


def extract_validation(workers):
    verify_seal()
    table=pd.read_csv(OUT/'validation_records.csv',dtype={'event_id':str})
    if table.event_id.nunique()<24:raise RuntimeError('Fewer than 24 acquired paired events')
    results=[];errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(extract_worker,row,OUT/'validation/cache'):row for row in table.to_dict('records')}
        for n,future in enumerate(as_completed(fs),1):
            try:results.append(future.result())
            except Exception as exc:
                row=fs[future]
                errors.append(dict(event_id=row['event_id'],route=row['route'],error=str(exc)))
            if n%4==0 or n==len(fs):print('Validation extraction',n,'/',len(fs),flush=True)
    save_json(OUT/'validation/extraction_failures.json',errors)
    pd.DataFrame(results).to_csv(OUT/'validation/extraction_receipt.csv',index=False)
    if errors:
        raise RuntimeError('Objective extraction QC failures recorded. Resolve using frozen reserves before any scores.')
    print('Validation extraction complete. No prediction/processing scores computed yet.',flush=True)


def predict_validation():
    verify_seal()
    import torch
    from coherencygraph_das.causal_validation_model import CausalSpectralModel
    torch.set_num_threads(4)
    table=pd.read_csv(OUT/'validation_records.csv',dtype={'event_id':str}).sort_values(['event_id','route'])
    features=[];records=[]
    for row in table.itertuples():
        with np.load(OUT/f'validation/cache/{row.event_id}_{row.route.replace("-","")}.npz') as z:
            features.extend(z['features'])
            records.extend([dict(event_id=row.event_id,route=row.route,cutoff_index=i) for i in range(4)])
    x=np.array(features)
    folder=OUT/'validation/predictions';folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(folder/'rows.csv',index=False)
    for tag,lags in [('sparse',SPARSE),('dense',DENSE)]:
        with np.load(OUT/f'development/models/{tag}_full_ridge_f64.npz') as m:
            values=((x-m['mean'])/m['std']).reshape(len(x),-1)
            p=(values@m['coef'].T+m['intercept']).reshape(len(x),8,4,len(lags),2)
        np.save(folder/f'{tag}_full_ridge.npy',p[...,0]+1j*p[...,1])
        for seed in [19,43,71]:
            saved=torch.load(OUT/f'development/models/{tag}_recurrence_seed{seed}.pt',weights_only=True,map_location='cpu')
            model=CausalSpectralModel(lags).eval();model.load_state_dict(saved['state'])
            values=((x-saved['mean'].numpy())/saved['std'].numpy()).astype('float32')
            probabilities=[]
            with torch.no_grad():
                for start in range(0,len(x),48):
                    _,q=model(torch.tensor(values[start:start+48]))
                    probabilities.append(q.numpy())
            np.save(folder/f'{tag}_recurrence_seed{seed}.npy',np.concatenate(probabilities))
    print('Locked models applied to',len(x),'event-route-cutoff samples.',flush=True)


def score_worker(row):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):return score_record(row)


def score_record(row):
    from coherencygraph_das.covariance_numerics import geometry_fit,_generalized_weight
    from coherencygraph_das.processor_audit import correlation,remove_loading,moments
    dest=OUT/f'validation/case_records/{row["event_id"]}_{row["route"].replace("-","")}.parquet'
    if dest.exists():return str(dest)
    folder=OUT/'validation/predictions'
    index=pd.read_csv(folder/'rows.csv',dtype={'event_id':str})
    selected=np.flatnonzero(index.event_id.eq(row['event_id']) & index.route.eq(row['route']))
    if len(selected)!=4:raise RuntimeError('Prediction-row alignment failed')
    predictions={f'{tag}_recurrence_seed{seed}':np.load(folder/f'{tag}_recurrence_seed{seed}.npy',mmap_mode='r')[selected]
                 for tag in ['sparse','dense'] for seed in [19,43,71]}
    ridges={tag:np.load(folder/f'{tag}_full_ridge.npy',mmap_mode='r')[selected] for tag in ['sparse','dense']}
    with np.load(OUT/f'validation/cache/{row["event_id"]}_{row["route"].replace("-","")}.npz') as z:
        rc=z['context'];rn=z['reference'];rt=z['target'];first=z['first'];second=z['second']
        truth=z['targets']
    results=[]
    for ci,block,band in np.ndindex(4,8,4):
        context=rc[ci,block,band];reference=rn[ci,block,band];target=rt[ci,block,band]
        D=np.diag(np.maximum(context.diagonal().real,1e-15))
        scale=np.sqrt(D.diagonal()[:,None]*D.diagonal()[None,:])
        kernels={name:upper_kernel(p[ci,block,band]) for name,p in predictions.items()}
        projections={}
        for tag,lags in [('sparse',SPARSE),('dense',DENSE)]:
            p,_,_,_,fit=geometry_fit(ridges[tag][ci,block,band],lags)
            if not fit['fit_converged']:raise RuntimeError('Ridge projection failed numerical gate')
            name=f'{tag}_full_ridge';kernels[name]=upper_kernel(p);projections[name]=fit['fit_gap']
        persistence=moments(correlation(remove_loading(context))[0],DENSE)
        p,_,_,_,fit=geometry_fit(persistence,DENSE)
        if not fit['fit_converged']:raise RuntimeError('Persistence projection failed numerical gate')
        kernels['local_persistence']=upper_kernel(p);projections['local_persistence']=fit['fit_gap']
        matrices={name:.25*kernel*scale+.75*D for name,kernel in kernels.items()}
        matrices.update(diagonal=D,fixed_shrinkage=.25*context+.75*D)
        off=~np.eye(32,dtype=bool)
        for name,matrix in matrices.items():
            weight=_generalized_weight(matrix,reference,.0001)
            denom=float((weight.conj()@reference@weight).real)
            kernel=kernels.get(name)
            for endpoint,tgt in [('full',target),('half0',first[ci,block,band]),('half1',second[ci,block,band])]:
                numerator=float((weight.conj()@tgt@weight).real)
                if min(numerator,denom)<=0:raise RuntimeError('Nonpositive power; report invalid, do not delete cases')
                observed=correlation(remove_loading(tgt))[0]
                lag_truth=moments(observed,DENSE)
                results.append(dict(event_id=row['event_id'],route=row['route'],cutoff=[20,40,60,80][ci],block=block,band=band,
                    method=name,endpoint=endpoint,ratio_db=float(10*np.log10(numerator/denom)),
                    covariance_entry_mse=float(np.mean(abs(kernel[off]-observed[off])**2)) if kernel is not None else np.nan,
                    complex_lag_mse=float(np.mean(abs(moments(kernel,DENSE)-lag_truth)**2)) if kernel is not None else np.nan,
                    projection_gap=projections.get(name,np.nan)))
    dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_parquet(dest,index=False)
    return str(dest)


def score(workers):
    verify_seal()
    table=pd.read_csv(OUT/'validation_records.csv',dtype={'event_id':str})
    prediction_seal=OUT/'validation/prediction_hashes.json'
    files=sorted((OUT/'validation/predictions').glob('*'))
    current={p.name:sha256(p) for p in files if p.is_file()}
    if prediction_seal.exists() and json.loads(prediction_seal.read_text())!=current:
        raise RuntimeError('Prediction arrays changed after scoring started')
    save_json(prediction_seal,current)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(score_worker,row) for row in table.to_dict('records')]
        for n,future in enumerate(as_completed(fs),1):
            future.result();print('Scored validation records',n,'/',len(fs),flush=True)
    summarize()


def summarize():
    verify_seal()
    from coherencygraph_das.covariance_numerics import boot
    files=sorted((OUT/'validation/case_records').glob('*.parquet'))
    data=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    expected=pd.read_csv(OUT/'validation_records.csv',dtype={'event_id':str})
    expected_pairs=set(zip(expected.event_id,expected.route))
    actual_pairs=set(zip(data.event_id,data.route))
    if expected_pairs!=actual_pairs or len(data)!=len(expected_pairs)*4*8*4*11*3:
        raise RuntimeError('Incomplete or duplicate validation cases; no scientific summary permitted')
    if data.duplicated(['event_id','route','cutoff','block','band','method','endpoint']).any():
        raise RuntimeError('Duplicate nested case')
    if not np.isfinite(data.ratio_db).all():raise RuntimeError('Nonfinite power scores')
    data['model']=data.method.str.replace(r'_seed\d+$','',regex=True)
    # Seed averaging occurs within each case, not in the resampling units.
    cases=data.groupby(['event_id','route','cutoff','block','band','endpoint','model'],as_index=False)[['ratio_db','covariance_entry_mse','complex_lag_mse']].mean()
    cases.to_parquet(OUT/'validation/cases.parquet',index=False)
    for stratum in ['band','cutoff','route']:
        cases.groupby(['event_id',stratum,'endpoint','model'],as_index=False)[
            ['ratio_db','covariance_entry_mse','complex_lag_mse']].mean().to_csv(
                OUT/f'validation/{stratum}_event_scores.csv',index=False)
    events=cases.groupby(['event_id','endpoint','model'],as_index=False)[['ratio_db','covariance_entry_mse','complex_lag_mse']].mean()
    events.to_csv(OUT/'validation/event_scores.csv',index=False)
    candidate=pd.read_csv(OUT/'locked_candidate_order.csv',dtype={'event_id':str}).set_index('event_id')
    used=candidate.loc[sorted(events.event_id.unique())].reset_index()
    components=dict(zip(used.event_id,source_components(used,50)))
    comparisons=[]
    for endpoint,g in events.groupby('endpoint'):
        wide=g.pivot(index='event_id',columns='model',values='ratio_db')
        for model,reference in [('dense_recurrence','sparse_recurrence'),('dense_recurrence','dense_full_ridge'),
                                ('dense_recurrence','fixed_shrinkage')]+[(m,'diagonal') for m in wide.columns if m!='diagonal']:
            v=wide[model]-wide[reference]
            for resampling in ['earthquake','source_50km']:
                groups=None if resampling=='earthquake' else [components[e] for e in v.index]
                comparisons.append(dict(endpoint=endpoint,model=model,reference=reference,resampling=resampling,
                                        **boot(v,groups,replicates=5000,seed=20260922)))
    report=pd.DataFrame(comparisons);report.to_csv(OUT/'validation/paired_comparisons.csv',index=False)
    print(report.query('endpoint=="full" and resampling=="earthquake"')[['model','reference','mean','low','high']].to_string(index=False),flush=True)
    primary=report.query('endpoint=="full" and resampling=="earthquake" and model=="dense_recurrence" and reference=="sparse_recurrence"').iloc[0]
    benchmarks=report.query('endpoint=="full" and resampling=="earthquake" and model=="dense_recurrence" and reference in ["dense_full_ridge","fixed_shrinkage"]')
    save_json(OUT/'validation/decision.json',dict(events=int(used.event_id.nunique()),
        source_components_50km=int(len(set(components.values()))),primary_support_replication=bool(len(used)>=24 and primary.low>0),
        ai_superiority_over_both_baselines=bool(len(benchmarks)==2 and (benchmarks.low>0).all()),
        submission_ready=False,note='Separate fixed-window task, not operational earthquake prediction.'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','extract-development','train','download','seal','extract-validation','predict','score','summarize'])
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='extract-development':extract_development(args.workers)
    elif args.stage=='train':train()
    elif args.stage=='download':download()
    elif args.stage=='seal':seal()
    elif args.stage=='extract-validation':extract_validation(args.workers)
    elif args.stage=='predict':predict_validation()
    elif args.stage=='score':score(args.workers)
    elif args.stage=='summarize':summarize()


if __name__=='__main__':main()
