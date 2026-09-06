"""Geometry-explicit, development-selected Amendment-09 experiments.

Historical test exposure is unchanged. Numerical bounds are conditional ranges,
not confidence or prediction intervals. Cached local matrices are sufficient
statistics of the exact processing channels, not the containing block.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import time
import numpy as np
import pandas as pd
import yaml
from . import final_revision as historical
from .critical_revision import ROOT, bundle, components, complex_values, Q, boot, upper_kernel_matrix
from .critical_experiments import _scale, _score_array, _partition, MatchedModel, predict_model
from .config import write_json, sha256

OUT = ROOT / 'reports/submission_revision'
MOD = ROOT / 'models/submission_revision'
SPEC = ROOT / 'configs/protocol_amendment_09_submission.yaml'
CFG = yaml.safe_load(SPEC.read_text())
for directory in [OUT, MOD]:
    directory.mkdir(parents=True, exist_ok=True)


def cache_path(ds, i):
    return historical.CACHE / f'{ds.event_ids[i]}_{ds.routes[i].replace("-", "")}.npz'


def local_lag_targets(loaded, lags, loading=.001):
    """Invert known trace loading, then reproduce pair-normalized lag moments."""
    loaded = np.asarray(loaded, np.complex128)
    n = loaded.shape[-1]
    level = np.trace(loaded, axis1=-2, axis2=-1).real / n / (1 + loading)
    matrix = loaded - loading * level[..., None, None] * np.eye(n)
    power = matrix.diagonal(axis1=-2, axis2=-1).real
    if np.any(power <= 0):
        raise ValueError('Nonpositive unloaded channel power')
    corr = matrix / np.sqrt(power[..., :, None] * power[..., None, :])
    return np.stack([np.diagonal(corr, offset=int(d), axis1=-2, axis2=-1).mean(-1) for d in lags], -1)


def audit():
    _, ds, roles, groups = bundle()
    manifest = pd.DataFrame(dict(event_id=ds.event_ids, route=ds.routes, role=roles, source_bin=groups))
    original = pd.read_csv(historical.OUT/'event_roles.csv', dtype=str)
    assert manifest[['event_id','route','role']].astype(str).equals(original[['event_id','route','role']])
    manifest.to_csv(OUT/'unchanged_event_roles.csv', index=False)
    frozen = OUT/'protocol.sha256'
    digest = sha256(SPEC)
    if frozen.exists():
        assert frozen.read_text().strip() == digest, 'Protocol changed after analysis freeze'
    else:
        frozen.write_text(digest + '\n')
    times = pd.read_parquet(ROOT/'reports/critical_review/raw_timing.parquet')
    # Actual round-to-sample end of all context inputs, plus shared noise end.
    times['safe_context_stop_exclusive'] = np.rint(times.global_cutoff * 25) / 25
    times['safe_latest_waveform_input'] = np.maximum(times.safe_context_stop_exclusive, np.rint(5.5*25)/25)
    latest = times.groupby(['event_id','route']).safe_latest_waveform_input.transform('max')
    times['audited_waveform_lead'] = times.target_first - latest
    assert (times.audited_waveform_lead > 0).all()
    times.to_parquet(OUT/'waveform_availability.parquet', index=False)
    write_json(OUT/'starting_audit.json', dict(protocol_sha256=digest,
        prior_numerical_checks=331, role_counts=manifest.groupby('role').event_id.nunique().to_dict(),
        waveform_rows=len(times), minimum_lead_seconds=float(times.audited_waveform_lead.min()),
        pick_availability='unknown, retrospective; not an operational benchmark'))
    print('Protocol frozen; waveform availability verified', len(times), 'blocks', flush=True)


def neural(x, y, ds, train, groups, lags, seed, tag):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    path = MOD/f'{tag}_state_psd_seed{seed}.pt'
    receipt = path.with_suffix('.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def new():
        torch.manual_seed(seed)
        m = MatchedModel(x.shape[-1], 'state_psd')
        basis = np.exp(1j*np.asarray(lags)[:,None]*Q)
        m.basis_re = torch.tensor(basis.real, dtype=torch.float32)
        m.basis_im = torch.tensor(basis.imag, dtype=torch.float32)
        return m.to(device)
    started = time.perf_counter()
    if path.exists() and receipt.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        m = new(); m.load_state_dict(saved['state'])
        mean, std = saved['mean'].numpy(), saved['std'].numpy()
    else:
        tr, va = _partition(ds, train, groups, seed)
        assert not set(ds.event_ids[tr]) & set(ds.event_ids[va])
        normalized, _, _ = _scale(x, tr)
        values = torch.tensor(normalized, device=device)
        target = torch.tensor(y, device=device)
        def epoch(m, opt, ix, rng):
            m.train(); events = np.unique(ds.event_ids[ix]); rng.shuffle(events); losses = []
            for start in range(0, len(events), 12):
                batch = ix[np.isin(ds.event_ids[ix], events[start:start+12])]
                opt.zero_grad(set_to_none=True)
                p, _ = m(values[batch]); loss = ((p-target[batch])**2).mean()
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
                losses.append(float(loss.detach()))
            return float(np.mean(losses))
        m = new(); opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(seed); best = np.inf; best_epoch = 1; stale = 0; history = []
        for e in range(CFG['training']['maximum_epochs']):
            loss = epoch(m, opt, tr, rng); m.eval()
            with torch.no_grad(): p, _ = m(values[va])
            score = float(_score_array(p.cpu().numpy(), y[va]).mean())
            history.append(dict(epoch=e+1, loss=loss, validation_nrmse=score))
            if score < best-1e-5: best=score; best_epoch=e+1; stale=0
            else: stale += 1
            if stale >= CFG['training']['patience']: break
        normalized, mean, std = _scale(x, train); values = torch.tensor(normalized, device=device)
        m = new(); opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(seed)
        refit = [epoch(m, opt, train, rng) for _ in range(best_epoch)]
        torch.save(dict(state={k:v.cpu() for k,v in m.state_dict().items()}, mean=torch.tensor(mean), std=torch.tensor(std)), path)
        write_json(receipt, dict(seed=seed, selected_epoch=best_epoch, history=history,
            refit_losses=refit, parameters=sum(p.numel() for p in m.parameters()),
            training_events=sorted(set(ds.event_ids[train])), internal_validation_events=sorted(set(ds.event_ids[va])),
            lags=list(map(int,lags)), protocol_sha256=sha256(SPEC), checkpoint_sha256=sha256(path),
            seconds=time.perf_counter()-started))
    m.eval(); p, q = predict_model(m, x, mean, std)
    np.save(MOD/f'{tag}_state_psd_seed{seed}_predictions.npy', p)
    np.save(MOD/f'{tag}_state_psd_seed{seed}_probabilities.npy', q)
    print(tag, 'seed', seed, 'complete', round(time.perf_counter()-started,1), 's', flush=True)
    return p


def ridge(x, y, ds, train, groups, kind, tag):
    # Reuse the already tested estimator; only new output paths are redirected.
    previous = historical.MOD
    try:
        historical.MOD = MOD
        return historical.ridge(x, y, ds, train, groups, kind, tag)
    finally:
        historical.MOD = previous


def train():
    _, ds, roles, groups = bundle(); tr = np.flatnonzero(roles == 'model_development')
    safe = ds.features.copy()
    dense = np.load(historical.MOD/'dense_targets.npy')
    local = np.zeros_like(dense)
    valid = np.flatnonzero(np.isin(roles, ['model_development','calibration','architecture_test']))
    for i in valid:
        with np.load(ROOT/'data/processed/critical_review'/cache_path(ds,i).name) as z:
            safe[i] = z['global_cutoff_features']
        with np.load(cache_path(ds,i)) as z:
            assert np.array_equal(z['positions'], np.arange(484,516))
            local[i] = components(local_lag_targets(z['target_matrix'], np.arange(1,32)))
    np.save(MOD/'local_dense_targets.npy', local)
    sparse_lags = np.array(CFG['experiments']['local_sparse_lags'])
    experiments = [('waveform_safe',safe,ds.targets,historical.LAGS),
                   ('block_dense',ds.features,dense,np.arange(1,32)),
                   ('local_sparse',ds.features,local[...,sparse_lags-1,:],sparse_lags),
                   ('local_dense',ds.features,local,np.arange(1,32))]
    frames = []
    for tag, x, y, lags in experiments:
        for kind in ['block_ridge','full_ridge']:
            p = ridge(x,y,ds,tr,groups,kind,tag)
            frames.append(historical.event_table(p,y,ds,roles,kind,tag))
        predictions = []
        for seed in CFG['training']['seeds']:
            p = neural(x,y,ds,tr,groups,lags,seed,tag); predictions.append(p)
            frames.append(historical.event_table(p,y,ds,roles,f'state_psd_seed{seed}',tag))
        ensemble = np.mean(predictions,axis=0)
        np.save(MOD/f'{tag}_state_psd_seed_mean_predictions.npy',ensemble)
        frames.append(historical.event_table(ensemble,y,ds,roles,'state_psd_seed_mean',tag))
        pd.concat(frames).to_csv(OUT/'prediction_event_scores.csv',index=False)
    historical.paired(pd.concat(frames),'block_ridge').to_csv(OUT/'prediction_comparisons.csv',index=False)


def geometry_fit(values,lags):
    """Polish difficult simplex fits with a checked active-set QP fallback.

    Only solver accuracy changes; no data, tolerance or scientific objective is
    selected here. The original projected-gradient fit is always tried first.
    """
    from scipy.optimize import minimize
    p,A,z,q,fit=historical.fit_grid(np.asarray(values),lags)
    fit['fit_polish_steps']=0
    if not fit['fit_converged']:
        active=set(np.flatnonzero(p>1e-10))
        for step in range(32):
            gradient=A.T@(A@p-z);active.add(int(np.argmin(gradient)))
            indices=np.array(sorted(active));B=A[:,indices];initial=p[indices];initial/=initial.sum()
            result=minimize(lambda v:.5*np.sum((B@v-z)**2),initial,
                jac=lambda v:B.T@(B@v-z),bounds=[(0,None)]*len(indices),
                constraints={'type':'eq','fun':lambda v:v.sum()-1,'jac':lambda v:np.ones(len(v))},
                method='SLSQP',options={'ftol':1e-14,'maxiter':500})
            candidate=np.zeros_like(p);candidate[indices]=np.maximum(result.x,0);candidate/=candidate.sum()
            if np.sum((A@candidate-z)**2)>np.sum((A@p-z)**2)+1e-12:break
            p=candidate;gradient=A.T@(A@p-z);gap=float(p@gradient-gradient.min())
            fit.update(fit_gap=gap,fit_converged=gap<1e-8,fit_polish_steps=step+1,fit_residual=float(abs(A@p-z).max()))
            if fit['fit_converged']:break
    if not fit['fit_converged']:
        # Equality-constrained active-set polishing avoids relying on an
        # objective-change stopping rule when spectral peaks are ill-conditioned.
        active=set(np.flatnonzero(p>1e-10))
        for step in range(2000):
            indices=np.array(sorted(active));B=A[:,indices];n=len(indices)
            kkt=np.block([[B.T@B,np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
            solution=np.linalg.lstsq(kkt,np.r_[B.T@z,1.],rcond=1e-14)[0][:n]
            if np.any(solution < -1e-11):
                current=p[indices];direction=solution-current;neg=direction < -1e-14
                alpha=min(1.,float(np.min(-current[neg]/direction[neg])))
                candidate=current+alpha*direction;p[:]=0;p[indices]=np.maximum(candidate,0);p/=p.sum()
                active=set(np.flatnonzero(p>1e-12))
            else:
                p[:]=0;p[indices]=np.maximum(solution,0);p/=p.sum()
                gradient=A.T@(A@p-z);gap=float(p@gradient-gradient.min())
                fit.update(fit_gap=gap,fit_converged=gap<1e-8,fit_active_steps=step+1,fit_residual=float(abs(A@p-z).max()))
                if fit['fit_converged']:break
                entering=int(np.argmin(gradient))
                if entering in active:break
                active.add(entering)
    if not fit['fit_converged']:
        # Feasible exact-line-search steps provide a final deterministic escape
        # from a degenerate active-set cycle without relaxing the gap threshold.
        for step in range(50000):
            residual=A@p-z;gradient=A.T@residual;entering=int(np.argmin(gradient))
            gap=float(p@gradient-gradient[entering])
            if gap<1e-8:break
            direction=A[:,entering]-A@p
            alpha=min(1.,gap/max(float(direction@direction),1e-30))
            p*=1-alpha;p[entering]+=alpha
        gradient=A.T@(A@p-z);gap=float(p@gradient-gradient.min())
        fit.update(fit_gap=gap,fit_converged=gap<1e-8,fit_line_search_steps=step+1,fit_residual=float(abs(A@p-z).max()))
    return p,A,z,q,fit


def conditional_ranges(values, lags, delta, unseen=(4,10), axes=('real','imaginary')):
    p,A,z,q,fit = geometry_fit(np.asarray(values),lags)
    # Component-specific residual allowance guarantees the fitted simplex is
    # feasible; the development disagreement allowance describes sensitivity.
    eps = abs(A@p-z) + np.asarray(delta)
    if eps.shape != z.shape or np.any(eps < 0): raise ValueError('Invalid component tolerance')
    H = np.r_[A,-A]; b = np.r_[z+eps,-z+eps]; rows = []
    for lag in unseen:
        if lag in lags: raise ValueError('Identification endpoint must be genuinely unseen')
        for axis in axes:
            c = np.cos(lag*q) if axis == 'real' else np.sin(lag*q)
            lo = historical.validated_min(c,H,b); hi = historical.validated_min(-c,H,b)
            valid = lo['valid'] and hi['valid'] and fit['fit_converged']
            lower = lo['outer']; upper = -hi['outer'] if hi['outer'] is not None else None
            sign = 1 if valid and lower>1e-7 else (-1 if valid and upper < -1e-7 else 0)
            rows.append(dict(lag=int(lag),axis=axis,valid=bool(valid),lower=lower,upper=upper,
                width=upper-lower if valid else None,identified=sign!=0,sign=sign,
                tolerance_min=float(eps.min()),tolerance_max=float(eps.max()),**fit,
                low_gap=lo['gap'],high_gap=hi['gap']))
    return rows


def select_design():
    _, ds, roles, _ = bundle(); config = CFG['identifiability']
    tr = np.flatnonzero(roles == 'model_development')
    samples=[]
    for i in tr:
        with np.load(cache_path(ds,i)) as z:
            samples.append(z['first_gamma'].mean(0)-z['second_gamma'].mean(0))
    d = np.asarray(samples)
    delta = np.maximum(np.quantile(abs(np.stack([d.real,d.imag],-1)),.9,axis=0), config['tolerance_floor'])
    np.save(MOD/'development_component_tolerance.npy',delta)
    selected = sorted(set(ds.event_ids[tr]),key=lambda e:hashlib.sha256(f'20260907:{e}'.encode()).hexdigest())[:config['selection_events']]
    assert not set(selected) & set(ds.event_ids[roles == 'architecture_test'])
    candidates = list(combinations(config['candidate_pool'],config['budget']))
    rows=[]; detail=[]
    # Persist each completed candidate, so interruption never changes selection.
    checkpoint=OUT/'development_design_candidates.csv'
    if checkpoint.exists(): rows=pd.read_csv(checkpoint).to_dict('records')
    done={r['lags'] for r in rows}
    for lags in candidates:
        label=','.join(map(str,lags))
        if label in done: continue
        ids=[list(historical.MEASURE).index(d) for d in lags]; widths=[]; invalid=0
        for i in tr[np.isin(ds.event_ids[tr],selected)]:
            with np.load(cache_path(ds,i)) as z: target=z['target_gamma'].mean(0)
            for band in range(4):
                tolerance=delta[band,ids].T.reshape(-1)
                bounds=conditional_ranges(target[band,ids],lags,tolerance,axes=('real',))
                for b in bounds:
                    invalid+=not b['valid']; widths.append(b['width'] if b['valid'] else 2.)
        rows.append(dict(lags=label,mean_width=float(np.mean(widths)),invalid=invalid,cases=len(widths)))
        pd.DataFrame(rows).to_csv(checkpoint,index=False)
        print('development design',len(rows),'/',len(candidates),flush=True)
    choice=min(rows,key=lambda r:(r['mean_width'],tuple(map(int,r['lags'].split(',')))))
    write_json(OUT/'locked_design.json',dict(lags=list(map(int,choice['lags'].split(','))),
        selection_events=selected,development_events=sorted(set(ds.event_ids[tr])),
        candidate_count=len(rows),objective=choice['mean_width'],delta_sha256=sha256(MOD/'development_component_tolerance.npy'),
        protocol_sha256=sha256(SPEC),test_evaluated_during_selection=False))


def evaluate_design():
    _, ds, roles, _ = bundle(); choice=json.loads((OUT/'locked_design.json').read_text())
    delta=np.load(MOD/'development_component_tolerance.npy'); rows=[]
    designs={'historical':CFG['identifiability']['historical_lags'],
             'manual':CFG['identifiability']['manual_lags'],'algorithmic':choice['lags'],
             'augmented':CFG['identifiability']['candidate_pool']}
    for i in np.flatnonzero(roles == 'architecture_test'):
        with np.load(cache_path(ds,i)) as z: target=z['target_gamma'].mean(0)
        for name,lags in designs.items():
            ids=[list(historical.MEASURE).index(d) for d in lags]
            for band in range(4):
                for b in conditional_ranges(target[band,ids],lags,delta[band,ids].T.reshape(-1)):
                    rows.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],band=band,design=name,**b))
        print('conditional ranges',ds.event_ids[i],ds.routes[i],flush=True)
    table=pd.DataFrame(rows);table.to_parquet(OUT/'conditional_ranges.parquet',index=False)
    table.groupby(['design','axis']).agg(cases=('lag','size'),valid=('valid','sum'),identified=('identified','sum'),mean_width=('width','mean')).to_csv(OUT/'conditional_ranges_summary.csv')


def processing():
    from .review_revision import _generalized_weight, _ratio
    _,ds,roles,_=bundle(); rows=[]; fits=[]
    seeds=CFG['training']['seeds']
    probabilities={}
    for tag in ['block_dense','local_sparse','local_dense']:
        for seed in seeds:
            probabilities[tag,seed]=np.load(MOD/f'{tag}_state_psd_seed{seed}_probabilities.npy')
    for seed in seeds:
        probabilities['block_sparse',seed]=np.load(ROOT/f'models/critical_review/matched_state_psd_seed{seed}_all_probabilities.npy')
    ridge_values={tag:np.load(MOD/f'{tag}_block_ridge_predictions.npy') for tag in ['block_dense','local_sparse','local_dense']}
    ridge_values['block_sparse']=np.load(historical.MOD/'primary_block_ridge_predictions.npy')
    lags={'block_sparse':historical.LAGS,'block_dense':np.arange(1,32),
          'local_sparse':np.array(CFG['experiments']['local_sparse_lags']),'local_dense':np.arange(1,32)}
    for n,i in enumerate(np.flatnonzero(roles=='architecture_test')):
        with np.load(cache_path(ds,i)) as z:
            rc=z['context_matrix'];rt=z['target_matrix'];rn=z['noise_matrix']
            h1=z['first_matrix'];h2=z['second_matrix'];pos=z['positions']
        off=~np.eye(len(pos),dtype=bool)
        for block,band in np.ndindex(8,4):
            D=np.diag(np.maximum(rc[block,band].diagonal().real,1e-15))
            scale=np.sqrt(D.diagonal()[:,None]*D.diagonal()[None,:])
            kernels={(tag,'recurrence',seed):upper_kernel_matrix(prob[i,block,band],pos) for (tag,seed),prob in probabilities.items()}
            for tag,values in ridge_values.items():
                prob,_,_,_,fit=historical.fit_grid(complex_values(values[i,block,band]),lags[tag])
                kernels[tag,'block ridge',0]=upper_kernel_matrix(prob,pos)
                fits.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],block=block,band=band,support=tag,**fit))
            methods={('reference','Diagonal',0):D,('reference','Classical shrinkage',0):.25*rc[block,band]+.75*D}
            methods.update({key:.25*kernel*scale+.75*D for key,kernel in kernels.items()})
            observed=rt[block,band]/np.sqrt(rt[block,band].diagonal().real[:,None]*rt[block,band].diagonal().real[None,:])
            for (support,model,seed),cov in methods.items():
                w=_generalized_weight(cov,rn[block,band],.0001)
                for endpoint,target in [('full',rt[block,band]),('half0',h1[block,band]),('half1',h2[block,band])]:
                    rows.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],block=block,band=band,support=support,model=model,seed=seed,endpoint=endpoint,
                        ratio_db=float(10*np.log10(_ratio(target,rn[block,band],w))),
                        covariance_entry_mse=float(np.mean(abs(kernels[support,model,seed][off]-observed[off])**2)) if (support,model,seed) in kernels else None))
        print('geometry-matched processing',n+1,'/ 48',flush=True)
    table=pd.DataFrame(rows);table.to_parquet(OUT/'geometry_matched_processing.parquet',index=False)
    pd.DataFrame(fits).to_csv(OUT/'processing_projection_checks.csv',index=False)
    summarise_processing()


def summarise_processing():
    frame=pd.read_parquet(OUT/'geometry_matched_processing.parquet')
    frame['evaluation']=np.where(frame.endpoint=='full','full','split_target')
    frame['method']=frame.support+' / '+frame.model
    mapping=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet').set_index('event_id').sensitivity_component_25km.to_dict()
    event=frame.groupby(['evaluation','event_id','method','seed']).ratio_db.mean().reset_index()
    event.to_csv(OUT/'processing_event_seed_scores.csv',index=False)
    result=[];delete=[]
    for endpoint,g in event.groupby('evaluation'):
        # Seed means are averaged inside earthquakes, never resampled as events.
        e=g.groupby(['event_id','method']).ratio_db.mean().unstack()
        for reference in ['reference / Diagonal','reference / Classical shrinkage','block_sparse / recurrence','local_sparse / recurrence','local_sparse / block ridge','block_sparse / block ridge']:
            for method in e:
                v=e[method]-e[reference];labels=np.array([mapping[str(ev)] for ev in v.index])
                result.append(dict(endpoint=endpoint,method=method,reference=reference,**boot(v,labels)))
                for component in sorted(set(labels)):
                    keep=labels!=component
                    delete.append(dict(endpoint=endpoint,method=method,reference=reference,omitted_component=component,retained_events=int(keep.sum()),mean=float(v[keep].mean())))
    pd.DataFrame(result).to_csv(OUT/'processing_comparisons.csv',index=False)
    pd.DataFrame(delete).to_csv(OUT/'processing_component_deletion.csv',index=False)


def verify():
    import subprocess,sys
    checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value: raise AssertionError(name)
    _,ds,roles,_=bundle()
    check('protocol unchanged',sha256(SPEC)==(OUT/'protocol.sha256').read_text().strip())
    a=json.loads((OUT/'starting_audit.json').read_text())
    check('all waveform leads positive',a['minimum_lead_seconds']>0)
    dev=set(ds.event_ids[roles=='model_development']);test=set(ds.event_ids[roles=='architecture_test'])
    for path in MOD.glob('*_seed*.json'):
        receipt=json.loads(path.read_text())
        check(path.stem+' development only',set(receipt['training_events'])==dev and not set(receipt['internal_validation_events'])&test)
        check(path.stem+' current checkpoint',sha256(path.with_suffix('.pt'))==receipt['checkpoint_sha256'])
    choice=json.loads((OUT/'locked_design.json').read_text())
    check('design selected only on development',set(choice['selection_events'])<=dev and not set(choice['selection_events'])&test)
    check('unseen design endpoints',not set(choice['lags'])&{4,10})
    check('complete finite search',len(pd.read_csv(OUT/'development_design_candidates.csv'))==45)
    r=pd.read_parquet(OUT/'conditional_ranges.parquet')
    check('all range fits and bounds validated',r.valid.all())
    check('complex range count',len(r)==48*4*4*2*2)
    check('identified only if zero excluded',(~r.identified | (r.lower>1e-7) | (r.upper < -1e-7)).all())
    p=pd.read_parquet(OUT/'geometry_matched_processing.parquet')
    check('three seeds for every recurrence support',p[p.model=='recurrence'].groupby('support').seed.nunique().eq(3).all())
    check('all processing finite',np.isfinite(p.ratio_db).all())
    tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_submission_revision.py'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'pytest.txt').write_text(tests.stdout+tests.stderr)
    check('new unit tests',tests.returncode==0)
    write_json(OUT/'verification.json',dict(checks=checks,tests=tests.stdout))
    print('Submission revision verified',len(checks),'checks',flush=True)


def inference():
    mapping=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet').set_index('event_id').sensitivity_component_25km.to_dict()
    frame=pd.read_csv(OUT/'prediction_event_scores.csv',dtype={'event_id':str})
    summaries=[]; deletions=[];seeds=[]
    for endpoint,g in frame.groupby('endpoint'):
        e=g.groupby(['event_id','model']).nrmse.mean().unstack()
        for reference in ['block_ridge','full_ridge']:
            for model in e:
                v=e[model]-e[reference]; labels=np.array([mapping[str(ev)] for ev in v.index])
                for mode,group in [('component',labels),('earthquake',e.index)]:
                    summaries.append(dict(endpoint=endpoint,model=model,reference=reference,inference=mode,**boot(v,group)))
                for component in sorted(set(labels)):
                    keep=labels!=component
                    deletions.append(dict(endpoint=endpoint,model=model,reference=reference,omitted_component=component,retained_events=int(keep.sum()),mean=float(v[keep].mean())))
        values=[float(e[f'state_psd_seed{s}'].mean()) for s in CFG['training']['seeds']]
        seeds.append(dict(endpoint=endpoint,individual_seed_mean=float(np.mean(values)),seed_sample_sd=float(np.std(values,ddof=1)),seed_min=min(values),seed_max=max(values),prediction_ensemble_score=float(e.state_psd_seed_mean.mean())))
    pd.DataFrame(summaries).to_csv(OUT/'prediction_inference.csv',index=False)
    pd.DataFrame(deletions).to_csv(OUT/'prediction_component_deletion.csv',index=False)
    pd.DataFrame(seeds).to_csv(OUT/'seed_variation.csv',index=False)
    bounds=pd.read_parquet(OUT/'conditional_ranges.parquet'); effects=[]
    for axis,g in bounds.groupby('axis'):
        for quantity in ['identified','width']:
            e=g.groupby(['event_id','design'])[quantity].mean().unstack()
            for design in ['manual','algorithmic','augmented']:
                v=e[design]-e.historical
                effects.append(dict(axis=axis,quantity=quantity,design=design,**boot(v,[mapping[str(ev)] for ev in v.index])))
    pd.DataFrame(effects).to_csv(OUT/'design_paired_effects.csv',index=False)


def synthetic():
    rng=np.random.default_rng(20260907); rows=[]
    lags=np.array(CFG['identifiability']['manual_lags'])
    for family in ['white','smooth','narrow','off_grid']:
        for replicate in range(20):
            q=Q if family!='off_grid' else Q+.371/257
            p=np.ones(len(Q)) if family=='white' else np.exp((2 if family=='smooth' else 15)*np.cos(q-rng.uniform(-np.pi,np.pi)))
            p/=p.sum(); target=np.exp(1j*lags[:,None]*q)@p
            delta=rng.uniform(.005,.03,2*len(lags))
            noise=rng.uniform(-1,1,2*len(lags))*delta
            observed=target+noise[:len(lags)]+1j*noise[len(lags):]
            for r in conditional_ranges(observed,lags,delta):
                truth=np.exp(1j*r['lag']*q)@p
                truth=float(truth.real if r['axis']=='real' else truth.imag)
                rows.append(dict(family=family,replicate=replicate,truth=truth,covered=r['valid'] and r['lower']<=truth<=r['upper'],incorrect=r['identified'] and np.sign(truth)!=r['sign'],**r))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'complex_known_truth.csv',index=False)
    frame.groupby('family').agg(cases=('lag','size'),valid=('valid','sum'),covered=('covered','sum'),identified=('identified','sum'),incorrect=('incorrect','sum')).to_csv(OUT/'complex_known_truth_summary.csv')
    assert frame.loc[frame.family!='off_grid','covered'].all()


def validate_raw_local():
    import h5py
    from .labels import RAW_DATASET,_read_window
    from .spectral import robust_linear_pick_model,multitaper_fourier,band_snapshots,lag_coherency
    cfg,ds,roles,_=bundle();meta=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet').set_index('event_id')
    selected=sorted(set(ds.event_ids[roles=='model_development']))[:2]; rows=[]
    for i in np.flatnonzero(np.isin(ds.event_ids,selected)):
        row=meta.loc[str(ds.event_ids[i])];route=ds.routes[i];prefix='terra' if route=='TERRA' else 'kkfls'
        picks=pd.read_csv(ROOT/'data/provenance/picks'/f'{ds.event_ids[i]}.csv')
        ch=picks.channel.to_numpy(float);s=pd.to_numeric(picks['TERRA_s_sec' if route=='TERRA' else 'KKFLS_s_sec'],errors='coerce').to_numpy(float)
        with np.load(cache_path(ds,i)) as z: cached=local_lag_targets(z['target_matrix'],np.arange(1,32))
        with h5py.File(row[prefix+'_path'],'r') as h:
            for block,start in enumerate(cfg['spectral']['block_starts']):
                ref,slope,_=robust_linear_pick_model(ch,s,start,1000)
                if not np.isfinite(ref):ref,slope=float(np.nanmedian(s)),0.
                x=_read_window(h[RAW_DATASET],ref+3,ref+10,25.,slice(start+484,start+516))
                tr=multitaper_fourier(x,25.,2.5,3,slope*(np.arange(484,516)-499.5))
                for band,bounds in enumerate(cfg['spectral']['bands_hz']):
                    direct=lag_coherency(band_snapshots(*tr,bounds),np.arange(1,32))
                    difference=float(np.max(abs(direct-cached[block,band])))
                    rows.append(dict(event_id=ds.event_ids[i],route=route,block=block,band=band,max_complex_difference=difference))
                    assert difference<2e-6
    pd.DataFrame(rows).to_csv(OUT/'direct_raw_local_validation.csv',index=False)


def regenerate():
    """Independently regenerate all new saved predictions from CPU parameters."""
    import torch
    torch.set_num_threads(4)
    _,ds,roles,_=bundle();safe=ds.features.copy();rows=[]
    for i in np.flatnonzero(np.isin(roles,['model_development','calibration','architecture_test'])):
        with np.load(ROOT/'data/processed/critical_review'/cache_path(ds,i).name) as z:safe[i]=z['global_cutoff_features']
    for tag in ['waveform_safe','block_dense','local_sparse','local_dense']:
        x=safe if tag=='waveform_safe' else ds.features
        for seed in CFG['training']['seeds']:
            path=MOD/f'{tag}_state_psd_seed{seed}.pt';receipt=json.loads(path.with_suffix('.json').read_text())
            saved=torch.load(path,map_location='cpu',weights_only=True)
            model=MatchedModel(137,'state_psd');lags=np.asarray(receipt['lags'])
            model.basis_re=torch.tensor(np.exp(1j*lags[:,None]*Q).real,dtype=torch.float32)
            model.basis_im=torch.tensor(np.exp(1j*lags[:,None]*Q).imag,dtype=torch.float32)
            model.load_state_dict(saved['state']);model.eval()
            p,q=predict_model(model,x,saved['mean'].numpy(),saved['std'].numpy())
            for quantity,value in [('predictions',p),('probabilities',q)]:
                expected=np.load(MOD/f'{tag}_state_psd_seed{seed}_{quantity}.npy')
                error=float(abs(value-expected).max());assert error<2e-5
                rows.append(dict(endpoint=tag,model='recurrence',seed=seed,quantity=quantity,max_absolute_difference=error,tolerance=2e-5))
        for kind in ['block_ridge','full_ridge']:
            with np.load(MOD/f'{tag}_{kind}_parameters.npz') as z:
                xnorm=((x-z['mean'])/z['std']).astype('float32')
                a=xnorm.reshape(-1,137) if kind=='block_ridge' else xnorm.reshape(len(x),-1)
                p=a@z['coef'].T+z['intercept']
            expected=np.load(MOD/f'{tag}_{kind}_predictions.npy');p=p.reshape(expected.shape)
            error=float(abs(p-expected).max());assert error<2e-5
            rows.append(dict(endpoint=tag,model=kind,seed=0,quantity='predictions',max_absolute_difference=error,tolerance=2e-5))
    pd.DataFrame(rows).to_csv(OUT/'cpu_parameter_regeneration.csv',index=False)
    print('CPU parameter regeneration passed',len(rows),'checks',flush=True)
