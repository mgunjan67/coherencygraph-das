"""Bounded G1 closeout: same audit, empirical withheld checks, complete numerics.

Never trains models, changes frozen choices, or writes historical evidence.
All case counts are dependent; summaries first average within event/route.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from coherencygraph_das import submission_revision as r
OUT = ROOT/'reports/submission_closeout/audit_validation'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')

def detailed_ranges(values, lags, delta):
    """Exact original conditional_ranges operations, with retained diagnostics."""
    p, A, z, q, fit = r.geometry_fit(np.asarray(values), lags)
    residual = abs(A@p-z)
    eps = residual + np.asarray(delta)
    assert eps.shape == z.shape and np.all(np.asarray(delta) > 0)
    H = np.r_[A, -A]; b = np.r_[z+eps, -z+eps]
    rows = []
    for lag in [4, 10]:
        assert lag not in lags
        for axis in ['real', 'imaginary']:
            c = np.cos(lag*q) if axis == 'real' else np.sin(lag*q)
            lo = r.historical.validated_min(c, H, b)
            hi = r.historical.validated_min(-c, H, b)
            valid = lo['valid'] and hi['valid'] and fit['fit_converged']
            lower = lo['outer']; upper = -hi['outer'] if hi['outer'] is not None else None
            sign = 1 if valid and lower > 1e-7 else (-1 if valid and upper < -1e-7 else 0)
            rows.append(dict(lag=lag, axis=axis, valid=bool(valid), lower=lower,
                upper=upper, width=upper-lower if valid else None, sign=sign,
                identified=sign != 0, **fit,
                tolerance_min=float(eps.min()), tolerance_max=float(eps.max()),
                simplex_sum_error=float(abs(p.sum()-1)), simplex_min=float(p.min()),
                fitted_feasible_violation=float(max(0., (H@p-b).max())),
                **{'low_'+k:v for k,v in lo.items()}, **{'high_'+k:v for k,v in hi.items()}))
    residual_rows = []
    n = len(lags)
    for j, (res, d, e) in enumerate(zip(residual, delta, eps)):
        residual_rows.append(dict(observed_lag=int(lags[j % n]),
            component='real' if j < n else 'imaginary',
            residual=float(res), development_delta=float(d),
            residual_to_delta=float(res/d), total_allowance=float(e),
            allowance_enlargement_factor=float(e/d)))
    return rows, residual_rows

def empirical_fields(row, measured):
    measured = float(measured)
    valid = row['valid']
    identified = row['identified']
    distance = max(row['lower']-measured, measured-row['upper'], 0.) if valid else None
    return dict(measured_component=measured, measured_abs=abs(measured),
        measured_sign=int(np.sign(measured)), measured_exact_zero=measured == 0.,
        interval_contains=bool(valid and distance == 0.),
        distance_outside=distance,
        identified_sign_agreement=bool(identified and np.sign(measured) == row['sign']),
        identified_sign_disagreement=bool(identified and np.sign(measured) != row['sign']))

def sources():
    paths = [r.SPEC, r.OUT/'conditional_ranges.parquet', r.OUT/'complex_known_truth.csv',
        r.OUT/'locked_design.json', r.OUT/'unchanged_event_roles.csv',
        r.MOD/'development_component_tolerance.npy',
        ROOT/'src/coherencygraph_das/submission_revision.py',
        ROOT/'src/coherencygraph_das/final_revision.py',
        r.historical.OUT/'known_truth_validation.csv',
        r.historical.OUT/'measurement_design_bounds.parquet',
        r.historical.OUT/'split_window_bounds.csv']
    roles = pd.read_csv(r.OUT/'unchanged_event_roles.csv', dtype=str)
    records = roles[roles.role == 'architecture_test'][['event_id', 'route']].to_dict('records')
    assert len(records) == 48 and len({v['event_id'] for v in records}) == 24
    for rec in records:
        rec['cache'] = (r.historical.CACHE/f"{rec['event_id']}_{rec['route'].replace('-', '')}.npz").relative_to(ROOT).as_posix()
        paths.append(ROOT/rec['cache'])
    return records, {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'specification.json').exists(), 'Closeout spec already exists; never overwrite'
    records, hashes = sources()
    write(OUT/'specification.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='bounded retrospective closeout, not preregistration',
        endpoints=[4,10], axes=['real','imaginary'], records=records, sources=hashes,
        full_window='unchanged original full-window input intervals versus withheld full-window components',
        opposite_half='first-half input intervals using unchanged full-development delta, versus second-half components',
        half_support='target waveform split at floor(n_samples/2); same fitted passage; each half has independent finite-window spectral estimation, not independent earthquakes',
        exclusions='none; exact-zero and arbitrarily small measured values retained',
        aggregation='bands and lags within event-route; paired routes within complete earthquake; descriptive, no bootstrap/p-values',
        near_zero='no added exclusion threshold',
        solver='unchanged geometry_fit and validated_min; original acceptance thresholds',
        existing_stress_tests='reuse older known-truth/grid/scalar-allowance/split-window cases with explicit historical boundary',
        current_synthetic='regenerate exact original 80 seeded spectra (320 components), same RNG and allowances',
        script_sha256=sha(Path(__file__))))

def check_sources():
    spec = json.loads((OUT/'specification.json').read_text())
    for path, digest in spec['sources'].items():
        assert sha(ROOT/path) == digest, f'Historical input changed: {path}'
    return spec

def one_record(rec):
    path = OUT/'records'/f"{rec['event_id']}_{rec['route']}.parquet"
    respath = path.with_name(path.stem+'_residuals.parquet')
    if path.exists() and respath.exists():
        return path.name, 'cached'
    delta = np.load(r.MOD/'development_component_tolerance.npy')
    choice = json.loads((r.OUT/'locked_design.json').read_text())
    designs = {'historical':r.CFG['identifiability']['historical_lags'],
        'manual':r.CFG['identifiability']['manual_lags'], 'algorithmic':choice['lags'],
        'augmented':r.CFG['identifiability']['candidate_pool']}
    unique = {}
    for name, lags in designs.items():
        unique.setdefault(tuple(lags), []).append(name)
    rows = []; residuals = []
    with threadpool_limits(limits=1), np.load(ROOT/rec['cache']) as z:
        measured_lags = z['measured_lags'].tolist()
        estimates = {key:z[key+'_gamma'].mean(0) for key in ['target','first','second']}
        for support, input_key, check_key in [('full_window','target','target'),('opposite_half','first','second')]:
            for lags, names in unique.items():
                ids = [measured_lags.index(v) for v in lags]
                for band in range(4):
                    result, res = detailed_ranges(estimates[input_key][band,ids], lags, delta[band,ids].T.reshape(-1))
                    for name in names:
                        prefix = dict(event_id=rec['event_id'], route=rec['route'], band=band,
                            design=name, support=support, fit_inputs=input_key, measured_check=check_key,
                            duplicate_design='manual=algorithmic' if name in ['manual','algorithmic'] else '')
                        residuals.extend([dict(**prefix, **v) for v in res])
                        for bnd in result:
                            value = estimates[check_key][band,measured_lags.index(bnd['lag'])]
                            value = value.real if bnd['axis']=='real' else value.imag
                            rows.append(dict(**prefix, **bnd, **empirical_fields(bnd,value)))
    pd.DataFrame(rows).to_parquet(path, index=False)
    pd.DataFrame(residuals).to_parquet(respath, index=False)
    return path.name, len(rows)

def evaluate(workers):
    spec = check_sources(); (OUT/'records').mkdir(exist_ok=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_record, rec) for rec in spec['records']]
        for i, f in enumerate(as_completed(futures),1):
            print(i, '/48', f.result(), flush=True)
    write(OUT/'evaluation_receipt.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-start, workers=workers, blas_threads_per_worker=1,
        script_sha256=sha(Path(__file__)), specification_sha256=sha(OUT/'specification.json')))

def synthetic():
    check_sources(); rows=[]; residuals=[]; rng=np.random.default_rng(20260907)
    lags=np.array(r.CFG['identifiability']['manual_lags'])
    with threadpool_limits(limits=1):
        for family in ['white','smooth','narrow','off_grid']:
            for replicate in range(20):
                q=r.Q if family!='off_grid' else r.Q+.371/257
                p=np.ones(len(r.Q)) if family=='white' else np.exp((2 if family=='smooth' else 15)*np.cos(q-rng.uniform(-np.pi,np.pi)))
                p/=p.sum(); target=np.exp(1j*lags[:,None]*q)@p
                delta=rng.uniform(.005,.03,2*len(lags))
                noise=rng.uniform(-1,1,2*len(lags))*delta
                observed=target+noise[:len(lags)]+1j*noise[len(lags):]
                bounds, res = detailed_ranges(observed,lags,delta)
                residuals.extend([dict(family=family,replicate=replicate,**v) for v in res])
                for v in bounds:
                    truth=np.exp(1j*v['lag']*q)@p
                    truth=float(truth.real if v['axis']=='real' else truth.imag)
                    rows.append(dict(family=family,replicate=replicate,truth=truth,
                        covered=bool(v['valid'] and v['lower']<=truth<=v['upper']),
                        incorrect=bool(v['identified'] and np.sign(truth)!=v['sign']), **v))
    pd.DataFrame(rows).to_parquet(OUT/'current_synthetic_recomputed.parquet',index=False)
    pd.DataFrame(residuals).to_csv(OUT/'current_synthetic_residuals.csv',index=False)
    print('Recomputed original synthetic cases:',len(rows),flush=True)

def distribution(values):
    a=np.asarray(values,float); a=a[np.isfinite(a)]
    return dict(n=len(a), minimum=float(a.min()), median=float(np.median(a)),
        q90=float(np.quantile(a,.9)), q95=float(np.quantile(a,.95)), maximum=float(a.max()))

def summarise():
    check_sources()
    cases=pd.concat([pd.read_parquet(p) for p in sorted((OUT/'records').glob('*.parquet')) if not p.stem.endswith('_residuals')],ignore_index=True)
    residuals=pd.concat([pd.read_parquet(p) for p in sorted((OUT/'records').glob('*_residuals.parquet'))],ignore_index=True)
    assert len(cases)==6144 and not cases.duplicated(['event_id','route','band','design','support','lag','axis']).any()
    cases.to_parquet(OUT/'empirical_case_checks.parquet',index=False)
    cases.to_csv(OUT/'empirical_case_checks.csv',index=False)
    residuals.to_parquet(OUT/'moment_residuals.parquet',index=False)
    summary=[]
    for key,g in cases.groupby(['support','design','axis']):
        identified=g[g.identified]; invalid=~g.valid
        summary.append(dict(support=key[0],design=key[1],axis=key[2],cases=len(g),
            earthquakes=g.event_id.nunique(),valid=int(g.valid.sum()),invalid=int(invalid.sum()),
            fit_failures=int((~g.fit_converged).sum()),identified=int(g.identified.sum()),
            valid_unresolved=int((g.valid & ~g.identified).sum()),
            sign_agreements=int(g.identified_sign_agreement.sum()),sign_disagreements=int(g.identified_sign_disagreement.sum()),
            contained=int(g.interval_contains.sum()), fraction_contained=float(g.interval_contains.mean()),
            max_outside_distance=float(g.distance_outside.max()),mean_outside_distance=float(g.distance_outside.mean()),
            identified_min_abs_measured=float(identified.measured_abs.min()) if len(identified) else None,
            minimum_abs_measured=float(g.measured_abs.min()),exact_zero_count=int(g.measured_exact_zero.sum())))
    pd.DataFrame(summary).to_csv(OUT/'empirical_summary.csv',index=False)
    event=cases.groupby(['support','design','axis','event_id','route'])[['valid','identified','interval_contains','identified_sign_disagreement','distance_outside','width']].mean().groupby(['support','design','axis','event_id']).mean().reset_index()
    event.to_csv(OUT/'empirical_event_summary.csv',index=False)
    event.groupby(['support','design','axis']).agg(earthquakes=('event_id','nunique'),mean_containment_fraction=('interval_contains','mean'),mean_identified_fraction=('identified','mean'),mean_sign_disagreement_fraction=('identified_sign_disagreement','mean'),mean_outside_distance=('distance_outside','mean'),mean_width=('width','mean')).to_csv(OUT/'empirical_equal_event_summary.csv')
    original=pd.read_parquet(r.OUT/'conditional_ranges.parquet')
    old=original.set_index(['event_id','route','band','design','lag','axis']).sort_index()
    old.index=old.index.set_levels(old.index.levels[0].astype(str),level=0)
    new=cases[cases.support=='full_window'].set_index(['event_id','route','band','design','lag','axis']).sort_index()
    assert old.index.equals(new.index)
    reconcile=[]
    for col in ['lower','upper','width','fit_gap','fit_residual','low_gap','high_gap']:
        gap=float(np.max(abs(old[col]-new[col])))
        reconcile.append(dict(artifact='current_DAS_full_window',column=col,max_abs_difference=gap,passed=gap<1e-9))
    for col in ['valid','identified','sign']:
        reconcile.append(dict(artifact='current_DAS_full_window',column=col,max_abs_difference=int((old[col]!=new[col]).sum()),passed=bool((old[col]==new[col]).all())))
    syn=pd.read_parquet(OUT/'current_synthetic_recomputed.parquet')
    synold=pd.read_csv(r.OUT/'complex_known_truth.csv').set_index(['family','replicate','lag','axis']).sort_index()
    synnew=syn.set_index(['family','replicate','lag','axis']).sort_index()
    for col in ['truth','lower','upper','width','fit_gap','low_gap','high_gap']:
        gap=float(np.max(abs(synold[col]-synnew[col])))
        reconcile.append(dict(artifact='current_complex_synthetic',column=col,max_abs_difference=gap,passed=gap<1e-9))
    for col in ['valid','identified','sign','covered','incorrect']:
        reconcile.append(dict(artifact='current_complex_synthetic',column=col,max_abs_difference=int((synold[col]!=synnew[col]).sum()),passed=bool((synold[col]==synnew[col]).all())))
    pd.DataFrame(reconcile).to_csv(OUT/'old_new_reconciliation.csv',index=False)
    assert all(v['passed'] for v in reconcile)
    syn.groupby('family').agg(cases=('lag','size'),valid=('valid','sum'),covered=('covered','sum'),identified=('identified','sum'),incorrect=('incorrect','sum'),fit_converged=('fit_converged','sum')).to_csv(OUT/'current_synthetic_summary.csv')
    # Historical checks are not silently relabelled as Amendment-09 tests.
    legacy=pd.read_csv(r.historical.OUT/'known_truth_validation.csv')
    legacy.groupby('family').agg(cases=('lag','size'),valid=('valid','sum'),covered=('covered','sum'),identified=('identified','sum'),incorrect=('incorrect_certificate','sum'),fit_converged=('fit_converged','sum')).to_csv(OUT/'historical_synthetic_summary.csv')
    sensitivity=pd.read_parquet(r.historical.OUT/'measurement_design_bounds.parquet')
    primary=sensitivity[(sensitivity.design=='historical')&(sensitivity.grid==257)&np.isclose(sensitivity.offset,.01)].set_index(['event_id','route','band','lag'])
    rows=[]
    for key,g in sensitivity[sensitivity.design=='historical'].groupby(['grid','offset']):
        comp=g.set_index(['event_id','route','band','lag']).join(primary[['sign','identified']],rsuffix='_reference')
        rows.append(dict(grid=int(key[0]),scalar_offset=float(key[1]),cases=len(g),
            valid=int(g.valid.sum()),identified=int(g.identified.sum()),fit_converged=int(g.fit_converged.sum()),
            mean_width=float(g.width.mean()),empirical_containment=int(g.empirical_in_range.sum()),
            sign_decisions_changed=int((comp.sign!=comp.sign_reference).sum()),
            sign_reversals=int(((comp.sign*comp.sign_reference)<0).sum()),
            scope='historical scalar-residual enlargement; real lags1,2,4,10, not component-specific current intervals'))
    pd.DataFrame(rows).to_csv(OUT/'historical_grid_allowance_sensitivity.csv',index=False)
    split=pd.read_csv(r.historical.OUT/'split_window_bounds.csv')
    split.groupby('design').agg(cases=('lag','size'),valid=('valid','sum'),identified=('identified','sum'),contained=('in_range','sum'),sign_disagreement=('sign_disagreement','sum'),fit_converged=('fit_converged','sum')).to_csv(OUT/'historical_half_summary.csv')
    numerical=[]
    for name,g in [('current_DAS',cases),('current_complex_synthetic',syn),('historical_synthetic',legacy),('historical_grid_allowance',sensitivity),('historical_half',split)]:
        for col in ['fit_gap','low_primal_violation','high_primal_violation','low_dual_violation','high_dual_violation','low_gap','high_gap','low_rounding_margin','high_rounding_margin']:
            numerical.append(dict(family=name,quantity=col,**distribution(g[col])))
    pd.DataFrame(numerical).to_csv(OUT/'numerical_diagnostic_distributions.csv',index=False)
    residual_summary=[]
    for key,g in residuals.groupby(['support','design','component']):
        for col in ['residual','development_delta','residual_to_delta','total_allowance','allowance_enlargement_factor']:
            residual_summary.append(dict(support=key[0],design=key[1],component=key[2],quantity=col,**distribution(g[col])))
    pd.DataFrame(residual_summary).to_csv(OUT/'residual_distributions.csv',index=False)
    check_sources()
    write(OUT/'verification.json',dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),
        original_sources_unchanged=True, empirical_cases=len(cases), unique_fit_designs=3,
        manual_equals_algorithmic=True,current_synthetic_cases=len(syn),
        lower_level_checks_not_independent_samples=True,exclusions=0,
        current_empirical_invalid_cases=int((~cases.valid).sum()),
        current_synthetic_invalid_cases=int((~syn.valid).sum()),
        reconciliation_checks=len(reconcile),all_reconciliations_pass=True,
        source_definition='specification.json',summaries=summary))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','evaluate','synthetic','summarise']);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='evaluate': evaluate(args.workers)
    else: globals()[args.stage]()
