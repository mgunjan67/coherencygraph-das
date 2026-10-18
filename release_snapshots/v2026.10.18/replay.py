"""Replay existing derived results. No acquisition, fitting, or new simulation.

This file is copied as replay.py into the scientific package. It has no private
manuscript dependencies and accepts an explicit package root and output path.
"""
from pathlib import Path
import argparse, hashlib, json, math, platform, sys, shutil
import numpy as np
import pandas as pd

METHODS = ['context_0','context_0.1','context_0.25','context_0.5','context_0.75',
           'context_1','conventional','neural','ridge']
CONDITIONS = ['matched','delay_minus','delay_plus','amplitude_minus','amplitude_plus']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def boot(values, seed):
    x=np.asarray(values,float); ix=np.random.default_rng(seed).integers(0,len(x),(5000,len(x)))
    return np.r_[x.mean(),np.quantile(x[ix].mean(axis=1),[.025,.975])]

def assert_near(a,b,name,tol=1e-10):
    x,y=np.asarray(a,float),np.asarray(b,float)
    assert x.shape==y.shape and np.array_equal(np.isnan(x),np.isnan(y)),name
    keep=~np.isnan(x)
    delta=float(np.max(abs(x[keep]-y[keep]))) if np.any(keep) else 0.0
    if not np.isfinite(delta) or delta>tol:raise AssertionError(f'{name}: {delta} > {tol}')
    return delta

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def cook(root,out):
    base=root/'revisions/20260922_unseen_validation'
    frames=[pd.read_parquet(p) for p in sorted((base/'validation/case_records').glob('*.parquet'))]
    cases=pd.concat(frames,ignore_index=True)
    assert len(cases)==253440 and cases.event_id.nunique()==30
    assert not cases.duplicated(['event_id','route','cutoff','block','band','method','endpoint']).any()
    cases['model']=cases.method.str.replace(r'_seed\d+$','',regex=True)
    cols=['ratio_db','covariance_entry_mse','complex_lag_mse']
    merged=cases.groupby(['event_id','route','cutoff','block','band','endpoint','model'],as_index=False)[cols].mean()
    assert len(merged)==161280
    events=merged.groupby(['event_id','endpoint','model'],as_index=False)[cols].mean()
    saved=pd.read_csv(base/'validation/event_scores.csv',dtype={'event_id':str})
    events.event_id=events.event_id.astype(str)
    keys=['event_id','endpoint','model']; a=events.set_index(keys).sort_index(); b=saved.set_index(keys).sort_index()
    errors=[assert_near(a[k],b[k],f'Cook event {k}') for k in cols]
    events.to_csv(out/'cook_event_scores.csv',index=False)
    expected=pd.read_csv(base/'validation/paired_comparisons.csv'); rows=[]
    # Independently rebuild 50-km source components from catalog coordinates.
    ids=sorted(events.event_id.unique())
    cat=pd.read_csv(base/'locked_candidate_order.csv',dtype={'event_id':str}).set_index('event_id').loc[ids]
    coordinates=cat[['latitude_deg','longitude_deg','depth_km']].to_numpy(); adjacency=[set() for _ in ids]
    for i,(la,lo,z) in enumerate(coordinates):
        for j,(lb,lp,zb) in enumerate(coordinates[:i]):
            p1,p2=map(math.radians,(la,lb))
            hav=math.sin((p1-p2)/2)**2+math.cos(p1)*math.cos(p2)*math.sin(math.radians(lo-lp)/2)**2
            distance=math.hypot(12742*math.asin(math.sqrt(max(0,min(1,hav)))),z-zb)
            if distance<=50:adjacency[i].add(j);adjacency[j].add(i)
    components=[];seen=set()
    for i in range(len(ids)):
        if i in seen:continue
        todo=[i];members=[]
        while todo:
            j=todo.pop()
            if j in seen:continue
            seen.add(j);members.append(j);todo.extend(adjacency[j]-seen)
        components.append(np.array(sorted(members)))
    assert len(components)==23
    for r in expected.itertuples():
        wide=events[events.endpoint.eq(r.endpoint)].pivot(index='event_id',columns='model',values='ratio_db').loc[ids]
        v=(wide[r.model]-wide[r.reference]).to_numpy()
        if r.resampling=='earthquake':result=boot(v,20260922)
        elif r.resampling=='source_50km':
            rng=np.random.default_rng(20260922);draw=[]
            for _ in range(5000):
                ix=np.concatenate([components[k] for k in rng.integers(0,len(components),len(components))])
                draw.append(v[ix].mean())
            result=np.r_[v.mean(),np.quantile(draw,[.025,.975])]
        else:raise AssertionError(r.resampling)
        errors.append(assert_near(result,[r.mean,r.low,r.high],'Cook bootstrap'))
        rows.append(dict(endpoint=r.endpoint,model=r.model,reference=r.reference,resampling=r.resampling,
                         mean=result[0],low=result[1],high=result[2]))
    pd.DataFrame(rows).to_csv(out/'cook_comparisons.csv',index=False)
    mse=events[events.endpoint.eq('full')].groupby('model')[cols[1:]].mean()
    mse.to_csv(out/'correlation_prediction_errors.csv')
    return dict(status='PASS',seed_specific_rows=len(cases),model_family_rows=len(merged),
                earthquakes=len(ids),source_components=len(components),comparisons=len(rows),max_error=max(errors))

def diagnostics(root,out):
    base=root/'revisions/20261003_scientific_revision/tables'
    expected=pd.read_csv(base/'reference_comparisons.csv')
    events=pd.read_csv(base/'reference_event_results.csv',dtype={'event_id':str})
    rows=[];errors=[]
    for r in expected.itertuples():
        part=events[events.loading.eq(r.loading)&events.reference.eq(r.reference)]
        wide=part.pivot(index='event_id',columns='model',values=r.metric)
        other=r.comparison.replace('dense_minus_','')
        ans=boot(wide.dense_recurrence-wide[other],20261003)
        errors.append(assert_near(ans,[r.mean,r.low,r.high],'reference comparison'))
        rows.append(dict(loading=r.loading,reference=r.reference,comparison=r.comparison,metric=r.metric,
                         mean=ans[0],low=ans[1],high=ans[2]))
    pd.DataFrame(rows).to_csv(out/'reference_comparisons.csv',index=False)
    ctl=pd.read_csv(base/'control_event_scores.csv').pivot(index='event_id',columns='model',values='score')
    result=[]
    for r in pd.read_csv(base/'control_comparisons.csv').itertuples():
        ans=boot(ctl[r.model]-ctl[r.reference],20261003)
        errors.append(assert_near(ans,[r.mean,r.low,r.high],'support control'))
        result.append(dict(model=r.model,reference=r.reference,mean=ans[0],low=ans[1],high=ans[2]))
    pd.DataFrame(result).to_csv(out/'support_controls.csv',index=False)
    d=pd.read_csv(base/'decomposition_event_band.csv')
    e=d.groupby(['event_id','model'],as_index=False)[['heterogeneity','lag_error','total']].mean()
    e['heterogeneity_fraction']=e.heterogeneity/e.total
    assert_near(e.heterogeneity+e.lag_error,e.total,'decomposition identity')
    summaries=[]
    for r in pd.read_csv(base/'decomposition_summary.csv').itertuples():
        ans=boot(e[e.model.eq(r.model)][r.metric],20261003)
        errors.append(assert_near(ans,[r.mean,r.low,r.high],'decomposition summary'))
        summaries.append(dict(model=r.model,metric=r.metric,mean=ans[0],low=ans[1],high=ans[2]))
    pd.DataFrame(summaries).to_csv(out/'decomposition_summary.csv',index=False)
    # Replay, never regenerate, the saved 16-condition simulation.
    t=pd.read_parquet(base/'synthetic_trials.parquet'); assert len(t)==9600
    sr=[];expected=pd.read_csv(base/'synthetic_contrasts.csv');z=1.959963984540054
    for key,g in t.groupby(['heterogeneous','scenario','snapshots']):
        q=g.pivot(index='replicate',columns='method',values=['score_db','true_snr_db'])
        assert len(q)==200
        ds=q['score_db'].full_context-q['score_db'].fourier_context
        dt=q['true_snr_db'].full_context-q['true_snr_db'].fourier_context
        count=int(np.sum(np.sign(ds)!=np.sign(dt)));rate=count/200
        old=expected[expected.heterogeneous.eq(key[0])&expected.scenario.eq(key[1])&expected.snapshots.eq(key[2])].iloc[0]
        errors.append(assert_near([ds.mean(),dt.mean(),rate],[old.mean_score_difference,old.mean_snr_difference,old.sign_disagreement],'synthetic replay'))
        center=(rate+z*z/400)/(1+z*z/200);half=z*np.sqrt(rate*(1-rate)/200+z*z/(4*200**2))/(1+z*z/200)
        sr.append(dict(heterogeneous=bool(key[0]),scenario=key[1],snapshots=int(key[2]),realizations=200,
                       disagreement_count=count,disagreement_rate=rate,mc_low=center-half,mc_high=center+half,
                       mean_score_difference=float(ds.mean()),mean_snr_difference=float(dt.mean())))
    assert len(sr)==16
    pd.DataFrame(sr).to_csv(out/'synthetic_conditions.csv',index=False)
    return dict(status='PASS',reference_comparisons=len(rows),simulation_conditions=16,simulation_rows=len(t),
                new_simulation_draws=0,max_error=max(errors))

def later_cook(root,out):
    base=root/'revisions/20261010_research_session'
    cases=pd.read_parquet(base/'repeatability/case_metrics.parquet');assert len(cases)==7680
    cols=['first_half_residual_power','second_half_residual_power','residual_cross_product','normalized_repeatability']
    event=cases.groupby(['event_id','band'],as_index=False)[cols].mean()
    saved=pd.read_csv(base/'repeatability/complete_earthquake_metrics.csv')
    errors=[assert_near(event.set_index(['event_id','band']).sort_index()[k],saved.set_index(['event_id','band']).sort_index()[k],'repeatability event') for k in cols]
    rows=[]
    for band,g in event.groupby('band'):
        ans=boot(g.normalized_repeatability,20261010)
        rows.append(dict(band=int(band),mean=ans[0],low=ans[1],high=ans[2]))
    pd.DataFrame(rows).to_csv(out/'repeatability_by_band.csv',index=False)
    events=pd.read_csv(base/'gauge_overlap/all_control_event_scores.csv',dtype={'event_id':str}).pivot(index='event_id',columns='model',values='score')
    records=pd.read_csv(root/'revisions/20260922_unseen_validation/validation_records.csv',dtype={'event_id':str})
    gauge=records.groupby('event_id').gauge_m.first().reindex(events.index)
    results=[]
    for r in pd.read_csv(base/'gauge_overlap/planned_comparisons.csv').itertuples():
        v=events[r.model]-events[r.reference]
        if r.gauge_stratum!='all':v=v[np.isclose(gauge,float(r.gauge_stratum.replace('m','')),atol=.01)]
        ans=boot(v,20261010)
        errors.append(assert_near(ans,[r.mean,r.low,r.high],'gauge-support comparison'))
        results.append(dict(model=r.model,reference=r.reference,gauge_stratum=r.gauge_stratum,mean=ans[0],low=ans[1],high=ans[2]))
    pd.DataFrame(results).to_csv(out/'gauge_support.csv',index=False)
    return dict(status='PASS',repeatability_cases=len(cases),gauge_contrasts=len(results),max_error=max(errors))

def greek(root,out):
    base=root/'revisions/20261010_forecast_processing/development_outcomes'
    frames=[]
    for p in sorted(base.glob('2024-*/cases.parquet')):
        d=pd.read_parquet(p);assert len(d)==97200
        assert not d.duplicated(['cutoff','aperture','delay_step','wavelet_hz','condition','method']).any()
        frames.append(d)
    assert len(frames)==8
    data=pd.concat(frames,ignore_index=True);assert len(data)==777600
    keys=['date','method','condition'];metrics=['snr_gain_db','signal_relative_mse','signal_power_ratio_db']
    daily=data.groupby(keys)[metrics].mean().sort_index()
    expected=pd.read_csv(base/'mismatch_date_summary.csv').set_index(keys).sort_index()
    errors=[assert_near(daily[k],expected[k],f'Greek {k}') for k in metrics]
    daily.to_csv(out/'greek_date_summary.csv')
    matched=daily.xs('matched',level='condition').snr_gain_db.unstack('method').sort_index()
    fractions=[0,.1,.25,.5,.75,1];chosen={}
    for day in matched.index:
        train=matched.drop(index=day)
        chosen[day]=max(fractions,key=lambda f:train[f'context_{f:g}'].mean())
    assert all(v==.75 for v in chosen.values())
    matched['selected_context']=[matched.loc[d,f'context_{chosen[d]:g}'] for d in matched.index]
    matched.to_csv(out/'greek_matched_date_scores.csv')
    results=[]
    for r in json.loads((base/'summary.json').read_text())['comparisons']:
        v=matched.neural-matched[r['comparator']];ans=boot(v,20261010)
        errors.append(assert_near(ans,[r['neural_minus_comparator_db'],r['descriptive_lower95'],r['descriptive_upper95']],'Greek primary contrast'))
        results.append(dict(comparator=r['comparator'],mean=ans[0],low=ans[1],high=ans[2]))
    pd.DataFrame(results).to_csv(out/'greek_primary_contrasts.csv',index=False)
    full=[];pareto=[]
    for condition in CONDITIONS:
        part=daily.xs(condition,level='condition')
        rows=[]
        for method in METHODS:
            v=part.xs(method,level='method');snr=boot(v.snr_gain_db,20261012);err=boot(v.signal_relative_mse,20261012)
            rows.append(dict(condition=condition,method=method,snr_gain_db=snr[0],snr_lower95=snr[1],snr_upper95=snr[2],
                             relative_mse=err[0],mse_lower95=err[1],mse_upper95=err[2],relative_rms_percent=100*np.sqrt(err[0]),
                             signal_power_ratio_db=float(v.signal_power_ratio_db.mean())))
        full.extend(rows)
        if condition!='matched':
            for r in rows:
                better=[o['method'] for o in rows if o['method']!=r['method'] and
                        o['snr_gain_db']>=r['snr_gain_db']-1e-12 and o['relative_mse']<=r['relative_mse']+1e-12 and
                        (o['snr_gain_db']>r['snr_gain_db']+1e-12 or o['relative_mse']<r['relative_mse']-1e-12)]
                pareto.append(dict(condition=condition,method=r['method'],dominated_by='|'.join(better),nondominated=not better))
    summary=pd.DataFrame(full);summary.to_csv(out/'all_method_condition_summary.csv',index=False)
    pd.DataFrame(pareto).to_csv(out/'pareto_checks.csv',index=False)
    old=pd.read_csv(root/'expected/all_method_condition_summary.csv')
    a=summary.set_index(['condition','method']).sort_index();b=old.set_index(['condition','method']).sort_index()
    for k in a.columns:errors.append(assert_near(a[k],b[k],f'Greek grid {k}'))
    return dict(status='PASS',cases=len(data),blocks=len(frames),date_clusters=4,full_grid_rows=len(full),
                selection=chosen,neural_minus_fixed_context025_descriptive=float((matched.neural-matched['context_0.25']).mean()),
                max_error=max(errors))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();root=args.package_root.resolve();out=args.output_dir.resolve()
    assert root.is_dir() and root!=out
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new or empty output directory; existing output is not overwritten.')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'ASSET_MANIFEST.json').read_text())
    for r in manifest['files']:
        p=(root/r['path']).resolve()
        assert p.is_relative_to(root) and p.is_file(),r['path']
        assert sha(p)==r['sha256'],r['path']
    checks={}
    for name,fn in [('cook',cook),('diagnostics',diagnostics),('later_cook',later_cook),('greek',greek)]:
        print('Replaying',name,flush=True);checks[name]=fn(root,out)
    # Export the remaining display source tables without pretending that copying them
    # recomputes waveform extraction, training, or the original figure artwork.
    display=json.loads((root/'DISPLAY_INVENTORY.json').read_text())
    exported=set()
    for item in display['tables']+display['figures']:
        for rel in item['source_assets']:
            asset=root/rel
            assert asset.exists(),rel
            paths=list(asset.rglob('*.csv')) if asset.is_dir() else ([asset] if asset.suffix=='.csv' else [])
            for p in paths:
                key=p.relative_to(root).as_posix()
                if key in exported:continue
                target=out/'display_source_data'/key;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,target);exported.add(key)
        if 'expected_output' in item and item['expected_output'].endswith('.csv'):
            p=root/item['expected_output'];assert p.is_file()
            target=out/'display_source_data'/item['expected_output'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target);exported.add(item['expected_output'])
    # Every display's exact source-data inventory accompanies regenerated core tables.
    shutil.copy2(root/'DISPLAY_INVENTORY.json',out/'DISPLAY_INVENTORY.json')
    report=dict(status='PASS_CLEAN_DERIVED_ASSET_REPLAY',checks=checks,verified_files=len(manifest['files']),
                python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,
                acquisition=False,training=False,new_simulation=False,independent_human_replication=False,
                additional_display_source_csvs_exported=len(exported),
                display_export_scope='Other archived source tables exported unchanged; core reductions recomputed; original artwork not re-extracted.',
                raw_extraction_scope='UNVERIFIED_IN_THIS_CLOSEOUT',full_end_to_end_scope='UNVERIFIED_IN_THIS_CLOSEOUT')
    dump(out/'REPLAY_RECEIPT.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
