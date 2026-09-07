"""Frozen, retrospective audit-versus-geometry validation; never trains models."""
from pathlib import Path
import argparse, hashlib, itertools, json, os, sys, time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from coherencygraph_das import submission_revision as r
OUT=ROOT/'reports/audit_design_utility';SPEC=ROOT/'configs/audit_design_utility.json'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2),encoding='utf8')
def definitions(lags):
    dist=[min(abs(t-d) for d in lags) for t in [4,10]]
    return dict(H0_direct_count=len(set(lags)&{4,10}),H1_mean_nearest_distance=float(np.mean(dist)),H2_max_nearest_distance=max(dist),H3_max_lag=max(lags))
def freeze():
    OUT.mkdir(exist_ok=True,parents=True);assert not (OUT/'frozen_receipt.json').exists(),'Already frozen; do not overwrite'
    cfg=json.loads(SPEC.read_text());choice=json.loads((r.OUT/'locked_design.json').read_text())
    candidates=list(itertools.combinations(cfg['candidate_pool'],cfg['budget']))
    dev=pd.read_csv(r.OUT/'development_design_candidates.csv');dev['tuple']=dev.lags.map(lambda s:tuple(map(int,s.split(','))))
    assert len(dev)==45 and set(dev.tuple)==set(candidates) and not dev.invalid.any()
    selected=min(dev.to_dict('records'),key=lambda row:(row['mean_width'],row['tuple']))['tuple']
    assert list(selected)==choice['lags']==[1,2,3,5,8,13,21,34]
    assert sha(r.SPEC)==choice['protocol_sha256'] and sha(r.MOD/'development_component_tolerance.npy')==choice['delta_sha256']
    _,ds,roles,_=r.bundle();development=sorted(set(ds.event_ids[roles=='model_development']),key=lambda e:hashlib.sha256(f'20260907:{e}'.encode()).hexdigest())[:8]
    assert development==choice['selection_events'];assert not set(development)&set(ds.event_ids[roles=='architecture_test'])
    records=[]
    for i in range(len(ds.event_ids)):
        event=str(ds.event_ids[i]);cohort='development' if event in development else ('retrospective' if roles[i]=='architecture_test' else None)
        if cohort:records.append(dict(event_id=event,route=str(ds.routes[i]),cohort=cohort,path=r.cache_path(ds,i).relative_to(ROOT).as_posix()))
    assert len(records)==64 and len({x['event_id'] for x in records if x['cohort']=='retrospective'})==24
    protected=[r.SPEC,r.OUT/'locked_design.json',r.OUT/'development_design_candidates.csv',r.OUT/'conditional_ranges.parquet',r.OUT/'prediction_event_scores.csv',r.OUT/'processing_comparisons.csv',r.OUT/'unchanged_event_roles.csv',r.MOD/'development_component_tolerance.npy',ROOT/'src/coherencygraph_das/submission_revision.py',ROOT/'src/coherencygraph_das/final_revision.py']
    source={p.relative_to(ROOT).as_posix():sha(p) for p in protected}
    source.update({x['path']:sha(ROOT/x['path']) for x in records})
    candidates=[dict(design_id=f'D{i+1:02d}',lags=list(x),**definitions(x)) for i,x in enumerate(candidates)]
    heur={key:min(candidates,key=lambda d:(d[field],d['lags']))['design_id'] for key,field in [('H1','H1_mean_nearest_distance'),('H2','H2_max_nearest_distance')]}
    receipt=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),specification_sha256=sha(SPEC),script_sha256=sha(Path(__file__)),candidate_count=45,candidates=candidates,selection_event_ids=development,evaluation_lags=[4,10],selected_design=list(selected),selected_by_audit=next(d['design_id'] for d in candidates if d['lags']==list(selected)),heuristic_selected=heur,allowance_sha256=choice['delta_sha256'],protocol_sha256=choice['protocol_sha256'],sources=source,records=records,retrospective_ranking_calculated=False)
    write(OUT/'frozen_receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['sources','records','candidates']},indent=2))
def check_frozen():
    frozen=json.loads((OUT/'frozen_receipt.json').read_text());assert sha(SPEC)==frozen['specification_sha256']
    for p,digest in frozen['sources'].items():assert sha(ROOT/p)==digest,p
    return frozen
def one_design(design,records):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        path=OUT/'ranges'/f"{design['design_id']}.parquet"
        if path.exists():return design['design_id'],'cached'
        delta=np.load(r.MOD/'development_component_tolerance.npy');lags=design['lags'];ids=[list(r.historical.MEASURE).index(d) for d in lags];rows=[]
        for record in records:
            with np.load(ROOT/record['path']) as z:target=z['target_gamma'].mean(0)
            for band in range(4):
                for bounds in r.conditional_ranges(target[band,ids],lags,delta[band,ids].T.reshape(-1)):
                    rows.append(dict(design_id=design['design_id'],event_id=record['event_id'],route=record['route'],cohort=record['cohort'],band=band,**bounds))
        pd.DataFrame(rows).to_parquet(path,index=False)
        return design['design_id'],len(rows)
def evaluate(workers):
    frozen=check_frozen();(OUT/'ranges').mkdir(exist_ok=True)
    if not (OUT/'evaluation_started.json').exists():write(OUT/'evaluation_started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),receipt_sha256=sha(OUT/'frozen_receipt.json')))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(one_design,d,frozen['records']) for d in frozen['candidates']]
        for k,f in enumerate(as_completed(jobs),1):print(k,'/45',f.result(),flush=True)
def event_average(g,column):
    return g.groupby(['event_id','route'])[column].mean().groupby('event_id').mean().mean()
def summarise():
    f=check_frozen();assert len(list((OUT/'ranges').glob('D*.parquet')))==45
    ranges=pd.concat([pd.read_parquet(OUT/'ranges'/f"{d['design_id']}.parquet") for d in f['candidates']],ignore_index=True)
    ranges.to_parquet(OUT/'all_design_ranges.parquet',index=False)
    ranges['objective_width']=ranges.width.where(ranges.valid,2.)
    dev=pd.read_csv(r.OUT/'development_design_candidates.csv').set_index('lags');rows=[]
    for d in f['candidates']:
        g=ranges[ranges.design_id==d['design_id']];retro=g[g.cohort=='retrospective'];real=retro[retro.axis=='real'];imag=retro[retro.axis=='imaginary'];development=g[(g.cohort=='development')&(g.axis=='real')]
        key=','.join(map(str,d['lags']));old=float(dev.loc[key,'mean_width']);recomputed=event_average(development,'objective_width');assert abs(old-recomputed)<1e-9,(d['design_id'],old,recomputed)
        rows.append(dict(design_id=d['design_id'],lag_set=key,J_dev_real_width=old,J_dev_recomputed=recomputed,J_retro_real_width=event_average(real,'objective_width'),retro_real_identified_fraction=event_average(real,'identified'),retro_imag_width=event_average(imag,'objective_width'),retro_imag_identified_fraction=event_average(imag,'identified'),valid_case_fraction=event_average(retro,'valid'),failure_fraction=1-event_average(retro,'fit_converged'),**definitions(d['lags']),selected_by_audit=d['design_id']==f['selected_by_audit'],selected_by_H1=d['design_id']==f['heuristic_selected']['H1'],selected_by_H2=d['design_id']==f['heuristic_selected']['H2']))
    table=pd.DataFrame(rows)
    for column,name in [('J_dev_real_width','dev_rank'),('J_retro_real_width','retro_rank')]:
        ordered=table.sort_values([column,'design_id']);rank={k:i+1 for i,k in enumerate(ordered.design_id)};table[name]=table.design_id.map(rank)
    table.to_csv(OUT/'all45_designs.csv',index=False)
    selected=table[table.selected_by_audit].iloc[0];best=table.loc[table.J_retro_real_width.idxmin()];rho=float(spearmanr(table.J_dev_real_width,table.J_retro_real_width).statistic)
    associations={}
    for field in ['H0_direct_count','H1_mean_nearest_distance','H2_max_nearest_distance','H3_max_lag']:
        associations[field]=dict(unique_values=int(table[field].nunique()),rho=None if table[field].nunique()==1 else float(spearmanr(table[field],table.J_retro_real_width).statistic),interpretation='descriptive; no p-value')
    choices={'Audit':f['selected_by_audit'],**f['heuristic_selected'],'Retrospective best':best.design_id}
    historical=','.join(map(str,r.CFG['identifiability']['historical_lags']));choices['Historical']=table[table.lag_set==historical].iloc[0].design_id
    summary=[]
    for rule,ident in choices.items():
        row=table[table.design_id==ident].iloc[0];summary.append(dict(selection_rule=rule,design_id=ident,lag_set=row.lag_set,J_retro=float(row.J_retro_real_width),retro_rank=int(row.retro_rank),gap_to_best=float(row.J_retro_real_width-best.J_retro_real_width)))
    pd.DataFrame(summary).to_csv(OUT/'selection_summary.csv',index=False)
    chosen_geometry=[table[table.design_id==f['heuristic_selected'][h]].iloc[0] for h in ['H1','H2']]
    if selected.retro_rank<=5 and rho>=.5 and any(abs(v.J_retro_real_width/selected.J_retro_real_width-1)<=.01 for v in chosen_geometry):outcome='B'
    elif selected.retro_rank<=5 and rho>=.5 and all(v.design_id!=selected.design_id and v.J_retro_real_width/selected.J_retro_real_width-1>=.05 for v in chosen_geometry):outcome='A'
    else:outcome='C'
    lag_ranks=[]
    for lag in [4,10]:
        values={d:event_average(g,'objective_width') for d,g in ranges[(ranges.cohort=='retrospective')&(ranges.axis=='real')&(ranges.lag==lag)].groupby('design_id')};order=sorted(values,key=lambda k:(values[k],k));lag_ranks.append(dict(lag=lag,selected_design=f['selected_by_audit'],rank=order.index(f['selected_by_audit'])+1,J=values[f['selected_by_audit']]))
    gap=float(selected.J_retro_real_width-best.J_retro_real_width)
    receipt=dict(outcome=outcome,selected_design=f['selected_design'],dev_rank=int(selected.dev_rank),retro_rank=int(selected.retro_rank),retro_percentile=100*(45-int(selected.retro_rank))/44,J_retro=float(selected.J_retro_real_width),best_J_retro=float(best.J_retro_real_width),gap_to_best=gap,relative_gap_percent=100*gap/best.J_retro_real_width,median_candidate_J=float(table.J_retro_real_width.median()),rho_descriptive=rho,top5_overlap=len(set(table.nsmallest(5,'dev_rank').design_id)&set(table.nsmallest(5,'retro_rank').design_id)),top10_overlap=len(set(table.nsmallest(10,'dev_rank').design_id)&set(table.nsmallest(10,'retro_rank').design_id)),selected_in_top={str(n):bool(selected.retro_rank<=n) for n in [1,5,10,12]},heuristic_associations=associations,selected_comparisons=summary,lag_sensitivity=lag_ranks,invalid_cases=int((~ranges.valid).sum()),processing_connection=json.loads(SPEC.read_text())['processing_connection'])
    write(OUT/'ranking_receipt.json',receipt);print(json.dumps(receipt,indent=2))
    # Independent low-level aggregation: explicit loops over endpoints, routes,
    # then complete earthquakes; no candidate table values in reconstruction.
    checked=[]
    for ident in sorted(set(['D01','D23','D45']+list(choices.values()))):
        raw=pd.read_parquet(OUT/'ranges'/f'{ident}.parquet');a=raw[(raw.cohort=='retrospective')&(raw.axis=='real')];values=[]
        for event in sorted(a.event_id.unique()):
            route_means=[]
            for route in ['TERRA','KKFL-S']:
                cells=a[(a.event_id==event)&(a.route==route)];assert len(cells)==8
                widths=[float(v.upper-v.lower) if v.valid else 2. for v in cells.itertuples()];route_means.append(sum(widths)/8)
            values.append(sum(route_means)/2)
        value=sum(values)/24;expected=float(table[table.design_id==ident].iloc[0].J_retro_real_width);assert abs(value-expected)<1e-12;checked.append(dict(design_id=ident,J_retro_recomputed=value,passed=True))
    # Confirm the previously reported selected/historical design ranges are not altered.
    old=pd.read_parquet(r.OUT/'conditional_ranges.parquet');comparisons=[]
    for label,ident in [('algorithmic',f['selected_by_audit']),('historical',choices['Historical'])]:
        new=ranges[(ranges.cohort=='retrospective')&(ranges.design_id==ident)]
        pair=new.merge(old[old.design==label],on=['event_id','route','band','lag','axis'],suffixes=('_new','_old'));assert len(pair)==768
        error=float(np.max(abs(pair.width_new-pair.width_old)));assert error<1e-9;comparisons.append(dict(design=label,max_width_difference=error))
    assert len(ranges)==46080 and ranges.groupby(['design_id','cohort','axis']).size().isin([128,384]).all()
    assert datetime.fromisoformat(f['frozen_utc'])<datetime.fromisoformat(json.loads((OUT/'evaluation_started.json').read_text())['started_utc'])
    check_frozen();write(OUT/'validation.json',dict(status='PASS',unique_eight_lag_designs=45,targets_absent=True,development_scores_preserved=True,all_sources_unchanged=True,lower_level_checks=checked,existing_range_checks=comparisons,cases=len(ranges),invalid_cases=int((~ranges.valid).sum()),frozen_before_evaluation=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','evaluate','summarise']);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='evaluate':evaluate(args.workers)
    else:summarise()
