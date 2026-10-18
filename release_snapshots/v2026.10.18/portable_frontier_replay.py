"""Independent portable recomputation of October13 saved-case frontiers.

No model, waveform, acquisition, fitting, threshold selection or bootstrap is run.
This does not import the audited frontier analysis or its helper functions.
"""
from pathlib import Path
import argparse,json,math,time
import numpy as np
import pandas as pd

METHODS=['context_0','context_0.1','context_0.25','context_0.5','context_0.75',
         'context_1','conventional','neural','ridge']
CONDITIONS=['matched','delay_minus','delay_plus','amplitude_minus','amplitude_plus']
KEY=['block','cutoff','aperture','delay_step','wavelet_hz','condition']
TOL=1e-12

def q7(x,p):
    a=np.sort(np.asarray(x,float));r=(len(a)-1)*p;i,j=math.floor(r),math.ceil(r)
    return float(a[i]+(r-i)*(a[j]-a[i]))

def near(a,b,label):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(label+' invalid aligned arrays')
    error=float(np.max(np.abs(a-b)/np.maximum(1,np.abs(b)),initial=0))
    if error>1e-10:raise AssertionError(f'{label}: {error}')
    return error

def classify(g,e):
    if abs(g)<=TOL and abs(e)<=TOL:return 'numerical_tie'
    if g>=-TOL and e<=TOL and (g>TOL or e<-TOL):return 'neural_dominance'
    if g<=TOL and e>=-TOL and (g<-TOL or e>TOL):return 'comparator_dominance'
    return 'tradeoff'

def replay(root,out):
    started=time.perf_counter()
    expected=root/'revisions/20261013_research_reassessment/frontier_stability'
    source=root/'revisions/20261010_forecast_processing/development_outcomes'
    saved_metric=pd.read_csv(expected/'date_method_metrics.csv')
    saved_front=pd.read_csv(expected/'date_frontiers.csv',keep_default_na=False)
    saved_pairs=pd.read_csv(expected/'exact_paired_fractions.csv')
    metrics=[];frontiers=[];partitions=[];errors=[];rows=0
    for day in [f'2024-08-{d:02d}' for d in range(4,8)]:
        frames=[pd.read_parquet(source/f'{day}_{h:02d}/cases.parquet') for h in (0,12)]
        assert all(len(a)==97200 for a in frames)
        data=pd.concat(frames,ignore_index=True);rows+=len(data)
        assert not data.duplicated(KEY+['method']).any()
        errors.append(near(data.output_snr_db,10*np.log10(data.signal_output_power/data.background_output_power),'power formula'))
        errors.append(near(data.snr_gain_db,data.output_snr_db-data.input_snr_db,'gain formula'))
        records={}
        for condition in CONDITIONS:
            for method in METHODS:
                a=data[data.condition.eq(condition)&data.method.eq(method)]
                assert len(a)==4320 and a.groupby('block').size().tolist()==[2160,2160]
                record=dict(date=day,condition=condition,method=method,
                    snr_gain_db=float(a.snr_gain_db.mean()),signal_relative_mse=float(a.signal_relative_mse.mean()),
                    snr_gain_q05_db=q7(a.snr_gain_db,.05),signal_error95=q7(a.signal_relative_mse,.95))
                old=saved_metric[saved_metric.date.eq(day)&saved_metric.condition.eq(condition)&saved_metric.method.eq(method)]
                assert len(old)==1
                columns=['snr_gain_db','signal_relative_mse','snr_gain_q05_db','signal_error95']
                errors.append(near([record[k] for k in columns],old[columns].iloc[0].tolist(),'date/method metrics'))
                metrics.append(record);records[(condition,method)]=record
        for condition in CONDITIONS[1:]:
            for kind,columns in [('mean',('snr_gain_db','signal_relative_mse')),
                                 ('marginal_tail',('snr_gain_q05_db','signal_error95'))]:
                for method in METHODS:
                    gain,error=[records[(condition,method)][k] for k in columns]
                    dominators=[]
                    for other in METHODS:
                        if other==method:continue
                        og,oe=[records[(condition,other)][k] for k in columns]
                        if og>=gain-TOL and oe<=error+TOL and (og>gain+TOL or oe<error-TOL):dominators.append(other)
                    old=saved_front[saved_front.date.eq(day)&saved_front.condition.eq(condition)&
                        saved_front.frontier.eq(kind)&saved_front.method.eq(method)]
                    assert len(old)==1
                    assert bool(old.iloc[0].nondominated)==(not dominators)
                    assert set(old.iloc[0].dominators.split('|'))-{''}==set(dominators)
                    errors.append(near([gain,error],old[['gain_coordinate_db','mse_coordinate']].iloc[0].tolist(),'frontier coordinate'))
                    frontiers.append(dict(date=day,condition=condition,method=method,frontier=kind,
                        nondominated=not dominators,dominators='|'.join(dominators),gain_coordinate_db=gain,mse_coordinate=error))
        for condition in CONDITIONS:
            neural=data[data.condition.eq(condition)&data.method.eq('neural')].set_index(KEY).sort_index()
            for method in METHODS:
                if method=='neural':continue
                other=data[data.condition.eq(condition)&data.method.eq(method)].set_index(KEY).sort_index()
                assert neural.index.equals(other.index) and len(neural)==4320
                np.testing.assert_array_equal(neural.input_snr_db,other.input_snr_db)
                gains=neural.snr_gain_db.to_numpy()-other.snr_gain_db.to_numpy()
                err=neural.signal_relative_mse.to_numpy()-other.signal_relative_mse.to_numpy()
                counts=dict.fromkeys(['neural_dominance','comparator_dominance','numerical_tie','tradeoff'],0)
                for g,e in zip(gains,err):counts[classify(float(g),float(e))]+=1
                old=saved_pairs[saved_pairs.date.eq(day)&saved_pairs.condition.eq(condition)&saved_pairs.comparator.eq(method)]
                assert len(old)==1 and sum(counts.values())==4320
                for name,count in counts.items():
                    assert int(old.iloc[0][name+'_count'])==count
                    errors.append(abs(float(old.iloc[0][name+'_fraction'])-count/4320))
                partitions.append(dict(date=day,condition=condition,comparator=method,cases=4320,**counts))
    assert rows==777600 and len(metrics)==180 and len(frontiers)==288 and len(partitions)==160
    metric_frame=pd.DataFrame(metrics);front_frame=pd.DataFrame(frontiers)
    metric_frame.to_csv(out/'frontier_date_method_metrics.csv',index=False)
    front_frame.to_csv(out/'frontier_date_memberships.csv',index=False)
    pd.DataFrame(partitions).to_csv(out/'frontier_paired_partitions.csv',index=False)
    stability=front_frame.groupby(['frontier','condition','method']).nondominated.sum().reset_index(name='nondominated_dates')
    stability.to_csv(out/'frontier_date_stability.csv',index=False)
    report=dict(status='PASS_ALL_FOUR_DATE_PORTABLE_FRONTIER_REPLAY',archived_rows=rows,
        date_method_summaries=180,frontier_memberships=288,paired_partitions=160,
        maximum_scaled_difference=max(errors),elapsed_seconds=time.perf_counter()-started,
        acquisition=False,training=False,new_waveforms=False,independent_human_replication=False,
        audited_analysis_imported=False,interpretation='Descriptive finite-grid and marginal-tail findings; no joint safety probability or statistical superiority inferred.')
    (out/'FRONTIER_REPLAY_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if (a.output_dir/'FRONTIER_REPLAY_RECEIPT.json').exists():raise FileExistsError('Preserve replay receipt')
    print(json.dumps(replay(a.package_root.resolve(),a.output_dir.resolve()),indent=2))
