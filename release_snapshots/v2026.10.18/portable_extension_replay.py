"""Independent saved-table replay of the two finite October18 panels.

No scientific scorer, summary helper, waveform array, parameter fitting, filter
execution, new acquisition or held-out evaluation is imported or run.
"""
from pathlib import Path
import argparse,json,math,time
import numpy as np
import pandas as pd

PANEL='revisions/20261018_no_human_closeout/implementation'
KEY=['block','cutoff','aperture','delay_step','wavelet_hz','condition','method']
CONDITIONS=['matched','delay_minus','delay_plus','amplitude_minus','amplitude_plus']
REC_METHODS=['neural','ridge','conventional','context_0.25','context_0.75','lcmv_context_0.25']
REC_METRICS=['component_ratio_gain_db','recorded_component_relative_mse',
    'component_power_relative_reference_db','background_output_power',
    'recorded_component_output_power','reference_to_input_power_ratio']
TOL=1e-10

def near(a,b,label):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(label+' nonfinite or misaligned')
    error=float(np.max(np.abs(a-b)/np.maximum(1,np.abs(b)),initial=0))
    if error>TOL:raise AssertionError(f'{label}: {error} exceeds {TOL}')
    return error

def q7(x,q):
    a=np.sort(np.asarray(x,float));rank=(len(a)-1)*q;lo,hi=math.floor(rank),math.ceil(rank)
    return float(a[lo]+(rank-lo)*(a[hi]-a[lo]))

def compare(frame,path,keys):
    expected=pd.read_csv(path)
    if set(frame.columns)!=set(expected.columns):raise AssertionError(str(path)+' column mismatch')
    actual=frame.set_index(keys).sort_index();saved=expected.set_index(keys).sort_index()
    if actual.index.has_duplicates or saved.index.has_duplicates or not actual.index.equals(saved.index):
        raise AssertionError(str(path)+' keyed rows mismatch')
    errors=[]
    for column in actual.columns:
        a,b=actual[column],saved[column]
        if pd.api.types.is_bool_dtype(a.dtype):
            if not np.array_equal(a.to_numpy(),b.to_numpy()):raise AssertionError(column+' boolean mismatch')
        elif pd.api.types.is_numeric_dtype(a.dtype):errors.append(near(a,b,path.name+'/'+column))
        else:
            if not np.array_equal(a.fillna('').astype(str),b.fillna('').astype(str)):
                raise AssertionError(column+' string mismatch')
    return dict(file=path.name,rows=len(frame),maximum_scaled_difference=max(errors,default=0.))

def nondominated(gain,error):
    result=[]
    for i,(g,e) in enumerate(zip(gain,error)):
        found=False
        for j,(og,oe) in enumerate(zip(gain,error)):
            if i!=j and og>=g-1e-12 and oe<=e+1e-12 and (og>g+1e-12 or oe<e-1e-12):
                found=True;break
        result.append(not found)
    return result

def ricker(root,out):
    panel=root/PANEL;old=root/'revisions/20261010_forecast_processing/development_outcomes'
    old_frames=[pd.read_parquet(p) for p in sorted(old.glob('2024-*/cases.parquet'))]
    assert len(old_frames)==8 and all(len(a)==97200 for a in old_frames)
    original=pd.concat(old_frames,ignore_index=True)
    extra=pd.read_parquet(panel/'lcmv_ricker_results/all_new_cases.parquet')
    assert len(original)==777600 and len(extra)==86400
    assert set(extra.method)=={'derivative_lcmv_context_0.25'}
    assert set(original.method).isdisjoint(set(extra.method))
    data=pd.concat([original,extra],ignore_index=True)
    assert len(data)==864000 and not data.duplicated(KEY).any()
    assert data.method.nunique()==10 and data.date.nunique()==4
    formula=max(near(data.output_snr_db,10*np.log10(data.signal_output_power/data.background_output_power),'Ricker power formula'),
        near(data.snr_gain_db,data.output_snr_db-data.input_snr_db,'Ricker gain formula'))
    records=[]
    for (date,condition,method),g in data.groupby(['date','condition','method'],sort=True):
        assert len(g)==4320
        records.append(dict(date=date,condition=condition,method=method,rows=len(g),
            snr_gain_mean=float(g.snr_gain_db.mean()),relative_mse_mean=float(g.signal_relative_mse.mean()),
            snr_gain_q05=q7(g.snr_gain_db,.05),relative_mse_q95=q7(g.signal_relative_mse,.95)))
    table=pd.DataFrame(records);assert len(table)==200
    checks=[compare(table,panel/'lcmv_ricker_results/expanded_date_condition_table.csv',['date','condition','method'])]
    table.to_csv(out/'expanded_date_condition_table.csv',index=False)
    table['mean_nondominated']=False;table['marginal_tail_nondominated']=False
    for (_,condition),indices in table.groupby(['date','condition'],sort=True).groups.items():
        if condition=='matched':continue
        g=table.loc[indices]
        table.loc[indices,'mean_nondominated']=nondominated(g.snr_gain_mean.tolist(),g.relative_mse_mean.tolist())
        table.loc[indices,'marginal_tail_nondominated']=nondominated(g.snr_gain_q05.tolist(),g.relative_mse_q95.tolist())
    checks.append(compare(table,panel/'finite_panel_summary/expanded_ricker_frontier_all_dates.csv',['date','condition','method']))
    table.to_csv(out/'expanded_ricker_frontier_all_dates.csv',index=False)
    counts=table[table.condition.ne('matched')].groupby(['condition','method'],sort=True).agg(
        mean_nondominated_dates=('mean_nondominated','sum'),tail_nondominated_dates=('marginal_tail_nondominated','sum')).reset_index()
    assert len(counts)==40
    checks.append(compare(counts,panel/'finite_panel_summary/expanded_ricker_frontier_counts.csv',['condition','method']))
    counts.to_csv(out/'expanded_ricker_frontier_counts.csv',index=False)
    return dict(status='PASS',old_rows=777600,new_rows=86400,combined_rows=len(data),date_condition_method_rows=len(table),
        frontier_count_rows=len(counts),formula_maximum_scaled_difference=formula,csv_comparisons=checks,
        original_rows_replaced=False)

def recorded(root,out):
    panel=root/PANEL
    cases=pd.read_parquet(panel/'recorded_component_results/all_cases.parquet')
    assert len(cases)==23040 and not cases.duplicated(['source_event','block','cutoff','aperture','method']).any()
    assert cases.source_event.nunique()==2 and cases.source_recording_date.nunique()==1
    assert cases.background_date.nunique()==4 and cases.block.nunique()==8 and set(cases.method)==set(REC_METHODS)
    assert set(cases.cutoff)==set(range(120,3601,120)) and set(cases.aperture)==set(range(8))
    assert (cases.groupby(['source_event','block','method']).size()==240).all()
    ps=cases.recorded_component_output_power;pn=cases.background_output_power
    pref=cases.conventional_reference_output_power;pin_s=cases.recorded_component_input_power;pin_n=cases.background_input_power
    assert min(ps.min(),pn.min(),pref.min(),pin_s.min(),pin_n.min())>0
    formula=max(near(cases.recorded_component_relative_mse,cases.output_error_power/pref,'Recorded relative error'),
        near(cases.reference_to_input_power_ratio,pref/pin_s,'Recorded cancellation ratio'),
        near(cases.component_input_ratio_db,10*np.log10(pin_s/pin_n),'Recorded input ratio'),
        near(cases.component_output_ratio_db,10*np.log10(ps/pn),'Recorded output ratio'),
        near(cases.component_ratio_gain_db,cases.component_output_ratio_db-cases.component_input_ratio_db,'Recorded gain'),
        near(cases.component_power_relative_reference_db,10*np.log10(ps/pref),'Recorded reference power'))
    assert np.isfinite(cases.linearity_scaled_error).all() and cases.linearity_scaled_error.max()<1e-10
    hours=cases.groupby(['source_event','block','background_date','method'],sort=True)[REC_METRICS].mean().reset_index()
    dates=hours.groupby(['source_event','background_date','method'],sort=True)[REC_METRICS].mean().reset_index()
    sources=dates.groupby(['source_event','method'],sort=True)[REC_METRICS].mean().reset_index()
    checks=[]
    for frame,name,keys in [(hours,'recorded_source_hour_method.csv',['source_event','block','background_date','method']),
        (dates,'recorded_source_date_method.csv',['source_event','background_date','method']),
        (sources,'recorded_source_method.csv',['source_event','method'])]:
        checks.append(compare(frame,panel/'finite_panel_summary'/name,keys));frame.to_csv(out/name,index=False)
    cancellation=cases.groupby(['source_event','aperture'],sort=True).agg(
        reference_input_ratio_min=('reference_to_input_power_ratio','min'),
        reference_input_ratio_max=('reference_to_input_power_ratio','max'),
        component_input_power=('recorded_component_input_power','mean'),
        conventional_reference_power=('conventional_reference_output_power','mean')).reset_index()
    assert len(cancellation)==16
    checks.append(compare(cancellation,panel/'finite_panel_summary/all_reference_cancellation_diagnostics.csv',['source_event','aperture']))
    cancellation.to_csv(out/'all_reference_cancellation_diagnostics.csv',index=False)
    records=[]
    for (source,date),g in dates.groupby(['source_event','background_date'],sort=True):
        table=g.set_index('method');neural=table.loc['neural']
        for method in REC_METHODS[1:]:
            records.append(dict(source_event=source,background_date=date,comparator=method,
                neural_minus_component_ratio_gain_db=float(neural.component_ratio_gain_db-table.loc[method].component_ratio_gain_db),
                neural_minus_relative_mse=float(neural.recorded_component_relative_mse-table.loc[method].recorded_component_relative_mse)))
    paired=pd.DataFrame(records);assert len(paired)==40
    checks.append(compare(paired,panel/'finite_panel_summary/recorded_neural_paired_date_contrasts.csv',['source_event','background_date','comparator']))
    paired.to_csv(out/'recorded_neural_paired_date_contrasts.csv',index=False)
    return dict(status='PASS',rows=len(cases),source_events=2,source_recording_dates=1,background_dates=4,
        hour_method_rows=len(hours),date_method_rows=len(dates),source_method_rows=len(sources),
        cancellation_rows=16,paired_contrast_rows=40,formula_maximum_scaled_difference=formula,
        maximum_retained_linearity_scaled_error=float(cases.linearity_scaled_error.max()),csv_comparisons=checks,
        interpretation='Known added recorded-wavefield component relative to conventional output, not clean earthquake recovery or independent eight-event evidence.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.package_root.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError('Preserve earlier extension replay')
    out.mkdir(parents=True);started=time.perf_counter()
    checks=dict(ricker=ricker(root,out),recorded=recorded(root,out))
    report=dict(status='PASS_PORTABLE_TWO_PANEL_SAVED_TABLE_REPLAY',checks=checks,tolerance=TOL,
        elapsed_seconds=time.perf_counter()-started,scorer_or_summary_imported=False,raw_or_processed_waveforms_read=False,
        filters_reexecuted=False,new_training=False,new_acquisition=False,heldout_assessment=False,
        independent_human_replication=False,scope='Recompute derived statistics/formulas/frontiers from saved scalar Parquets; waveform-to-case replay is separate and was not rerun.')
    (out/'EXTENSION_REPLAY_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
