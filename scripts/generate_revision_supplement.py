"""Generate supplemental tables and all additional manuscript numbers."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from coherencygraph_das.critical_revision import OUT,MODELS,boot
from coherencygraph_das.critical_paper import GEN,latex_table,LABELS

info=json.loads((OUT/'manuscript_summary.json').read_text())
changes=[]
historical=pd.read_csv(OUT/'historical_number_check.csv')
for r in historical.itertuples():changes.append(dict(quantity=r.model+' '+r.role+' historical NRMSE',before=r.old_nrmse,after=r.new_nrmse,interpretation='Exact regeneration; same historical endpoint'))
for r in pd.read_csv(OUT/'matched_summary.csv').itertuples():changes.append(dict(quantity=r.model+' revised all-cell NRMSE',before=np.nan,after=r.mean,interpretation='New retrospective comparator; not the historical primary'))
for r in pd.read_csv(OUT/'downstream_summary.csv').itertuples():
    if r.reference=='Diagonal':changes.append(dict(quantity=f'{r.method}; {r.frame}; {r.evaluation}; difference dB',before=np.nan,after=r.mean,interpretation=f'Component interval [{r.low},{r.high}]'))
pd.DataFrame(changes).to_csv(OUT/'NUMERICAL_CHANGELOG.csv',index=False)
support=pd.read_parquet(OUT/'raw_spectral_support.parquet');timing=pd.read_parquet(OUT/'raw_timing.parquet')
support=support.merge(timing[['event_id','route','block','s_reference','global_cutoff']],on=['event_id','route','block'],validate='many_to_one')
bands=[(.5,1),(1,2),(2,4),(4,8)];expanded=[]
for r in support.itertuples():
    first={'noise':.5,'context':r.s_reference-12,'target':r.s_reference+3,'equal_7s':r.s_reference-5,'global_cutoff':r.global_cutoff-7}[r.window]
    first=max(0,round(first*25));hz=np.fft.rfftfreq(r.samples,1/25)
    for b,(low,high) in enumerate(bands):
        bins=int(((hz>=low)&(hz<high)).sum());expanded.append(dict(event_id=r.event_id,route=r.route,block=r.block,window=r.window,
            sample_start=first,sample_stop_exclusive=first+r.samples,samples=r.samples,band=b,bins=bins,tapers=3,snapshots=3*bins,
            equal_snapshot_weight=1/(3*bins),half_bandwidth_hz=r.half_bandwidth_hz))
pd.DataFrame(expanded).to_parquet(OUT/'raw_window_band_support.parquet',index=False)
cohort=pd.read_csv(OUT/'cohort_summary.csv');dist=pd.read_csv(OUT/'cross_role_distances.csv')
temporal=json.loads((OUT/'temporal_split_specification.json').read_text())
sen=pd.read_csv(OUT/'estimator_sensitivity_summary.csv');r=sen[(sen.variant=='global_cutoff')&(sen.model=='state_psd')].iloc[0]
coverage=pd.read_csv(OUT/'revised_joint_coverage.csv');cv=coverage[coverage.unit=='earthquake']
vals=dict(TestComponents=str(int(cohort[cohort.role=='architecture_test'].components_25km.iloc[0])),
    NearestDistance=f'{dist[(dist.role_1=="architecture_test")&(dist.role_2=="model_development")].minimum_hypocentral_km.iloc[0]:.2f}',
    RawRecords=str(info['timing']['records']),LateBlocks=str(info['timing']['nonpositive_route_leads']),TimingBlocks=str(info['timing']['blocks']),
    CutoffDelta=f'{r.delta_mean:.4f}',CutoffLow=f'{r.delta_low:.4f}',CutoffHigh=f'{r.delta_high:.4f}',
    TemporalTrain=str(len(temporal['training_events'])),TemporalTest=str(len(temporal['test_events'])),
    OffDiagonalCount=str(int(info['geometry']['ordered_offdiagonal'])),BeyondLagCount=str(int(info['geometry']['beyond_max_lag'])))
for nominal,key in [(.8,'CoverageEighty'),(.9,'CoverageNinety'),(.95,'CoverageNinetyFive')]:
    r=cv[cv.nominal==nominal].iloc[0];vals[key]=f'{int(r.successes)}/{int(r.test_units)}'
GEN.joinpath('additional_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in vals.items())+'\n')
# A compact history table preserves the original endpoint exactly.
h=pd.read_csv(OUT/'historical_number_check.csv');h=h[h.role=='architecture_test']
latex_table(GEN/'s_historical.tex',['Historical model','Old NRMSE','Regenerated'],[
    [r.model,f'{r.old_nrmse:.6f}',f'{r.new_nrmse:.6f}'] for r in h.itertuples()])
compute=pd.read_csv(OUT/'matched_compute.csv');rows=[]
for name,g in compute.groupby('kind',sort=False):
    epochs='--' if g.selected_epoch.isna().all() else '/'.join(str(int(x)) for x in g.selected_epoch)
    rows.append([LABELS.get(name,name),f'{int(g.parameters.iloc[0]):,}',epochs,f'{g.training_seconds.sum():.1f}'])
latex_table(GEN/'s_compute.tex',['Model','Parameters','Epochs by seed','Total fit (s)'],rows)
latex_table(GEN/'s_sensitivity.tex',['Input / target','Difference vs full ridge','Component interval'],[
    [r.variant.replace('_',' ').replace('normalize','normalise'),f'{r.delta_mean:.4f}',f'[{r.delta_low:.4f}, {r.delta_high:.4f}]'] for r in sen[sen.model=='state_psd'].itertuples()])
inter=pd.read_csv(OUT/'internal_summary.csv');rows=[]
for scope in inter.scope.unique():
    for r in inter[(inter.scope==scope)&(inter.model.isin(['full_ridge','state_psd']))].itertuples():
        name={'pooled_group_folds':'Four component folds','chronological':'Latest-group-date stress','purged_temporal':'Purged temporal stress'}[scope]
        rows.append([name,'Ridge' if r.model=='full_ridge' else 'State PSD',f'{r.mean_nrmse:.4f}',f'{r.delta_low:.4f}, {r.delta_high:.4f}'])
latex_table(GEN/'s_internal.tex',['Validation','Model','NRMSE','Paired interval vs ridge'],rows)
missing=pd.read_csv(OUT/'missingness_summary.csv');rows=[]
names={'historical_unaugmented':'Historical, no augmentation','historical_mask_dropout':'Historical, mask + dropout','indicator_only':'Revised indicator only','mlp_imputation':'Revised MLP imputation'}
for name,g in missing.groupby('model',sort=False):
    for k in [1,2,4]:
        part=g[g.missing_count==k].set_index('endpoint')
        rows.append([names[name],k,*[f'{part.loc[e,"mean"]:.3f}' for e in ['all_blocks','missing_only','retained_only']]])
latex_table(GEN/'s_missingness.tex',['Model','Removed','All','Missing','Retained'],rows)
latex_table(GEN/'s_coverage.tex',['Unit','Nominal','Rank','Threshold','Joint success'],[
    [r.unit,f'{r.nominal:.0%}'.replace('%',r'\%'),int(r.order),r'$\infty$' if np.isinf(r.threshold) else f'{r.threshold:.4f}',f'{r.successes}/{r.test_units}'] for r in coverage.itertuples()])
rank=pd.read_csv(OUT/'uncertainty_ranking.csv')
latex_table(GEN/'s_ranking.tex',['Score','Spearman correlation','Component interval'],[[r.score,f'{r.spearman:.3f}',f'[{r.low:.3f}, {r.high:.3f}]'] for r in rank.itertuples()])
down=pd.read_csv(OUT/'downstream_summary.csv');part=down[(down.frame=='corrected_frame')&(down.reference=='Diagonal')&(down.method!='Diagonal')]
latex_table(GEN/'s_downstream.tex',['Evaluation','Matrix','Delta (dB)','Component interval'],[
    ['Full target' if r.evaluation=='full_window_in_sample' else 'Two-way half test',r.method.replace('Learned corrected sign','Historical learned').replace('Learned revised recurrence','Revised learned'),f'{r.mean:.4f}',f'[{r.low:.4f}, {r.high:.4f}]'] for r in part.itertuples()])
syn=pd.read_csv(OUT/'stationary_estimator_summary.csv');rows=[]
for (duration,est),g in syn.groupby(['duration','estimator']):
    rows.append([f'{duration:g}',est.replace('pair_normalized','Pair normalised').replace('pooled_power','Power pooled'),*[f'{x:.4f}' for x in g.sort_values('band').mse]])
latex_table(GEN/'s_stationary.tex',['Duration (s)','Estimator','0.5--1','1--2','2--4','4--8'],rows)
# Summarise sub-block estimates at the complete-earthquake level; no pseudoreplication.
sub=pd.read_parquet(OUT/'raw_subblocks.parquet');sub['squared_difference']=(sub.real-sub.full_real)**2+(sub.imag-sub.full_imag)**2
subsum=sub.groupby(['event_id','subblock_channels']).squared_difference.mean().reset_index();subsum.to_csv(OUT/'subblock_event_disagreement.csv',index=False)
latex_table(GEN/'s_subblocks.tex',['Channels','Pairs at lag 3','Pairs at lag 89','Mean squared disagreement'],[
    [int(n),int(n)-3,int(n)-89,f'{g.squared_difference.mean():.5f}'] for n,g in subsum.groupby('subblock_channels')])
fits=pd.read_csv(OUT/'simplex_solver_diagnostics.csv');rows=[]
for name,g in fits.groupby('model'):
    rows.append([LABELS[name],len(g),int(g.converged.sum()),f'{g.duality_gap.max():.2g}'])
latex_table(GEN/'s_solver.tex',['Input','Fits','Gap below $10^{-8}$','Maximum gap'],rows)
# Historical held-lag results are not renamed as new matched-input experiments.
held=pd.read_csv(ROOT/'reports/methodological_audit/held_separation_summary.csv');rows=[]
for pattern,g in held.groupby('pattern'):
    models=['Local-state PSD','Persistence','Climatology','Ridge + linear_complex']
    rows.append([pattern.replace('_',' '),*[f'{g[g.model==m].nrmse.iloc[0]:.3f}' if len(g[g.model==m]) else '--' for m in models]])
latex_table(GEN/'s_held_lags.tex',['Historical mask','Local-state','Persistence*','Climatology*','Ridge interp.'],rows)
strata=pd.read_csv(OUT/'geoscientific_strata.csv');rows=[]
meta=pd.read_parquet(OUT/'event_metadata.parquet').set_index('event_id').sensitivity_component_25km.to_dict()
for field in ['band','quality','gauge']:
    for v,g in strata.groupby(field):
        event=g.groupby('event_id').delta.mean();b=boot(event,[meta[str(x)] for x in event.index])
        label=['0.5--1 Hz','1--2 Hz','2--4 Hz','4--8 Hz'][int(v)] if field=='band' else f'{float(v):.2f} m period' if field=='gauge' else str(v)+' input quality'
        rows.append([label,f'{b["mean"]:.4f}',f'[{b["low"]:.4f}, {b["high"]:.4f}]',b['events']])
latex_table(GEN/'s_strata.tex',['Stratum','State - block ridge','Component interval','Events'],rows)
print('Generated supplemental tables and additional numerical macros.')
