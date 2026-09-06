"""Single numerical source for Amendment-07 manuscript tables and figures."""
from __future__ import annotations
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .critical_revision import ROOT,OUT,OLD,MODELS,SEED,LAGS,Q,bundle,metrics,complex_values,boot,coverage_interval
from .config import write_json

PAPER=ROOT/'manuscript/cageo_submission'
FIG=OUT/'figures'
GEN=PAPER/'generated'
LABELS={'block_ridge':'Per-block ridge','full_ridge':'Full-context ridge','residual_ridge':'Residual ridge',
        'full_mlp_direct':'Full-context MLP, direct','full_mlp_psd':'Full-context MLP, PSD',
        'state_direct':'State recurrence, direct','state_psd':'State recurrence, PSD',
        'full_ridge_simplex':'Full ridge + spectral fit','full_mlp_direct_simplex':'MLP + spectral fit',
        'state_direct_simplex':'State + spectral fit'}


def latex_table(path,headers,rows,align=None):
    if align is None:align='l'+'r'*(len(headers)-1)
    text='\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n'+r'\midrule'+'\n'
    text+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    path.write_text(text,encoding='utf8')


def summarize():
    _,ds,roles,groups=bundle();GEN.mkdir(parents=True,exist_ok=True)
    meta=pd.read_parquet(OUT/'event_metadata.parquet');comp=meta.set_index('event_id').sensitivity_component_25km.to_dict()
    frame=pd.read_parquet(OUT/'matched_model_metrics.parquet')
    projection=pd.read_parquet(OUT/'simplex_projection_metrics.parquet')
    frame=pd.concat([frame,projection],ignore_index=True)
    frame['component']=frame.event_id.map(comp)
    test=frame[frame.role=='architecture_test']
    means=[];paired=[];effects=[]
    for name,g in test.groupby('model'):
        event=g.groupby('event_id').nrmse.mean();sg=np.array([comp[e] for e in event.index])
        means.append(dict(model=name,mse=g.mse.mean(),**boot(event,sg)))
        for reference in ['block_ridge','full_ridge']:
            ref=test[test.model==reference].groupby('event_id').nrmse.mean()
            delta=event-ref
            for unit,gr in [('earthquake',None),('component_25km',sg)]:
                paired.append(dict(model=name,reference=reference,unit=unit,**boot(delta,gr)))
            for e,d in delta.items():effects.append(dict(model=name,reference=reference,event_id=e,component=comp[e],delta=d))
    summary=pd.DataFrame(means);pairs=pd.DataFrame(paired);effect=pd.DataFrame(effects)
    summary.to_csv(OUT/'matched_summary.csv',index=False);pairs.to_csv(OUT/'matched_paired_intervals.csv',index=False)
    effect.to_csv(OUT/'matched_event_effects.csv',index=False)
    # Influence: leave one complete source component out without refitting.
    influence=[]
    for name,g in effect[effect.reference=='block_ridge'].groupby('model'):
        for c in g.component.unique():
            a=g[g.component!=c]
            influence.append(dict(model=name,omitted_component=c,remaining_events=len(a),mean_delta=a.delta.mean()))
    pd.DataFrame(influence).to_csv(OUT/'component_influence.csv',index=False)
    sen=pd.read_parquet(OUT/'estimator_sensitivity_metrics.parquet');sens=[]
    for (variant,name),g in sen[sen.role=='architecture_test'].groupby(['variant','model']):
        e=g.groupby('event_id').nrmse.mean();reference=sen[(sen.role=='architecture_test')&(sen.variant==variant)&(sen.model=='full_ridge')].groupby('event_id').nrmse.mean()
        sens.append(dict(variant=variant,model=name,mean_nrmse=e.mean(),**{f'delta_{k}':v for k,v in boot(e-reference,[comp[x] for x in e.index]).items()}))
    pd.DataFrame(sens).to_csv(OUT/'estimator_sensitivity_summary.csv',index=False)
    internal=pd.read_parquet(OUT/'internal_validation.parquet');internal_rows=[]
    for scope in ['pooled_group_folds','chronological','purged_temporal']:
        part=internal[internal.internal_fold.str.startswith('group_fold_')] if scope=='pooled_group_folds' else internal[internal.internal_fold==scope]
        if part.empty:continue
        for name,g in part.groupby('model'):
            e=g.groupby('event_id').nrmse.mean();ref=part[part.model=='full_ridge'].groupby('event_id').nrmse.mean()
            internal_rows.append(dict(scope=scope,model=name,mean_nrmse=e.mean(),**{f'delta_{k}':v for k,v in boot(e-ref,[comp[x] for x in e.index]).items()}))
    pd.DataFrame(internal_rows).to_csv(OUT/'internal_summary.csv',index=False)
    downstream=pd.read_parquet(OUT/'downstream_revision.parquet')
    downstream['evaluation']=downstream.evaluation.str.replace(r'half_crossfit_[01]','half_crossfit',regex=True)
    down=[]
    for (evaluation,fr),part in downstream.groupby(['evaluation','frame']):
        wide=part.groupby(['event_id','method']).ratio_db.mean().unstack('method')
        for name in wide.columns:
            for ref in ['Diagonal','Classical shrinkage']:
                d=wide[name]-wide[ref]
                down.append(dict(evaluation=evaluation,frame=fr,method=name,reference=ref,**boot(d,[comp[x] for x in d.index])))
    down=pd.DataFrame(down);down.to_csv(OUT/'downstream_summary.csv',index=False)
    missing=pd.read_parquet(OUT/'exhaustive_missingness.parquet');missing_rows=[]
    for keys,g in missing.groupby(['model','missing_count','endpoint']):
        e=g.groupby('event_id').nrmse.mean()
        missing_rows.append(dict(model=keys[0],missing_count=keys[1],endpoint=keys[2],**boot(e,[comp[x] for x in e.index])))
    pd.DataFrame(missing_rows).to_csv(OUT/'missingness_summary.csv',index=False)
    # Geoscientific error contrasts; fixed bands and development-median quality.
    train=np.flatnonzero(roles=='model_development');te=np.flatnonzero(roles=='architecture_test')
    power=np.array([np.mean(np.load(p)['context_noise_power_ratio']) for p in ds.paths])
    threshold=float(np.median(power[train]));pstate=np.load(MODELS/'matched_state_psd_all_predictions.npy');pridge=np.load(MODELS/'matched_block_ridge_all_predictions.npy')
    strata=[];examples=[]
    for i in te:
        for b in range(4):
            mask=np.zeros((4,8),bool);mask[b]=True
            a=metrics(pstate[i],ds.targets[i],mask);r=metrics(pridge[i],ds.targets[i],mask)
            strata.append(dict(event_id=ds.event_ids[i],route=ds.routes[i],band=b,quality='high' if power[i]>=threshold else 'low',
                gauge=float(ds.features[i,0,136]*24),delta=a['nrmse']-r['nrmse'],state_nrmse=a['nrmse'],ridge_nrmse=r['nrmse']))
    pd.DataFrame(strata).to_csv(OUT/'geoscientific_strata.csv',index=False)
    # Display events selected mechanically from paired-effect quartiles.
    event_effect=effect[(effect.model=='state_psd')&(effect.reference=='block_ridge')].sort_values('delta')
    chosen=[]
    for q in [.25,.5,.75]:
        row=event_effect.iloc[int(round(q*(len(event_effect)-1)))];chosen.append(dict(quantile=q,event_id=row.event_id,delta=row.delta))
        i=next(i for i in te if ds.event_ids[i]==row.event_id and ds.routes[i]=='TERRA')
        with np.load(ds.paths[i]) as z:context=z['context_gamma']
        for li,lag in enumerate(LAGS):
            for name,p in [('context',context),('target',complex_values(ds.targets[i])),('ridge',complex_values(pridge[i])),('state',complex_values(pstate[i]))]:
                examples.append(dict(quantile=q,event_id=row.event_id,route='TERRA',block=3,band=1,lag=int(lag),
                    separation_m=lag*9.5714288,series=name,real=float(p[3,1,li].real),imag=float(p[3,1,li].imag)))
    pd.DataFrame(examples).to_csv(OUT/'representative_examples.csv',index=False)
    # Direct/projection agreement and input support are recorded in one long table.
    long=[]
    for path in sorted(MODELS.glob('matched_*_all_predictions.npy')):
        if '_seed' in path.name:continue
        name=path.name[len('matched_'):-len('_all_predictions.npy')];pred=np.load(path)
        for i in te:
            p,t=complex_values(pred[i]),complex_values(ds.targets[i])
            for block,b,l in np.ndindex(8,4,8):
                long.append(dict(model=name,seed='ensemble_or_classical',cohort='architecture_test',event_id=ds.event_ids[i],
                    source_group_id=groups[i],component=comp[ds.event_ids[i]],route=ds.routes[i],block=block,band=b,lag=int(LAGS[l]),
                    target_mask='all_32',missingness_mask='none',prediction_real=float(p[block,b,l].real),prediction_imag=float(p[block,b,l].imag),
                    target_real=float(t[block,b,l].real),target_imag=float(t[block,b,l].imag)))
    pd.DataFrame(long).to_parquet(OUT/'revised_predictions_long.parquet',index=False)
    # One central JSON and generated macros are the numerical manuscript source.
    timing=json.loads((OUT/'timing_summary.json').read_text());ambiguity=pd.read_csv(OUT/'real_feasible_ranges.csv')
    syn=pd.read_csv(OUT/'synthetic_abstention_validation.csv');geom=pd.read_csv(OUT/'actual_downstream_geometry.csv')
    info=dict(matched_summary=summary.to_dict('records'),paired_intervals=pairs.to_dict('records'),
        sensitivity=sens,internal=internal_rows,downstream=down.to_dict('records'),missingness=missing_rows,
        timing=timing,ambiguity_mean_width=ambiguity.groupby('lag').width.mean().to_dict(),
        ambiguity_sign_identified=int(ambiguity.sign_identified.sum()),ambiguity_cases=len(ambiguity),
        synthetic_bound_cases=len(syn),synthetic_truth_covered=int(syn.truth_covered.sum()),
        synthetic_sign_certified=int(syn.sign_identified.sum()),synthetic_wrong_sign=int(syn.wrong_certified_sign.sum()),
        geometry=geom.iloc[0].to_dict(),quality_threshold=threshold,display_events=chosen,
        historical_mask_unchanged=bool(np.all(pd.read_csv(OUT/'reliability_mask_audit.csv').reliable==pd.read_csv(OUT/'reliability_mask_audit.csv').historical_reliable)))
    info=json.loads(json.dumps(info,default=lambda value:value.item() if isinstance(value,np.generic) else str(value)))
    write_json(OUT/'manuscript_summary.json',info)
    vals={}
    for name in ['block_ridge','full_ridge','full_mlp_direct','full_mlp_psd','state_direct','state_psd']:
        row=summary[summary.model==name].iloc[0];key=''.join(x.title() for x in name.split('_'))
        vals[key]=f'{row["mean"]:.4f}'
    for ref,tag in [('block_ridge','Block'),('full_ridge','Full')]:
        r=pairs[(pairs.model=='state_psd')&(pairs.reference==ref)&(pairs.unit=='component_25km')].iloc[0]
        vals[f'Delta{tag}']=f'{r["mean"]:.4f}';vals[f'Delta{tag}Low']=f'{r.low:.4f}';vals[f'Delta{tag}High']=f'{r.high:.4f}'
    vals.update(TimingPercent=f'{timing["fraction_nonpositive_route_leads"]*100:.1f}',
        AmbiguityCases=str(len(ambiguity)),AmbiguitySigns=str(int(ambiguity.sign_identified.sum())),
        SyntheticCases=str(len(syn)),SyntheticCovered=str(int(syn.truth_covered.sum())),
        SyntheticCertified=str(int(syn.sign_identified.sum())),SyntheticWrong=str(int(syn.wrong_certified_sign.sum())))
    for method,tag in [('Learned revised recurrence','Learned'),('Classical shrinkage','Classical'),('Oracle same estimate','Oracle')]:
        r=down[(down.evaluation=='full_window_in_sample')&(down.frame=='corrected_frame')&(down.method==method)&(down.reference=='Diagonal')].iloc[0]
        vals[tag+'Db']=f'{r["mean"]:.4f}';vals[tag+'DbLow']=f'{r.low:.4f}';vals[tag+'DbHigh']=f'{r.high:.4f}'
    GEN.joinpath('numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in vals.items())+'\n',encoding='utf8')
    rows=[]
    for name in ['block_ridge','full_ridge','residual_ridge','full_mlp_direct','full_mlp_psd','state_direct','state_psd']:
        r=summary[summary.model==name].iloc[0];d=pairs[(pairs.model==name)&(pairs.reference=='block_ridge')&(pairs.unit=='component_25km')].iloc[0]
        rows.append([LABELS[name],f'{r["mean"]:.4f}',f'{r.mse:.6f}',f'{d["mean"]:.4f}',f'[{d.low:.4f}, {d.high:.4f}]'])
    latex_table(GEN/'main_results.tex',['Model','NRMSE','Complex MSE',r'$\Delta$ vs block ridge',r'95\% component CI'],rows)
    cohort=pd.read_csv(OUT/'cohort_summary.csv')
    latex_table(GEN/'cohort.tex',['Role','Events','Bins','25-km components','M range'],[
        [r.role.replace('model_development','Development').replace('architecture_test','Retrospective test').replace('calibration','Calibration').replace('confirmation','Prior consistency'),r.events,r.source_bins,r.components_25km,f'{r.magnitude_min:.1f}--{r.magnitude_max:.1f}'] for r in cohort.itertuples()])
    latex_table(GEN/'sensitivity.tex',['Estimator/input','Full ridge','MLP direct','State PSD'],[
        [v.replace('_',' ').replace('normalize','normalise'),*[f'{next(r for r in sens if r["variant"]==v and r["model"]==m)["mean_nrmse"]:.4f}' for m in ['full_ridge','full_mlp_direct','state_psd']]] for v in ['equal_7s','global_cutoff','pool_before_normalize']])
    print(summary[['model','mean','low','high']].to_string(index=False),flush=True)


def build_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    FIG.mkdir(parents=True,exist_ok=True)
    blue,orange,green,gray='#2166ac','#d95f02','#1b9e77','#69717a'
    def save(fig,stem):
        for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{stem}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.07)
        plt.close(fig)
    def letter(ax,s):ax.text(-.10,1.08,s,transform=ax.transAxes,fontweight='bold',va='bottom')
    timing=pd.read_parquet(OUT/'raw_timing.parquet');meta=pd.read_parquet(OUT/'event_metadata.parquet')
    fig,axs=plt.subplots(1,2,figsize=(6.5,3.2),layout='constrained')
    role_names={'model_development':'Development (93)','calibration':'Calibration (24)','architecture_test':'Retrospective test (24)','confirmation':'Prior consistency (30)'}
    for i,(role,name) in enumerate(role_names.items()):
        part=meta[meta.analysis_role==role]
        axs[0].scatter(pd.to_datetime(part.archive_date),np.full(len(part),i),s=15,alpha=.55,label=name)
    axs[0].set_yticks(range(4),list(role_names.values()));axs[0].tick_params(axis='x',rotation=35)
    axs[0].set_title('Whole-earthquake roles',pad=12);letter(axs[0],'(a)')
    for route,c in [('TERRA',blue),('KKFL-S',orange)]:
        axs[1].hist(timing[timing.route==route].full_route_waveform_lead,bins=np.arange(-17,2,.8),alpha=.5,color=c,label=route)
    axs[1].axvline(0,color='black',ls='--');axs[1].set_xlabel('Target start - latest route input (s)');axs[1].set_ylabel('Block predictions')
    axs[1].set_title('Historical waveform timing',pad=12);axs[1].legend(loc='upper left');letter(axs[1],'(b)')
    save(fig,'fig01_timing')
    fig=plt.figure(figsize=(6.5,4.4));ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
    boxes=[(.02,.70,.29,.23,'Raw phase + arrival tables\n25 Hz; 8 spatial blocks\n1,000 channels per block\nRetrospective S references',blue),
        (.355,.70,.29,.23,'Window-local estimation\nDetrend; DPSS tapers; FFT\nCorrect fitted phase\nNormalise pairs; lag mean',blue),
        (.69,.70,.29,.23,'Full-route features\n8 blocks x 137 inputs\nContext, noise and powers\nPick fit + static metadata',blue),
        (.02,.35,.29,.23,'Matched predictors\nRidge; full-context MLP\nGated spatial recurrence\nBidirectional block mixing',orange),
        (.355,.35,.29,.23,'Two output heads\nDirect: 8 complex values\nSpectral: 257 probabilities\nThen evaluate 8 lags',orange),
        (.69,.35,.29,.23,'Compare with later targets\n4 bands x 8 lags\nAll 32 cells are scored\nNo target feature inputs',orange),
        (.18,.03,.64,.18,'Inference and application boundary\nAverage within earthquake; resample source components\nBound unseen cross-terms; abstain if sign is unresolved',green)]
    for x,y,w,h,text,col in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',edgecolor=col,facecolor=col+'12',linewidth=1.3))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8,linespacing=1.5)
    for a,b in [((.315,.815),(.35,.815)),((.65,.815),(.685,.815)),((.315,.465),(.35,.465)),((.65,.465),(.685,.465)),((.835,.34),(.70,.22))]:
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':gray,'lw':1.2})
    # Explicit input-to-predictor elbow, with no arrow through the target box.
    ax.plot([.835,.835,.165],[.69,.635,.635],color=gray,lw=1.2)
    ax.annotate('',xy=(.165,.59),xytext=(.165,.635),arrowprops={'arrowstyle':'-|>','color':gray,'lw':1.2})
    save(fig,'fig02_workflow')
    s=pd.read_csv(OUT/'matched_summary.csv');p=pd.read_csv(OUT/'matched_paired_intervals.csv')
    order=['block_ridge','full_ridge','full_mlp_direct','full_mlp_psd','state_direct','state_psd']
    fig,ax=plt.subplots(2,1,figsize=(6.5,5.5),layout='constrained',gridspec_kw={'height_ratios':[1.5,1]})
    for i,name in enumerate(order):
        r=s[s.model==name].iloc[0]
        ax[0].errorbar(r['mean'],i,xerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=green if name=='state_psd' else blue,capsize=3)
    ax[0].set_yticks(range(len(order)),[LABELS[x] for x in order]);ax[0].invert_yaxis();ax[0].set_xlabel('Mean event NRMSE with source-component interval')
    ax[0].set_title('All 32 cells; 24 retrospective test earthquakes',pad=13);letter(ax[0],'(a)')
    e=pd.read_csv(OUT/'matched_event_effects.csv');e=e[(e.model=='state_psd')&(e.reference=='block_ridge')].sort_values('delta')
    ax[1].scatter(np.arange(len(e)),e.delta,color=green,s=22);ax[1].axhline(0,color='black',ls='--',lw=1)
    ax[1].set_xlabel('Complete earthquakes ordered by paired difference');ax[1].set_ylabel('State PSD - block ridge\nNRMSE difference');letter(ax[1],'(b)')
    save(fig,'fig03_comparison')
    examples=pd.read_csv(OUT/'representative_examples.csv')
    fig,ax=plt.subplots(3,2,figsize=(6.5,6.2),layout='constrained',sharex=True)
    styles={'target':('black','o'),'context':(gray,'x'),'ridge':(orange,'s'),'state':(blue,'^')}
    for row,q in enumerate([.25,.5,.75]):
        part=examples[examples['quantile']==q]
        for col,component in enumerate(['real','imag']):
            for name,(color,marker) in styles.items():
                g=part[part.series==name]
                ax[row,col].plot(g.separation_m,g[component],color=color,marker=marker,ms=3,lw=1,label=name)
            ax[row,col].axhline(0,color='#dddddd',lw=.6);ax[row,col].set_xscale('log')
            ax[row,col].set_title(f'Event {part.event_id.iloc[0]} | '+('real' if col==0 else 'imaginary'),pad=9)
            if col==0:ax[row,col].set_ylabel(f'Effect quartile {q:g}\nCoherency')
            if row==2:ax[row,col].set_xlabel('Along-fibre separation (m)')
    handles,labels=ax[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    save(fig,'fig04_examples')
    ambiguity=pd.read_csv(OUT/'real_feasible_ranges.csv');nonunique=pd.read_csv(OUT/'synthetic_nonuniqueness.csv')
    fig,ax=plt.subplots(2,2,figsize=(6.5,5.0),layout='constrained')
    ax[0,0].plot(Q,(1+.9*np.cos(Q))/257,color=blue,label=r'$p_+$');ax[0,0].plot(Q,(1-.9*np.cos(Q))/257,color=orange,label=r'$p_-$')
    ax[0,0].set_title('Synthetic positive spectra',pad=12);ax[0,0].set_xlabel('Spectral coordinate (rad/channel)');ax[0,0].set_ylabel('Probability');ax[0,0].legend(frameon=False);letter(ax[0,0],'(a)')
    x=[0,1,3,5,8];sel=nonunique[nonunique.lag.isin(x)]
    ax[0,1].plot(sel.lag,sel.plus,'o-',color=blue,label=r'$\gamma_+$');ax[0,1].plot(sel.lag,sel.minus,'s--',color=orange,label=r'$\gamma_-$')
    ax[0,1].set_title('Same supervised values',pad=12);ax[0,1].set_xlabel('Channel lag');ax[0,1].set_ylabel('Real coherency');ax[0,1].legend(frameon=False);letter(ax[0,1],'(b)')
    for n,lag in enumerate([1,2,4,10]):
        g=ambiguity[ambiguity.lag==lag]
        for low,high in zip(g.lower,g.upper):ax[1,0].plot([n,n],[low,high],color=blue,alpha=.025)
        ax[1,0].plot([n-.2,n+.2],[g.lower.median()]*2,color='black');ax[1,0].plot([n-.2,n+.2],[g.upper.median()]*2,color='black')
    ax[1,0].axhline(0,color=gray,ls='--');ax[1,0].set_xticks(range(4),[1,2,4,10]);ax[1,0].set_ylim(-1.05,1.05)
    ax[1,0].set_title('Real-target feasible ranges',pad=12);ax[1,0].set_xlabel('Unsupervised channel lag');ax[1,0].set_ylabel('Compatible real coherency');letter(ax[1,0],'(c)')
    geom=pd.read_csv(OUT/'actual_downstream_geometry.csv').iloc[0]
    counts=[geom.supervised_entries,geom.ordered_offdiagonal-geom.supervised_entries]
    ax[1,1].bar(['Supervised','Unsupervised'],counts,color=[green,orange]);ax[1,1].set_ylabel('Ordered off-diagonal entries')
    ax[1,1].set_title('Actual 32-channel aperture',pad=12);letter(ax[1,1],'(d)')
    save(fig,'fig05_ambiguity')
    down=pd.read_csv(OUT/'downstream_summary.csv')
    fig,ax=plt.subplots(1,2,figsize=(6.5,3.4),layout='constrained')
    for i,name in enumerate(['Learned revised recurrence','Classical shrinkage']):
        for j,scope in enumerate(['full_window_in_sample','half_crossfit']):
            r=down[(down.method==name)&(down.evaluation==scope)&(down.frame=='corrected_frame')&(down.reference=='Diagonal')].iloc[0]
            ax[0].errorbar(r['mean'],i+(j-.5)*.2,xerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='os'[j],color=[blue,orange][j],capsize=3,label=['Full window','Two-way half test'][j] if i==0 else None)
    ax[0].axvline(0,color='black',ls='--');ax[0].set_yticks([0,1],['Learned','Classical shrinkage']);ax[0].set_xlabel('Difference from diagonal (dB)');ax[0].legend(loc='upper left',frameon=False)
    ax[0].set_title('Attainable-input diagnostic',pad=13);letter(ax[0],'(a)')
    for i,(scope,name,label) in enumerate([('full_window_in_sample','Oracle same estimate','Same full estimate'),('half_crossfit','Oracle same half','Same half'),('half_crossfit','Oracle other half','Other half')]):
        r=down[(down.method==name)&(down.evaluation==scope)&(down.frame=='corrected_frame')&(down.reference=='Diagonal')].iloc[0]
        ax[1].errorbar(i,r['mean'],yerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=green,capsize=4)
    ax[1].set_xticks(range(3),['Same full\nestimate','Same\nhalf','Other\nhalf']);ax[1].set_ylabel('Difference from diagonal (dB)');ax[1].set_title('Target-informed oracle',pad=13);letter(ax[1],'(b)')
    save(fig,'fig06_downstream')
    # Supplement: separate endpoints avoid an unreadable multi-purpose panel.
    miss=pd.read_csv(OUT/'missingness_summary.csv')
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.2),layout='constrained')
    colors={'historical_unaugmented':gray,'historical_mask_dropout':blue,'indicator_only':orange,'mlp_imputation':green}
    nice={'historical_unaugmented':'No augmentation','historical_mask_dropout':'Mask + dropout','indicator_only':'Indicator only','mlp_imputation':'MLP imputation'}
    for ax,endpoint in zip(axs,['all_blocks','missing_only','retained_only']):
        for name,c in colors.items():
            g=miss[(miss.endpoint==endpoint)&(miss.model==name)].sort_values('missing_count')
            ax.plot(g.missing_count,g['mean'],'o-',color=c,label=nice[name],ms=3)
        ax.set_title(endpoint.replace('_',' '));ax.set_xticks([1,2,4]);ax.set_xlabel('Removed blocks');ax.set_ylabel('NRMSE')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,'figS01_missingness')
    coverage=pd.read_csv(OUT/'joint_conformal_coverage.csv');coverage=coverage[coverage.endpoint=='all_32']
    fig,ax=plt.subplots(figsize=(6.5,3.6),layout='constrained')
    for j,(unit,col) in enumerate([('joint_earthquake','black'),('TERRA',blue),('KKFL-S',orange)]):
        g=coverage[coverage.unit==unit];offset=(j-1)*.006
        ax.errorbar(g.nominal+offset,g.coverage,yerr=[g.coverage-g.ci_low,g.ci_high-g.coverage],fmt='o',capsize=3,color=col,label=unit.replace('_',' '))
    ax.plot([.75,1],[.75,1],color=gray,ls='--');ax.set_xlim(.77,.98);ax.set_ylim(.2,1.02);ax.set_xlabel('Nominal error-bound coverage');ax.set_ylabel('Observed coverage (24 earthquakes)');ax.legend(frameon=False,loc='lower right')
    save(fig,'figS02_uncertainty')
    print(f'figures: {FIG}',flush=True)


def main(stage):
    if stage=='summary':summarize()
    elif stage=='figures':build_figures()
    elif stage in ['release','verify']:
        from .critical_release import main as release_main
        release_main(stage)
    else:raise ValueError(stage)
