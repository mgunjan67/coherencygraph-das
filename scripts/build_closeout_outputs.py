"""Generate closeout tables, vector figures and numerical macros from receipts."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reports/submission_closeout';F=R/'figures';F.mkdir(parents=True,exist_ok=True)
P=ROOT/'manuscript/cageo_closeout';G=P/'generated';G.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
numbers={};sources={}
def put(name,value,source,fmt='.4f'):
    numbers[name]=format(value,fmt) if isinstance(value,(float,np.floating)) else str(value)
    sources[name]=source
def triple(prefix,row,source):
    for key in ['mean','low','high']:put(prefix+key.title(),float(row[key]),source)
def table(name,headers,rows,widths):
    spec=''.join('p{'+str(w)+r'\textwidth}' for w in widths)
    text='{\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    text+=''.join(' & '.join(map(str,row))+' \\\\\n' for row in rows)+'\\bottomrule\n\\end{tabular}}\n'
    (G/(name+'.tex')).write_text(text,encoding='utf8')
def save(fig,name):
    for ext in ['pdf','svg','png']:fig.savefig(F/(name+'.'+ext),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
def interval(row,digits=3):return f"{row['mean']:.{digits}f} [{row['low']:.{digits}f}, {row['high']:.{digits}f}]"

def main():
    pred=pd.read_csv(ROOT/'reports/submission_revision/prediction_both_ridge_comparisons.csv')
    comps=pd.read_csv(R/'processing/processing_all_comparisons.csv')
    pv=json.loads((R/'processing/verification.json').read_text())
    cost=json.loads((R/'examples/cost_environment.json').read_text())
    synth=pd.read_csv(R/'audit_validation/current_synthetic_summary.csv')
    empirical=pd.read_csv(R/'audit_validation/empirical_summary.csv')
    residual=pd.read_csv(R/'audit_validation/residual_distributions.csv')
    diagnostics=pd.read_csv(R/'audit_validation/numerical_diagnostic_distributions.csv')
    receipt=json.loads((ROOT/'models/submission_revision/local_dense_state_psd_seed19.json').read_text())
    put('CloseoutParameters',receipt['parameters'],'models/submission_revision/local_dense_state_psd_seed19.json')
    put('CloseoutScipy',cost['dependencies']['scipy'],'examples/cost_environment.json')
    for key,field in [('CloseoutHistoricalFailed','historical_failed_repaired'),('CloseoutHistoricalFits','historical_projections'),('CloseoutAcceptedFits','accepted_final_projections')]:put(key,pv[field],'processing/verification.json')
    put('CloseoutProjectionMeanChange',pv['max_paired_mean_correction_db'],'processing/verification.json','.2e')
    put('CloseoutProjectionCaseChange',pv['max_historical_case_correction_db'],'processing/verification.json','.6f')
    put('CloseoutResidualMax',residual.query('quantity=="residual"')['maximum'].max(),'audit_validation/residual_distributions.csv','.2e')
    put('CloseoutResidualRatio',residual.query('quantity=="residual_to_delta"')['maximum'].max(),'audit_validation/residual_distributions.csv','.2e')
    for support,name in [('full_window','CloseoutFullSigns'),('opposite_half','CloseoutHalfSigns')]:
        put(name,int(empirical.query('support==@support and design=="algorithmic" and axis=="real"').identified.iloc[0]),'audit_validation/empirical_summary.csv')
    def comp(method,reference,endpoint='full'):
        rows=comps.query('endpoint==@endpoint and method==@method and reference==@reference');assert len(rows)==1
        return rows.iloc[0]
    contrasts={'CloseoutBlockRidgeGain':('local_dense / block ridge','local_sparse / block ridge'),'CloseoutFullRidgeGain':('local_dense / full-context ridge','local_sparse / full-context ridge'),'CloseoutFullVersusNeural':('local_dense / full-context ridge','local_dense / recurrence'),'CloseoutFullVersusContext':('local_dense / full-context ridge','reference / Classical shrinkage')}
    for prefix,(a,b) in contrasts.items():triple(prefix,comp(a,b),'processing/processing_all_comparisons.csv')
    for key,field in [('CloseoutAuditSeconds','single_call_median_seconds'),('CloseoutCohortSeconds','cohort_seconds')]:put(key,cost[field],'examples/cost_environment.json','.3f')
    put('CloseoutPeakMemory',cost['peak_observed_process_rss_bytes']/1024**2,'examples/cost_environment.json','.1f')

    endpoints=[('waveform_safe','Earlier waveform'),('block_dense','Block dense 31'),('local_sparse','Local sparse 5'),('local_dense','Local dense 31')]
    rows=[]
    for endpoint,label in endpoints:
        for ref,ref_label in [('block_ridge','Blockwise'),('full_ridge','Full-context')]:
            row=pred.query('endpoint==@endpoint and reference==@ref').iloc[0]
            rows.append([label,ref_label,f'{row.reference_score:.4f}',f'{row.ensemble_score:.4f}',interval(row,4)])
    table('closeout_prediction',['Endpoint','Ridge design','Ridge score','Ensemble',r'$\Delta$ NRMSE [95\% interval]'],rows,[.21,.17,.10,.12,.32])
    fig,ax=plt.subplots(figsize=(10.5,4.25));fig.subplots_adjust(left=.25,right=.98,bottom=.24,top=.86)
    for ref,color,mark,dy,label in [('block_ridge','#1671a5','o',-.13,'Blockwise ridge'),('full_ridge','#c95728','s',.13,'Full-context ridge')]:
        for i,(endpoint,_) in enumerate(endpoints):
            row=pred.query('endpoint==@endpoint and reference==@ref').iloc[0]
            ax.errorbar(row['mean'],i+dy,xerr=[[row['mean']-row.low],[row.high-row['mean']]],fmt=mark,color=color,markersize=6,capsize=3,label=label if i==0 else None)
    ax.set_yticks(range(4),[x[1] for x in endpoints]);ax.invert_yaxis();ax.axvline(0,color='.3',ls='--',lw=1)
    ax.set_xlabel('Ensemble minus named ridge NRMSE');ax.grid(axis='x',alpha=.18)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=2,frameon=False)
    save(fig,'fig03_prediction')

    validation=[]
    for family,label in [('in_class','On-grid truth'),('off_grid','Off-grid stress')]:
        ss=synth.query('family!="off_grid"') if family=='in_class' else synth.query('family=="off_grid"')
        validation.append([label,int(ss.cases.sum()),int(ss.valid.sum()),int(ss.covered.sum()),int(ss.identified.sum()),int(ss.incorrect.sum())])
    old=pd.read_csv(R/'audit_validation/historical_synthetic_summary.csv').query('family=="nonstationary"').iloc[0]
    validation.append(['Nonstationary stress$^a$',int(old.cases),int(old.valid),int(old.covered),int(old.identified),int(old.incorrect)])
    for support,prefix in [('full_window','Full window'),('opposite_half','Opposite half')]:
        for axis,suffix in [('real','Re'),('imaginary','Im')]:
            row=empirical.query('support==@support and design=="algorithmic" and axis==@axis').iloc[0]
            validation.append([prefix+' '+suffix,int(row.cases),int(row.valid),int(row.contained),int(row.identified),int(row.sign_disagreements)])
    table('closeout_validation',['Check','Cases','Valid','Inside','Signs','Wrong$^b$'],validation,[.32,.11,.11,.11,.11,.11])
    with (G/'closeout_validation.tex').open('a',encoding='utf8') as f:f.write('\n\\par\\smallskip\\footnotesize $^a$Historical scalar-allowance fixture; not a component-allowance replication. $^b$Incorrect known-truth signs or disagreement with measured signs. Manual and selected lag designs coincide.\n')
    rows=[]
    for support,label in [('full_window','Full'),('opposite_half','Opposite half')]:
        for design,dlabel in [('historical','Historical'),('algorithmic','Selected/manual'),('augmented','Augmented')]:
            for axis,alabel in [('real','Re'),('imaginary','Im')]:
                r=empirical.query('support==@support and design==@design and axis==@axis').iloc[0]
                rows.append([label,dlabel,alabel,int(r.valid),int(r.identified),int(r.sign_disagreements),int(r.contained)])
    table('closeout_empirical_complete',['Window','Design','Part','Valid','Signs','Wrong','Inside'],rows,[.19,.22,.07,.10,.10,.10,.10])

    ex=pd.read_csv(R/'examples/fixed_geometry_examples.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True);fig.subplots_adjust(left=.22,right=.98,top=.84,bottom=.20,wspace=.25)
    items=[('uniform',4,'Uniform, lag 4'),('uniform',10,'Uniform, lag 10'),('uniform_plus_dc',4,'Mixture, lag 4'),('uniform_plus_dc',10,'Mixture, lag 10')]
    for ax,axis,label in zip(axes,['real','imaginary'],['(a) Real component','(b) Imaginary component']):
        for i,(fixture,lag,_) in enumerate(items):
            row=ex.query('fixture==@fixture and lag==@lag and axis==@axis').iloc[0];c='#1671a5' if fixture=='uniform' else '#c95728'
            ax.hlines(i,row.lower,row.upper,color=c,lw=3);ax.vlines([row.lower,row.upper],i-.09,i+.09,color=c,lw=1.4);ax.scatter(row.truth,i,marker='D',s=28,color='black',zorder=5)
        ax.axvline(0,color='.55',ls='--',lw=1);ax.set_xlim(-1.07,1.07);ax.set_title(label,pad=16);ax.set_xlabel('Conditional component range');ax.grid(axis='x',alpha=.18)
    axes[0].set_yticks(range(4),[x[2] for x in items]);axes[0].invert_yaxis()
    fig.legend(handles=[Line2D([0],[0],color='#1671a5',lw=3,label='Uniform spectrum'),Line2D([0],[0],color='#c95728',lw=3,label='Uniform + DC mixture'),Line2D([0],[0],marker='D',color='black',ls='',label='Known value')],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.57,-.01))
    save(fig,'fig04_conditional_design')

    methods=[('block_sparse / recurrence','Block sparse recurrence'),('block_dense / recurrence','Block dense recurrence'),('local_sparse / recurrence','Local sparse recurrence'),('local_dense / recurrence','Local dense recurrence'),('local_sparse / block ridge','Local sparse block ridge'),('local_dense / block ridge','Local dense block ridge'),('local_sparse / full-context ridge','Local sparse full ridge'),('local_dense / full-context ridge','Local dense full ridge'),('reference / Classical shrinkage','Fixed context shrinkage')]
    rows=[]
    for method,label in methods:rows.append([label,interval(comp(method,'reference / Diagonal')),interval(comp(method,'reference / Diagonal','split_target'))])
    table('closeout_processing',['Method',r'Full target $G_m$ (dB)',r'Split target $G_m$ (dB)'],rows,[.38,.28,.28])
    fig,axes=plt.subplots(1,2,figsize=(11,6.5),sharey=True);fig.subplots_adjust(left=.29,right=.97,bottom=.14,top=.91,wspace=.22)
    for ax,endpoint,title in zip(axes,['full','split_target'],['(a) Full target','(b) Average of target halves']):
        for i,(method,_) in enumerate(methods):
            row=comp(method,'reference / Diagonal',endpoint)
            color='#1671a5' if 'recurrence' in method else ('#c95728' if 'ridge' in method else '#237d62')
            marker='o' if 'recurrence' in method else ('s' if 'ridge' in method else 'D')
            ax.errorbar(row['mean'],i,xerr=[[row['mean']-row.low],[row.high-row['mean']]],fmt=marker,color=color,capsize=3,markersize=6)
        ax.axvline(0,color='.35',ls='--',lw=1);ax.grid(axis='x',alpha=.18);ax.set_title(title,pad=16);ax.set_xlabel('Power change from diagonal (dB)')
    axes[0].set_yticks(range(len(methods)),[x[1] for x in methods]);axes[0].invert_yaxis()
    save(fig,'fig05_geometry_processing')

    # Complete comparator results are editable supplementary tables.
    all_methods=sorted(comps.method.unique());rows=[]
    for m in all_methods:
        if m=='reference / Diagonal':continue
        rows.append([m.replace('_',' ').replace(' / ',' /\\newline '),interval(comp(m,'reference / Diagonal')),interval(comp(m,'reference / Diagonal','split_target'))])
    table('closeout_processing_complete',['Method',r'Full target $G_m$ (dB)',r'Split target $G_m$ (dB)'],rows,[.40,.27,.27])
    rows=[]
    for p,(a,b) in contrasts.items():rows.append([a.replace('local_dense / ','Dense local ').replace(' / ',' '),b.replace('local_sparse / ','Sparse local ').replace('local_dense / ','Dense local ').replace('reference / Classical shrinkage','Context shrinkage'),interval(comp(a,b))])
    table('closeout_processing_contrasts',['Method','Reference',r'Paired difference (dB)'],rows,[.31,.31,.31])
    rows=[['One audit call',3,'1 fit + 8 LPs',f"{cost['single_call_median_seconds']:.3f} (median)"],['One selected design',1,'192 fits + 1536 LPs',f"{cost['cohort_seconds']:.3f}"]]
    table('closeout_cost',['Calculation','Repeats','Solves per repeat','Wall time (s)'],rows,[.26,.11,.32,.22])
    macro_text=[]
    for k,v in sorted(numbers.items()):
        if 'e-' in v:
            a,b=v.split('e-');v=r'\ensuremath{'+a+r'\times10^{-'+str(int(b))+'}}'
        macro_text.append('\\newcommand{\\'+k+'}{'+v+'}')
    (G/'closeout_numbers.tex').write_text('\n'.join(macro_text)+'\n',encoding='utf8')
    (R/'manuscript_numbers.json').write_text(json.dumps({'values':numbers,'sources':sources},indent=2))
    pd.DataFrame([dict(claim_id=k,manuscript_location='main or Supplementary Text S17',claim_text=k+' = '+v,estimand='See referenced source column; component-event reduction preserved',cohort='Historical or 24 retrospective earthquakes as indicated by source',waveform_support='Retrospective supplied picks; fixed original windows',target_support='Block lag or exact-local support, as indicated by source',processing_geometry='Channels 484-515 for processing; not applicable to audit-only rows',source_artifact=sources[k],generation_command='python scripts/build_closeout_outputs.py',status='GENERATED',permitted_interpretation='Bounded retrospective comparison, not independent confirmation') for k,v in numbers.items()]).to_csv(R/'claim_evidence_generated.csv',index=False)
    print('Generated',len(numbers),'macros, 7 tables, 3 revised figures; Figure 2 preserved')
if __name__=='__main__':main()
