"""Generate amendment-08 tables and vector figures only from saved results."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports/final_revision';G=ROOT/'manuscript/cageo_submission/generated';F=R/'figures';F.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from coherencygraph_das.critical_paper import latex_table
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
blue='#176595';orange='#b14d17';green='#28785b'
def save(fig,name):
    fig.canvas.draw();fig.canvas.draw()
    fig.set_layout_engine('none')
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
def fmt(r):return f'{r["mean"]:.4f} [{r.low:.4f}, {r.high:.4f}]'

comp=pd.read_csv(R/'corrected_comparisons.csv');rows=[];num={}
labels={'equal_7s':'Equal 7-s','global_cutoff':'Common cutoff','pool_before_normalize':'Pooled (new target)'}
for endpoint,label in labels.items():
    g=comp[comp.endpoint==endpoint].set_index('model');r=g.loc['state_psd']
    rows.append([label,f'{g.loc["block_ridge","score"]:.4f}',f'{g.loc["full_ridge","score"]:.4f}',f'{r.score:.4f}',fmt(r)])
    tag={'equal_7s':'Equal','global_cutoff':'Common','pool_before_normalize':'Pooled'}[endpoint]
    for key in ['mean','low','high']:num['Final'+tag+key.title()]=f'{r[key]:.4f}'
latex_table(G/'final_controls.tex',['Endpoint','Block ridge','Full ridge','Recurrence',r'$\Delta$ [95\% CI]'],rows)
primary=comp[comp.endpoint=='primary']
latex_table(G/'final_primary.tex',['Model','NRMSE',r'$\Delta$ vs block ridge [95\% CI]'],[[r.model.replace('_',' '),f'{r.score:.4f}',fmt(r)] for _,r in primary.iterrows()])

bounds=pd.read_csv(R/'measurement_design_summary.csv');hist=bounds[(bounds.design=='historical')&(bounds.grid==257)&np.isclose(bounds.offset,.01)].iloc[0]
num['FinalBoundSigns']=str(int(hist.identified));num['FinalBoundCases']=str(int(hist.cases))
rows=[];plot=[]
for _,r in bounds.iterrows():
    label={'historical':'Historical 8','same_budget':'Near-neighbour 8','augmented':'Augmented 10'}[r.design]
    kind='measured' if r.directly_measured else 'unseen'
    rows.append([label,str(int(r.grid)),f'{r.offset:.3f}',kind,f'{int(r.identified)}/{int(r.cases)}',f'{r.mean_width:.3f}',str(int(r.wrong_sign))])
    if not r.directly_measured:plot.append((f'{label}\nM={int(r.grid)}, delta={r.offset:g}',r.identified/r.cases,r.mean_width))
latex_table(G/'final_bounds.tex',['Lag design','Grid',r'$\delta$','Entry','Signs/cases','Width','Wrong signs'],rows)
fig,axs=plt.subplots(1,2,figsize=(7.1,4.6),layout='constrained',sharey=True)
for i,(label,fraction,width) in enumerate(plot):
    axs[0].barh(i,fraction,color=blue);axs[1].barh(i,width,color=green)
axs[0].set_yticks(range(len(plot)),[x[0] for x in plot]);axs[0].invert_yaxis();axs[0].set_xlim(0,1);axs[1].set_xlim(0,2)
axs[0].set_xlabel('Fraction with identified sign');axs[1].set_xlabel('Mean compatible range width')
axs[0].set_title('(a) Genuinely unseen entries');axs[1].set_title('(b) Same target-informed cases')
save(fig,'fig07_design')
for design in ['same_budget','augmented']:
    r=bounds[(bounds.design==design)&(~bounds.directly_measured)].iloc[0]
    num['Final'+('Near' if design=='same_budget' else 'Augmented')+'Signs']=str(int(r.identified));num['Final'+('Near' if design=='same_budget' else 'Augmented')+'Cases']=str(int(r.cases))

syn=pd.read_csv(R/'known_truth_summary.csv')
latex_table(G/'final_synthetics.tex',['Condition','Cases','In range','Signs','Incorrect','Mean width'],[[r.family.replace('_',' '),str(r.cases),str(r.covered),str(r.identified),str(r.incorrect),f'{r.mean_width:.3f}'] for r in syn.itertuples()])
geo=pd.read_csv(R/'geometry_inventory.csv').drop_duplicates('geometry')
latex_table(G/'final_geometry.tex',['Geometry','Channels',r'\shortstack{Span\\(intervals)}',r'\shortstack{Required\\entries}',r'\shortstack{Historical\\observed}',r'\shortstack{Dense\\observed}'],[[r.geometry,'32',str(r.aperture),str(r.ordered_entries),str(r.historical_supervised),str(r.dense_supervised)] for r in geo.itertuples()])
down=pd.read_csv(R/'geometry_processing_summary.csv');selected=down[(down.endpoint=='full')&(down.reference=='Diagonal')];names={'historical recurrence ensemble':'Historical ensemble','historical recurrence seed19':'Historical, seed 19','dense31 recurrence seed19':'Dense supervision, seed 19','historical block ridge + simplex':'Historical ridge + simplex','dense31 block ridge + simplex':'Dense ridge + simplex','Classical shrinkage':'Classical shrinkage'}
latex_table(G/'final_processing.tex',['Geometry','Method',r'Power difference (dB) [95\% CI]'],[[r.geometry,names[r.model],fmt(r)] for _,r in selected.iterrows() if r.model in names])
half=down[(down.endpoint=='opposite_half')&(down.reference=='Diagonal')&(down.model=='Target opposite half')]
latex_table(G/'final_oracles.tex',['Geometry',r'Opposite-half oracle (dB) [95\% CI]'],[[r.geometry,fmt(r)] for _,r in half.iterrows()])
for _,r in half.iterrows():
    for k in ['mean','low','high']:num['FinalOracle'+r.geometry.title()+k.title()]=f'{r[k]:.4f}'
dense=pd.read_csv(R/'dense_prediction_comparisons.csv')
latex_table(G/'final_dense_prediction.tex',['Model','Dense-31 NRMSE',r'$\Delta$ [95\% CI]'],[[r.model.replace('_',' '),f'{r.score:.4f}',fmt(r)] for _,r in dense.iterrows()])
heads=pd.read_csv(R/'head_comparisons.csv');heads=heads[heads.model!=heads.reference]
cov=pd.read_csv(R/'covariance_diagnostics.csv').query('geometry=="small"').set_index('model')
a=cov.loc['historical recurrence seed19'];b=cov.loc['dense31 recurrence seed19']
(G/'final_covariance_text.tex').write_text(
    f'At the small aperture, the mean squared complex off-diagonal correlation-kernel error against the noisy target estimate is {a.entry_mse:.4f} for historical seed-19 supervision and {b.entry_mse:.4f} for dense-31 supervision. '
    f'For the corresponding shrunk covariance matrices, minimum eigenvalues over the evaluated cases are {a.min_eigenvalue:.6f} and {b.min_eigenvalue:.6f} in optical-phase power units; maximum condition numbers are {a.max_condition:.1f} and {b.max_condition:.1f}. '
    'Kernel error is evaluated before shrinkage, whereas conditioning describes the matrices passed to the loaded processing calculation. These descriptive aggregates do not establish statistical superiority or physical strain accuracy.\n')
latex_table(G/'final_heads.tex',['Head family',r'Paired NRMSE difference [95\% CI]'],[[r.model.replace('_',' '),fmt(r)] for _,r in heads.iterrows()])
split=pd.read_csv(R/'split_window_bounds_summary.csv')
latex_table(G/'final_split.tex',['Design','Cases','Signs','In range','Sign disagreement','Mean width'],[[r.design.replace('_',' '),str(r.cases),str(r.identified),str(r.in_range),str(r.sign_disagreement),f'{r.width:.3f}'] for r in split.itertuples()])
effects=pd.read_csv(R/'paired_design_effects.csv')
latex_table(G/'final_design_effects.tex',['Quantity','Design minus historical',r'Effect [95\% CI]'],[[r.quantity,r.design.replace('_',' '),fmt(r)] for _,r in effects.iterrows()])
small_pair=down[(down.geometry=='small')&(down.endpoint=='full')&(down.reference=='historical recurrence seed19')&(down.model=='dense31 recurrence seed19')].iloc[0]
small_gain=selected[(selected.geometry=='small')&(selected.model=='dense31 recurrence seed19')].iloc[0]
small_classic=selected[(selected.geometry=='small')&(selected.model=='Classical shrinkage')].iloc[0]
small_vs_classic=down[(down.geometry=='small')&(down.endpoint=='full')&(down.reference=='Classical shrinkage')&(down.model=='dense31 recurrence seed19')].iloc[0]
for prefix,r in [('DenseGain',small_pair),('DenseVersusClassical',small_vs_classic)]:
    for k in ['mean','low','high']:num['Final'+prefix+k.title()]=f'{r[k]:.4f}'
interpretation=(f'At the fixed small aperture, changing from eight-lag to dense-31 supervision changes the seed-19 recurrence processing endpoint by {small_pair["mean"]:.4f}~dB [{small_pair.low:.4f}, {small_pair.high:.4f}]. '
    f'The dense model changes the endpoint relative to diagonal processing by {small_gain["mean"]:.4f}~dB [{small_gain.low:.4f}, {small_gain.high:.4f}], whereas classical shrinkage gives {small_classic["mean"]:.4f}~dB [{small_classic.low:.4f}, {small_classic.high:.4f}]. '
    f'The paired dense-minus-classical difference is {small_vs_classic["mean"]:.4f}~dB [{small_vs_classic.low:.4f}, {small_vs_classic.high:.4f}], favouring classical shrinkage. '
    'These are paired diagnostic differences, not detection gains. The full geometry and covariance-error tables retain all methods, including unfavourable comparisons.\n')
(G/'final_processing_interpretation.tex').write_text(interpretation)
sel=selected[(selected.geometry=='small')&selected.model.isin(names)]
fig,ax=plt.subplots(figsize=(7,3.4),layout='constrained')
for i,(_,r) in enumerate(sel.iterrows()):ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=green if 'dense31' in r.model else blue,capsize=3)
ax.set_yticks(range(len(sel)),[names[x] for x in sel.model]);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--');ax.set_xlabel('Target / reference-noise power difference from diagonal (dB)');ax.set_title('Fixed 32-consecutive-channel aperture; 24 earthquakes')
save(fig,'fig08_small_processing')

# Explicitly separate measured-target diagnostic from the prediction branch.
fig,ax=plt.subplots(figsize=(7.2,5.6));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
box_checks=[]
def box(x,y,w,h,text,color):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor=color+'10',edgecolor=color,lw=1.2)
    ax.add_patch(patch)
    label=ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.5,linespacing=1.4)
    box_checks.append((patch,label))
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color='#555555',lw=1.2))
box(.2,.84,.6,.14,'Raw phase and arrival tables\nWindow-local detrending / DPSS\nRetrospective passage correction',blue)
box(.02,.57,.43,.18,'PREDICTION INPUT\nContext and noise features\n8 blocks x 137 values\nLater targets excluded',blue)
box(.55,.57,.43,.18,'DIAGNOSTIC INPUT\nMeasured later coherency\nComplex mean of eight blocks\nOne vector per route and band',green)
box(.02,.29,.43,.18,'Ridge / MLP / recurrence\nPer band and block:\n8 complex values OR\n257 spectral probabilities',blue)
box(.55,.29,.43,.18,'Target-informed feasible set\nDeclared grid and tolerance\nValidated outer bounds\nReal cross-terms only',green)
box(.02,.03,.43,.16,'Score against later targets\nAverage routes per earthquake\nPaired component uncertainty',blue)
box(.55,.03,.43,.16,'Conditional sign or abstention\nNot a predictive certificate\nCompare measured lag designs',green)
for a,b in [((.32,.84),(.235,.76)),((.68,.84),(.765,.76)),((.235,.56),(.235,.48)),((.765,.56),(.765,.48)),((.235,.28),(.235,.20)),((.765,.28),(.765,.20))]:arrow(a,b)
fig.canvas.draw()
for patch,label in box_checks:
    outer=patch.get_window_extent();inner=label.get_window_extent()
    assert outer.x0<=inner.x0 and outer.y0<=inner.y0 and outer.x1>=inner.x1 and outer.y1>=inner.y1, label.get_text()
save(fig,'fig02_workflow')

# Preserve the mechanically chosen examples; identify the actual baseline.
ex=pd.read_csv(ROOT/'reports/critical_review/representative_examples.csv')
fig,axes=plt.subplots(3,2,figsize=(7,6.3),layout='constrained',sharex=True,sharey='col')
styles={'target':('black','o','Later measurement'),'context':('#888888','x','Context persistence'),'ridge':(orange,'s','Per-block ridge'),'state':(blue,'^','Spectral recurrence')}
for row,quant in enumerate([.25,.5,.75]):
    part=ex[np.isclose(ex['quantile'],quant)]
    for col,component in enumerate(['real','imag']):
        ax=axes[row,col]
        for name,(color,marker,label) in styles.items():
            g=part[part.series==name];ax.plot(g.separation_m,g[component],marker=marker,color=color,label=label,ms=3,lw=1)
        ax.set_xscale('log');ax.axhline(0,color='#dddddd');ax.set_title(f'Event {part.event_id.iloc[0]}: {component}')
        if col==0:ax.set_ylabel(f'Effect quartile {quant:g}\nCoherency')
        if row==2:ax.set_xlabel('Along-fibre separation (m)')
handles,labs=axes[0,0].get_legend_handles_labels();fig.legend(handles,labs,loc='outside lower center',ncol=2,frameon=False)
save(fig,'fig04_examples')
(G/'final_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in num.items())+'\n')
pd.DataFrame([dict(macro=k,value=v) for k,v in num.items()]).to_csv(R/'manuscript_number_sources.csv',index=False)
print('Final tables and figures generated.')
