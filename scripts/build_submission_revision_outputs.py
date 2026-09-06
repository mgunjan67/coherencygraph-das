"""Build all Amendment-09 figures and manuscript numbers from saved results."""
from pathlib import Path
import sys,json,struct,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from coherencygraph_das.critical_paper import latex_table
R=ROOT/'reports/submission_revision'; G=ROOT/'manuscript/cageo_submission/generated'; F=R/'figures'
F.mkdir(exist_ok=True)
if not G.parent.exists():G=R/'generated_tables'
G.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
BLUE='#176595'; ORANGE='#b14d17'; GREEN='#28785b'; PURPLE='#735193'; numbers={}

def save(fig,name):
    fig.canvas.draw()
    for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def ci(row):return f'{row["mean"]:.4f} [{row.low:.4f}, {row.high:.4f}]'
def macros(prefix,row,keys=('mean','low','high')):
    for key in keys:numbers['Rev'+prefix+key.title()]=f'{row[key]:.4f}'

def coastlines():
    with zipfile.ZipFile(ROOT/'data/provenance/ne_10m_coastline.zip') as archive:
        data=archive.read(next(x for x in archive.namelist() if x.endswith('.shp')))
    cursor=100
    while cursor<len(data):
        _,length=struct.unpack_from('>ii',data,cursor);record=data[cursor+8:cursor+8+2*length];cursor+=8+2*length
        if len(record)<44 or struct.unpack_from('<i',record)[0]!=3:continue
        xmin,ymin,xmax,ymax=struct.unpack_from('<4d',record,4)
        if xmax<-158 or xmin>-146 or ymax<56 or ymin>64:continue
        parts,n=struct.unpack_from('<2i',record,36)
        offsets=list(struct.unpack_from(f'<{parts}i',record,44))+[n]
        points=np.frombuffer(record,dtype='<f8',count=2*n,offset=44+4*parts).reshape(-1,2)
        for a,b in zip(offsets[:-1],offsets[1:]):yield points[a:b]


meta=pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet')
roles=pd.read_csv(R/'unchanged_event_roles.csv',dtype={'event_id':str}).drop_duplicates('event_id')[['event_id','role']]
meta=meta.merge(roles,on='event_id')
fig,axs=plt.subplots(1,2,figsize=(7.3,4.5),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
ax=axs[0]
for line in coastlines():ax.plot(line[:,0],line[:,1],color='.6',lw=.6,zorder=0)
for role,label,color,marker in [('model_development','Development',BLUE,'o'),('calibration','Calibration',GREEN,'s'),('architecture_test','Retrospective test',ORANGE,'^'),('confirmation','Prior consistency','.45','x')]:
    g=meta[meta.role==role];ax.scatter(g.longitude_deg,g.latitude_deg,s=20,label=f'{label} ({len(g)})',color=color,marker=marker,alpha=.75)
ax.set(xlim=(-158,-146),ylim=(56,64),xlabel='Longitude (degrees W)',ylabel='Latitude (degrees N)',title='(a) Earthquake locations')
ax.set_xticks([-156,-152,-148],['156','152','148'])
ax.set_aspect(1/np.cos(np.deg2rad(60)));ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False,ncol=2)
ax=axs[1];ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.set_title('(b) Processing geometry')
for y,title,positions,color in [(.78,'Historical processor: 32 channels',np.linspace(0,999,32),BLUE),(.43,'Matched processor: 32 channels',np.arange(32),GREEN)]:
    x=.05+.90*(positions-positions.min())/(positions.max()-positions.min())
    ax.plot([.05,.95],[y,y],color=color,lw=1);ax.scatter(x,np.full(32,y),s=9,color=color)
    ax.text(.5,y+.09,title,ha='center',fontsize=10)
    span=int(positions.max()-positions.min());ax.text(.5,y-.10,f'Span: {span} channel intervals',ha='center',fontsize=10)
ax.text(.5,.13,'Exact local targets use the same\n32 consecutive processing channels.\nSchematic only; no cable route is inferred.',ha='center',va='center',fontsize=9,linespacing=1.5)
save(fig,'fig01_setting_geometry')

fig,ax=plt.subplots(figsize=(7.3,4.6));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
boxes=[]
def box(x,y,w,h,title,body,color):
    rect=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor=color+'12',edgecolor=color,lw=1.4);ax.add_patch(rect)
    a=ax.text(x+w/2,y+h-.035,title,ha='center',va='top',weight='bold',fontsize=10,color=color)
    b=ax.text(x+w/2,y+h/2-.035,body,ha='center',va='center',fontsize=9,linespacing=1.45)
    boxes.append((rect,a,b))
box(.025,.62,.44,.33,'Prediction branch','Context features + supplied picks\nHistorical / earlier-waveform variants\nDevelopment-only fitting; later-lag scoring',BLUE)
box(.535,.62,.44,.33,'Measurement-audit branch','Measured later-window lag moments\nDevelopment-selected lag design\nFit a stationary spectral feasible set',GREEN)
box(.535,.18,.44,.32,'Validated conditional ranges','Bound real and imaginary cross-terms\nCheck optimisation residuals and dual gaps\nIdentify a sign or explicitly abstain',GREEN)
box(.025,.18,.44,.32,'Fixed-geometry processing','Historical-context predictors only\nShared channels, powers and noise frame\nCompare sparse, dense and classical inputs',BLUE)
for x in [.245,.755]:ax.annotate('',xy=(x,.51),xytext=(x,.61),arrowprops={'arrowstyle':'->','lw':1.5,'color':'.3'})
ax.text(.5,.05,'Separate outputs: prediction error | conditional ambiguity | processing utility',ha='center',fontsize=10)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for rect,*texts in boxes:
    boundary=rect.get_window_extent(renderer)
    for t in texts:
        extent=t.get_window_extent(renderer)
        assert extent.x0>=boundary.x0 and extent.x1<=boundary.x1 and extent.y0>=boundary.y0 and extent.y1<=boundary.y1,t.get_text()
save(fig,'fig02_audit_workflow')

pred=pd.read_csv(R/'prediction_comparisons.csv'); frames=[]
labels={'waveform_safe':'Earlier-waveform input','block_dense':'Block-averaged 31 lags','local_sparse':'Exact local 5 lags','local_dense':'Exact local 31 lags'}
for endpoint,label in labels.items():
    g=pred[pred.endpoint==endpoint].set_index('model');r=g.loc['state_psd_seed_mean']
    frames.append([label,f'{g.loc["block_ridge","score"]:.4f}',f'{g.loc["full_ridge","score"]:.4f}',f'{r.score:.4f}',ci(r)])
    macros(''.join(x.title() for x in endpoint.split('_')),r)
    numbers['Rev'+''.join(x.title() for x in endpoint.split('_'))+'Score']=f'{r.score:.4f}'
latex_table(G/'revision_prediction.tex',['Target/input','Block ridge','Full ridge','Ensemble',r'$\Delta$ vs block ridge [95\% CI]'],frames)
seed=pd.read_csv(R/'seed_variation.csv')
latex_table(G/'revision_seeds.tex',['Endpoint','Seed 19','Seed 43','Seed 71','Across-seed SD'],[[labels[r.endpoint]]+[f'{pred[(pred.endpoint==r.endpoint)&(pred.model==f"state_psd_seed{s}")].iloc[0].score:.4f}' for s in [19,43,71]]+[f'{r.seed_sample_sd:.4f}'] for r in seed.itertuples()])
fig,axs=plt.subplots(1,2,figsize=(7.3,3.4),layout='constrained')
for i,endpoint in enumerate(['waveform_safe','block_dense','local_sparse','local_dense']):
    g=pred[pred.endpoint==endpoint].set_index('model');r=g.loc['state_psd_seed_mean']
    axs[0].errorbar(r['mean'],i,xerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=BLUE,capsize=3)
    for offset,s in zip([-.13,0,.13],[19,43,71]):axs[1].scatter(g.loc[f'state_psd_seed{s}','score'],i+offset,color=[BLUE,ORANGE,GREEN][[19,43,71].index(s)],marker='os^'[[19,43,71].index(s)],s=30,label=f'Seed {s}' if i==0 else None)
axs[0].set_yticks(range(4),[labels[x] for x in labels]);axs[0].invert_yaxis();axs[0].axvline(0,color='.5',ls='--');axs[0].set(xlabel='Ensemble minus ridge NRMSE',title='(a) Paired prediction error')
axs[1].set_yticks(range(4),['']*4);axs[1].invert_yaxis();axs[1].set(xlabel='Individual-seed NRMSE',title='(b) Training seeds');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,fontsize=8,frameon=False)
save(fig,'fig03_prediction')

bounds=pd.read_csv(R/'conditional_ranges_summary.csv');choice=json.loads((R/'locked_design.json').read_text())
numbers['RevSelectedLags']=','.join(map(str,choice['lags']))
names={'historical':'Historical 8','manual':'Manual 8','algorithmic':'Selected 8','augmented':'Augmented 10'}
latex_table(G/'revision_bounds.tex',['Design','Real signs/cases','Imaginary signs/cases','Real width','Imaginary width'],[[names[name],f'{int(g.loc["real","identified"])}/{int(g.loc["real","cases"])}',f'{int(g.loc["imaginary","identified"])}/{int(g.loc["imaginary","cases"])}',f'{g.loc["real","mean_width"]:.3f}',f'{g.loc["imaginary","mean_width"]:.3f}'] for name in names for g in [bounds[bounds.design==name].set_index('axis')]])
for name in names:
    r=bounds[(bounds.design==name)&(bounds.axis=='real')].iloc[0]
    numbers['Rev'+name.title()+'Signs']=str(int(r.identified));numbers['Rev'+name.title()+'Cases']=str(int(r.cases))
effects=pd.read_csv(R/'design_paired_effects.csv');macros('DesignWidth',effects[(effects.axis=='real')&(effects.quantity=='width')&(effects.design=='algorithmic')].iloc[0]);macros('DesignSigns',effects[(effects.axis=='real')&(effects.quantity=='identified')&(effects.design=='algorithmic')].iloc[0])
fig,axs=plt.subplots(1,2,figsize=(7.3,3.3),layout='constrained',sharey=True)
for i,name in enumerate(names):
    for offset,axis,color in [(-.13,'real',BLUE),(.13,'imaginary',ORANGE)]:
        r=bounds[(bounds.design==name)&(bounds.axis==axis)].iloc[0]
        axs[0].barh(i+offset,r.identified/r.cases,height=.23,color=color,label=axis.title() if i==0 else None)
        axs[1].barh(i+offset,r.mean_width,height=.23,color=color)
axs[0].set_yticks(range(4),list(names.values()));axs[0].invert_yaxis();axs[0].set(xlim=(0,.1),xlabel='Fraction with identified sign',title='(a) Unseen sign decisions')
axs[1].set(xlim=(0,2),xlabel='Mean compatible range width',title='(b) Conditional ambiguity')
axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2,fontsize=9,frameon=False)
save(fig,'fig04_conditional_design')

down=pd.read_csv(R/'processing_comparisons.csv');take=down[(down.endpoint=='full')&(down.reference=='reference / Diagonal')]
ordered=['block_sparse / recurrence','block_dense / recurrence','local_sparse / recurrence','local_dense / recurrence','local_dense / block ridge','reference / Classical shrinkage']
nice=['Block sparse recurrence','Block dense recurrence','Local sparse recurrence','Local dense recurrence','Local dense block ridge','Classical shrinkage']
latex_table(G/'revision_processing.tex',['Fixed 32-channel method',r'Power change from diagonal (dB) [95\% CI]'],[[label,ci(take[take.method==method].iloc[0])] for method,label in zip(ordered,nice)])
for prefix,method,reference in [('BlockDenseGain','block_dense / recurrence','block_sparse / recurrence'),('LocalDenseGain','local_dense / recurrence','local_sparse / recurrence'),('LocalDenseRidgeGain','local_dense / block ridge','local_sparse / block ridge'),('LocalVersusClassical','local_dense / recurrence','reference / Classical shrinkage')]:
    r=down[(down.endpoint=='full')&(down.method==method)&(down.reference==reference)].iloc[0];macros(prefix,r)
fig,ax=plt.subplots(figsize=(7.3,3.8),layout='constrained')
for i,method in enumerate(ordered):
    r=take[take.method==method].iloc[0];ax.errorbar(r['mean'],i,xerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=GREEN if 'dense' in method else BLUE,capsize=3)
ax.set_yticks(range(len(ordered)),nice);ax.invert_yaxis();ax.axvline(0,color='.5',ls='--');ax.set_xlabel('Target / reference-noise power change from diagonal (dB)');ax.set_title('Fixed 32-channel geometry')
save(fig,'fig05_geometry_processing')

ev=pd.read_csv(R/'processing_event_seed_scores.csv',dtype={'event_id':str});ev=ev[ev.evaluation=='full']
fig,axs=plt.subplots(1,2,figsize=(7.3,3.4),layout='constrained')
for j,(sparse,dense,label,color) in enumerate([('block_sparse / recurrence','block_dense / recurrence','Block targets',BLUE),('local_sparse / recurrence','local_dense / recurrence','Exact local targets',ORANGE)]):
    means=[]
    for s in [19,43,71]:
        e=ev[ev.seed==s].pivot(index='event_id',columns='method',values='ratio_db');means.append(float((e[dense]-e[sparse]).mean()))
    axs[0].plot([19,43,71],means,'o-',label=label,color=color)
delete=pd.read_csv(R/'processing_component_deletion.csv')
for j,(method,reference,label,color) in enumerate([('block_dense / recurrence','block_sparse / recurrence','Block targets',BLUE),('local_dense / recurrence','local_sparse / recurrence','Exact local targets',ORANGE)]):
    g=delete[(delete.endpoint=='full')&(delete.method==method)&(delete.reference==reference)]
    axs[1].scatter(g['mean'],np.full(len(g),j),color=color,s=27,alpha=.65)
axs[0].axhline(0,color='.5',ls='--');axs[0].set(xlabel='Training seed',ylabel='Dense minus sparse (dB)',title='(a) Every fixed seed');axs[0].set_xticks([19,43,71]);axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False,fontsize=8)
axs[1].set_yticks([0,1],['Block','Local']);axs[1].axvline(0,color='.5',ls='--');axs[1].set(xlabel='Dense minus sparse (dB)',title='(b) Component deletion')
save(fig,'fig06_robustness')
audit=json.loads((R/'starting_audit.json').read_text());numbers['RevMinimumLead']=f'{audit["minimum_lead_seconds"]:.2f}'
raw=pd.read_csv(R/'direct_raw_local_validation.csv');numbers['RevRawLocalMax']=f'{raw.max_complex_difference.max():.2e}'
latex_table(G/'revision_synthetics.tex',['Condition','Cases','Valid','Truth in range','Signs','Incorrect'],[[r.family.replace('_',' '),str(r.cases),str(r.valid),str(r.covered),str(r.identified),str(r.incorrect)] for r in pd.read_csv(R/'complex_known_truth_summary.csv').itertuples()])
(G/'revision_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in numbers.items())+'\n')
(R/'manuscript_numbers.json').write_text(json.dumps(numbers,indent=2))
paper=ROOT/'manuscript/cageo_submission'
if (paper/'supplement.tex').exists():
    import re
    def expanded(path):
        content=path.read_text(encoding='utf8')
        def include(match):
            target=paper/match.group(1)
            if not target.suffix:target=target.with_suffix('.tex')
            return '' if target.name=='revision_inventory.tex' else expanded(target)
        return re.sub(r'\\input\{([^}]+)\}',include,content)
    source=expanded(paper/'supplement.tex')
    inventory=dict(text_sections=source.count('\\section{'),tables=source.count('\\begin{table}'),figures=source.count('\\begin{figure}'))
    (G/'revision_inventory.tex').write_text(f'The complete supplementary inventory comprises {inventory["text_sections"]} numbered Text sections, {inventory["tables"]} Tables and {inventory["figures"]} Figures. All are included in this PDF; complete numerical source tables accompany the public scientific assets.\n')
    (R/'supplement_inventory.json').write_text(json.dumps(inventory,indent=2))
print('Generated six main figures and result-linked tables/macros')
