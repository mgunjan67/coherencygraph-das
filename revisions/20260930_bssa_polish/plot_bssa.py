"""BSSA artwork from frozen source tables; no score estimation or model changes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'revisions/20260922_unseen_validation'
FIG=HERE/'figures'
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':11})
sources=[OLD/'validation/paired_comparisons.csv',OLD/'tables/descriptive_strata.csv']
rows=pd.read_csv(sources[0]);strata=pd.read_csv(sources[1])
def panel(ax,label):
    ax.text(-.04,1.035,f'({label})',transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=11)
def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=300)
    plt.close(fig)
def row(model,reference):
    s=rows[(rows.endpoint=='full')&(rows.resampling=='earthquake')&(rows.model==model)&(rows.reference==reference)]
    assert len(s)==1
    return s.iloc[0]
fig,axes=plt.subplots(2,1,figsize=(6.5,5.8),layout='constrained',gridspec_kw={'height_ratios':[3,5]})
models=['dense_recurrence','sparse_recurrence','dense_full_ridge','sparse_full_ridge','fixed_shrinkage','local_persistence']
labels=['Dense recurrence','Sparse recurrence','Dense ridge','Sparse ridge','Fixed shrinkage','Lag persistence']
for p,ax in enumerate(axes):
    pairs=[('dense_recurrence',x) for x in ['sparse_recurrence','dense_full_ridge','fixed_shrinkage']] if p==0 else [(x,'diagonal') for x in models]
    names=['Sparse recurrence','Dense ridge','Fixed shrinkage'] if p==0 else labels
    for i,pair in enumerate(pairs):
        r=row(*pair);ax.errorbar(r['mean'],i,xerr=[[r['mean']-r['low']],[r['high']-r['mean']]],fmt='o',color='#20639B',capsize=4)
    ax.set_yticks(range(len(names)),names);ax.set_ylim(len(names)-.5,-.5)
    ax.axvline(0,color='.4',ls='--');ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('Dense recurrence minus comparator (dB)' if p==0 else 'Method minus diagonal processing (dB)')
    panel(ax,chr(97+p))
save(fig,'fixed_window_benchmarks')
fig,axes=plt.subplots(1,2,figsize=(6.5,4.4),layout='constrained')
refs=['sparse_recurrence','dense_full_ridge','fixed_shrinkage']
names=['Dense minus sparse recurrence','Dense recurrence minus ridge','Dense recurrence minus shrinkage']
for p,(ax,kind,ticks,xlabel) in enumerate(zip(axes,['band','cutoff'],[['0.5–1','1–2','2–4','4–8'],['20','40','60','80']],['Frequency band (Hz)','Context cutoff (s)'])):
    for k,(ref,col,mark) in enumerate(zip(refs,['#20639B','#C7522A','#278579'],['o','s','^'])):
        s=strata[(strata.stratum==kind)&(strata.model=='dense_recurrence')&(strata.reference==ref)].sort_values('value')
        assert len(s)==4
        ax.errorbar(np.arange(4)+(k-1)*.12,s.mean_db,yerr=[s.mean_db-s.low_db,s.high_db-s.mean_db],fmt=mark+'-',color=col,capsize=3,label=names[k])
    ax.axhline(0,color='.4',ls='--');ax.set_xticks(range(4),ticks);ax.set_xlabel(xlabel);ax.grid(axis='y',alpha=.15);panel(ax,chr(97+p))
axes[0].set_ylabel('Power-score difference (dB)')
h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',frameon=False,fontsize=10)
save(fig,'fixed_window_strata')
fig,axes=plt.subplots(3,1,figsize=(6.5,5.5),layout='constrained',gridspec_kw={'height_ratios':[1.5,1,1.5]})
ax=axes[0]
for y,lo,hi,col in [(2,.5,5.5,'#8A96A8'),(1,6,20,'#20639B'),(0,21,28,'#C7522A')]:ax.add_patch(Rectangle((lo,y-.2),hi-lo,.4,color=col))
ax.axvline(20,ls='--',color='.4');ax.set(xlim=(0,30),ylim=(-.5,2.5),yticks=[0,1,2],yticklabels=['Target','Context','Reference'],xlabel='Time from record start (s)');panel(ax,'a')
ax=axes[1];ax.set(xlim=(0,1000),ylim=(-.5,1.1));ax.axis('off')
ax.add_patch(Rectangle((0,.2),1000,.4,facecolor='#E7EEF5',edgecolor='#20639B'))
ax.add_patch(Rectangle((484,.2),32,.4,color='#20639B'))
ax.annotate('32 local channels',xy=(500,.6),xytext=(720,.95),ha='center',arrowprops={'arrowstyle':'-','color':'.4'},fontsize=10)
ax.text(500,-.15,'1,000-channel block; eight fixed locations',ha='center',fontsize=10);panel(ax,'b')
ax=axes[2];ax.set(xlim=(-.02,1.02),ylim=(0,1));ax.axis('off');panel(ax,'c')
for x,title,body,col in [(0,'Earlier inputs','Block coherencies\nLocal power ratios\nRoute / block / gauge','#20639B'),(.35,'Matched training','5 versus 31 lags\nSame local aperture\n93 earthquakes','#278579'),(.70,'Locked test','30 new earthquakes\nProcessing and errors\nNon-neural baselines','#C7522A')]:
    ax.add_patch(FancyBboxPatch((x,.08),.29,.82,boxstyle='round,pad=.004',facecolor='white',edgecolor=col))
    ax.text(x+.145,.75,title,ha='center',fontweight='bold',fontsize=10)
    ax.text(x+.145,.40,body,ha='center',va='center',linespacing=1.5,fontsize=10)
for x in [.30,.65]:ax.annotate('',xy=(x+.04,.48),xytext=(x,.48),arrowprops={'arrowstyle':'->','color':'.4'})
save(fig,'fixed_window_design')
receipt={'operation':'Artwork-only regeneration from frozen tables','sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in FIG.glob('fixed_window*')}}
(HERE/'production/artwork_receipt.json').write_text(json.dumps(receipt,indent=2))
