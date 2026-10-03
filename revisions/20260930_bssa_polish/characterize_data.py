"""Describe frozen input records; never select events, alter masks or fit models."""
from pathlib import Path
import sys, json, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import h5py
from scipy.signal import detrend, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'revisions/20260922_unseen_validation'
sys.path.insert(0,str(ROOT/'src'))
from coherencygraph_das.causal_validation import DATASET, BLOCKS, metadata, sha256
DEST=HERE/'data_quality'
BANDS=[(.5,1),(1,2),(2,4),(4,8)]

def records():
    frames=[]
    for role,name in [('development','development_records.csv'),('validation','validation_records.csv')]:
        f=pd.read_csv(OLD/name,dtype={'event_id':str});f['role']=role;frames.append(f)
    return pd.concat(frames,ignore_index=True)

def process(row):
    receipt=DEST/'records'/f"{row['role']}_{row['event_id']}_{row['route']}.json"
    if receipt.exists():
        prior=json.loads(receipt.read_text())
        if prior['script_sha256']==sha256(__file__):return prior
    blocks=[]; windows=[]
    with h5py.File(row['raw_path'],'r') as h:
        info=metadata(h);raw=h[DATASET]
        for bi,start in enumerate(BLOCKS):
            x=np.asarray(raw[:2200,start:start+1000],dtype=np.float64)
            finite=np.isfinite(x);std=np.nanstd(x,axis=0)
            median=float(np.nanmedian(std));lo=np.nanmin(x,axis=0);hi=np.nanmax(x,axis=0)
            extrema=np.mean((x==lo)|(x==hi),axis=0)
            blocks.append(dict(block=bi+1,finite_fraction=float(finite.mean()),channels=x.shape[1],
                flat_channels=int(np.sum(std==0)),near_flat_channels=int(np.sum((std>0)&(std<median*.001))),
                high_variance_channels=int(np.sum(std>median*100)),
                repeated_extrema_channels=int(np.sum((std>0)&(extrema>=.01))),
                rms_q25=float(np.quantile(std,.25)),rms_median=median,rms_q75=float(np.quantile(std,.75))))
            local=x[:,484:516]
            for cutoff in [20,40,60,80]:
                for name,a,b in [('reference',.5,5.5),('context',cutoff-14,cutoff),('target',cutoff+1,cutoff+8)]:
                    z=local[int(np.ceil(a*25)):int(np.ceil(b*25))]
                    # Welch is for visualization only; primary inference retains its DPSS estimator.
                    f,p=welch(z,fs=25,nperseg=min(128,len(z)),axis=0,detrend='linear')
                    p=np.mean(p,axis=1)
                    bandpower=[float(np.sum(p[(f>=l)&(f<u)])*(f[1]-f[0])) for l,u in BANDS]
                    windows.append(dict(block=bi+1,cutoff=cutoff,window=name,
                        rms=float(np.sqrt(np.mean(detrend(z,axis=0,type='linear')**2))),
                        **{f'band_{i}':v for i,v in enumerate(bandpower)}))
    result=dict(event_id=row['event_id'],route=row['route'],role=row['role'],raw_path=row['raw_path'],
                script_sha256=sha256(__file__),metadata=info,blocks=blocks,windows=windows)
    receipt.write_text(json.dumps(result,indent=2))
    return result

def figures(file_table,block_table,window_table,cohort):
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','pdf.fonttype':42,
                        'axes.spines.top':False,'axes.spines.right':False})
    colors={'development':'#1667A4','validation':'#C25322'}
    fig,ax=plt.subplots(2,2,figsize=(7.5,6.6),layout='constrained')
    for role in colors:
        c=cohort[cohort.role.eq(role)]
        marker='o' if role=='development' else 's'
        style='-' if role=='development' else '--'
        ax[0,0].scatter(c.homer_hypocentral_km,c.depth_km,label=role.capitalize(),color=colors[role],marker=marker,s=20,alpha=.65)
        b=block_table[block_table.role.eq(role)].groupby(['event_id','route']).rms_median.median()
        vals=np.sort(np.log10(np.maximum(b,1e-30)))
        ax[0,1].plot(vals,np.arange(1,len(vals)+1)/len(vals),color=colors[role],linestyle=style,label=role.capitalize())
        w=window_table[window_table.role.eq(role)]
        context=w[w.window.eq('context')].groupby(['event_id','route'])[[f'band_{i}' for i in range(4)]].mean()
        shares=context.div(context.sum(axis=1),axis=0)
        ax[1,0].plot(range(4),shares.median(),marker=marker,linestyle=style,color=colors[role],label=role.capitalize())
        event_rms=w.groupby(['event_id','route','window']).rms.mean().unstack()
        ratio=20*np.log10(event_rms.target/event_rms.reference)
        v=np.sort(ratio)
        ax[1,1].plot(v,np.arange(1,len(v)+1)/len(v),color=colors[role],linestyle=style,label=role.capitalize())
    ax[0,0].set(xlabel='Hypocentral distance from Homer (km)',ylabel='Depth (km)')
    ax[0,1].set(xlabel='Log10 temporal standard deviation (rad)',ylabel='Fraction of route records')
    ax[1,0].set(xticks=range(4),xticklabels=['0.5–1','1–2','2–4','4–8'],xlabel='Frequency band (Hz)',ylabel='Median context band-power fraction',ylim=(0,1))
    ax[1,1].set(xlabel='Target/reference RMS ratio (dB)',ylabel='Fraction of route records')
    for i,a in enumerate(ax.flat):
        a.text(-.12,1.04,f'({chr(97+i)})',transform=a.transAxes,fontweight='bold');a.grid(alpha=.18)
    ax[0,0].legend(loc='upper right',fontsize=9)
    for ext in ['pdf','png','svg']:fig.savefig(HERE/f'figures/data_quality_overview.{ext}',dpi=250)
    plt.close(fig)
    # First locked primary event: selected by pre-outcome hash order, never by waveform appearance.
    event=str(pd.read_csv(OLD/'locked_candidate_order.csv',dtype={'event_id':str}).query('candidate_role == "primary"').iloc[0].event_id)
    frame=file_table[file_table.event_id.eq(event)]
    fig,axs=plt.subplots(2,2,figsize=(7.5,6.7),layout='constrained')
    for ri,route in enumerate(['TERRA','KKFL-S']):
        row=frame[frame.route.eq(route)].iloc[0]
        with h5py.File(row.raw_path,'r') as h:
            raw=h[DATASET]
            wide=np.asarray(raw[:2200,200:1200:5],float)
            local=np.asarray(raw[:2200,684:716],float)
        for ci,x in enumerate([wide,local]):
            a=axs[ri,ci];z=x-np.median(x,axis=0,keepdims=True)
            scale=max(float(np.median(np.abs(z)))*1.4826,1e-15)
            high=(999 if ci==0 else 31)*row.spacing_m/1000
            im=a.imshow((z/scale).T,aspect='auto',origin='lower',extent=[0,88,0,high],cmap='RdBu_r',vmin=-3,vmax=3,rasterized=True)
            a.set(xlabel='Time from record start (s)',ylabel=f'{route}: optical offset (km)')
            for lo,hi,col in [(.5,5.5,'#888888'),(6,20,'#1667A4'),(21,28,'#C25322')]:
                a.axvspan(lo,hi,alpha=.12,color=col);a.axvline(lo,color=col,lw=.8);a.axvline(hi,color=col,lw=.8)
            a.text(-.14,1.04,f'({chr(97+ri*2+ci)})',transform=a.transAxes,fontweight='bold')
    fig.colorbar(im,ax=axs,shrink=.8,label='Median-removed phase / robust scale')
    for ext in ['pdf','png','svg']:fig.savefig(HERE/f'figures/raw_wavefield_example.{ext}',dpi=250)
    plt.close(fig)
    return event

def main():
    DEST.mkdir(exist_ok=True);(DEST/'records').mkdir(exist_ok=True)
    rows=records(); results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(process,r) for r in rows.to_dict('records')]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%10==0:print('Characterized',len(results),'/',len(rows),flush=True)
    files=[];blocks=[];windows=[]
    for r in results:
        ident={k:r[k] for k in ['event_id','route','role']}
        files.append(dict(**ident,raw_path=r['raw_path'],**r['metadata']))
        blocks.extend(dict(**ident,**b) for b in r['blocks'])
        windows.extend(dict(**ident,**w) for w in r['windows'])
    files=pd.DataFrame(files).sort_values(['role','event_id','route']);blocks=pd.DataFrame(blocks);windows=pd.DataFrame(windows)
    dev=pd.read_csv(OLD/'excluded_previously_used_events.csv',dtype={'event_id':str})
    dev=dev[dev.event_id.isin(rows[rows.role.eq('development')].event_id)].copy();dev['role']='development'
    val=pd.read_csv(OLD/'locked_candidate_order.csv',dtype={'event_id':str}).query('candidate_role == "primary"').copy();val['role']='validation'
    cohort=pd.concat([dev,val],ignore_index=True)
    for name,frame in [('file_quality',files),('block_quality',blocks),('window_characteristics',windows),('cohort_characteristics',cohort)]:
        frame.to_csv(DEST/f'{name}.csv',index=False);frame.to_parquet(DEST/f'{name}.parquet',index=False)
    summary={}
    for role in ['development','validation']:
        f=files[files.role.eq(role)];b=blocks[blocks.role.eq(role)];c=cohort[cohort.role.eq(role)]
        summary[role]=dict(events=int(f.event_id.nunique()),files=len(f),blocks=len(b),
            finite_min=float(b.finite_fraction.min()),finite_max=float(b.finite_fraction.max()),
            flat_channels=int(b.flat_channels.sum()),near_flat_channels=int(b.near_flat_channels.sum()),
            high_variance_channels=int(b.high_variance_channels.sum()),repeated_extrema_channels=int(b.repeated_extrema_channels.sum()),
            channel_records=int(b.channels.sum()),magnitude_range=[float(c.magnitude.min()),float(c.magnitude.max())],
            depth_range=[float(c.depth_km.min()),float(c.depth_km.max())],
            distance_range=[float(c.homer_hypocentral_km.min()),float(c.homer_hypocentral_km.max())])
    summary['example_event']=figures(files,blocks,windows,cohort)
    summary['interpretation']='Retrospective descriptive diagnostics; no exclusion, reweighting, model tuning or alteration of frozen results. Repeated extrema do not establish clipping without instrument limits.'
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
