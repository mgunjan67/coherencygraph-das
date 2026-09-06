"""Raw-waveform audit and estimator sensitivities for Amendment 07."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import time
import h5py
import numpy as np
import pandas as pd

from .critical_revision import ROOT, OUT, bundle, components
from .config import write_json
from .data import _features
from .labels import _read_window, RAW_DATASET
from .spectral import robust_linear_pick_model, multitaper_fourier, band_snapshots, lag_coherency, cross_spectral_matrix

CACHE=ROOT/'data/processed/critical_review'


def _cache_stem(value):
    """Read a Windows-produced receipt on either Windows or POSIX."""
    return Path(str(value).replace('\\', '/')).stem


def pooled_coherency(x,lags):
    out=[]
    for lag in lags:
        a,b=x[:,:-lag],x[:,lag:]
        out.append(np.mean(a*np.conj(b))/max(np.sqrt(np.mean(abs(a)**2)*np.mean(abs(b)**2)),np.finfo(float).eps))
    return np.asarray(out)


def _one_record(row,route,old_path,cfg):
    started=time.perf_counter()
    output=CACHE/f'{row["event_id"]}_{route.replace("-","")}.npz'
    receipt=output.with_suffix('.json')
    if output.exists() and receipt.exists(): return json.loads(receipt.read_text())
    prefix='terra' if route=='TERRA' else 'kkfls'
    picks=pd.read_csv(row['picks_path'])
    ch=picks.channel.to_numpy(float)
    s=pd.to_numeric(picks['TERRA_s_sec' if route=='TERRA' else 'KKFLS_s_sec'],errors='coerce').to_numpy(float)
    spectral=cfg['spectral']; starts=spectral['block_starts']; n=spectral['block_channels']; fs=25.
    fits=[]
    for start in starts:
        ref,slope,cov=robust_linear_pick_model(ch,s,start,n)
        if not np.isfinite(ref): ref,slope,cov=float(np.nanmedian(s)),0.,0.
        fits.append((ref,slope,cov))
    with np.load(old_path) as z: old={k:z[k] for k in z.files}
    variants={key:{k:np.array(v,copy=True) for k,v in old.items()} for key in ['equal_7s','global_cutoff','pool_before_normalize']}
    timings=[]; quality=[]; subblocks=[]; power=[]
    corrected_noise=np.empty_like(old['noise_anchor'])
    first_anchor=np.empty_like(old['target_anchor']); second_anchor=np.empty_like(first_anchor)
    full_target=np.empty_like(first_anchor); full_context=np.empty_like(first_anchor)
    raw_pdiff=0.; raw_prediction_support_pass=True
    global_cutoff=min(f[0]+3 for f in fits)-1.
    refs=[x[0] for x in fits]
    with h5py.File(row[f'{prefix}_path'],'r') as h:
        raw=h[RAW_DATASET]
        global_latest=max(min(raw.shape[0],round((r+2)*fs))/fs for r in refs)
        for block,(start,(ref,slope,cov)) in enumerate(zip(starts,fits)):
            cols=slice(start,start+n)
            delay=slope*(np.arange(n)-(n-1)/2)
            intervals={'noise':(.5,5.5),'context':(ref-12,ref+2),'target':(ref+3,ref+10),
                       'equal_7s':(ref-5,ref+2),'global_cutoff':(global_cutoff-7,global_cutoff)}
            waves={name:_read_window(raw,a,b,fs,cols) for name,(a,b) in intervals.items()}
            transforms={name:multitaper_fourier(wave,fs,2.5,3,delay if name!='noise' else None) for name,wave in waves.items()}
            noise_corrected=multitaper_fourier(waves['noise'],fs,2.5,3,delay)
            half=len(waves['target'])//2
            halves=[multitaper_fourier(waves['target'][:half],fs,2.5,3,delay),
                    multitaper_fourier(waves['target'][half:],fs,2.5,3,delay)]
            stop=min(raw.shape[0],round((ref+2)*fs))
            target_first=max(0,round((ref+3)*fs))
            # Conditional invariance: arrival fit is explicitly held fixed. This
            # tests the waveform transform, not the undocumented arrival picker.
            if block==0:
                a=max(0,round((ref-12)*fs)); b=stop
                observed=np.asarray(raw[a:min(raw.shape[0],b+50),cols])
                changed=observed.copy()
                changed[b-a:]=np.random.default_rng(20260906).normal(size=changed[b-a:].shape)*1000
                f1,_=multitaper_fourier(observed[:b-a],fs,2.5,3,delay)
                f2,_=multitaper_fourier(changed[:b-a],fs,2.5,3,delay)
                raw_pdiff=max(raw_pdiff,float(np.max(abs(f1-f2))))
            timings.append(dict(event_id=row['event_id'],route=route,block=block,s_reference=ref,
                slope_seconds_channel=slope,pick_coverage=cov,context_first=max(0,round((ref-12)*fs))/fs,
                context_stop_exclusive=stop/fs,target_first=target_first/fs,
                local_waveform_lead=target_first/fs-max(stop/fs,5.5),
                full_route_waveform_lead=target_first/fs-max(global_latest,5.5),
                global_cutoff=global_cutoff,global_cutoff_lead=target_first/fs-global_cutoff,
                pick_computation_available_time='unknown',
                window_clipped=any(a<0 or b*fs>raw.shape[0] for a,b in intervals.values())))
            for band,bounds in enumerate(spectral['bands_hz']):
                snapshots={name:band_snapshots(*tr,bounds) for name,tr in transforms.items()}
                nc=band_snapshots(*noise_corrected,bounds)
                anchors=old['anchor_local']
                corrected_noise[block,band]=cross_spectral_matrix(nc[:,anchors],.001)
                full_target[block,band]=cross_spectral_matrix(snapshots['target'][:,anchors],.001)
                full_context[block,band]=cross_spectral_matrix(snapshots['context'][:,anchors],.001)
                first_anchor[block,band]=cross_spectral_matrix(band_snapshots(*halves[0],bounds)[:,anchors],.001)
                second_anchor[block,band]=cross_spectral_matrix(band_snapshots(*halves[1],bounds)[:,anchors],.001)
                for name in ['equal_7s','global_cutoff']:
                    variants[name]['context_gamma'][block,band]=lag_coherency(snapshots[name],spectral['channel_lags'])
                    variants[name]['context_noise_power_ratio'][block,band]=float(np.mean(abs(snapshots[name][:,anchors])**2)/max(np.mean(abs(snapshots['noise'][:,anchors])**2),1e-15))
                for key,name in [('context_gamma','context'),('noise_gamma','noise'),('target_gamma','target')]:
                    variants['pool_before_normalize'][key][block,band]=pooled_coherency(snapshots[name],spectral['channel_lags'])
                reconstructed=lag_coherency(snapshots['target'],spectral['channel_lags'])
                for li,lag in enumerate(spectral['channel_lags']):
                    quality.append(dict(event_id=row['event_id'],route=route,block=block,band=band,lag=lag,
                        channel_pairs=n-lag,snapshots=snapshots['target'].shape[0],tapers=3,
                        bins=snapshots['target'].shape[0]//3,finite_fraction=float(np.isfinite(waves['target']).mean()),
                        target_cache_difference=float(abs(reconstructed[li]-old['target_gamma'][block,band,li]))))
                for size in [250,500,1000]:
                    for offset in range(0,n,size):
                        gam=lag_coherency(snapshots['target'][:,offset:offset+size],spectral['channel_lags'])
                        for li,lag in enumerate(spectral['channel_lags']):
                            subblocks.append(dict(event_id=row['event_id'],route=route,block=block,band=band,
                                subblock_channels=size,subblock_start=offset,lag=lag,pairs=size-lag,
                                real=float(gam[li].real),imag=float(gam[li].imag),
                                full_real=float(reconstructed[li].real),full_imag=float(reconstructed[li].imag)))
            for name,wave in waves.items():
                power.append(dict(event_id=row['event_id'],route=route,block=block,window=name,
                    samples=len(wave),duration_sec=len(wave)/fs,NW=2.5,half_bandwidth_hz=2.5/(len(wave)/fs)))
    values={f'{key}_features':_features(value,route) for key,value in variants.items()}
    values['pool_before_normalize_targets']=components(variants['pool_before_normalize']['target_gamma'])
    values.update(noise_anchor_corrected=corrected_noise,target_anchor=full_target,context_anchor=full_context,
                  target_first_anchor=first_anchor,target_second_anchor=second_anchor)
    np.savez_compressed(output,**values)
    for name,rows in [('timing',timings),('quality',quality),('subblocks',subblocks),('spectral_support',power)]:
        pd.DataFrame(rows).to_parquet(CACHE/f'{output.stem}_{name}.parquet',index=False)
    result=dict(event_id=row['event_id'],route=route,path=str(output),seconds=time.perf_counter()-started,
        conditional_future_perturbation_difference=raw_pdiff,verified_pick_availability=False)
    write_json(receipt,result)
    return result


def run_raw_audit():
    cfg,ds,roles,_=bundle(); CACHE.mkdir(parents=True,exist_ok=True)
    cohort=pd.read_parquet(ROOT/'reports/audit/frozen_cohort.parquet');cohort.event_id=cohort.event_id.astype(str)
    lookup=cohort.set_index('event_id').to_dict('index')
    ix=np.flatnonzero(np.isin(roles,['model_development','calibration','architecture_test']))
    result=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(_one_record,dict(event_id=str(ds.event_ids[i]),**lookup[str(ds.event_ids[i])]),str(ds.routes[i]),str(ds.paths[i]),cfg) for i in ix]
        for n,future in enumerate(as_completed(futures),1):
            result.append(future.result())
            if n%10==0 or n==len(futures): print(f'raw audit {n}/{len(futures)}',flush=True)
    pd.DataFrame(result).to_csv(OUT/'raw_audit_manifest.csv',index=False)
    for name in ['timing','quality','subblocks','spectral_support']:
        frame=pd.concat([pd.read_parquet(CACHE/f"{_cache_stem(r['path'])}_{name}.parquet") for r in result],ignore_index=True)
        frame.to_parquet(OUT/f'raw_{name}.parquet',index=False)
        if name=='timing':
            write_json(OUT/'timing_summary.json',dict(records=len(result),blocks=len(frame),
                nonpositive_local_leads=int((frame.local_waveform_lead<=0).sum()),
                nonpositive_route_leads=int((frame.full_route_waveform_lead<=0).sum()),
                fraction_nonpositive_route_leads=float((frame.full_route_waveform_lead<=0).mean()),
                route_lead_quantiles=frame.full_route_waveform_lead.quantile([0,.25,.5,.75,1]).to_dict(),
                clipped_blocks=int(frame.window_clipped.sum()),
                conditional_cutoff_max_difference=max(r['conditional_future_perturbation_difference'] for r in result),
                task='retrospective prediction conditional on arrival estimates; online causal claim withdrawn'))
    print('raw audit complete',flush=True)
