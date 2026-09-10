"""Fixed geometry illustration, actual API example and bounded CPU timing."""
from pathlib import Path
import os,sys,json,time,platform,threading,statistics,importlib.metadata as metadata
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import pandas as pd
import psutil
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from coherencygraph_das.closeout_api import audit_moments
from coherencygraph_das.critical_revision import bundle,Q
from coherencygraph_das import submission_revision as sr
torch.set_num_threads(1)
OUT=ROOT/'reports/submission_closeout/examples';OUT.mkdir(parents=True,exist_ok=True)

def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2),encoding='utf8')

def main():
    assert (ROOT/'reports/submission_closeout/CLOSEOUT_SPECIFICATION.md').exists()
    lags=np.array([1,2,3,5,8,13,21,34]);positions=np.arange(484,516);unseen=np.array([4,10])
    rows=[]
    with threadpool_limits(limits=1):
        for name in ['uniform','uniform_plus_dc']:
            p=np.ones(257)/257
            if name=='uniform_plus_dc':p*=.1;p[np.argmin(abs(Q))]+=.9
            y=np.exp(1j*lags[:,None]*Q)@p
            result=audit_moments(lags,y,positions,np.full(16,.005),unseen)
            truth=np.exp(1j*unseen[:,None]*Q)@p
            for row in result['intervals']:
                t=truth[list(unseen).index(row['lag'])]
                t=float(t.real if row['axis']=='real' else t.imag)
                assert row['valid'] and row['lower']<=t<=row['upper']
                rows.append(dict(fixture=name,lag=row['lag'],axis=row['axis'],lower=row['lower'],upper=row['upper'],width=row['width'],truth=t,status=row['status'],residual_max=max(result['residual_components'])))
            write(name+'.json',result)
        pd.DataFrame(rows).to_csv(OUT/'fixed_geometry_examples.csv',index=False)
        _,ds,roles,_=bundle()
        locked=json.loads((sr.OUT/'locked_design.json').read_text())
        # Allowance schema is deliberately resolved from the existing lock.
        allowance=np.load(sr.MOD/'development_component_tolerance.npy')
        assert locked['lags']==lags.tolist()
        use=np.flatnonzero((ds.event_ids.astype(str)=='11741048')&(ds.routes=='TERRA'))
        assert len(use)==1 and roles[use[0]]=='model_development'
        inputs=[]
        for i in [int(use[0])]+list(np.flatnonzero(roles=='architecture_test')):
            with np.load(sr.cache_path(ds,i)) as z:
                mean=z['target_gamma'].mean(0);measured=z['measured_lags'].tolist()
            ids=[measured.index(int(d)) for d in lags]
            for band in ([0] if i==use[0] else range(4)):
                # Native allowance is [band, measured lag, real/imaginary].
                delta=np.r_[allowance[band,ids,0],allowance[band,ids,1]]
                inputs.append((str(ds.event_ids[i]),str(ds.routes[i]),band,mean[band,ids],delta))
        warm=audit_moments(lags,inputs[0][3],positions,inputs[0][4],unseen)
        write('real_development_example.json',dict(event_id=inputs[0][0],route=inputs[0][1],band=0,selection='first frozen development event, first route/band; specified before outcomes',**warm))
        process=psutil.Process();memory=[process.memory_info().rss];stop=threading.Event()
        def sample():
            while not stop.wait(.1):memory.append(process.memory_info().rss)
        monitor=threading.Thread(target=sample,daemon=True);monitor.start()
        times=[]
        for repeat in range(3):
            begin=time.perf_counter();a=audit_moments(lags,inputs[0][3],positions,inputs[0][4],unseen);times.append(time.perf_counter()-begin)
            assert all(r['valid'] for r in a['intervals'])
        begin=time.perf_counter();records=[]
        for event,route,band,y,delta in inputs[1:]:
            r=audit_moments(lags,y,positions,delta,unseen)
            records += [dict(event_id=event,route=route,band=band,**{k:v for k,v in x.items() if k not in ['low','high']}) for x in r['intervals']]
            if len(records)%64==0:print('Cost/optimizer replay',len(records)//4,'/192',flush=True)
        seconds=time.perf_counter()-begin;stop.set();monitor.join()
        pd.DataFrame(records).to_csv(OUT/'timed_cohort_intervals.csv',index=False)
        assert len(records)==768 and all(r['valid'] for r in records)
        environment=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),processor=platform.processor(),logical_cpus=psutil.cpu_count(),cpu_only=True,worker_processes=1,pytorch_threads=torch.get_num_threads(),blas_threadpools=threadpool_info(),dependencies={name:metadata.version(name) for name in ['numpy','scipy','pandas','torch','scikit-learn','threadpoolctl','psutil']},timing_conditions='Serial cached-data optimizer replay; concurrent local closeout jobs may contend for CPU. Imports, warm-up, cache reads and neural training excluded. Each call includes simplex fitting and eight LP solves.',single_call_repetitions_seconds=times,single_call_median_seconds=statistics.median(times),cohort_seconds=seconds,cohort_fit_calls=192,cohort_lp_calls=1536,cohort_events=24,cohort_routes=2,cohort_bands=4,peak_observed_process_rss_bytes=max(memory),memory_sampling_seconds=.1)
        write('cost_environment.json',environment);print(json.dumps(environment,indent=2),flush=True)
if __name__=='__main__':main()
