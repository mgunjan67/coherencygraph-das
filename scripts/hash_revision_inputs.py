"""Checksums for all raw records and picks referenced by the frozen cohort."""
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from coherencygraph_das.config import sha256
OUT=ROOT/'reports/critical_review'
cohort=pd.read_parquet(ROOT/'reports/audit/frozen_cohort.parquet')
def one(spec):
    event,kind,path,url=spec;p=Path(path)
    return dict(event_id=str(event),kind=kind,filename=p.name,url=url,bytes=p.stat().st_size,sha256=sha256(p),local_path=str(p))
work=[(r.event_id,k,getattr(r,k+'_path'),getattr(r,k+'_url')) for r in cohort.itertuples() for k in ['terra','kkfls','picks']]
with ThreadPoolExecutor(max_workers=4) as pool:
    results=[]
    for i,r in enumerate(pool.map(one,work)):
        results.append(r)
        if i%75==0:print(f'Checksummed {i+1}/{len(work)} inputs',flush=True)
pd.DataFrame(results).to_csv(OUT/'raw_input_checksums.csv',index=False)
print('Raw/pick checksum manifest complete:',len(results))
