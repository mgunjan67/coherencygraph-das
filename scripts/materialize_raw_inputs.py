"""Resumable public-input downloader, explicit events only, checksummed.

python scripts/materialize_raw_inputs.py --root /data/cidas --events 11710081
Use --existing-only to rebase a manifest onto already downloaded files.
"""
from pathlib import Path
import sys,argparse,shutil,urllib.request,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from coherencygraph_das.config import sha256
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--events',nargs='+',required=True);p.add_argument('--existing-only',action='store_true');a=p.parse_args()
destination=Path(a.root).resolve();destination.mkdir(parents=True,exist_ok=True)
manifest=pd.read_csv(ROOT/'reports/critical_review/raw_input_checksums.csv',dtype={'event_id':str})
chosen=manifest[manifest.event_id.isin(a.events)]
if set(a.events)-set(chosen.event_id):raise SystemExit('Requested event missing from checksummed manifest.')
for r in chosen.itertuples():
    folder=destination/r.event_id;folder.mkdir(exist_ok=True);target=folder/r.filename
    if target.exists():
        if sha256(target)==r.sha256:continue
        raise SystemExit(f'Existing file checksum mismatch, not overwritten: {target}')
    if a.existing_only:raise SystemExit(f'Missing existing input: {target}')
    if shutil.disk_usage(destination).free-r.bytes<100*1024**3:raise SystemExit('Download would violate the 100-GiB free-space reserve.')
    partial=target.with_suffix(target.suffix+'.part');offset=partial.stat().st_size if partial.exists() else 0
    headers={'User-Agent':'CoherencyGraph-reproducibility/1.1'}
    if offset:headers['Range']=f'bytes={offset}-'
    with urllib.request.urlopen(urllib.request.Request(r.url,headers=headers),timeout=45) as response:
        mode='ab' if offset and response.status==206 else 'wb'
        with partial.open(mode) as output:shutil.copyfileobj(response,output,length=1024*1024)
    if partial.stat().st_size!=r.bytes or sha256(partial)!=r.sha256:raise SystemExit(f'Checksum mismatch; partial file retained for inspection: {partial}')
    partial.replace(target)
    print('Verified',target,flush=True)
cohort_path=ROOT/'reports/audit/frozen_cohort.parquet';cohort=pd.read_parquet(cohort_path);cohort.event_id=cohort.event_id.astype(str)
for r in chosen.itertuples():cohort.loc[cohort.event_id==r.event_id,r.kind+'_path']=str(destination/r.event_id/r.filename)
# Rebase local paths only; never change cohort roles, source bins or values.
cohort.to_parquet(cohort_path,index=False)
print('Manifest local paths rebased; event roles and scientific metadata unchanged.')
