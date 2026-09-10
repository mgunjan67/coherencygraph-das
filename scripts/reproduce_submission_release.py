"""Reconstruct the versioned final scientific release; never train or tune."""
from pathlib import Path
import argparse, json, os, subprocess, sys, urllib.request, shutil
from reproduce_closeout_assets import sha, safe_extract, manifest_files, run_logged

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,default=ROOT/'reproduced_submission')
    parser.add_argument('--asset-cache',type=Path,help='Optional directory containing the four checksum-pinned ZIP files')
    parser.add_argument('--expected-commit',help='Local release-candidate verification only; otherwise require the public release tag')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/submission_release.json').read_text())
    git=['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT)]
    head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    expected=args.expected_commit or subprocess.check_output(git+['rev-parse',cfg['release']+'^{commit}'],text=True).strip()
    if head!=expected:raise RuntimeError('Source version mismatch')
    if subprocess.check_output(git+['status','--porcelain','--untracked-files=no'],text=True).strip():raise RuntimeError('Tracked source has uncommitted changes')
    work=args.work.resolve()
    if work.exists():raise FileExistsError('Refusing to overwrite '+str(work))
    work.mkdir(parents=True); logs=work/'verification_logs';logs.mkdir()
    source=work/'release_source.zip'
    subprocess.run(git+['archive','--format=zip','--output='+str(source),head],check=True)
    records=[]
    for asset in cfg['assets']:
        local=args.asset_cache/asset['name'] if args.asset_cache else work/asset['name']
        if not local.exists():
            with urllib.request.urlopen(asset['url'],timeout=60) as src,local.open('wb') as dst:shutil.copyfileobj(src,dst)
        if sha(local)!=asset['sha256']:raise RuntimeError('Archive checksum mismatch: '+asset['name'])
        safe_extract(local,work)
        count=manifest_files(work,work/asset['manifest'])
        records.append(dict(name=asset['name'],sha256=sha(local),verified_members=count))
        print('Verified',asset['name'],count,flush=True)
    safe_extract(source,work)
    # Source export may have LF/CRLF serialization differences. Only restore an
    # exact pre-existing receipt hash, never accept different scientific content.
    specifications=[work/'reports/audit_design_utility/frozen_receipt.json',work/'reports/submission_closeout/audit_validation/specification.json',work/'reports/submission_closeout/processing/specification.json']
    import hashlib
    for p in specifications:
        receipt=json.loads(p.read_text())
        expected_sources=receipt.get('input_sha256',receipt.get('sources',{})).copy()
        if 'script_sha256' in receipt:expected_sources['scripts/evaluate_audit_design_utility.py']=receipt['script_sha256']
        if p.name=='frozen_receipt.json':expected_sources['configs/audit_design_utility.json']=receipt['specification_sha256']
        for rel,digest in expected_sources.items():
            path=work/rel
            if sha(path)==digest:continue
            raw=path.read_bytes();lf=raw.replace(b'\r\n',b'\n')
            candidates=[v for v in [lf,lf.replace(b'\n',b'\r\n')] if hashlib.sha256(v).hexdigest()==digest]
            if not candidates:raise RuntimeError('Frozen input mismatch: '+rel)
            path.write_bytes(candidates[0])
    env=os.environ.copy();env.update(PYTHONPATH=str(work/'src'),CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg')
    commands=[
      ['examples/closeout_datafree.py'],
      ['scripts/run_submission_revision.py','regenerate'],
      ['scripts/run_submission_revision.py','verify'],
      ['scripts/closeout_prediction_aggregation.py'],
      ['scripts/closeout_audit_validation.py','summarise'],
      ['scripts/closeout_matched_processing.py','summarise'],
      ['scripts/build_submission_revision_outputs.py'],
      ['scripts/generate_revision_supplement.py'],
      ['scripts/build_final_revision_outputs.py'],
      ['scripts/evaluate_audit_design_utility.py','summarise'],
      ['scripts/build_audit_design_utility.py'],
      ['scripts/build_closeout_outputs.py'],
      ['scripts/draw_submission_architecture.py'],
      ['scripts/verify_submission_numbers.py'],
      ['scripts/verify_submission_release.py'],
      ['-m','pytest','-o','addopts=','tests/test_closeout_api.py','tests/test_closeout_processing.py','tests/test_closeout_audit_validation.py','tests/test_closeout_prediction.py','tests/test_submission_release_api.py','--junitxml=final_release_tests.xml']]
    runs=[]
    for i,command in enumerate(commands):
        row,p=run_logged([sys.executable,*command],work,env,logs,f'{i:02d}_{Path(command[0]).stem}');runs.append(row)
        if p.returncode:raise RuntimeError('Numerical reproduction failed: '+str(command)+'; '+row['log'])
    receipt=dict(status='PASS',source_commit=head,release=cfg['release'],assets=records,runs=runs,no_training=True,no_raw_reextraction=True,no_hosted_AI_required=True,verification_scope='Cached prediction regeneration, diagnostic and bootstrap reconstruction, source-table/figure regeneration, public numeric contract and tests; not independent human replication')
    (work/'FINAL_RELEASE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('FINAL SCIENTIFIC REPRODUCTION PASS',flush=True)

if __name__=='__main__':main()
