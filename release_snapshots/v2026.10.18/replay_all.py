"""Portable numerical replay orchestration; no network/model fitting."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--with-inference-smoke',action='store_true')
    a=p.parse_args();root=a.package_root.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError('Use new output; do not erase prior evidence')
    out.mkdir(parents=True);started=time.perf_counter()
    manifest=json.loads((root/'ASSET_MANIFEST.json').read_text())
    for entry in manifest['files']:
        target=(root/entry['path']).resolve()
        assert target.is_relative_to(root) and target.is_file()
        assert sha(target)==entry['sha256'],entry['path']
    commands=[['replay.py','--package-root',str(root),'--output-dir',str(out/'core')],
              ['portable_frontier_replay.py','--package-root',str(root),'--output-dir',str(out/'frontier')]]
    extension=root/'revisions/20261018_no_human_closeout/implementation'
    if (extension/'lcmv_ricker_results/all_new_cases.parquet').exists() or (extension/'recorded_component_results/all_cases.parquet').exists():
        if not (extension/'lcmv_ricker_results/all_new_cases.parquet').is_file() or not (extension/'recorded_component_results/all_cases.parquet').is_file():
            raise RuntimeError('Incomplete finite extension assets; both declared panels required')
        commands.append(['portable_extension_replay.py','--package-root',str(root),'--output-dir',str(out/'extension')])
    if a.with_inference_smoke:commands.append(['smoke_archived_inference.py','--package-root',str(root),'--output-dir',str(out/'inference')])
    results=[]
    for i,command in enumerate(commands):
        args=[sys.executable,str(root/command[0]),*command[1:]]
        r=subprocess.run(args,cwd=root,text=True,capture_output=True,timeout=300)
        (out/f'{i:02d}_{command[0]}.log').write_text(r.stdout+'\nSTDERR\n'+r.stderr,encoding='utf-8')
        results.append(dict(script=command[0],exit_code=r.returncode))
        if r.returncode:raise RuntimeError('Replay failure: '+command[0])
    report=dict(status='PASS_PORTABLE_CORE_FRONTIER_AND_REQUESTED_SMOKE',verified_files=len(manifest['files']),
        elapsed_seconds=time.perf_counter()-started,commands=results,network=False,training=False,
        raw_waveforms=False,independent_human_replication=False,
        finite_extension_replayed=any(x['script']=='portable_extension_replay.py' for x in results),
        scope='Core derived reductions, all-four-date frontier recomputation, present finite-extension scalar tables and optional single statistical-case checkpoint inference. Original training/raw extraction/figure artwork not regenerated.')
    (out/'PORTABLE_REPLAY_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
