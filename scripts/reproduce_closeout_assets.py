"""One-command, manifest-driven scientific asset reconstruction; no training.

Never changes old source checkouts, releases or tags. Existing public ZIP bytes
may be reused, but their SHA-256 is always checked before extraction. Download
fallback is public and unauthenticated. No credentials, publication or private
manuscript data are used.
"""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, os, shutil, subprocess, sys, time, urllib.request, zipfile
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')

def safe_destination(root,name):
    clean=PurePosixPath(name.replace('\\','/'))
    if clean.is_absolute() or '..' in clean.parts or any(':' in part for part in clean.parts):
        raise ValueError(f'Unsafe archive path: {name}')
    dest=(root/str(clean)).resolve()
    if not dest.is_relative_to(root.resolve()):raise ValueError(f'Archive escape: {name}')
    return dest

def safe_extract(path,root):
    count=0
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, f'CRC error: {path}'
        members=z.infolist()
        for info in members:
            if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Symbolic links are unsupported')
            safe_destination(root,info.filename)
        for info in members:
            dest=safe_destination(root,info.filename)
            if info.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
            count+=1
    return count

def manifest_files(work,path):
    data=json.loads(path.read_text())
    rows=data if isinstance(data,list) else data['files']
    for row in rows:
        p=safe_destination(work,row['path'])
        assert p.is_file() and sha(p)==row['sha256'],row['path']
        if 'bytes' in row:assert p.stat().st_size==row['bytes']
    return len(rows)

def run_logged(argv,cwd,env,logs,name):
    start=time.perf_counter()
    p=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True,encoding='utf8',errors='replace')
    log=logs/(name+'.txt');log.write_text(p.stdout+'\n'+p.stderr,encoding='utf8')
    row=dict(command=argv,cwd=str(cwd),exit_code=p.returncode,seconds=time.perf_counter()-start,
        log=str(log),log_sha256=sha(log))
    print(name,'exit',p.returncode,'seconds',round(row['seconds'],2),flush=True)
    return row,p

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=ROOT/'configs/closeout_reproduction_manifest.json')
    parser.add_argument('--work',type=Path,default=ROOT/'publication/github/closeout_reproduction')
    parser.add_argument('--source-repository',help='Optional already available read-only Git object source; default is public remote')
    parser.add_argument('--reuse-prepared',action='store_true',help='Resume commands only after recorded reconstruction and input checks')
    args=parser.parse_args();cfg=json.loads(args.manifest.read_text());work=args.work.resolve()
    reports=ROOT/'reports/submission_closeout/reproduction';reports.mkdir(parents=True,exist_ok=True)
    if (reports/'verification.json').exists() and not (reports/'initial_verification_attempt.json').exists():
        shutil.copyfile(reports/'verification.json',reports/'initial_verification_attempt.json')
        if (reports/'verification_environment.json').exists():
            shutil.copyfile(reports/'verification_environment.json',reports/'initial_environment_attempt.json')
    attempt=1
    while (reports/f'logs_attempt_{attempt:02d}').exists():attempt+=1
    logs=reports/f'logs_attempt_{attempt:02d}';logs.mkdir()
    record_path=reports/'reconstruction.json'
    protected={args.manifest.resolve():sha(args.manifest)}
    for a in cfg['assets']:
        local=ROOT/a['local_candidate']
        if local.exists():protected[local]=sha(local)
    # Core original numerical tables and source remain read-only during isolated commands.
    for rel in ['src/coherencygraph_das/submission_revision.py','src/coherencygraph_das/final_revision.py','reports/submission_revision/conditional_ranges.parquet','reports/submission_revision/prediction_event_scores.csv','reports/submission_revision/processing_comparisons.csv']:
        protected[ROOT/rel]=sha(ROOT/rel)
    if args.reuse_prepared:
        rec=json.loads(record_path.read_text());assert rec['work']==str(work)
        assert rec['manifest_sha256']==sha(args.manifest)
        for path,digest in rec.get('execution_source_hashes',rec['pinned_source_hashes']).items():
            # Public source includes some generated illustrations/tables. Their
            # PDF metadata can change during the requested regeneration; only
            # executable/config/test source is immutable on a resumed run.
            if PurePosixPath(path).parts[0] in {'src','scripts','configs','tests'}:
                assert sha(work/path)==digest,path
    else:
        if work.exists():raise FileExistsError(f'Refusing to overwrite existing directory: {work}')
        if shutil.disk_usage(work.parent).free<5*1024**3:raise RuntimeError('At least 5GiB free is required')
        work.mkdir(parents=True)
        archive=reports/'pinned_source.zip'
        source=args.source_repository
        if source is None:
            source_dir=work/'.source_git'
            p=subprocess.run(['git','clone','--no-checkout',cfg['repository'],str(source_dir)],capture_output=True,text=True)
            assert p.returncode==0,p.stderr
            source=str(source_dir)
        source=Path(source).resolve()
        git=['git','-c','safe.directory='+source.as_posix(),'-C',str(source)]
        resolved=subprocess.check_output(git+['rev-parse',cfg['source_commit']+'^{commit}'],text=True).strip()
        assert resolved==cfg['source_commit']
        tree=subprocess.check_output(git+['rev-parse',resolved+'^{tree}'],text=True).strip()
        subprocess.run(git+['archive','--format=zip','--output='+str(archive),resolved],check=True)
        safe_extract(archive,work)
        source_hashes={info.filename:sha(work/info.filename) for info in zipfile.ZipFile(archive).infolist() if not info.is_dir()}
        assets=[]
        for a in cfg['assets']:
            local=ROOT/a['local_candidate'];downloaded=False
            if not local.exists():
                local=reports/a['name'];downloaded=True
                request=urllib.request.Request(a['url'],headers={'User-Agent':'CoherencyGraph reproducibility verifier'})
                with urllib.request.urlopen(request,timeout=60) as src,local.open('wb') as dst:shutil.copyfileobj(src,dst)
            digest=sha(local)
            assert digest==a['sha256'],f'Checksum mismatch before extraction: {a["id"]}'
            print('Verified pre-extraction SHA:',a['id'],flush=True)
            count=safe_extract(local,work)
            inner=manifest_files(work,work/a['manifest'])
            assets.append(dict(id=a['id'],name=a['name'],sha256=digest,downloaded_this_run=downloaded,
                reused_public_asset_bytes=not downloaded,extracted_files=count,manifest_entries_verified=inner))
        changed=[p for p,d in source_hashes.items() if sha(work/p)!=d]
        # Asset overlays contain historical copies of source/docs. Pin source finally.
        safe_extract(archive,work)
        assert all(sha(work/p)==d for p,d in source_hashes.items())
        assert not (work/'manuscript').exists(),'Private manuscript leaked into scientific reconstruction'
        rec=dict(status='PREPARED',created_utc=datetime.now(timezone.utc).isoformat(),work=str(work),
            manifest_sha256=sha(args.manifest),source_commit=resolved,source_tree=tree,
            source_origin=str(source),source_archive_sha256=sha(archive),pinned_source_hashes=source_hashes,
            source_files_restored_after_asset_overlay=changed,assets=assets,raw_waveforms_absent=True,
            private_manuscripts_absent=True)
        write(record_path,rec)
    # Git and archive packaging can serialize identical text with different
    # newlines. The scientific receipts pin bytes, not only parsed content.
    # Restore only a candidate serialization whose hash EXACTLY matches a
    # pre-existing receipt; never replace a receipt or accept changed content.
    frozen=json.loads((work/'reports/audit_design_utility/frozen_receipt.json').read_text())
    expected=dict(frozen['sources'])
    expected['configs/audit_design_utility.json']=frozen['specification_sha256']
    expected['scripts/evaluate_audit_design_utility.py']=frozen['script_sha256']
    serializations=[]
    for rel,digest in expected.items():
        path=safe_destination(work,rel)
        if sha(path)==digest:continue
        original=path.read_bytes()
        candidates=[('LF',original.replace(b'\r\n',b'\n')),
            ('CRLF',original.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))]
        matches=[(name,value) for name,value in candidates if hashlib.sha256(value).hexdigest()==digest]
        assert len(matches)==1,f'Frozen source/content mismatch not explained by newline serialization: {rel}'
        name,value=matches[0];path.write_bytes(value)
        serializations.append(dict(path=rel,source_export_sha256=hashlib.sha256(original).hexdigest(),
            frozen_sha256=digest,serialization=name,text_content_changed=False))
    assert all(sha(work/rel)==digest for rel,digest in expected.items())
    write(reports/f'frozen_serialization_attempt_{attempt:02d}.json',serializations)
    rec['frozen_text_serializations']=serializations
    rec['execution_source_hashes']={rel:sha(work/rel) for rel in rec['pinned_source_hashes']}
    write(record_path,rec)
    env=os.environ.copy();env.update(PYTHONPATH=str(work/'src'),CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLBACKEND='Agg',PIP_DISABLE_PIP_VERSION_CHECK='1')
    # Exercise the documented editable package installation without changing
    # the shared virtual environment or downloading/replacing its dependencies.
    # The analytical dependencies are the separately recorded installed runtime.
    install_target=work/'_editable_install_check'
    install_runs=[]
    if not install_target.exists():
        row,p=run_logged([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',
            '--target',str(install_target),'--editable',str(work)],work,env,logs,'installation_editable')
        install_runs.append(row)
        assert p.returncode==0,p.stderr
    import_env=dict(env);import_env['PYTHONPATH']=''
    import_code="import site,json;from pathlib import Path;site.addsitedir("+repr(str(install_target))+");import coherencygraph_das;actual=Path(coherencygraph_das.__file__).resolve();assert actual.is_relative_to(Path("+repr(str(work))+"));print(json.dumps(dict(installed_source=str(actual))))"
    row,p=run_logged([sys.executable,'-c',import_code],work,import_env,logs,'installation_import')
    install_runs.append(row);assert p.returncode==0,p.stderr
    write(reports/f'installation_attempt_{attempt:02d}.json',dict(status='PASS',
        installation='editable public-source package installed in isolated target with existing recorded scientific dependencies',
        dependencies_reinstalled=False,shared_virtual_environment_modified=False,runs=install_runs))
    # Verify execution imports the isolated source, not another editable install.
    environment_code="import sys,platform,json,torch,numpy,scipy,pandas,sklearn,coherencygraph_das;from pathlib import Path;print(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,torch_built_cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),execution='CPU model construction and CPU tensor execution in regeneration; installed wheel may support CUDA',default_tensor_device=str(torch.tensor(0.).device),numpy=numpy.__version__,scipy=scipy.__version__,pandas=pandas.__version__,sklearn=sklearn.__version__,package_file=str(Path(coherencygraph_das.__file__).resolve())),indent=2))"
    runs=[]
    row,p=run_logged([sys.executable,'-c',environment_code],work,env,logs,'00_environment');runs.append(row)
    assert p.returncode==0
    environment=json.loads(p.stdout);assert Path(environment['package_file']).is_relative_to(work)
    write(reports/'verification_environment.json',environment)
    row,p=run_logged([sys.executable,'-m','pip','freeze','--all'],work,env,logs,'01_actual_pip_freeze');runs.append(row)
    assert p.returncode==0
    # No dependency replacement is performed or hidden; actual wheel versions are recorded.
    for i,command in enumerate(cfg['commands'],2):
        row,p=run_logged([sys.executable,*command],work,env,logs,f'{i:02d}_'+Path(command[0]).stem)
        runs.append(row);write(reports/f'command_runs_attempt_{attempt:02d}.json',runs)
        if p.returncode:
            result=dict(status='BLOCKED',failed_command=command,runs=runs,environment=environment)
            write(reports/f'verification_attempt_{attempt:02d}.json',result)
            write(reports/'verification.json',result)
            raise RuntimeError(f'Verification command failed: {command}; see {row["log"]}')
    xml=ET.parse(work/'closeout_test_results.xml').getroot()
    suites=[xml] if xml.tag=='testsuite' else list(xml.findall('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    counts['passed']=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    skips=[dict(name=c.attrib.get('name'),reason=c.find('skipped').attrib.get('message')) for c in xml.iter('testcase') if c.find('skipped') is not None]
    for path,digest in protected.items():assert sha(path)==digest,f'Original input changed: {path}'
    result=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=cfg['source_commit'],source_tree=rec['source_tree'],assets=rec['assets'],
        environment=environment,tests=counts,skips=skips,runs=runs,original_inputs_unchanged=True,
        original_files_verified=len(protected),new_training=False,raw_reextraction=False,
        private_manuscripts_used=False,public_publication=False,
        editable_installation_verified=True,dependency_environment_reused=True,
        frozen_text_serializations=rec.get('frozen_text_serializations',[]),
        note='Existing scientific public release reconstructed from checksum-verified cached ZIP bytes and pinned Git objects. Exact pre-existing frozen scientific byte hashes are restored by documented newline serialization only. This is a fresh isolated numerical run, not a fresh network download or independent raw-data re-extraction.')
    write(reports/f'verification_attempt_{attempt:02d}.json',result)
    write(reports/'verification.json',result)
    print('PUBLIC SCIENTIFIC REPRODUCTION PASS',counts,flush=True)

if __name__=='__main__':main()
