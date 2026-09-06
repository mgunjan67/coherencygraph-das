"""Local release preparation and verification; never publishes or invents a DOI."""
from pathlib import Path
import sys,subprocess,json,shutil,zipfile,platform,importlib.metadata,re,os
import numpy as np
import pandas as pd
from .critical_revision import ROOT,OUT,MODELS,bundle,metrics
from .config import sha256,write_json

RELEASE=ROOT/'releases/20260906_review_revision'
SOURCE_DIRS=['src','scripts','configs','tests','docs']
ROOT_FILES=['README.md','LICENSE','CITATION.cff','pyproject.toml','requirements-revision.lock.txt',
            'CRITICAL_REVIEW_RESPONSE.md','CLAIMS_LEDGER.md','AUTHOR_ACTIONS.md']

def run_logged(command,name,cwd=ROOT):
    env=os.environ.copy();env['PYTHONPATH']=str(cwd/'src');env['PYTHONUTF8']='1'
    p=subprocess.run(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
    (OUT/name).write_text(p.stdout,encoding='utf8')
    if p.returncode:raise RuntimeError(f'{name}: exit {p.returncode}; see complete saved log')
    return p.stdout

def verify():
    # Retain exact versions without exposing pip configuration or credentials.
    packages=['numpy','scipy','pandas','pyarrow','h5py','scikit-learn','matplotlib','torch','PyYAML','pytest','PyMuPDF','pypdf']
    versions={}
    for package in packages:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    lines=['# Exact observed analytical environment. CUDA build is platform-specific.',
           '# For CPU, install the matching public PyTorch CPU wheel separately.']
    lines.extend(k+'=='+v for k,v in versions.items())
    (ROOT/'requirements-revision.lock.txt').write_text('\n'.join(lines)+'\n')
    import torch
    write_json(OUT/'runtime_environment_revision.json',dict(python=sys.version,platform=platform.platform(),packages=versions,
        cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
    test=run_logged([sys.executable,'-m','pytest'],'test_log_revision.txt')
    run_logged([sys.executable,'scripts/demo_revision.py','--output','reports/critical_review/demo'],'demo_test_log.txt')
    # Recompute table scores directly from saved prediction tensors.
    _,ds,roles,_=bundle();idx=np.flatnonzero(roles=='architecture_test');rows=[]
    table=pd.read_csv(OUT/'matched_summary.csv')
    for name in ['block_ridge','full_ridge','residual_ridge','full_mlp_direct','full_mlp_psd','state_direct','state_psd']:
        p=np.load(MODELS/f'matched_{name}_all_predictions.npy')
        value=np.mean([metrics(p[i],ds.targets[i])['nrmse'] for i in idx])
        expected=float(table[table.model==name]['mean'].iloc[0]);delta=abs(value-expected)
        assert delta<1e-10
        rows.append(dict(model=name,regenerated=value,reported=expected,absolute_difference=delta))
    pd.DataFrame(rows).to_csv(OUT/'final_numerical_verification.csv',index=False)
    paper=ROOT/'manuscript/cageo_submission';inventory={}
    for name in ['main','supplement']:
        source=(paper/f'{name}.tex').read_text();inventory[name]=dict(
            sections=len(re.findall(r'\\section\{',source)),figures=len(re.findall(r'\\begin\{figure\}',source)),
            tables=len(re.findall(r'\\begin\{table\}',source)),labels=re.findall(r'\\label\{([^}]+)\}',source))
        assert len(inventory[name]['labels'])==len(set(inventory[name]['labels']))
        refs=re.findall(r'\\ref\{([^}]+)\}',source);assert set(refs)<=set(inventory[name]['labels'])
    assert inventory['supplement']['sections']==8 and inventory['supplement']['tables']==13 and inventory['supplement']['figures']==2
    assert max(len(x) for x in (paper/'highlights.txt').read_text().splitlines())<=85
    build=json.loads((OUT/'latex_build_report.json').read_text());assert all(not r['warnings'] for r in build)
    visual=json.loads((OUT/'pdf_qa/visual_inspection.json').read_text())
    assert visual['all_pages_inspected'] is True
    for doc in visual['documents']:
        assert sha256(paper/'output'/f'{doc["document"]}.pdf')==doc['pdf_sha256']
    public=OUT/'public_repository_verification.json'
    verified=public.exists() and json.loads(public.read_text()).get('verified_public_browsable_source') is True
    result=dict(unit_tests=test.strip().splitlines()[-1],numerical_checks=len(rows),inventory=inventory,
        public_repository_verified=verified,author_approval_verified=False,decision='NOT READY',
        reason='Public browsable repository and final author approvals are not verified.',pdf_build=build,
        visually_inspected_pages=sum(d['pages'] for d in visual['documents']))
    write_json(OUT/'final_verification.json',result)
    print(json.dumps(result,indent=2))

def copy_selected(source,destination):
    for p in sorted(source.rglob('*')):
        rel=p.relative_to(source)
        if not p.is_file() or any(s in ['__pycache__','.pytest_cache'] for s in rel.parts):continue
        if p.name=='release_package_manifest.json':continue  # Receipt is external to the ZIP it hashes.
        if 'pdf_qa' in rel.parts and p.suffix.lower() not in ['.json','.txt']:continue
        if p.suffix in ['.pyc','.aux','.log','.out','.blg','.spl']:continue
        dest=destination/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

def prepare_release():
    RELEASE.mkdir(parents=True,exist_ok=True);code=RELEASE/'code_repository';full=RELEASE/'reproducibility'
    code.mkdir(exist_ok=True);full.mkdir(exist_ok=True)
    for folder in SOURCE_DIRS:copy_selected(ROOT/folder,code/folder)
    for name in ROOT_FILES:
        if not (ROOT/name).exists():raise FileNotFoundError(name)
        shutil.copy2(ROOT/name,code/name)
    copy_selected(ROOT/'manuscript/cageo_submission',code/'manuscript/cageo_submission')
    copy_selected(ROOT/'reports/critical_review/figures',code/'reports/critical_review/figures')
    # No unreviewed proprietary binaries, raw waveforms, tokens or account data.
    copy_selected(code,full)
    for folder in ['data/processed','models/critical_review','models/methodological_audit','reports']:
        copy_selected(ROOT/folder,full/folder)
    # Preserve original selection metadata without rewriting the frozen protocol.
    from .config import load_config
    from .cohort import _source_path
    original_config=load_config(ROOT/'configs/protocol_v1.0.yaml')
    for key in ['source_manifest','source_split_table']:
        source=_source_path(original_config,key);destination=full/'data/provenance'/source.name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        assert sha256(source)==sha256(destination)
    cohort=pd.read_parquet(ROOT/'reports/audit/frozen_cohort.parquet')
    for r in cohort.itertuples():
        p=Path(r.picks_path)
        if p.exists():
            target=full/'data/source_picks'/str(r.event_id)/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    manifests=[]
    for name,directory in [('code_repository',code),('reproducibility',full)]:
        rows=[]
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.name!='SHA256SUMS.csv':rows.append(dict(path=path.relative_to(directory).as_posix(),bytes=path.stat().st_size,sha256=sha256(path)))
        pd.DataFrame(rows).to_csv(directory/'SHA256SUMS.csv',index=False)
        archive=RELEASE/f'CoherencyGraph_DAS_{name}_20260906.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(directory.rglob('*')):
                if path.is_file():z.write(path,path.relative_to(directory).as_posix())
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise RuntimeError('ZIP integrity failure')
        manifests.append(dict(kind=name,path=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=sha256(archive),files=len(rows)))
    # Isolated extraction proves that the example does not rely on original cwd.
    isolated=RELEASE/'isolated_verification';isolated.mkdir(exist_ok=True)
    with zipfile.ZipFile(RELEASE/'CoherencyGraph_DAS_reproducibility_20260906.zip') as z:z.extractall(isolated)
    checksum_rows=pd.read_csv(isolated/'SHA256SUMS.csv')
    for row in checksum_rows.itertuples():
        if sha256(isolated/row.path)!=row.sha256:raise RuntimeError('Extracted checksum mismatch: '+row.path)
    run_logged([sys.executable,'scripts/demo_revision.py','--output','demo_output'],'code_only_demo_log.txt',code)
    run_logged([sys.executable,'-m','pytest','tests/test_critical_revision.py','tests/test_spectral.py'],'code_only_unit_log.txt',code)
    run_logged([sys.executable,'scripts/demo_revision.py','--output','demo_output'],'isolated_demo_log.txt',isolated)
    run_logged([sys.executable,'-m','pytest'],'isolated_full_test_log.txt',isolated)
    run_logged([sys.executable,'scripts/verify_revision_checkpoints.py'],'isolated_checkpoint_log.txt',isolated)
    write_json(OUT/'release_package_manifest.json',dict(archives=manifests,publicly_published=False,
        isolated_demo_passed=True,isolated_checkpoint_regeneration_passed=True,
        extracted_checksums_verified=len(checksum_rows),code_only_demo_passed=True,code_only_unit_tests_passed=True,
        isolated_full_test_suite_passed=True,
        journal_requires_browsable_code=True,zip_alone_satisfies_public_repository_rule=False))
    print(json.dumps(manifests,indent=2),flush=True)

def main(stage):
    if stage=='verify':verify()
    elif stage=='release':prepare_release()
    else:raise ValueError(stage)
