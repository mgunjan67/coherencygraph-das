"""Prepare code-only public source; exclude manuscripts and account credentials."""
from pathlib import Path
import shutil,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'releases/20260906_review_revision/code_repository'
DEST=ROOT/'publication/github/coherencygraph-das'
if DEST.exists():raise SystemExit('Staging directory already exists; inspect it rather than overwrite.')
DEST.mkdir(parents=True)
folders=['src','scripts','configs','tests','docs','reports/critical_review/figures']
for folder in folders:
    for p in (SOURCE/folder).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts:continue
        out=DEST/p.relative_to(SOURCE);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
for filename in ['LICENSE','CITATION.cff','pyproject.toml','requirements-revision.lock.txt','CLAIMS_LEDGER.md']:
    shutil.copy2(SOURCE/filename,DEST/filename)
print(DEST)
