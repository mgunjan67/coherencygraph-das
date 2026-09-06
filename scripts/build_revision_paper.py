"""Build all revised LaTeX deliverables and retain complete compiler logs."""
from pathlib import Path
import subprocess,shutil,json,re
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'manuscript/cageo_submission'; OUT=PAPER/'output'
REPORT=ROOT/'reports/critical_review';OUT.mkdir(exist_ok=True)
tex=shutil.which('pdflatex');bib=shutil.which('bibtex')
if not tex or not bib:raise SystemExit('Install a LaTeX distribution with elsarticle and BibTeX.')
records=[]
for name in ['main','supplement','cover_letter','review']:
    commands=[[tex,'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory=output',name+'.tex']]
    if name in ['main','review']:commands += [[bib,'output/'+name]]
    commands += [[tex,'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory=output',name+'.tex']]*2
    logs=[]
    for command in commands:
        p=subprocess.run(command,cwd=PAPER,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace')
        logs.append('$ '+' '.join(command)+'\n'+p.stdout)
        if p.returncode:
            (REPORT/f'build_{name}.txt').write_text('\n'.join(logs),encoding='utf8')
            print(p.stdout[-6000:]);raise SystemExit(p.returncode)
    (REPORT/f'build_{name}.txt').write_text('\n'.join(logs),encoding='utf8')
    last=(OUT/f'{name}.log').read_text(encoding='utf8',errors='replace')
    warnings=[x for x in last.splitlines() if 'Overfull' in x or 'undefined' in x or 'Float too large' in x]
    records.append(dict(document=name,bytes=(OUT/f'{name}.pdf').stat().st_size,warnings=warnings))
    print(name,records[-1],flush=True)
(REPORT/'latex_build_report.json').write_text(json.dumps(records,indent=2))
