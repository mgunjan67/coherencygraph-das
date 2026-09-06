"""Record actual page inspection, not an automated substitute for visual review."""
from pathlib import Path
import argparse,hashlib,json
from datetime import datetime,timezone
import fitz

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--confirm-inspected',action='store_true')
args=parser.parse_args()
if not args.confirm_inspected:raise SystemExit('Inspect every current rendered PDF page before recording this receipt.')
qa=ROOT/'reports/critical_review/pdf_qa'
rows=json.loads((qa/'page_checks.json').read_text())
assert not any(r['outside_page_blocks'] for r in rows)
documents=[]
for name in ['main','supplement','cover_letter','review']:
    pdf=ROOT/'manuscript/cageo_submission/output'/f'{name}.pdf'
    count=len(fitz.open(pdf))
    assert sum(r['document']==name for r in rows)==count
    pages=[]
    for page in range(1,count+1):
        png=qa/name/f'page-{page:0{len(str(count))}d}.png'
        assert png.exists()
        pages.append(dict(page=page,render_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),inspected=True))
    documents.append(dict(document=name,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages=count,page_inspection=pages))
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),reviewer='Codex visual inspection of rendered pages',
    independent_human_review=False,documents=documents,all_pages_inspected=True,
    findings=['No observed clipped text, overlapping plot labels, missing glyphs or reversed workflow arrows.',
              'Figure float pages retain deliberate margins; references were tightened to remove an orphan final page.',
              'Author order, correspondence emails, equation signs, captions, table headers and line-number placement inspected.',
              'Submission remains NOT READY pending public-repository verification and author approval.'])
(qa/'visual_inspection.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('Recorded visual inspection:',sum(d['pages'] for d in documents),'pages across',len(documents),'PDFs')
