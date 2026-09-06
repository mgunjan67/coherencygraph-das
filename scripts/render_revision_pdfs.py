"""Render every final PDF page and record extractable text/bounds for visual QA."""
from pathlib import Path
import shutil,subprocess,json
import fitz
ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'reports/critical_review/pdf_qa';QA.mkdir(exist_ok=True)
poppler=shutil.which('pdftoppm')
if not poppler:raise SystemExit('pdftoppm (Poppler) is required for page rendering.')
rows=[]
for name in ['main','supplement','cover_letter','review']:
    pdf=ROOT/'manuscript/cageo_submission/output'/f'{name}.pdf';dest=QA/name;dest.mkdir(exist_ok=True)
    subprocess.run([poppler,'-r','115','-png',str(pdf),str(dest/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    doc=fitz.open(pdf);texts=[]
    for i,page in enumerate(doc):
        text=page.get_text();texts.append(f'\n--- PAGE {i+1} ---\n'+text)
        outside=[]
        for block in page.get_text('blocks'):
            if block[6]==0 and (block[0]<8 or block[1]<8 or block[2]>page.rect.width-8 or block[3]>page.rect.height-8):outside.append(block[:4])
        rows.append(dict(document=name,page=i+1,width=page.rect.width,height=page.rect.height,characters=len(text),outside_page_blocks=outside))
    (QA/f'{name}_text.txt').write_text(''.join(texts),encoding='utf8')
(QA/'page_checks.json').write_text(json.dumps(rows,indent=2))
print([(name,sum(r['document']==name for r in rows)) for name in ['main','supplement','cover_letter','review']])
print('Outside-page text blocks:',sum(len(r['outside_page_blocks']) for r in rows))
