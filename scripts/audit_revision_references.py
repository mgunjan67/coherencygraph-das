"""Read-only Crossref metadata audit for references actually cited in the revision."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import re,json,urllib.request,urllib.parse,unicodedata,difflib,time
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'manuscript/cageo_submission';OUT=ROOT/'reports/critical_review'
text=PAPER.joinpath('main.tex').read_text();keys=set()
for group in re.findall(r'\\cite\w*\{([^}]+)\}',text):keys.update(group.split(','))
bib=PAPER.joinpath('references.bib').read_text();entries={}
for chunk in re.split(r'(?=@\w+\{)',bib):
    match=re.match(r'@\w+\{([^,]+),',chunk)
    if match:entries[match.group(1)]=chunk
def field(entry,name):
    m=re.search(r'\b'+name+r'\s*=\s*\{([^\n]+?)\}(?:,|\s*$)',entry,re.M)
    return m.group(1) if m else ''
def canon(value):return re.sub(r'[^a-z0-9]','',unicodedata.normalize('NFKD',value).lower())
def audit(key):
    entry=entries[key];doi=field(entry,'doi');record=dict(key=key,doi=doi,bib_title=field(entry,'title'),utc=datetime.now(timezone.utc).isoformat())
    if not doi:
        record.update(status='non-Crossref source',url=field(entry,'url'));return record
    url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='');record['url']=url
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CoherencyGraph-reference-audit/1.1'}),timeout=30) as response:
            raw=response.read();data=json.loads(raw)['message']
        (OUT/f'reference_{key}.json').write_bytes(raw)
        title=data.get('title',[''])[0];record.update(status=200,registered_title=title,registered_year=data.get('published',{}).get('date-parts'),
            title_similarity=difflib.SequenceMatcher(None,canon(title),canon(record['bib_title'])).ratio(),authors=data.get('author',[]))
    except Exception as exc:record.update(status=getattr(exc,'code','failed'),error=str(exc))
    return record
results=[]
prior_path=OUT/'reference_audit.json';prior={r['key']:r for r in json.loads(prior_path.read_text())} if prior_path.exists() else {}
for key in sorted(keys):
    old=prior.get(key)
    if key=='Shi2024CookInletPicks':
        evidence=json.loads((OUT/'public_source_picks_metadata.txt').read_text())
        results.append(dict(key=key,doi=field(entries[key],'doi'),status='verified via Zenodo API (DataCite DOI)',
            registered_title=evidence['metadata']['title'],url='https://zenodo.org/api/records/11642688',license=evidence['metadata']['license']));continue
    if old and old.get('status')==200 and old.get('title_similarity')==1:results.append(old);continue
    results.append(audit(key));time.sleep(.5)
(OUT/'reference_audit.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print([(r['key'],r['status'],r.get('title_similarity')) for r in results])
