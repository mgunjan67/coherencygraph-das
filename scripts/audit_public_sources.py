"""Read-only public-source checks; records exact response status, never assumes access."""
from pathlib import Path
from datetime import datetime,timezone
import json,re,urllib.request,urllib.error,hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/critical_review'
URLS={
 'guide':'https://www.sciencedirect.com/journal/computers-and-geosciences/publish/guide-for-authors',
 'project':'https://fiberlab.uw.edu/projects/alaska-cook-inlet/',
 'archive_index':'https://dasway.ess.washington.edu/gci/index.html',
 'picks_metadata':'https://zenodo.org/api/records/11642688',
 'companion_metadata':'https://zenodo.org/api/records/22325016',
}
def fetch(url,limit=2000000):
    started=datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CoherencyGraph-reproducibility-audit/0.2'}),timeout=25) as r:
            content=r.read(limit)
            return dict(url=url,utc=started,status=r.status,bytes_read=len(content),sha256=hashlib.sha256(content).hexdigest()),content
    except Exception as e:
        return dict(url=url,utc=started,status=getattr(e,'code',None),error=f'{type(e).__name__}: {e}'),b''

if __name__=='__main__':
    results=[]
    for name,url in URLS.items():
        record,body=fetch(url);record['name']=name;results.append(record)
        if body:
            (OUT/f'public_source_{name}.txt').write_bytes(body)
        print(name,record.get('status'),record.get('error',''),flush=True)
        if name=='archive_index' and body:
            text=body.decode('utf8',errors='replace')
            links=re.findall(r'(?:href|src)=[\"\']([^\"\']+)[\"\']',text)
            selected=sorted(x for x in links if '2024-' in x)[:2]
            for link in selected:
                absolute=urllib.parse.urljoin(url,link)
                rr,bb=fetch(absolute);rr['name']='listed_2024_page';results.append(rr)
                if bb:
                    linked=re.findall(r'href=[\"\']([^\"\']+\.h5)[\"\']',bb.decode('utf8',errors='replace'))
                    if not linked:
                        event_links=re.findall(r'href=[\"\']([^\"\']+)[\"\']',bb.decode('utf8',errors='replace'))
                        event_links=[x for x in event_links if re.search(r'/[0-9]{6,}/',x)]
                        for event_link in sorted(set(event_links))[:1]:
                            event_url=urllib.parse.urljoin(absolute,event_link)
                            er,eb=fetch(event_url);er['name']='listed_2024_event';results.append(er)
                            if eb:
                                linked=re.findall(r'href=[\"\']([^\"\']+\.h5)[\"\']',eb.decode('utf8',errors='replace'))
                                linked=[urllib.parse.urljoin(event_url,x) for x in linked]
                    for waveform in linked[:2]:
                        obj=urllib.parse.urljoin(absolute,waveform)
                        rec,_=fetch(obj,1);rec['name']='listed_2024_waveform';results.append(rec)
    (OUT/'public_access_audit.json').write_text(json.dumps(results,indent=2),encoding='utf8')
