"""Independent 50-km connectivity and component-bootstrap replay of saved scores."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revisions/20260922_unseen_validation'


def main():
    candidates=pd.read_csv(OUT/'locked_candidate_order.csv',dtype={'event_id':str}).set_index('event_id')
    saved=pd.read_csv(OUT/'validation/event_scores.csv',dtype={'event_id':str})
    event_ids=sorted(saved.event_id.unique())
    cohort=candidates.loc[event_ids]
    coordinates=list(cohort[['latitude_deg','longitude_deg','depth_km']].itertuples(index=False,name=None))
    adjacency=[set() for _ in event_ids]
    for i,(lat1,lon1,z1) in enumerate(coordinates):
        for j,(lat2,lon2,z2) in enumerate(coordinates[:i]):
            phi1,phi2=map(math.radians,(lat1,lat2))
            hav=math.sin((phi1-phi2)/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lon1-lon2)/2)**2
            arc=12742*math.asin(math.sqrt(max(0,min(1,hav))))
            if math.hypot(arc,z1-z2)<=50:
                adjacency[i].add(j);adjacency[j].add(i)
    components=[];visited=set()
    for root in range(len(event_ids)):
        if root in visited:continue
        pending=[root];members=[]
        while pending:
            current=pending.pop()
            if current in visited:continue
            visited.add(current);members.append(current)
            pending.extend(adjacency[current]-visited)
        components.append(np.array(sorted(members)))
    comparisons=pd.read_csv(OUT/'validation/paired_comparisons.csv')
    errors=[]
    for row in comparisons[comparisons.resampling.eq('source_50km')].itertuples():
        wide=saved[saved.endpoint.eq(row.endpoint)].pivot(index='event_id',columns='model',values='ratio_db').loc[event_ids]
        values=(wide[row.model]-wide[row.reference]).to_numpy()
        rng=np.random.default_rng(20260922)
        draws=[]
        for _ in range(5000):
            chosen=rng.integers(len(components),size=len(components))
            # Explicit event concatenation, not the production totals/counts shortcut.
            indices=np.concatenate([components[group] for group in chosen])
            draws.append(values[indices].mean())
        low,high=np.quantile(draws,[.025,.975])
        error=max(abs(low-row.low),abs(high-row.high),abs(values.mean()-row.mean))
        assert error<1e-10
        assert len(components)==row.groups
        errors.append(dict(endpoint=row.endpoint,model=row.model,reference=row.reference,maximum_error=float(error)))
    receipt=dict(passed=True,events=len(event_ids),source_components_50km=len(components),
                 component_members=[[event_ids[i] for i in component] for component in components],
                 comparisons_verified=len(errors),maximum_error=max(r['maximum_error'] for r in errors),
                 method='Scalar haversine plus graph traversal; explicit whole-event concatenation per bootstrap draw',
                 checks=errors,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'validation/group_bootstrap_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['component_members','checks']},indent=2))


if __name__=='__main__':main()
