"""Raw-data-free independent replay of event reductions and bootstrap intervals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revisions/20260922_unseen_validation'

def main():
    saved=pd.read_csv(OUT/'validation/event_scores.csv',dtype={'event_id':str})
    records=pd.read_csv(OUT/'validation_records.csv',dtype={'event_id':str})
    assert records.event_id.nunique()==30 and len(records)==60
    errors=[]
    for event in sorted(records.event_id.unique()):
        cases=pd.concat([pd.read_parquet(OUT/f'validation/case_records/{event}_{route}.parquet') for route in ['TERRA','KKFLS']])
        assert not cases.duplicated(['event_id','route','cutoff','block','band','method','endpoint']).any()
        for model in saved.model.unique():
            methods=[f'{model}_seed{s}' for s in [19,43,71]] if model.endswith('recurrence') else [model]
            for endpoint in ['full','half0','half1']:
                chosen=cases[cases.method.isin(methods)&cases.endpoint.eq(endpoint)]
                assert len(chosen)==2*4*8*4*len(methods)
                row=saved[saved.event_id.eq(event)&saved.model.eq(model)&saved.endpoint.eq(endpoint)]
                assert len(row)==1
                errors.append(abs(chosen.ratio_db.mean()-row.ratio_db.iloc[0]))
    assert max(errors)<1e-10
    comparisons=pd.read_csv(OUT/'validation/paired_comparisons.csv')
    bootstrap_errors=[]
    for row in comparisons[comparisons.resampling.eq('earthquake')].itertuples():
        wide=saved[saved.endpoint.eq(row.endpoint)].pivot(index='event_id',columns='model',values='ratio_db')
        v=(wide[row.model]-wide[row.reference]).to_numpy()
        rng=np.random.default_rng(20260922)
        draws=np.array([v[rng.integers(len(v),size=len(v))].mean() for _ in range(5000)])
        lo,hi=np.quantile(draws,[.025,.975])
        bootstrap_errors.extend([abs(v.mean()-row.mean),abs(lo-row.low),abs(hi-row.high)])
    assert max(bootstrap_errors)<1e-10
    result=dict(passed=True,events=30,route_files=60,maximum_event_error=max(errors),maximum_bootstrap_error=max(bootstrap_errors),
        scope='Cached case-score reductions and earthquake intervals only. Does not replace raw-waveform or model-training verification.')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
