"""Check final reporting from existing complete-event arrays; no model fitting."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports/submission_revision'
SEED, DRAWS = 20260906, 5000
PROTOCOL = '4a27ec9f71d5df9e2b3880c122815772ce0ac3b430378e0dbe17835fb1570184'

def run():
    assert hashlib.sha256((ROOT/'configs/protocol_amendment_09_submission.yaml').read_bytes()).hexdigest() == PROTOCOL
    events = pd.read_csv(R/'prediction_event_scores.csv', dtype={'event_id':str})
    table = pd.read_csv(R/'prediction_both_ridge_comparisons.csv')
    meta = pd.read_parquet(ROOT/'reports/critical_review/event_metadata.parquet').set_index('event_id')
    macros = json.loads((R/'manuscript_numbers.json').read_text())
    rows = []
    for endpoint, g in events.groupby('endpoint'):
        assert set(g.route)=={'TERRA','KKFL-S'}
        assert g.groupby(['event_id','model']).route.nunique().eq(2).all()
        matrix = g.groupby(['event_id','model']).nrmse.mean().unstack()
        assert len(matrix)==24
        labels = meta.loc[matrix.index,'sensitivity_component_25km'].to_numpy()
        groups, inverse = np.unique(labels, return_inverse=True)
        assert len(groups)==11
        strongest = matrix[['block_ridge','full_ridge']].mean().idxmin()
        for reference in ['block_ridge','full_ridge']:
            delta = (matrix.state_psd_seed_mean-matrix[reference]).to_numpy()
            totals = np.bincount(inverse,weights=delta)
            sizes = np.bincount(inverse)
            # Same historical component draw stream and event-weighted reduction,
            # expressed here independently of the inference helper.
            sampled = np.random.default_rng(SEED).integers(0,len(groups),(DRAWS,len(groups)))
            draws = totals[sampled].sum(axis=1)/sizes[sampled].sum(axis=1)
            actual = np.array([delta.mean(),*np.quantile(draws,[.025,.975])])
            row = table[(table.endpoint==endpoint)&(table.reference==reference)].iloc[0]
            np.testing.assert_allclose(actual,row[['mean','low','high']].to_numpy(float),atol=1e-12,rtol=0)
            np.testing.assert_allclose(row.ensemble_score,matrix.state_psd_seed_mean.mean(),atol=1e-12,rtol=0)
            np.testing.assert_allclose(row.reference_score,matrix[reference].mean(),atol=1e-12,rtol=0)
            assert row.strongest_observed_ridge==strongest
            if reference==strongest:
                prefix='RevStrong'+''.join(x.title() for x in endpoint.split('_'))
                for key,value in zip(['Mean','Low','High'],actual):assert macros[prefix+key]==f'{value:.4f}'
            rows.append(dict(endpoint=endpoint,reference=reference,mean=float(actual[0]),low=float(actual[1]),high=float(actual[2]),events=24,components=11,verified=True))
    assert len(rows)==8
    graph=json.loads((R/'figures/fig02_dataflow.json').read_text())
    assert all(x['text_fits'] for x in graph['checks'])
    assert not any(e['source']=='prediction' and e['target']=='audit' for e in graph['edges'])
    sources=['prediction_event_scores.csv','prediction_inference.csv','prediction_both_ridge_comparisons.csv']
    result=dict(status='PASS',bootstrap_seed=SEED,bootstrap_draws=DRAWS,protocol_sha256=PROTOCOL,source_sha256={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in sources},comparisons=rows,architecture_no_prediction_to_audit=True)
    (R/'reporting_correction_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
