"""Post hoc descriptive check of existing event scores against the new QC flags."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
events=pd.read_csv(ROOT/'revisions/20260922_unseen_validation/validation/event_scores.csv',dtype={'event_id':str})
flags=pd.read_csv(HERE/'data_quality/high_variance_channel_locations.csv',dtype={'event_id':str})
flagged=set(flags.loc[flags.role.eq('validation'),'event_id'])
pivot=events[events.endpoint.eq('full')].pivot(index='event_id',columns='model',values='ratio_db')
rng=np.random.default_rng(20260930);rows=[]
for name,mask in [('all_frozen_events',np.ones(len(pivot),bool)),('no_high_variance_flags',~pivot.index.isin(flagged)),('with_high_variance_flags',pivot.index.isin(flagged))]:
    part=pivot.loc[mask]
    for baseline in ['sparse_recurrence','dense_full_ridge','fixed_shrinkage']:
        v=(part.dense_recurrence-part[baseline]).to_numpy()
        boot=v[rng.integers(0,len(v),size=(5000,len(v)))].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        rows.append(dict(subset=name,events=len(v),model='dense_recurrence',reference=baseline,mean=float(v.mean()),low=float(lo),high=float(hi)))
frame=pd.DataFrame(rows);frame.to_csv(HERE/'data_quality/posthoc_flag_sensitivity.csv',index=False)
(HERE/'data_quality/posthoc_flag_sensitivity.json').write_text(json.dumps(dict(seed=20260930,draws=5000,flagged_events=sorted(flagged),results=rows,note='Post hoc descriptive sensitivity, not a prespecified confirmation test. Original 30-event scores and bootstrap remain unchanged.'),indent=2))
row=frame[(frame.subset=='no_high_variance_flags')&(frame.reference=='sparse_recurrence')].iloc[0]
text=f'The high-variance flags affect {len(flagged)} validation earthquakes. A post hoc descriptive check retains the {row.events} earthquakes without these flags and recomputes the mean of their existing dense-minus-sparse event scores. The difference is {row["mean"]:.3f} [{row.low:.3f}, {row.high:.3f}]~dB (95\\% complete-earthquake percentile interval; 5,000 draws, seed 20260930). This subset was defined after the new data audit. It does not replace the frozen 30-earthquake estimate or its bootstrap, and it does not establish that the flagged channels are harmless. The small flagged subset is retained in the source tables rather than promoted to a separate confirmation test.\n'
(HERE/'manuscript/generated_quality_sensitivity.tex').write_text(text)
print(frame.to_string(index=False))
