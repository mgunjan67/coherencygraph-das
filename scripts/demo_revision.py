"""Offline synthetic example: observed-lag prediction is not identification.

Usage: python scripts/demo_revision.py --output demo_output
"""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from coherencygraph_das.critical_revision import Q,LAGS,feasible_bounds,upper_kernel_matrix
def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plus=(1+.9*np.cos(Q))/257;minus=(1-.9*np.cos(Q))/257;basis=np.exp(1j*LAGS[:,None]*Q)
    observed=basis@plus;other=basis@minus
    bounds=feasible_bounds(observed,[1,2,4,10])
    assert np.max(abs(observed-other))<1e-12
    assert all(np.linalg.eigvalsh(upper_kernel_matrix(p,np.arange(32))).min()>-1e-10 for p in [plus,minus])
    assert not bounds[0]['sign_identified']
    result=dict(synthetic=True,seed='analytic construction; no random seed',supervised_max_difference=float(np.max(abs(observed-other))),
        nearest_lag_values=[float((np.exp(1j*Q)@p).real) for p in [plus,minus]],bounds=bounds,
        expected_decision='abstain at lag one: both signs are compatible')
    (output/'result.json').write_text(json.dumps(result,indent=2))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,3),layout='constrained')
    for j,r in enumerate(bounds):ax.plot([j,j],[r['lower'],r['upper']],lw=4)
    ax.axhline(0,color='black',ls='--');ax.set_xticks(range(4),[r['lag'] for r in bounds]);ax.set_xlabel('Unseen channel lag');ax.set_ylabel('Feasible real coherency')
    fig.savefig(output/'feasible_ranges.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['supervised_max_difference','nearest_lag_values','expected_decision']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='demo_output');a=p.parse_args();run(a.output)
