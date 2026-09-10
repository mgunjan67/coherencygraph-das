"""Run the actual component-allowance audit without scientific data assets."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from coherencygraph_das.closeout_api import audit_moments

lags=np.array([1,2,3,5,8,13,21,34])
q=2*np.pi*np.arange(-128,129)/257
spectrum=np.ones(257)*.1/257
spectrum[128]+=.9
moments=np.exp(1j*lags[:,None]*q)@spectrum
result=audit_moments(lags,moments,np.arange(484,516),np.full(16,.005),[4,10])
assert result['fit']['accepted']
assert all(row['valid'] for row in result['intervals'])
for row in result['intervals']:
    truth=.9 if row['axis']=='real' else 0.
    assert row['lower']<=truth<=row['upper']
print(json.dumps(result,indent=2))
