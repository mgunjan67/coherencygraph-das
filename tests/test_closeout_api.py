import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coherencygraph_das.closeout_api import audit_moments

def test_reject_wrong_allowance_shape():
    with pytest.raises(ValueError):audit_moments([1],[0j],[0,1],[.1])

def test_reject_duplicate_coordinates():
    with pytest.raises(ValueError):audit_moments([1],[0j],[0,0],[.1,.1])

def test_inclass_truth_and_geometry():
    q=np.sort(np.fft.fftfreq(257)*2*np.pi);lags=[1,2,3,5,8,13,21,34]
    p=np.ones(257)*.1/257;p[np.argmin(abs(q))]+=.9
    r=audit_moments(lags,np.exp(1j*np.asarray(lags)[:,None]*q)@p,np.arange(484,516),np.ones(16)*.005)
    assert 4 in r['unsupported_required_lags'] and 34 not in r['required_lags']
    assert r['fit']['accepted'] and r['solver_calls']==8
    for x in r['intervals']:
        truth=.9 if x['axis']=='real' else 0.
        assert x['valid'] and x['lower']<=truth<=x['upper']
        assert x['directly_measured'] is False
