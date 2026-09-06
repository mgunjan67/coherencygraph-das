import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coherencygraph_das.submission_revision import local_lag_targets,conditional_ranges
from coherencygraph_das.spectral import lag_coherency,cross_spectral_matrix
from coherencygraph_das.critical_revision import Q


def test_local_targets_equal_direct_snapshot_estimator():
    rng=np.random.default_rng(812)
    x=rng.normal(size=(30,32))+1j*rng.normal(size=(30,32))
    x*=np.linspace(.5,3,32)[None,:]
    for lags in [np.arange(1,32),np.array([3,5,8,13,21])]:
        expected=lag_coherency(x,lags)
        actual=local_lag_targets(cross_spectral_matrix(x,.001),lags)
        np.testing.assert_allclose(actual,expected,atol=3e-8,rtol=1e-6)


def test_exact_support_does_not_use_surrounding_channels():
    rng=np.random.default_rng(182)
    x=rng.normal(size=(100,1000))+1j*rng.normal(size=(100,1000))
    a=local_lag_targets(cross_spectral_matrix(x[:,484:516],.001),[1,5,31])
    x[:,:484]=1e4; x[:,516:]=-1e4j
    b=local_lag_targets(cross_spectral_matrix(x[:,484:516],.001),[1,5,31])
    np.testing.assert_array_equal(a,b)


def test_complex_ranges_contain_known_truth():
    p=np.exp(3*np.cos(Q-.31));p/=p.sum()
    lags=np.array([1,2,3,5,8,13,21,34])
    values=np.exp(1j*lags[:,None]*Q)@p
    for row in conditional_ranges(values,lags,np.full(2*len(lags),.005)):
        truth=np.exp(1j*row['lag']*Q)@p
        truth=truth.real if row['axis']=='real' else truth.imag
        assert row['valid'] and row['lower']<=truth<=row['upper']


def test_measured_lags_cannot_be_labelled_unseen():
    with pytest.raises(ValueError,match='genuinely unseen'):
        conditional_ranges(np.array([.1+0j]),[4],np.ones(2)*.01)


def test_zero_information_abstains():
    lags=np.array([3,5,8,13,21,34,55,89])
    for row in conditional_ranges(np.zeros(8,complex),lags,np.full(16,.005)):
        assert row['valid'] and not row['identified'] and row['lower']<0<row['upper']
