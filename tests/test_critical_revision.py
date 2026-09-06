import itertools
import numpy as np
from coherencygraph_das.critical_revision import (Q,LAGS,metrics,spectrum_fit,upper_kernel_matrix,
    conformal_quantile,boot,feasible_bounds)
from coherencygraph_das.spectral import lag_coherency,cross_spectral_matrix,multitaper_fourier


def test_complex_metric_and_event_weighting():
    t=np.ones((2,4,8),complex);p=t+1j
    r=metrics(p,t)
    assert r['mse']==1 and r['nrmse']==1
    r=boot([1,3,10],groups=['a','a','b'],replicates=500)
    assert r['mean']==14/3 and r['groups']==2 and r['events']==3


def test_raw_receipt_names_are_cross_platform():
    from coherencygraph_das.critical_raw import _cache_stem
    assert _cache_stem(r'D:\old\audit\11710081_TERRA.npz')=='11710081_TERRA'
    assert _cache_stem('/data/audit/11710081_TERRA.npz')=='11710081_TERRA'


def test_known_asymmetric_covariance_and_upper_lag_sign():
    p=np.exp(4*np.cos(Q-.35));p/=p.sum()
    c=upper_kernel_matrix(p,np.arange(16))
    assert np.linalg.eigvalsh(c).min()>-1e-12
    assert np.allclose(c,c.conj().T)
    assert np.allclose(np.diag(c),1)
    assert np.allclose(c[0,3],np.exp(3j*Q)@p)
    # Deterministic spectral snapshots exactly reproduce C under X_i conj(X_j).
    evals,evecs=np.linalg.eigh(c)
    x=np.sqrt(16)*np.diag(np.sqrt(np.maximum(evals,0)))@evecs.T
    assert np.allclose(cross_spectral_matrix(x),c,atol=1e-6)
    got=lag_coherency(x,[1,3,5])
    assert np.allclose(got,[c[0,i] for i in [1,3,5]],atol=1e-6)
    assert np.allclose(lag_coherency(x[:,::-1],[1,3,5]),got.conj(),atol=1e-6)


def test_future_replacement_and_known_delay():
    fs=25.;time=np.arange(1000)/fs
    delays=np.array([0.,.2,.4]);x=np.cos(2*np.pi*2*(time[:,None]-delays))
    a,_=multitaper_fourier(x[:350],fs,2.5,3,delays)
    changed=x.copy();changed[350:]=np.random.default_rng(8).normal(size=changed[350:].shape)
    b,f=multitaper_fourier(changed[:350],fs,2.5,3,delays)
    assert np.array_equal(a,b)
    keep=abs(f-2)<.2
    gamma=lag_coherency(a[:,keep,:].reshape(-1,3),[1])
    assert abs(gamma[0].imag)<.01 and gamma[0].real>.99


def test_positive_spectral_nonuniqueness_and_periodicity():
    plus=(1+.9*np.cos(Q))/257;minus=(1-.9*np.cos(Q))/257
    B=np.exp(1j*LAGS[:,None]*Q)
    assert min(plus.min(),minus.min())>0
    assert np.max(abs(B@(plus-minus)))<1e-12
    assert np.isclose(np.exp(1j*Q)@(plus-minus),.9)
    assert np.allclose(np.exp(1j*(LAGS[:,None]+257)*Q),B)
    old=np.linspace(-np.pi,np.pi,65,endpoint=False)
    assert np.allclose(np.exp(1j*(LAGS[:,None]+65)*old),-np.exp(1j*LAGS[:,None]*old))


def test_simplex_solver_and_bounds_contain_known_truth():
    p=np.exp(5*np.cos(Q-.15));p/=p.sum();y=np.exp(1j*LAGS[:,None]*Q)@p
    fit,r=spectrum_fit(y)
    assert fit.min()>=0 and np.isclose(fit.sum(),1)
    assert r['observed_rmse']<1e-6
    for row in feasible_bounds(y,[1,2]):
        truth=np.cos(row['lag']*Q)@p
        assert row['lower']<=truth<=row['upper']
    for row in feasible_bounds(np.zeros(8,complex),[1,2]):
        assert row['lower']<0<row['upper']
        assert not row['sign_identified']


def test_conformal_joint_quantiles_and_mask_counts():
    scores=np.arange(1,25)
    assert [conformal_quantile(scores,c) for c in [.8,.9,.95]]==[(20.,20),(23.,23),(24.,24)]
    assert np.isinf(conformal_quantile(scores,.999)[0])
    assert [len(list(itertools.combinations(range(8),k))) for k in [1,2,4]]==[8,28,70]


def test_processor_known_covariance_and_common_frame_invariance():
    from scipy.linalg import eigh
    from coherencygraph_das.review_revision import _generalized_weight,_ratio
    p=np.exp(4*np.cos(Q-.4));p/=p.sum()
    signal=upper_kernel_matrix(p,np.arange(12))+.1*np.eye(12)
    noise=np.diag(np.linspace(.8,1.5,12))
    weight=_generalized_weight(signal,noise,0.)
    optimum=eigh(signal,noise,eigvals_only=True)[-1]
    assert np.isclose(_ratio(signal,noise,weight),optimum,rtol=1e-6)
    phase=np.diag(np.exp(1j*np.linspace(0,2,12)))
    s2=phase@signal@phase.conj().T;n2=phase@noise@phase.conj().T
    w2=_generalized_weight(s2,n2,0.)
    assert np.isclose(_ratio(s2,n2,w2),optimum,rtol=1e-6)
