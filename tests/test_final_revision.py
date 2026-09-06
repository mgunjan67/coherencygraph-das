import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coherencygraph_das.final_revision import bounds, validated_min, fit_grid
from coherencygraph_das.critical_revision import Q,LAGS,upper_kernel_matrix,boot,metrics,components

def test_dual_outer_bound_known_simplex_and_tolerance():
    c=np.array([-.7,.2,.8]);H=np.array([[1.,0.,0.],[-1.,0.,0.]])
    b=np.array([.4,-.2]);r=validated_min(c,H,b)
    # Exact minimum: .4*(-.7)+.6*.2 = -.16.
    assert r['valid'] and r['outer']<=-.16+1e-12
    assert -.16-r['outer']<1e-7

def test_infeasible_solver_is_not_certificate():
    r=validated_min(np.array([0.,1.]),np.ones((1,2)),np.array([-.1]))
    assert not r['valid'] and r['status']==2

def test_nonunique_sign_abstains_and_contains_both_truths():
    p=(1+.9*np.cos(Q))/257;y=np.exp(1j*LAGS[:,None]*Q)@p
    r=bounds(y,LAGS,unseen=[1])[0]
    assert r['valid'] and r['sign']==0 and r['lower']<=-.45 and r['upper']>=.45

def test_direct_neighbour_measurement_identifies_conditional_sign():
    p=(1+.9*np.cos(Q))/257;lags=np.array([1,2,3,5,8,13,21,34])
    r=bounds(np.exp(1j*lags[:,None]*Q)@p,lags,unseen=[1])[0]
    assert r['valid'] and r['directly_measured'] and r['sign']==1
    assert r['lower']<=.45<=r['upper']

def test_grid_fit_sign_and_complete_geometry():
    p=np.exp(3*np.cos(Q-.7));p/=p.sum();pos=np.arange(32)
    C=upper_kernel_matrix(p,pos)
    assert np.allclose(C,C.conj().T) and np.linalg.eigvalsh(C).min()>-1e-10
    assert np.allclose(C[0,1],np.exp(1j*Q)@p)
    d=abs(pos[:,None]-pos[None,:]);off=~np.eye(32,dtype=bool)
    assert np.isin(d[off],np.arange(1,32)).all()
    assert not np.isin(d[off],LAGS).all()

def test_component_estimator_is_event_weighted_not_group_weighted():
    result=boot([0,0,0,4],['a','a','a','b'],replicates=100)
    assert result['mean']==1 and result['groups']==2
    t=np.ones((8,4,8),complex)*(1+1j);p=2*t
    m=metrics(components(p),components(t))
    assert np.isclose(m['nrmse'],1) and np.isclose(m['mse'],2,rtol=1e-6)
