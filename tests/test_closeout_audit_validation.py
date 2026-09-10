"""Regression checks for the closeout wrapper, not new scientific fixtures."""
import importlib.util
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('closeout_audit_validation',ROOT/'scripts/closeout_audit_validation.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def test_empirical_check_keeps_zero_and_tiny_opposite_sign():
    bound=dict(valid=True,identified=True,sign=1,lower=.1,upper=.4)
    zero=audit.empirical_fields(bound,0.)
    tiny=audit.empirical_fields(bound,-1e-20)
    assert zero['measured_exact_zero'] and zero['identified_sign_disagreement']
    assert tiny['identified_sign_disagreement'] and not tiny['interval_contains']
    assert np.isclose(tiny['distance_outside'],.1)

def test_unresolved_is_not_disagreement_and_invalid_is_not_containment():
    unresolved=dict(valid=True,identified=False,sign=0,lower=-.2,upper=.4)
    check=audit.empirical_fields(unresolved,0.)
    assert check['interval_contains'] and not check['identified_sign_disagreement']
    invalid=dict(valid=False,identified=False,sign=0,lower=None,upper=None)
    check=audit.empirical_fields(invalid,.2)
    assert not check['interval_contains'] and check['distance_outside'] is None

def test_detailed_wrapper_preserves_original_simplex_lp_results():
    lags=[1,2,3,5,8,13,21,34]
    values=np.zeros(8,dtype=complex);delta=np.full(16,.005)
    with threadpool_limits(limits=1):
        actual,residuals=audit.detailed_ranges(values,lags,delta)
        original=audit.r.conditional_ranges(values,lags,delta)
    assert len(actual)==4 and len(residuals)==16
    for a,b in zip(actual,original):
        assert a['lag']==b['lag'] and a['axis']==b['axis']
        assert a['valid']==b['valid'] and a['sign']==b['sign']
        for key in ['lower','upper','width','fit_gap','low_gap','high_gap']:
            assert abs(a[key]-b[key])<1e-12
        assert a['low_primal_violation']<=1e-7 and a['high_primal_violation']<=1e-7
