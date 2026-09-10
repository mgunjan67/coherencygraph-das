import numpy as np
from coherencygraph_das.closeout_api import audit_moments
from threadpoolctl import threadpool_limits


def test_measured_lag_is_never_unseen_identification():
    with threadpool_limits(limits=1):
        result = audit_moments([1, 2], [.9, .9], [0, 1, 2], np.full(4, .005), [1])
    assert all(r['directly_measured'] and not r['identified'] for r in result['intervals'])
    assert {r['status'] for r in result['intervals']} == {'directly_measured'}
    assert {r['axis'] for r in result['intervals']} == {'real', 'imaginary'}
    assert 'jointly attainable' in result['interval_interpretation']
    assert 'does not validate' in result['interval_interpretation']


def test_failed_fit_status(monkeypatch):
    import coherencygraph_das.submission_revision as implementation
    original = implementation.geometry_fit
    def rejected(*args, **kwargs):
        p, a, z, q, fit = original(*args, **kwargs)
        return p, a, z, q, {**fit, 'fit_converged': False}
    monkeypatch.setattr(implementation, 'geometry_fit', rejected)
    result = audit_moments([1], [0j], [0, 1], [.005, .005], [4])
    assert all(r['status'] == 'failed_fit' and not r['identified'] for r in result['intervals'])


def test_failed_bound_status(monkeypatch):
    import coherencygraph_das.final_revision as implementation
    monkeypatch.setattr(implementation, 'validated_min', lambda *args: {'valid': False, 'outer': None})
    result = audit_moments([1], [0j], [0, 1], [.005, .005], [4])
    assert all(r['status'] == 'failed_bound' and not r['identified'] for r in result['intervals'])
