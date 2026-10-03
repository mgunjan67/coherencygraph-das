"""Independent numerical replay fixtures; do not modify sealed analysis tests."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from coherencygraph_das.causal_validation import upper_kernel
from coherencygraph_das.covariance_numerics import _generalized_weight


def replay():
    file = Path(__file__).resolve().parents[1]/'scripts/verify_causal_validation.py'
    spec = importlib.util.spec_from_file_location('replay', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('seed', [4, 11, 27])
def test_explicit_fourier_covariance(seed):
    module = replay()
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(257)*.2)
    actual = module.direct_matrix(p)
    np.testing.assert_allclose(actual, upper_kernel(p), atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() > -1e-12
    np.testing.assert_allclose(np.diag(actual), 1, atol=1e-12)


@pytest.mark.parametrize('seed', [4, 11, 27])
def test_whitening_replay_matches_generalized_eigenvector(seed):
    module = replay()
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(32, 40))+1j*rng.normal(size=(32, 40))
    y = rng.normal(size=(32, 10))+1j*rng.normal(size=(32, 10))
    signal = x@x.conj().T+np.eye(32)*.001
    reference = y@y.conj().T+np.eye(32)*.001
    expected = _generalized_weight(signal, reference, .0001)
    actual = module.direct_weight(signal, reference)
    assert abs(np.vdot(actual, expected)) > 1-1e-9
    target = x[:, :20]@x[:, :20].conj().T
    ratios = [(w.conj()@target@w).real/(w.conj()@reference@w).real for w in [actual, expected]]
    assert abs(10*np.log10(ratios[0]/ratios[1])) < 1e-7


def test_unloading_and_normalization_by_direct_entries():
    module = replay()
    rng = np.random.default_rng(51)
    x = rng.normal(size=(32, 20))+1j*rng.normal(size=(32, 20))
    matrix = x@x.conj().T
    loaded = matrix+.001*np.trace(matrix).real/32*np.eye(32)
    expected = matrix/np.sqrt(np.diag(matrix).real[:, None]*np.diag(matrix).real[None, :])
    np.testing.assert_allclose(module.unloaded_correlation(loaded), expected, atol=1e-12)
