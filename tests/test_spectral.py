from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coherencygraph_das.spectral import (
    lag_coherency,
    normalize_coherency,
    spectral_mixture_operator,
)


def test_spectral_mixture_operator_is_hermitian_psd() -> None:
    rng = np.random.default_rng(7)
    probabilities = rng.random(65)
    operator = spectral_mixture_operator(probabilities, np.arange(32))
    assert np.allclose(operator, operator.conj().T, atol=1e-12)
    assert np.allclose(np.diag(operator), 1.0)
    assert np.linalg.eigvalsh(operator).min() > -1e-10


def test_lag_coherency_recovers_perfect_delayed_signal() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(size=(200, 1)) + 1j * rng.normal(size=(200, 1))
    snapshots = np.repeat(base, 20, axis=1)
    gamma = lag_coherency(snapshots, [1, 3, 7])
    assert np.allclose(gamma, 1.0 + 0j, atol=1e-6)


def test_normalization_has_unit_diagonal() -> None:
    matrix = np.array([[4, 1 + 1j], [1 - 1j, 9]], dtype=complex)
    gamma = normalize_coherency(matrix)
    assert np.allclose(np.diag(gamma), 1.0)
    assert np.allclose(gamma, gamma.conj().T)

