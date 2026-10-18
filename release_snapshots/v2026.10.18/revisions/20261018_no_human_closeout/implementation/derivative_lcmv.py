"""Established derivative-constrained LCMV, not a new trained model.

Only pure numerical operations live here: no waveform reads, model loading,
label creation, tuning, outcome selection or final-assessment authorization.
The constraint convention is C^H w = g and the filtered output is w^H x.
For real g=(1,0), this gives w^H a=1 and w^H (q*a)=0.

References: Frost (1972), doi:10.1109/PROC.1972.8817; Er and Cantoni (1983),
doi:10.1109/TASSP.1983.1164219. This is a fixed first-order LCMV comparator,
NOT a general worst-case/ellipsoidal robust-MVDR implementation.
"""
from __future__ import annotations

import numpy as np

BANDS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))


def _loaded_factor(covariance, loading):
    matrix = np.asarray(covariance, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("Square multichannel covariance required")
    if not np.isfinite(matrix).all() or not np.isfinite(loading) or loading < 0:
        raise ValueError("Finite covariance and nonnegative loading required")
    scale = float(np.trace(matrix).real / len(matrix))
    if scale <= 0 or not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12*scale):
        raise ValueError("Hermitian covariance with positive mean power required")
    loaded = matrix + loading*scale*np.eye(len(matrix))
    # A substantially indefinite input is invalid, not rescued by a large load.
    if np.linalg.eigvalsh(matrix)[0] < -1e-10*scale:
        raise ValueError("Covariance must be positive semidefinite")
    try:
        return loaded, np.linalg.cholesky(loaded)
    except np.linalg.LinAlgError as error:
        raise ValueError("Loaded covariance must be positive definite") from error


def _constraint_solution(factor, constraints, gains):
    c = np.asarray(constraints, dtype=np.complex128)
    g = np.asarray(gains, dtype=np.complex128)
    if (c.ndim != 2 or c.shape[0] != len(factor) or not 1 <= c.shape[1] <= c.shape[0]
            or g.shape != (c.shape[1],) or not np.isfinite(c).all()
            or not np.isfinite(g).all()):
        raise ValueError("Finite channel-by-constraint matrix and aligned gains required")
    if np.linalg.matrix_rank(c) != c.shape[1]:
        raise ValueError("Constraints must be full column rank; no silent dropping")
    z = np.linalg.solve(factor.conj().T, np.linalg.solve(factor, c))
    gram = c.conj().T @ z
    try:
        gram_factor = np.linalg.cholesky((gram + gram.conj().T)/2)
        multipliers = np.linalg.solve(gram_factor.conj().T, np.linalg.solve(gram_factor, g))
    except np.linalg.LinAlgError as error:
        raise ValueError("Constraint Gram matrix must be positive definite") from error
    weights = z @ multipliers
    # One algebraic constraint-refinement step improves low-rank loaded cases.
    # It remains in span(R^-1 C) and does not drop or relax any constraint.
    correction = g - c.conj().T @ weights
    weights += z @ np.linalg.solve(gram_factor.conj().T,
                                   np.linalg.solve(gram_factor, correction))
    residual = np.max(np.abs(c.conj().T @ weights - g))
    if not np.isfinite(weights).all() or residual > 1e-9*max(1.0, np.max(np.abs(g))):
        raise ValueError("LCMV solve failed its declared constraint tolerance")
    return weights


def lcmv_weights(covariance, constraints, gains, loading=0.001):
    """Minimize w^H R_loaded w subject to C^H w=g.

    Cholesky solves only; no matrix inverse or pseudoinverse fallback. Redundant
    constraints and indefinite/nonfinite covariances fail explicitly.
    """
    _, factor = _loaded_factor(covariance, loading)
    return _constraint_solution(factor, constraints, gains)


def centred_coordinate(channels):
    """Fixed index coordinate in [-1,1] for the archived linear-index design.

    It is not a replacement for measured ENU geometry in a real source study.
    Scaling the tangent does not change the constraint span or its solution.
    """
    if not isinstance(channels, (int, np.integer)) or channels < 2:
        raise ValueError("At least two channels required")
    coordinate = np.arange(channels, dtype=float) - (channels-1)/2
    return coordinate / np.max(np.abs(coordinate))


def derivative_constraints(response, coordinate):
    a = np.asarray(response, dtype=np.complex128)
    q = np.asarray(coordinate, dtype=float)
    if (a.ndim != 1 or len(a) < 2 or q.shape != a.shape or not np.isfinite(a).all()
            or not np.isfinite(q).all() or np.any(np.abs(a) == 0)
            or np.ptp(q) <= 0):
        raise ValueError("Aligned finite nonzero response and nonconstant real coordinate required")
    return np.column_stack((a, q*a)), np.array([1.0, 0.0], dtype=complex)


def derivative_frequency_weights(past_covariances, samples, delays,
                                 coordinate=None, sample_rate=25.0, loading=0.001):
    """Unit response plus one fixed spatial-gradient null at each active bin.

    For true response a*(1+alpha*q), preservation is exact. For response
    a*exp(-2j*pi*f*epsilon*q), the first delay derivative is zero at epsilon=0.
    Neither result guarantees preservation for arbitrary real DAS wavefields.
    Input matrices must already be the fixed context_0.25 shrinkage matrices
    for the proposed comparator; this helper does not choose that fraction.
    """
    delays = np.asarray(delays, dtype=float)
    matrices = np.asarray(past_covariances, dtype=complex)
    if (delays.ndim != 1 or len(delays) < 2 or matrices.shape != (4, len(delays), len(delays))
            or not np.isfinite(delays).all()):
        raise ValueError("Four aligned covariances and finite channel delays required")
    if (not isinstance(samples, (int, np.integer)) or samples < 16
            or not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("Positive sample rate and adequate integer sample count required")
    q = centred_coordinate(len(delays)) if coordinate is None else np.asarray(coordinate, dtype=float)
    # Validate q before potentially opening a field runner; there is no fallback.
    derivative_constraints(np.ones(len(delays), complex), q)
    frequencies = np.fft.rfftfreq(samples, 1/sample_rate)
    weights = np.zeros((len(frequencies), len(delays)), complex)
    for band, (low, high) in enumerate(BANDS):
        chosen = np.flatnonzero((frequencies >= low) & (frequencies < high))
        if not len(chosen):
            raise ValueError("At least one Fourier bin in each declared band required")
        _, factor = _loaded_factor(matrices[band], loading)
        for index in chosen:
            response = np.exp(-2j*np.pi*frequencies[index]*delays)
            constraints, gains = derivative_constraints(response, q)
            weights[index] = _constraint_solution(factor, constraints, gains)
    return weights


def affine_delay_response_error_bound(weights, response, coordinate, frequency, edge_delay):
    """A deterministic quadratic remainder bound for affine delay mismatch.

    When unit/tangent constraints hold, |w^H a_true-1| is bounded by
    0.5*(2*pi*f*edge_delay)^2*sum(|conj(w)*a|*q^2). This concerns only the
    declared affine phase family, not random wavefield/error probabilities.
    """
    w, a = np.asarray(weights, complex), np.asarray(response, complex)
    q = np.asarray(coordinate, float)
    if (w.ndim != 1 or a.shape != w.shape or q.shape != w.shape
            or not all(np.isfinite(x).all() for x in (w, a, q))
            or not np.isfinite(frequency) or frequency < 0 or not np.isfinite(edge_delay)):
        raise ValueError("Finite aligned weights/response/coordinate and valid frequency required")
    return float(0.5*(2*np.pi*frequency*edge_delay)**2 * np.sum(np.abs(w.conj()*a)*q*q))
