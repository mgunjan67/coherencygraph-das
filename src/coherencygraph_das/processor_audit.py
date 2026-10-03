"""Conditional processor-power audit; never a physical-coverage guarantee.

The stationary Fourier class and spectral solver are historical. The new
objective is a fixed processor power difference on the same local aperture.
"""
from __future__ import annotations

import numpy as np

from .covariance_numerics import geometry_fit, validated_min


def remove_loading(matrix: np.ndarray, loading: float = .001) -> np.ndarray:
    """Invert the archive's R + loading * trace(R)/n * I operation."""
    a = np.asarray(matrix, dtype=np.complex128)
    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or not np.isfinite(a).all():
        raise ValueError("Finite square matrices required")
    if loading < 0:
        raise ValueError("Loading must be nonnegative")
    scale = np.trace(a, axis1=-2, axis2=-1).real / a.shape[-1] / (1 + loading)
    return a - loading * scale[..., None, None] * np.eye(a.shape[-1])


def correlation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(matrix, dtype=np.complex128)
    d = np.diagonal(a, axis1=-2, axis2=-1).real
    if not np.isfinite(a).all() or np.any(d <= 0):
        raise ValueError("Finite matrix with positive diagonal required")
    return a / np.sqrt(d[..., :, None] * d[..., None, :]), d


def moments(corr: np.ndarray, lags: np.ndarray) -> np.ndarray:
    d = np.asarray(lags)
    if np.any(d != d.astype(int)) or np.any(d <= 0) or np.any(d >= corr.shape[-1]):
        raise ValueError("Lags must be positive integers inside the aperture")
    return np.stack([np.diagonal(corr, offset=int(k), axis1=-2, axis2=-1).mean(-1)
                     for k in d], axis=-1)


def power_objective(q: np.ndarray, positions: np.ndarray, weights: np.ndarray,
                    powers: np.ndarray, noise_power: float = 1.) -> np.ndarray:
    """c'p = w^H D^.5 C(p) D^.5 w / noise_power.

    C_ij(p) = sum_k p_k exp(i q_k (x_j-x_i)); hence the + sign below.
    No outcome-dependent fitting or weight selection occurs in this function.
    """
    x = np.asarray(positions, float)
    w = np.asarray(weights, complex)
    d = np.asarray(powers, float)
    q = np.asarray(q, float)
    if x.ndim != 1 or w.shape != x.shape or d.shape != x.shape or q.ndim != 1:
        raise ValueError("Incompatible one-dimensional coordinates/weights/powers")
    if not all(np.isfinite(v).all() for v in (x, w, d, q)) or np.any(d <= 0):
        raise ValueError("Finite inputs and positive powers required")
    if not np.isfinite(noise_power) or noise_power <= 0:
        raise ValueError("Positive noise power required")
    return np.abs(np.exp(1j * q[:, None] * x) @ (np.sqrt(d) * w)) ** 2 / noise_power


def conditional_power_bounds(values: np.ndarray, lags: np.ndarray,
                             allowances: np.ndarray, objective: np.ndarray) -> dict:
    values = np.asarray(values, complex)
    lags = np.asarray(lags)
    delta = np.asarray(allowances, float)
    c = np.asarray(objective, float)
    if values.ndim != 1 or values.shape != lags.shape or delta.shape != (2 * len(lags),):
        raise ValueError("Invalid values, lags or allowance dimensions")
    if np.any(lags <= 0) or np.any(lags != lags.astype(int)) or len(set(lags)) != len(lags):
        raise ValueError("Unique positive integer lags required")
    if not all(np.isfinite(v).all() for v in (values, delta, c)) or np.any(delta < 0):
        raise ValueError("Finite inputs and nonnegative allowances required")
    if c.shape != (257,):
        raise ValueError("Historical stationary grid has 257 frequencies")
    p, A, z, q, fit = geometry_fit(values, lags)
    eps = np.abs(A @ p - z) + delta
    H = np.r_[A, -A]
    b = np.r_[z + eps, -z + eps]
    lo = validated_min(c, H, b)
    hi = validated_min(-c, H, b)
    valid = bool(fit['fit_converged'] and lo['valid'] and hi['valid'])
    lower = lo['outer'] if valid else None
    upper = -hi['outer'] if valid else None
    return dict(valid=valid, lower=lower, upper=upper,
                width=upper-lower if valid else None,
                sign=(1 if lower > 1e-7 else -1 if upper < -1e-7 else 0) if valid else 0,
                fitted_contrast=float(c @ p),
                max_fit_residual=float(np.abs(A @ p-z).max()),
                min_allowance=float(delta.min()), max_allowance=float(delta.max()),
                low_gap=lo['gap'], high_gap=hi['gap'],
                fit_converged=bool(fit['fit_converged']), fit_gap=fit['fit_gap'])
