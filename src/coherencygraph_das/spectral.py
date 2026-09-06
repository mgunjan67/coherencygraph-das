from __future__ import annotations

import numpy as np
from scipy import signal


def robust_linear_pick_model(
    channels: np.ndarray,
    picks: np.ndarray,
    block_start: int,
    block_channels: int,
) -> tuple[float, float, float]:
    stop = block_start + block_channels
    keep = (
        np.isfinite(channels)
        & np.isfinite(picks)
        & (channels >= block_start)
        & (channels < stop)
    )
    x = channels[keep].astype(float)
    y = picks[keep].astype(float)
    coverage = float(len(x) / block_channels)
    if len(x) < 10:
        return float(np.nanmedian(y) if len(y) else np.nan), 0.0, coverage
    for _ in range(3):
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (intercept + slope * x)
        mad = np.median(np.abs(residual - np.median(residual))) * 1.4826
        if not np.isfinite(mad) or mad <= 1e-6:
            break
        retain = np.abs(residual - np.median(residual)) <= 3.5 * mad
        if retain.sum() < 10 or retain.all():
            break
        x, y = x[retain], y[retain]
    centre = block_start + 0.5 * (block_channels - 1)
    reference = float(intercept + slope * centre)
    return reference, float(slope), coverage


def multitaper_fourier(
    data_time_channel: np.ndarray,
    sample_rate_hz: float,
    time_bandwidth: float,
    tapers: int,
    delay_sec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data_time_channel, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 16:
        raise ValueError("data must have shape [time, channel] with at least 16 samples")
    finite = np.isfinite(data)
    if finite.mean() < 0.999:
        raise ValueError("finite-data fraction below frozen threshold")
    data = np.nan_to_num(data, copy=False)
    data = signal.detrend(data, axis=0, type="linear")
    windows = signal.windows.dpss(
        data.shape[0], NW=float(time_bandwidth), Kmax=int(tapers), sym=False
    )
    transformed = np.fft.rfft(windows[:, :, None] * data[None, :, :], axis=1)
    frequencies = np.fft.rfftfreq(data.shape[0], d=1.0 / sample_rate_hz)
    if delay_sec is not None:
        delay = np.asarray(delay_sec, dtype=float)
        if delay.shape != (data.shape[1],):
            raise ValueError("delay_sec must have one value per channel")
        transformed *= np.exp(
            2j * np.pi * frequencies[None, :, None] * delay[None, None, :]
        )
    return transformed, frequencies


def band_snapshots(
    transformed: np.ndarray,
    frequencies: np.ndarray,
    band: tuple[float, float] | list[float],
    taper_indices: np.ndarray | None = None,
) -> np.ndarray:
    low, high = map(float, band)
    keep_f = (frequencies >= low) & (frequencies < high)
    if not keep_f.any():
        raise ValueError(f"no Fourier bins in band {band}")
    selected = transformed if taper_indices is None else transformed[taper_indices]
    return selected[:, keep_f, :].reshape(-1, transformed.shape[-1])


def lag_coherency(snapshots: np.ndarray, lags: list[int] | np.ndarray) -> np.ndarray:
    x = np.asarray(snapshots)
    output: list[complex] = []
    eps = np.finfo(float).eps
    for lag in map(int, lags):
        if lag < 1 or lag >= x.shape[1]:
            output.append(np.nan + 1j * np.nan)
            continue
        left, right = x[:, :-lag], x[:, lag:]
        cross = np.mean(left * np.conj(right), axis=0)
        denom = np.sqrt(
            np.mean(np.abs(left) ** 2, axis=0)
            * np.mean(np.abs(right) ** 2, axis=0)
        )
        pair = cross / np.maximum(denom, eps)
        pair = pair[np.isfinite(pair)]
        output.append(complex(pair.mean()) if len(pair) else np.nan + 1j * np.nan)
    return np.asarray(output, dtype=np.complex64)


def cross_spectral_matrix(snapshots: np.ndarray, loading: float = 0.0) -> np.ndarray:
    x = np.asarray(snapshots)
    matrix = x.T @ np.conj(x) / max(1, x.shape[0])
    matrix = 0.5 * (matrix + matrix.conj().T)
    if loading:
        scale = float(np.real(np.trace(matrix)) / matrix.shape[0])
        matrix = matrix + float(loading) * max(scale, np.finfo(float).eps) * np.eye(
            matrix.shape[0]
        )
    return matrix.astype(np.complex64)


def normalize_coherency(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.maximum(np.real(np.diag(matrix)), np.finfo(float).eps)
    output = matrix / np.sqrt(diagonal[:, None] * diagonal[None, :])
    output = 0.5 * (output + output.conj().T)
    np.fill_diagonal(output, 1.0)
    return output.astype(np.complex64)


def spectral_mixture_operator(
    probabilities: np.ndarray,
    positions: np.ndarray,
    wavenumbers: np.ndarray | None = None,
) -> np.ndarray:
    """Legacy pre-audit convention, retained only for historical reproduction.

    Amendment 07 uses critical_revision.upper_kernel_matrix with centred FFT
    harmonics and the upper-diagonal sign. Do not use this legacy helper for
    revised covariance inference or advertise its default grid as corrected.
    """
    p = np.asarray(probabilities, dtype=float)
    p = np.maximum(p, 0.0)
    p /= p.sum(axis=-1, keepdims=True)
    if wavenumbers is None:
        wavenumbers = np.linspace(-np.pi, np.pi, p.shape[-1], endpoint=False)
    separation = positions[:, None] - positions[None, :]
    operator = np.einsum("k,ijk->ij", p, np.exp(1j * separation[:, :, None] * wavenumbers))
    operator = 0.5 * (operator + operator.conj().T)
    np.fill_diagonal(operator, 1.0)
    return operator


def effective_rank(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T)).real
    eigenvalues = np.maximum(eigenvalues, 0.0)
    denominator = float(np.sum(eigenvalues**2))
    return float(np.sum(eigenvalues) ** 2 / denominator) if denominator > 0 else np.nan
