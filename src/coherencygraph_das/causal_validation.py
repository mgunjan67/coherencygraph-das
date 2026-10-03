"""Separate fixed-window replication; does not alter the primary S-window analysis."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .spectral import multitaper_fourier, band_snapshots, lag_coherency, cross_spectral_matrix
from .processor_audit import remove_loading, correlation, moments

FS = 25.0
CUTOFFS = (20.0, 40.0, 60.0, 80.0)
BLOCKS = tuple(range(200, 7201, 1000))
LOCAL = np.arange(484, 516)
FEATURE_LAGS = np.array([3, 5, 8, 13, 21, 34, 55, 89])
SPARSE = np.array([3, 5, 8, 13, 21])
DENSE = np.arange(1, 32)
BANDS = ((.5, 1), (1, 2), (2, 4), (4, 8))
Q = np.sort(np.fft.fftfreq(257)*2*np.pi)
DATASET = 'Acquisition/Raw[0]/RawData'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def hypocentral_distances(a, b):
    x = np.deg2rad(a.latitude_deg.to_numpy())[:, None]
    y = np.deg2rad(b.latitude_deg.to_numpy())[None, :]
    longitude = np.deg2rad(a.longitude_deg.to_numpy()[:, None]-b.longitude_deg.to_numpy()[None, :])
    value = np.sin((x-y)/2)**2 + np.cos(x)*np.cos(y)*np.sin(longitude/2)**2
    arc = 2*6371*np.arcsin(np.sqrt(np.clip(value, 0, 1)))
    return np.hypot(arc, a.depth_km.to_numpy()[:, None]-b.depth_km.to_numpy()[None, :])


def source_components(frame, radius=25.0):
    distance = hypocentral_distances(frame, frame)
    parent = list(range(len(frame)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in zip(*np.where(np.triu(distance <= radius, 1))):
        a, b = root(int(a)), root(int(b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(len(frame))])


def candidate_order(catalog, exposed, maximum=50):
    frame = catalog[
        ~catalog.event_id.astype(str).isin(exposed.event_id.astype(str))
        & catalog.archive_date.str.startswith('2023')
        & catalog.terra_url.notna() & catalog.kkfls_url.notna()
        & catalog.magnitude.ge(2.5)
    ].copy()
    home = pd.DataFrame(dict(latitude_deg=[59.65], longitude_deg=[-151.55], depth_km=[0.]))
    frame['homer_hypocentral_km'] = hypocentral_distances(frame, home).ravel()
    frame['nearest_exposed_km'] = hypocentral_distances(frame, exposed).min(1)
    frame = frame[frame.homer_hypocentral_km.le(300) & frame.nearest_exposed_km.ge(25)].copy()
    frame['selection_hash'] = frame.event_id.map(lambda x: hashlib.sha256(f'20260922:{x}'.encode()).hexdigest())
    frame = frame.sort_values('selection_hash').reset_index(drop=True)
    distance = hypocentral_distances(frame, frame)
    chosen = []
    for i in range(len(frame)):
        if not chosen or np.min(distance[i, chosen]) >= 25:
            chosen.append(i)
        if len(chosen) >= maximum:
            break
    selected = frame.iloc[chosen].copy()
    selected['candidate_rank'] = np.arange(1, len(selected)+1)
    selected['candidate_role'] = np.where(selected.candidate_rank <= 30, 'primary', 'reserve')
    return frame, selected


def sample_slice(start, stop, n, fs=FS):
    """Half-open physical-time intervals, with no silent clipping."""
    left = int(math.ceil(start*fs-1e-10))
    right = int(math.ceil(stop*fs-1e-10))
    if left < 0 or right > n or right-left < 16:
        raise ValueError('Incomplete fixed window')
    return slice(left, right)


def window_features(context, reference, block, route, gauge):
    """Inputs depend on these two arrays only, never target data or picks."""
    values = []
    mats = []
    for data in (context, reference):
        fft, f = multitaper_fourier(data, FS, 2.5, 3, None)
        snapshots = [band_snapshots(fft, f, band) for band in BANDS]
        gamma = np.array([lag_coherency(x, FEATURE_LAGS) for x in snapshots])
        values.append(gamma)
        mats.append(np.array([cross_spectral_matrix(x[:, LOCAL], .001) for x in snapshots]))
    gc, gn = values
    rc, rn = mats
    pc = np.trace(rc, axis1=-2, axis2=-1).real/len(LOCAL)
    pn = np.trace(rn, axis1=-2, axis2=-1).real/len(LOCAL)
    ratios = np.log1p(np.maximum(pc/np.maximum(pn, 1e-15), 0))
    feature = np.r_[gc.real.ravel(), gc.imag.ravel(), gn.real.ravel(), gn.imag.ravel(),
                    ratios, np.linspace(-1, 1, 8)[block], float(route == 'TERRA'), gauge/24]
    if feature.shape != (135,) or not np.isfinite(feature).all():
        raise ValueError('Invalid fixed-window feature vector')
    return feature.astype('float32'), rc, rn


def target_matrices(target):
    split = len(target)//2
    output = []
    for data in (target, target[:split], target[split:]):
        fft, f = multitaper_fourier(data[:, LOCAL], FS, 2.5, 3, None)
        output.append(np.array([cross_spectral_matrix(band_snapshots(fft, f, b), .001) for b in BANDS]))
    return output


def metadata(handle):
    a = handle['Acquisition'].attrs
    raw = handle[DATASET]
    unit = handle['Acquisition/Raw[0]'].attrs.get('RawDataUnit', '')
    if isinstance(unit, bytes):
        unit = unit.decode()
    record = dict(sample_rate_hz=float(a['PulseRate']), spacing_m=float(a['SpatialSamplingInterval']),
                  gauge_m=float(a['GaugeLength']), samples=int(raw.shape[0]), channels=int(raw.shape[1]),
                  raw_unit=str(unit), start_time=str(a.get('MeasurementStartTime', '')),
                  output_rate_hz=float(handle['Acquisition/Raw[0]'].attrs.get('OutputDataRate', np.nan)))
    checks = [abs(record['sample_rate_hz']-25) <= .01, record['samples'] >= 2200,
              record['channels'] >= 8200, abs(record['spacing_m']-9.5714288) <= .02,
              min(abs(record['gauge_m']-17.55), abs(record['gauge_m']-23.93)) <= .15,
              ('rad' in record['raw_unit'].lower()) or ('phase' in record['raw_unit'].lower()),
              abs(record['output_rate_hz']-25) <= .01]
    if not all(checks):
        raise ValueError(f'Acquisition QC failed: {record}')
    return record


def extract_file(raw_path, route, output):
    output = Path(output)
    if output.exists():
        with np.load(output) as prior:
            if str(prior['raw_sha256']) != sha256(raw_path):
                raise RuntimeError('Cached extraction has different raw input')
            if str(prior['extractor_sha256']) != sha256(__file__):
                raise RuntimeError('Cached extraction has different source code')
        return dict(cache=str(output), reused=True)
    features = np.zeros((4, 8, 135), 'float32')
    arrays = {k: np.zeros((4, 8, 4, 32, 32), 'complex64') for k in ['context','reference','target','first','second']}
    finite_count = sample_count = 0
    with h5py.File(raw_path, 'r') as handle:
        info = metadata(handle)
        raw = handle[DATASET]
        for bi, start in enumerate(BLOCKS):
            # All requested windows fit in the first 88 seconds.
            data = np.asarray(raw[:2200, start:start+1000], dtype='float64')
            finite_count += int(np.isfinite(data).sum()); sample_count += data.size
            if np.isfinite(data).mean() < .999:
                raise ValueError('Finite coverage below threshold')
            reference = data[sample_slice(.5, 5.5, len(data))]
            for ci, cutoff in enumerate(CUTOFFS):
                context = data[sample_slice(cutoff-14, cutoff, len(data))]
                target = data[sample_slice(cutoff+1, cutoff+8, len(data))]
                x, rc, rn = window_features(context, reference, bi, route, info['gauge_m'])
                rt, r1, r2 = target_matrices(target)
                features[ci, bi] = x
                for key, value in zip(arrays, (rc, rn, rt, r1, r2)):
                    arrays[key][ci, bi] = value
    gamma = moments(correlation(remove_loading(arrays['target']))[0], DENSE)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, targets=gamma,
                        raw_sha256=sha256(raw_path), extractor_sha256=sha256(__file__), **arrays, **info)
    return dict(cache=str(output), reused=False, finite_fraction=finite_count/sample_count, **info)


def upper_kernel(p):
    p = np.asarray(p, float)
    p = np.maximum(p, 0)/max(np.maximum(p, 0).sum(), 1e-30)
    lag = np.arange(32)[None, :]-np.arange(32)[:, None]
    kernel = np.einsum('q,ijq->ij', p, np.exp(1j*lag[..., None]*Q))
    return (kernel+kernel.conj().T)/2


def storage_guard(directory, next_bytes, used_bytes, maximum=8*1024**3, reserve=100*1024**3):
    import shutil
    if next_bytes < 0 or used_bytes+next_bytes > maximum:
        raise RuntimeError('New raw-data budget would be exceeded')
    if shutil.disk_usage(directory).free-next_bytes < reserve:
        raise RuntimeError('100 GiB free-space reserve would be violated')
