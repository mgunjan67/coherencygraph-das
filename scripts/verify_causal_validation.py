"""Read-only scientific replay of the sealed, pick-free replication outputs.

This does not select models, change outcomes or redefine acceptance gates.
Only verification receipts are written.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from coherencygraph_das.causal_validation import sha256, hypocentral_distances
from coherencygraph_das.covariance_numerics import geometry_fit

OUT = ROOT/'revisions/20260922_unseen_validation'


def direct_matrix(p):
    q = np.sort(np.fft.fftfreq(257))*2*np.pi
    p = np.maximum(np.asarray(p, float), 0)
    p /= p.sum()
    vectors = np.exp(-1j*np.arange(32)[:, None]*q[None, :])
    return (vectors*p[None, :])@vectors.conj().T


def direct_weight(matrix, reference):
    """Whiten, solve an ordinary Hermitian problem, and map back."""
    reference = .5*(reference+reference.conj().T)
    loaded = reference + .0001*max(float(np.trace(reference).real/32), np.finfo(float).eps)*np.eye(32)
    lower = cholesky(loaded, lower=True)
    left = solve_triangular(lower, matrix, lower=True)
    white = solve_triangular(lower, left.conj().T, lower=True).conj().T
    white = .5*(white+white.conj().T)
    _, vectors = np.linalg.eigh(white)
    weight = solve_triangular(lower.conj().T, vectors[:, -1], lower=False)
    return weight/np.linalg.norm(weight)


def unloaded_correlation(matrix):
    m = np.asarray(matrix, complex).copy()
    m -= .001*np.trace(m).real/32/1.001*np.eye(32)
    d = np.diag(m).real
    return m/np.sqrt(d[:, None]*d[None, :])


def main():
    spec = importlib.util.spec_from_file_location('sealed_runner', ROOT/'scripts/run_causal_validation.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.verify_seal()
    records = pd.read_csv(OUT/'validation_records.csv', dtype={'event_id': str}).sort_values(['event_id', 'route'])
    assert records.groupby('event_id').route.apply(set).map(lambda x: x == {'TERRA', 'KKFL-S'}).all()
    assert 24 <= records.event_id.nunique() <= 30
    old = pd.read_csv(OUT/'excluded_previously_used_events.csv', dtype={'event_id': str})
    candidates = pd.read_csv(OUT/'locked_candidate_order.csv', dtype={'event_id': str})
    selected = candidates.set_index('event_id').loc[sorted(records.event_id.unique())].reset_index()
    assert not set(records.event_id) & set(old.event_id)
    minimum_old_distance = float(hypocentral_distances(selected, old).min())
    assert minimum_old_distance >= 25
    within = hypocentral_distances(selected, selected)+np.eye(len(selected))*1e9
    assert within.min() >= 25
    for row in records.itertuples():
        with np.load(OUT/f'validation/cache/{row.event_id}_{row.route.replace("-", "")}.npz') as z:
            assert str(z['raw_sha256']) == sha256(row.raw_path) == row.sha256

    source = records.iloc[0]
    key = f'{source.event_id}_{source.route.replace("-", "")}'
    case = pd.read_parquet(OUT/f'validation/case_records/{key}.parquet')
    rows = pd.read_csv(OUT/'validation/predictions/rows.csv', dtype={'event_id': str})
    ix = np.flatnonzero(rows.event_id.eq(source.event_id) & rows.route.eq(source.route) & rows.cutoff_index.eq(0))
    assert len(ix) == 1
    ix = int(ix[0])
    replay = []
    with np.load(OUT/f'validation/cache/{key}.npz') as z:
        for band in range(4):
            rc, rn = z['context'][0, 0, band], z['reference'][0, 0, band]
            d = np.maximum(np.diag(rc).real, 1e-15)
            diagonal = np.diag(d)
            kernels = {}
            for support in ['sparse', 'dense']:
                for seed in [19, 43, 71]:
                    name = f'{support}_recurrence_seed{seed}'
                    p = np.load(OUT/f'validation/predictions/{name}.npy', mmap_mode='r')[ix, 0, band]
                    kernels[name] = direct_matrix(p)
                lags = np.array([3, 5, 8, 13, 21]) if support == 'sparse' else np.arange(1, 32)
                name = f'{support}_full_ridge'
                values = np.load(OUT/f'validation/predictions/{name}.npy', mmap_mode='r')[ix, 0, band]
                p, _, _, _, fit = geometry_fit(values, lags)
                assert fit['fit_converged']
                kernels[name] = direct_matrix(p)
            corr = unloaded_correlation(rc)
            values = np.array([np.diag(corr, k).mean() for k in range(1, 32)])
            p, _, _, _, fit = geometry_fit(values, np.arange(1, 32))
            assert fit['fit_converged']
            kernels['local_persistence'] = direct_matrix(p)
            matrices = {name: .25*k*np.sqrt(d[:, None]*d[None, :])+.75*diagonal for name, k in kernels.items()}
            matrices.update(diagonal=diagonal, fixed_shrinkage=.25*rc+.75*diagonal)
            for name, matrix in matrices.items():
                w = direct_weight(matrix, rn)
                for endpoint, array in [('full', 'target'), ('half0', 'first'), ('half1', 'second')]:
                    target = z[array][0, 0, band]
                    ratio = float(10*np.log10((w.conj()@target@w).real/(w.conj()@rn@w).real))
                    actual = case[(case.cutoff == 20) & (case.block == 0) & (case.band == band) &
                                  case.method.eq(name) & case.endpoint.eq(endpoint)]
                    assert len(actual) == 1
                    expected = actual.iloc[0]
                    error = abs(ratio-expected.ratio_db)
                    assert error < 2e-5, (name, band, endpoint, error)
                    item = dict(event_id=source.event_id, route=source.route, band=band, method=name,
                                endpoint=endpoint, score_error_db=error)
                    if name in kernels:
                        observed = unloaded_correlation(target)
                        off = ~np.eye(32, dtype=bool)
                        cmse = np.mean(np.abs(kernels[name][off]-observed[off])**2)
                        lmse = np.mean([abs(np.diag(kernels[name]-observed, k).mean())**2 for k in range(1, 32)])
                        item.update(covariance_mse_error=abs(cmse-expected.covariance_entry_mse),
                                    lag_mse_error=abs(lmse-expected.complex_lag_mse))
                        assert item['covariance_mse_error'] < 1e-10
                        assert item['lag_mse_error'] < 1e-10
                    replay.append(item)
    replay = pd.DataFrame(replay)
    replay.to_csv(OUT/'validation/independent_score_replay.csv', index=False)

    # Reconstruct whole-earthquake means independently of the main groupby chain.
    saved = pd.read_csv(OUT/'validation/event_scores.csv', dtype={'event_id': str})
    errors = []
    for event_id in sorted(records.event_id.unique()):
        frames = [pd.read_parquet(OUT/f'validation/case_records/{event_id}_{route}.parquet') for route in ['KKFLS', 'TERRA']]
        data = pd.concat(frames, ignore_index=True)
        for model in saved.model.unique():
            methods = [f'{model}_seed{s}' for s in [19, 43, 71]] if model.endswith('recurrence') else [model]
            for endpoint in ['full', 'half0', 'half1']:
                chosen = data[data.method.isin(methods) & data.endpoint.eq(endpoint)]
                assert len(chosen) == 2*4*8*4*len(methods)
                value = chosen.ratio_db.mean()
                recorded = saved[saved.event_id.eq(event_id) & saved.model.eq(model) & saved.endpoint.eq(endpoint)].ratio_db
                assert len(recorded) == 1
                errors.append(abs(value-recorded.iloc[0]))
    assert max(errors) < 1e-10

    comparisons = pd.read_csv(OUT/'validation/paired_comparisons.csv')
    bootstrap_errors = []
    for endpoint in ['full', 'half0', 'half1']:
        table = saved[saved.endpoint.eq(endpoint)].pivot(index='event_id', columns='model', values='ratio_db')
        for row in comparisons[comparisons.endpoint.eq(endpoint) & comparisons.resampling.eq('earthquake')].itertuples():
            values = (table[row.model]-table[row.reference]).to_numpy()
            rng = np.random.default_rng(20260922)
            draws = np.array([values[rng.integers(len(values), size=len(values))].mean() for _ in range(5000)])
            low, high = np.quantile(draws, [.025, .975])
            bootstrap_errors.extend([abs(low-row.low), abs(high-row.high), abs(values.mean()-row.mean)])
    assert max(bootstrap_errors) < 1e-10
    receipt = dict(passed=True, events=int(records.event_id.nunique()), raw_files_verified=len(records),
                   minimum_distance_from_exposed_km=minimum_old_distance,
                   minimum_within_cohort_distance_km=float(within.min()),
                   direct_replay_cases=len(replay), maximum_score_error_db=float(replay.score_error_db.max()),
                   maximum_event_reduction_error=float(max(errors)),
                   maximum_bootstrap_replay_error=float(max(bootstrap_errors)),
                   independent_replay='Explicit complex Fourier outer products; Cholesky whitening plus ordinary eigensolve; direct event reduction',
                   caveat='Automated numerical replay, not independent human scientific review',
                   hashes={str(p.relative_to(OUT)): sha256(p) for p in [OUT/'pre_outcome_seal.json',
                       OUT/'validation/event_scores.csv', OUT/'validation/paired_comparisons.csv',
                       OUT/'validation/independent_score_replay.csv']},
                   verifier_sha256=sha256(__file__))
    (OUT/'validation/verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
