# Measurement-support audit, version 1.2.0

This is a retrospective computational study, not an operational earthquake warning system. Earlier test outcomes were inspected; no new external confirmation is claimed. Prediction, numerical admissibility, conditional identification and processing utility are separate outputs.

## Inputs, outputs and reproduction

Use Python 3.12 and the scientific dependencies in `requirements-revision.lock.txt`. Install with `python -m pip install -e ".[dev]"`. The exact recorded PyTorch wheel is CUDA-specific; use its CPU counterpart on CPU-only hosts. No ChatGPT or Codex API is needed.

Download the versioned scientific assets from release `v2026.09.07` (base commit `1cd1548fef28d0dece6182bb02bc3ff59505a882`), verify SHA-256, and extract at the repository root. Overlay the same release's `CoherencyGraph_DAS_reporting_correction_20260907.zip` after verifying its companion checksum. Its manifest pins the reporting-source commit. The base science and original tag are unchanged. Source code remains browsable in Git; the asset ZIP is not a substitute for source distribution. Raw HDF5 waveforms are not redistributed. Caches and original pick/role metadata support offline numerical reproduction.

Commands below use `python scripts/run_submission_revision.py STAGE`:

| Stage | Required inputs | Output and interpretation |
|---|---|---|
| `audit` | retained roles and cached timing | immutable protocol hash and every waveform input/target lead |
| `train` | original features and local covariance caches | four endpoints, two ridge models and three recurrence seeds each; development-only selection |
| `select_design` | development full/half lag caches | component allowances and locked finite-search design; no test outcomes |
| `evaluate_design` | locked design, allowances and test target caches | real/imaginary conditional ranges at unseen lags 4 and 10 |
| `processing` | checkpoints/predictions, local covariance caches | fixed-channel power diagnostics, classical controls and seed results |
| `inference` | saved complete-event results | component and event intervals, deletion sensitivity and seed variability |
| `synthetic` | declared configuration only | 320 complex-component known-truth checks; no empirical coverage claim |
| `validate_raw_local` | original HDF5 paths and public picks | independent local-target extraction; optional when raw files are unavailable |
| `regenerate` | saved neural/ridge parameters and features | 32 CPU parameter-to-output checks |
| `verify` | release assets and new source | frozen roles, selection, ranges, processing and unit-test checks |

Run stages in the table's dependency order. Training and candidate selection resume from completed outputs. Processing regenerates the full table. The frozen protocol is `configs/protocol_amendment_09_submission.yaml`; do not change it silently in an existing run. New experiments belong in separately named configurations/output directories.

Run `python scripts/build_submission_revision_outputs.py` for six scientific figures in PDF/SVG/PNG and numerical tables. When private manuscript sources are absent, generated tables go to `reports/submission_revision/generated_tables`. No manuscript text is required for public figure regeneration.

Run `python scripts/verify_reporting_correction.py` to reconstruct both named ridge contrasts from complete event/route scores and independently re-express the unchanged 5,000-draw component bootstrap. It checks Table 2/Figure 3 values and the architecture dataflow. The strongest observed ridge is a descriptive reporting choice, not development selection. Historical seed-19 Amendment-08 results are precursor provenance; final Amendment-09 experiments use all seeds 19, 43 and 71.

## Data conventions

### Frozen all-candidate utility addendum (7 September 2026)

This addendum evaluates the existing 45 designs, eight hash-selected development earthquakes and 24 retrospective earthquakes without retraining or changing the objective. After extracting the base, reporting-correction and audit-utility assets in that order:

```sh
python scripts/evaluate_audit_design_utility.py evaluate --workers 8
python scripts/evaluate_audit_design_utility.py summarise
python scripts/build_audit_design_utility.py
python -m pytest tests/test_audit_design_utility.py -q
```

Evaluation is resumable: supplied lower-level range files are cached. To recompute optimisations, copy the repository/assets to a new working directory and move its `reports/audit_design_utility/ranges` directory aside, then run the same commands. Do not overwrite the original frozen receipt. The `freeze` stage was run once before the all-design comparison; it deliberately refuses to replace that receipt. The spec and source/input SHA-256 hashes allow the computation to be audited without a hosted AI service.

Outputs in `reports/audit_design_utility`: all45_designs.csv, selection_summary.csv, all_design_ranges.parquet, individual range files, ranking_receipt.json, validation.json, frozen_receipt.json and evaluation_started.json; the generated table/macros and one PDF/SVG/PNG figure accompany them. Primary J is mean real compatible-range width, with original equal-route/equal-event aggregation and invalid-width fallback 2. All 46,080 numerical cases are valid. H0 is constant zero; H1/H2 minimise mean/worst nearest-target distance with numeric-lexicographic ties; H3 is descriptive only. H4/H5 are omitted, not tuned. All correlations are descriptive, without p-values.

Outcome B: the audit design is retrospective rank 1/45, rho=0.997233, with complete top-5/top-10 overlap, but H1 and H2 select exactly the same design. The result quantifies a geometry preference without demonstrating superior selection. The existing processing experiments do not supply matched outputs for all 45 candidates, so no forced processing correlation is reported.

- Complex arrays store real/imaginary components on the final axis where indicated. Targets have route-record, eight-block, four-band, lag, component dimensions.
- Lag reversal conjugates the moment. Kernel assembly uses `C[i,j] = gamma(x[j]-x[i])`.
- Local targets remove the recorded trace-diagonal loading before pair normalisation. They use precisely channels 484–515 inside each block. The direct raw and synthetic tests verify this operation.
- Earlier-waveform context is shared across the route and ends before all targets. Supplied picks remain retrospective, so operational causality is not established.
- Conditional ranges consume later measured moments, not predicted moments. Component allowances are development half-window sensitivities, not confidence levels.
- Reported processing is target/reference-noise power relative to a diagonal method, not detector SNR or detection probability.
- Earthquakes, retaining both routes and nested observations, are the inferential unit. Component bootstrap preserves component sizes; seeds are averaged within earthquakes.

## Numerical behaviour

The projected-gradient simplex fit requires a first-order gap below 1e-8. Difficult narrow spectra trigger deterministic constrained active-set and feasible line-search polishing. Failure is reported as abstention. Linear-programme outer bounds are checked using dual feasibility and a rounding allowance; a solver success flag alone is insufficient.

The four design choices are historical eight lags, manual eight lags, development-selected eight lags, and their ten-lag augmented pool. The selected design is optimal only among 45 declared candidates on the eight hash-selected development events; it is not a globally optimal physical array.

## Licensing and provenance

Software: BSD-3-Clause. Cook Inlet-derived measurements and model assets: retain CC BY-NC-SA 4.0 upstream restrictions. Public arrival tables: separately attributed CC BY 4.0. The Natural Earth coastline is an unmodified public-domain geographic backdrop; its ZIP is preserved in `data/provenance`. No private manuscript, cover letter, credentials, raw HDF5 or work-account material is included in the public release.
