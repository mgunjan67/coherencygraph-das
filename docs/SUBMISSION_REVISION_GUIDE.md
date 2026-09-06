# Measurement-support audit, version 1.2.0

This is a retrospective computational study, not an operational earthquake warning system. Earlier test outcomes were inspected; no new external confirmation is claimed. Prediction, numerical admissibility, conditional identification and processing utility are separate outputs.

## Inputs, outputs and reproduction

Use Python 3.12 and the scientific dependencies in `requirements-revision.lock.txt`. Install with `python -m pip install -e ".[dev]"`. The exact recorded PyTorch wheel is CUDA-specific; use its CPU counterpart on CPU-only hosts. No ChatGPT or Codex API is needed.

Download the versioned scientific assets, verify SHA-256, and extract at the repository root. Source code remains browsable in Git; the asset ZIP is not a substitute for source distribution. Raw HDF5 waveforms are not redistributed. Caches and original pick/role metadata support offline numerical reproduction.

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

## Data conventions

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
