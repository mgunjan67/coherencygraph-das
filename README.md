# CoherencyGraph-DAS

Geometry-explicit measurement-support audits for submarine DAS covariance learning. Version 1.2.0 distinguishes observed-lag prediction, covariance admissibility, conditional identification and processing utility. It is not an earthquake-forecasting or operational early-warning system.

Authors: Gunjan Kumar Mishra and Badri Raj Lamichhane. Corresponding author: Badri Raj Lamichhane, d6622300231@g.siit.tu.ac.th.

## Public source and scientific assets

Source is browsable at https://github.com/mgunjan67/coherencygraph-das. Download the scientific assets and SHA-256 file from release **v2026.09.07**, verify the checksum, and extract the ZIP into the repository root. The ZIP supplies data/model assets, not a substitute for the source repository. Manuscripts, correspondence, raw HDF5 and credentials are excluded.

Earlier configurations, results and test exposure are retained. Amendment 09 adds an earlier-waveform benchmark, three-seed dense supervision, exact-local 32-channel targets, development-only allowances and finite lag selection, complex conditional ranges, and complete-event processing comparisons. No fresh external confirmation is claimed.

## Install

Use Python 3.12 in a virtual environment:

```sh
python -m venv .venv
# Activate .venv, then:
python -m pip install -e ".[dev]"
```

The exact analytical environment is recorded in requirements-revision.lock.txt. Its CUDA-specific PyTorch build can be replaced by the matching CPU build. No OpenAI API or hosted AI service is needed for reproduction.

## Data-free example and tests

```sh
python scripts/demo_revision.py --output demo_output
python -m pytest tests/test_critical_revision.py tests/test_spectral.py tests/test_final_revision.py tests/test_submission_revision.py
```

The example constructs nonnegative spectra that agree at measured lags but disagree at an unseen lag. Synthetic fixtures are not earthquake observations.

## Cached-data verification

After extracting the scientific assets:

```sh
python scripts/run_submission_revision.py audit
python scripts/run_submission_revision.py regenerate
python scripts/run_submission_revision.py verify
python scripts/build_submission_revision_outputs.py
```

See [the complete revision guide](docs/SUBMISSION_REVISION_GUIDE.md) for every stage, input/output convention and expected behaviour. [The earlier command reference](docs/USER_GUIDE.md) documents historical experiments. Raw-extraction commands require the original publicly referenced waveforms; cached-data stages do not.

## Scientific boundaries

- Whole earthquakes retain their routes, blocks and windows across splits and inference.
- Test outcomes were inspected historically; retrospective comparisons are not pristine external validation.
- The earlier-waveform benchmark precedes targets but still uses retrospectively supplied picks.
- Conditional bounds consume measured later-window moments, not early predictions. They are not population confidence intervals.
- Most unseen signs remain unresolved. Positive semidefiniteness does not identify physical covariance.
- Dense supervision improves the fixed-aperture diagnostic; classical shrinkage remains competitive. Neural superiority is not universal.
- Power diagnostics are not detection probabilities or operational detector SNR.

## Licences

Code is BSD-3-Clause. Cook Inlet-derived data and models retain CC BY-NC-SA 4.0 restrictions; the separately attributed public pick release is CC BY 4.0. Natural Earth coastline data are public domain. See [licensing and provenance](docs/LICENSING_AND_PROVENANCE.md).

The associated manuscript remains unpublished. Software release is not a claim of journal acceptance. No collaborator bot or work-account repository is used.
