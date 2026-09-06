# CoherencyGraph-DAS

Research software for **retrospective sparse submarine DAS coherency prediction and covariance-identifiability analysis**. This is not a validated earthquake-prediction or operational early-warning system.

Authors: Gunjan Kumar Mishra and Badri Raj Lamichhane. Corresponding author: Badri Raj Lamichhane, d6622300231@g.siit.tu.ac.th.

## Public release scope

This repository publishes the reviewed analysis source, configurations, tests, documentation and generated scientific figures. Manuscript files and cover letters are intentionally not included in this initial code release. Large data/model assets are not stored in Git. A public asset-download location will be added after those assets are separately released; until then, only the data-free example and unit-test subset below can be reproduced from this repository alone.

The associated study remains subject to final author review. Code availability is not a claim of journal acceptance or submission readiness.

## Install

Use Python 3.12 in a virtual environment. Activate it before installing:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Dependencies are declared in `pyproject.toml`. `requirements-revision.lock.txt` records the analytical environment; its CUDA-specific PyTorch build can be replaced with a suitable CPU build. The offline example does not require a GPU or raw-waveform download.

## Reproducible offline example

```sh
python scripts/demo_revision.py --output demo_output
python -m pytest tests/test_critical_revision.py tests/test_spectral.py
```

The example constructs two strictly positive spectra that agree at supervised lags but yield opposite nearest-neighbour signs. It writes JSON and a figure and checks an abstention decision. These are synthetic calculations, not earthquake observations.

## Analysis and documentation

See `docs/TUTORIAL.md`, `docs/USER_GUIDE.md`, `docs/LICENSING_AND_PROVENANCE.md` and `CLAIMS_LEDGER.md`. Commands that use saved checkpoints, raw records or manuscript source require the separately documented assets. The complete local revision passed 34 tests and 19 CPU checkpoint/ensemble checks; the public data-free subset has 11 tests.

The workflow separates observed-lag prediction, positive-semidefinite admissibility, identification of unseen covariance entries and processing utility. Existing test analyses are retrospective. Source-component uncertainty weakens the apparent advantage over a strong blockwise ridge model; learned covariance does not establish array-processing gain.

## Licences

Original code: BSD-3-Clause (`LICENSE`). Scientific figures and Cook Inlet-derived numerical assets retain CC BY-NC-SA 4.0 restrictions; see the provenance documentation. Public arrival tables are separately attributed under CC BY 4.0. Raw HDF5 files are not redistributed. Do not apply the code licence to the entire mixed-content release.

AI assistance in code development, analysis and writing is disclosed in the accompanying study. The named authors retain responsibility for final review and interpretation.
