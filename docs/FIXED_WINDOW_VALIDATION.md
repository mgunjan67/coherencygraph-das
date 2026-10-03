# Fixed-window validation and data-quality release (2026-10-03)

This release adds the separate, locked 30-earthquake validation to the historical
measurement-support study. It preserves the original analyses and their negative
benchmark findings. It is **not** an earthquake-prediction system or an operational
early-warning evaluation. Neither journal acceptance nor independent human review
is implied by automated numerical checks.

## Obtain the scientific assets

Download `CoherencyGraph_DAS_fixed_window_QC_20261003.zip` and its `.sha256` file
from the `v2026.10.03` GitHub release. Verify SHA-256 before extracting the ZIP at
the repository root. The ZIP contains development caches, trained model weights,
validation predictions, nested case scores, event results, frozen protocol and
seals, QC tables, environment lock, and scientific figure assets. It does not
contain manuscripts, cover letters, submission packages, credentials, or raw
waveform HDF5 files.

The included `RELEASE_MANIFEST.json` lists every scientific asset and checksum.
Historical absolute drive paths in provenance identify the original execution;
they are not portable download destinations. Do not reinterpret a historical seal
as a newly executed verification. The earlier scientific assets remain available
on the earlier releases for reproducing the historical S-window experiments.

## Install and verify

Use Python 3.12 and the package installation instructions in the root README.
`revisions/20260922_unseen_validation/environment_lock.txt` records the original
analytical environment. For the raw-data-free verification below, NumPy, pandas,
SciPy, PyArrow, h5py and pytest are sufficient; neural training and prediction also
require PyTorch. No hosted AI service is required.

```sh
python -m pytest tests/test_causal_validation.py tests/test_causal_replay.py
python scripts/verify_public_fixed_window.py
python scripts/verify_causal_group_inference.py
```

The first independent replay reconstructs all 30 event means from the nested
case scores and repeats the complete-earthquake percentile intervals with 5,000
draws and seed 20260922. The second independently reconstructs 50-km source
connectivity and source-component bootstrap intervals. These cached-data checks
do not themselves validate the original waveform extraction or model training.
For those steps, use `run_causal_validation.py` and the raw-data verifiers after
obtaining the referenced waveforms and remapping local raw paths. Do not rerun
cohort selection or rewrite a frozen seal to accommodate missing local paths.

## Findings and interpretation

There are 93 development and 30 previously unused validation earthquakes, with
both cable routes in the separate fixed-window task. Dense-minus-sparse recurrence
is 2.222 [2.068, 2.388] dB; dense-minus-dense-ridge is -0.071
[-0.119, -0.020] dB; dense-minus-fixed-shrinkage is -0.113 [-0.233, 0.028] dB.
Intervals are complete-earthquake 95% bootstrap intervals. Thus dense supervision
outperforms sparse supervision, but neural superiority over both non-neural
baselines was not established. The task uses known earthquake archive records;
it is not continuous-record detection, detector SNR, or pre-rupture prediction.

The added QC describes finite coverage, channel variability, route/block
differences, and high-variance flags. The unflagged-event sensitivity is post hoc
and does not replace the frozen primary result. See source tables and protocol
for definitions rather than interpreting flags as diagnosed sensor failures.

## Licensing and attribution

Original source code is BSD-3-Clause. Derived scientific assets and figure outputs
are CC BY-NC-SA 4.0, retaining UW Fiber Lab / Cook Inlet source attribution and
upstream restrictions. The archive does not relicense third-party source data.
See `docs/LICENSING_AND_PROVENANCE.md`. Manuscript files remain private.
