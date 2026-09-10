# Final scientific release: v2026.09.10

This release separates observed-lag prediction, covariance admissibility,
conditional identifiability and retrospective processing utility. It does not
establish neural superiority, unique covariance recovery, operational warning,
prospective validation or physical covariance calibration.

## Installation

Check out tag `v2026.09.10`. Use Python 3.12 and a virtual environment, then
`python -m pip install -e ".[dev]"`. The analytical versions are recorded in
`requirements-revision.lock.txt`; the release also includes actual execution
environment manifests. Cached execution is CPU based; the original recorded
PyTorch wheel includes CUDA support. No hosted AI service is required.

## Data-free API example

`python examples/closeout_datafree.py`

This uses the final numerical API on two specified spectra with the same
geometry. Directly measured lags are never counted as unseen identification.
Component extrema are marginal, not jointly attainable endpoint matrices.
Residual enlargement establishes feasibility, not physical-model validation.

## Canonical cached reproduction

`python scripts/reproduce_submission_release.py`

The command refuses mismatched source versions, archive hashes, missing
artifacts, failed numerical checks and mismatched reported numeric precision.
It creates a new `reproduced_submission` directory, verifies four archives in
historical order, regenerates saved-parameter predictions without training,
reconstructs diagnostic summaries and bootstrap contrasts, generates tables
and figures, checks 84 encoded numeric claims and runs regression tests.
It writes `FINAL_RELEASE_VERIFICATION.json` only after all commands pass.
An existing destination is never overwritten. For cached downloads, pass
`--asset-cache PATH` containing the four named ZIPs. Use `--work PATH` for a
fresh subsequent run.

## Assets and provenance

`configs/submission_release.json` provides URLs, SHA-256 hashes and internal
manifests, in this order:

1. `CoherencyGraph_DAS_assets_20260907.zip`: historical scientific base.
2. `CoherencyGraph_DAS_reporting_correction_20260907.zip`: reporting overlay.
3. `CoherencyGraph_DAS_audit_utility_20260907.zip`: candidate-design audit.
4. `CoherencyGraph_DAS_scientific_closeout_20260910.zip`: final diagnostics,
   corrected failed ridge projections, newly evaluated full-context processing,
   current empirical compatibility, source tables and evidence ledger.

Historical results remain unchanged in the September 7 release. The current
projection tables retain 417 initial historical failures and 107 new initial
failures; all 9,216 final projections must pass the original tolerances.
Manual and development-selected lag designs coincide. The two empirical
checking windows remain retrospective, dependent comparisons.

## Raw-data validation and licensing

Original Cook Inlet HDF5 optical-phase waveforms and supplied picks must be
obtained from their original archive for raw re-extraction. They are not
redistributed here. Cached spectral sufficient statistics and trained parameter
files support the documented reproduction, not an independent raw-data rerun.
See `docs/LICENSING_AND_PROVENANCE.md`, `LICENSE`, and the original archive's
terms; this release does not relicense third-party raw data.

## AI assistance and scientific boundaries

ChatGPT/Codex assisted literature organisation, code development, analysis
execution, plotting and editing. Hosted version identifiers were not retained
and are not invented. Executable code and results define reproducibility;
automated checks are not independent human replication. Authors remain
responsible for the scientific content. Manuscripts, cover letters and private
correspondence are excluded from this scientific release.
