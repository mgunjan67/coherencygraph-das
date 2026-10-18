# Independent scientific gate: pre-result review

Date: 18 October 2026. Scope: read-only review of the frozen protocol, numerical implementation, source metadata/extraction records, current manuscript copy and reproducibility records. This is a separate AI-assisted code/method review, not independent human replication or journal peer review.

## Initial decision

No fatal mathematical or eligibility error was identified in the inspected extension. Final readiness is not established by this preliminary review: exact outcome tables, source displays, revised prose, compilation and the final release manifest remain to be checked after integration.

The extension is a controlled, exploratory processing evaluation. It does not remove the restricted-data-size/editorial-significance risk or establish clean real-earthquake recovery, timing accuracy, detection performance, early warning, earthquake prediction or Q1 acceptance.

## Mathematical checks

1. `derivative_lcmv.py` implements the correct loaded LCMV solution for `C^H w = g`, with output `w^H x`. The Cholesky sequence calculates `R_loaded^-1 C`, then solves the positive-definite constraint Gram system. One refinement stays within that same solution span. `g=(1,0)` is real, so the output has unit nominal response and zero tangent response under either equivalent constraint convention.
2. For `C=[a,q*a]`, a signal response `a*(1+alpha*q)` is preserved exactly. Expanding `a*exp(-i*2*pi*f*epsilon*q)` shows a zero first derivative at epsilon zero and a second-order remainder. These statements apply only to that fixed affine family. They do not prove worst-case robustness, arbitrary wavefield preservation or a geometry-correct delay-uncertainty model.
3. In the recorded-component study `q=(j-15.5)/15.5` is a channel-index coordinate. The manuscript must keep that statement; it must not identify `q` with the measured ENU cable tangent. Nominal steering itself uses measured ENU and an assumed AK135 S/s branch, not fitted source phases.
4. `recorded_component_metrics.py` computes channel-mean input component/background powers, candidate output powers, the corresponding ratio gain in dB, and normalized squared difference from a supplied conventional output. The source scale cancels from the declared gain and normalized error when weights are frozen; a source-amplitude sweep would not add information about these metrics.
5. Conventional reference error is zero by definition. This cannot rank conventional filtering as the best physical earthquake recovery. Reference-to-input component power is necessary to expose possible reference cancellation and is retained for every source/aperture, without outcome exclusions.
6. The full-DFT verifier is implemented independently of the filter/scoring/LCMV helpers. Its negative-frequency conjugation and explicit inverse DFT are consistent with real-valued time signals. Its stated scope is a deterministic 384-case replay, not every scientific row or human replication.

## Eligibility, timing and dependence

- The fixed sources are the two source-classified T earthquake examples in the cached public release, kef0131 and kef0140. Both are on 2 August 2024 and do not overlap the eight background intervals. Selection uses catalog/header metadata before sample decoding, not processor outcomes.
- Crops are detection minus two seconds through plus five seconds, with first included native sample rounded upward on the 250-Hz clock. Right boundaries are exclusive. They are not S-pick-centered windows.
- Offset-safe resampling uses the crop-local grid; the source is digitally placed on a relative target grid. This is not a claim about source/target clock synchronization or original source-onset accuracy.
- Source recordings contain unseparated ambient content and acquisition response; development backgrounds can also contain earthquakes. Only the digitally added recording is exactly known. No clean desired ground-motion truth is available.
- The source catalog's T classification supports source identity, not a new manual label or proof that all low-frequency crop energy is seismic. A source waveform display and modeled-arrival positions should make the actual transplanted recording inspectable without requiring new human onset annotation.
- Two same-day sources crossed with four recording-label background dates are not eight independent earthquake cases. No population confidence interval or pseudo-replicated test is justified. Existing Cook bootstrap intervals retain their own frozen interpretation.
- The 96-hour assessment remains disabled. The original 48-hour protocol, models, primary cohort and historic numerical outcomes are not replaced.

## Interpretation conditions for final manuscript

| Retained claim | Potential problem | Objective closure condition |
|---|---|---|
| Neural frontier advantages depend on the comparator set | Adding LCMV does not causally explain neural architecture behavior | Report both historical nine-method and extended ten-method memberships; avoid architecture-causal wording |
| A recorded-component test supplements wavelet controls | Source crops are not clean seismic truth and the reference is conventional | Use recorded-component/reference language consistently in abstract, figures and conclusions; show source records and cancellation diagnostics |
| No source-output tuning or outcome exclusion | Strong results could tempt narrower conditions or favorable case selection | Retain all planned rows, every source/date/condition and failures; check counts against the frozen protocol |
| Reproducible research | Current public archive is historical; clean-directory replay is not fresh installation or full retraining | State the replay scope precisely, map every new table/figure to retained files, verify final local manifest and separately obtain approval for matching public release |

## Reproducibility audit scope

The inspected reproducibility report inventories 32 archived checkpoints with matching training/selection JSON metadata. A clean-directory saved-result replay and one fixed statistical-feature inference smoke were executed in the existing analysis environment with user-site/PYTHONPATH disabled. This is meaningful mechanical evidence, not a fresh dependency installation, full training replay, full waveform acquisition replay or independent human reproduction. Those limits are explicitly recorded and should remain in the manuscript and final report.

The historical GitHub/Zenodo release has verified anonymous metadata/checksum availability but does not contain the new revision. The new version-matched local bundle must not be described as public until external publication is explicitly approved and retrieval/checksum equality is verified.

## Files inspected

- `protocol.json`, `scope_seal.json`
- `implementation/derivative_lcmv.py`
- `implementation/run_lcmv_ricker_extension.py`
- `implementation/run_recorded_extension.py`
- `implementation/recorded_component_metrics.py`
- `implementation/verify_recorded_direct_dft.py`
- `implementation/summarize_finite_panels.py`
- `evidence_audit/prepare_source_components.py`, `source_spec.json`, `source_eligibility.json`
- Historical `20261010_forecast_processing/frequency_filter.py` and `forecast_adapter.py`
- `reproducibility/REPRODUCIBILITY_CLOSEOUT_REPORT.txt`, `smoke_archived_inference.py`
- Current `manuscript/submission_body.tex` and `generated_extension_methods.tex`

Final numerical/claim review is pending completed results and manuscript integration, not permission to initiate a new research cycle.
