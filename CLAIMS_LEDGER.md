# Historical claims ledger — Amendment 07

Historical record, not the current release status. Version 1.2.0 adds the explicitly separated results in `docs/CURRENT_RESULTS.md` and `reports/submission_revision`. Original numerical claims and historical blocks below are retained for provenance.

This ledger distinguishes a verified calculation from a general scientific claim. All existing test analyses are retrospective. Artifact paths are relative to this repository; R = `reports/critical_review/`.

| Claim / manuscript location | Decision | Evidence and boundary |
|---|---|---|
| Abstract, Methods: online/causal early warning | **Removed; unsupported after timing audit** | R/raw_timing.parquet, timing_summary.json: 1,807/2,256 block endpoints have no positive full-route waveform lead. Picker input support is unknown. Conditional future replacement is not end-to-end causality. |
| Results 4.1: revised spectral recurrence beats full-context ridge | **Supported within retrospective task** | R/matched_summary.csv and matched_paired_intervals.csv: NRMSE 0.308220 vs 0.355212; difference -0.046992, component CI [-0.055674,-0.030144]. |
| Revised model beats strong blockwise ridge | **Weakened** | NRMSE 0.323425 for blockwise ridge; difference -0.015205, component CI [-0.022609,0.006909]. Ordinary earthquake CI excludes zero, but source-component CI does not. |
| Unique graph/hybrid architectural superiority | **Removed** | Matched direct/PSD MLP and recurrence controls, parameter counts, seed outputs. Different capacities prevent causal attribution to one architectural element. |
| Source groups are independent connected clusters | **Corrected** | Historical IDs are 0.2-degree by 30-km hard bins. R/cross_role_distances.csv shows nearest development/test pair 6.733 km. Test has 11 sensitivity components. |
| Reliability mask was selected without test information | **Corrected** | R/reliability_provenance.json: all 24 architecture-test events were among 117 mask-construction events. Development-only mask remains 28 cells; revised primary reports all 32. |
| Exact primary numerical reproduction | **Supported** | R/historical_number_check.csv: 22 checks, max NRMSE difference 8.57e-9. R/checkpoint_regeneration.json: 19 CPU model/ensemble reconstructions, max prediction difference 1.02e-6. |
| Positive spectral output is Hermitian PSD and unit diagonal | **Supported algebraically, tested numerically** | Eq. 2; tests/test_critical_revision.py. Correct upper-diagonal convention C_ij=gamma(r_j-r_i). No arbitrary local-kernel stitching. PSD alone does not imply invertibility. |
| Sparse lag prediction identifies full covariance / a physical wavenumber spectrum | **Removed** | R/spectral_nonuniqueness.json: constraint rank 17, nullity 240; strictly positive spectra agree at supervised lags but have opposite lag-one signs. Actual 32-channel geometry has zero directly supervised off-diagonals. |
| Feasible ranges support a conservative unseen-sign decision | **Supported under declared tolerance** | R/real_feasible_ranges.csv: 8/768 certified signs. R/synthetic_abstention_validation.csv: 320/320 truth values inside ranges, 29 certificates, zero wrong certificates under bounded synthetic errors. Not a calibrated earthquake confidence region. |
| Equal-duration / strict waveform cutoff preserve a gain | **Unsupported at component-interval level** | R/estimator_sensitivity_summary.csv. Common-cutoff difference -0.005265, CI [-0.022462,0.052217]. Picks remain retrospective. |
| Repeat disagreement is a noise floor / learning ceiling | **Removed** | R/repeat_ratio_reconciliation.csv separates ratio-of-means from mean-of-ratios and latent error from repeated-estimate disagreement. |
| Revised joint uncertainty is uniformly calibrated | **Removed** | R/revised_joint_coverage.csv: 12/24,21/24,23/24 at 80/90/95%; component-level 90/95% bounds are infinite with six calibration components. |
| Ensemble spread is a reliable confidence ranking | **Unsupported** | R/uncertainty_ranking.csv and uncertainty_risk_coverage.csv. Component-bootstrap intervals and simple input-quality reference are reported. |
| Block-dropout solves missing-channel reconstruction | **Narrowed** | R/exhaustive_missingness.parquet covers 8/28/70 unique masks, separately scoring missing, retained and all blocks. Historical training masks differ from revised controls; no unique graph advantage. |
| Learned covariance improves array processing | **Unsupported** | R/downstream_summary.csv: revised recurrence about +0.0063 dB with CI spanning zero; classical shrinkage about +0.1699 dB with positive interval. Historical corrected local-state result is separately retained. |
| Same-target oracle measures recoverable performance headroom | **Removed** | R/downstream_revision.parquet, corrected-frame full/half tests. Oracle fit and evaluation on the same covariance are in-sample; opposite-half evaluation is much smaller. |
| External temporal / independent-instrument transfer | **Not established** | R/public_access_audit.json: inspected listed 2024 HDF5 objects 403. Internal component and 12-train/19-test purged-date stress are explicitly retrospective. |
| Publicly reproducible / submission ready | **BLOCKED** | Public browsable source and final author approvals are unverified. Prepared ZIP files are not a public repository. |

Any manuscript change that enlarges these claims requires new evidence and a dated protocol, not revised wording alone.
