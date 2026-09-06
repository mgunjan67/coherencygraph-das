# Version 1.2.0 scientific result ledger

All claims refer to retrospective test earthquakes whose earlier outcomes had been inspected. Source tables, not this rounded summary, are authoritative.

| Question | Result | Source |
|---|---|---|
| Historical all-cell neural superiority | Unresolved versus blockwise ridge; the original -0.0152 difference and component interval [-0.0226, 0.0069] are preserved. | `reports/final_revision/baseline_reproduction.csv` |
| Earlier-waveform prediction | Ensemble NRMSE 0.3346; difference from blockwise ridge -0.0144 [-0.0320, 0.0499], unresolved. Supplied picks remain retrospective. | `reports/submission_revision/prediction_comparisons.csv` |
| Finite measurement selection | Development-only search of 45 designs selects lags 1,2,3,5,8,13,21,34, identical to the manual set. | `locked_design.json`, `development_design_candidates.csv` |
| Conditional unseen signs | Real signs: 2/384 historical versus 26/384 selected; imaginary signs: 0/384 for both. These are component-specific sensitivity ranges, not confidence intervals. | `conditional_ranges_summary.csv` |
| Fixed-aperture block-target supervision | Three-seed dense-minus-sparse processing difference +2.1653 dB [2.0168, 2.5573]. | `processing_comparisons.csv` |
| Exact-local-target supervision | Dense-minus-sparse difference +2.0943 dB [1.9157, 2.5438]. The corresponding ridge difference is +2.7030 dB [2.4980, 3.1918]. | same |
| Neural model versus classical shrinkage | Exact-local dense recurrence difference -0.0577 dB [-0.3737, 0.0391], unresolved; no universal superiority claim. | same |
| Numerical correctness | 320 complex known-truth component cases validated and contained truth; no incorrect signs. Initial fit failures and numerical polishing are preserved in the stress-test audit. | `complex_known_truth*.csv` |
| Parameter reproduction | 32 CPU checks reproduce neural/ridge output arrays from saved parameters. | `cpu_parameter_regeneration.csv` |

The geometry-support processing experiment intentionally retains historical context inputs. It is a retrospective support comparison, not a waveform-availability-safe operational processor. All processing values are target/reference-noise power differences, not detection SNR or earthquake prediction skill.

Optional new acquisitions, architectures or prospective warning tasks are separate studies. Public software availability does not imply journal acceptance.
