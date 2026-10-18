# Independent numerical review of finite panels

Read-only recomputation from completed saved Parquet/CSV outputs, 18 October 2026. No scientific scoring run, new experiment, outcome edit, threshold relaxation or model training was performed by this review.

## Completeness and invariants

- New Ricker comparator: 86,400 unique rows; exactly 17,280 rows in each of matched, both delay signs and both amplitude-gradient signs.
- Recorded-component panel: 23,040 unique finite rows; exactly 3,840 rows per method.
- All-case recorded linearity error maximum: `1.6423338989019761e-15`.
- Conventional normalized reference error is exactly zero for every recorded case, as required by its definition.
- All 32 checkpoints and all 32 metadata JSONs are present in the local scientific package. Recomputed checkpoint SHA256 hashes match the provenance inventory, with no mismatch in recorded checkpoint hashes where present. This check does not claim inference of all checkpoints or retraining.

## Extended Ricker frontier

Recomputed nondominance directly from `expanded_date_condition_table.csv`, without importing `summarize_finite_panels.py` or its `nondominated` function. Gain is maximized, distortion minimized, with the same declared `1e-12` numerical equality tolerance.

| Mismatch | Neural mean nondominated dates | Neural marginal-tail nondominated dates |
|---|---:|---:|
| Amplitude minus | 2/4 | 0/4 |
| Amplitude plus | 1/4 | 0/4 |
| Delay minus | 2/4 | 0/4 |
| Delay plus | 2/4 | 0/4 |

The new LCMV comparator has higher fifth-percentile gain and lower 95th-percentile error than neural filtering on every date for every mismatch condition. Mean domination is not universal. Marginal-tail coordinates are different quantiles and do not imply joint worst-case protection.

LCMV relative MSE in affine-amplitude cases is numerical zero (approximately `1e-31`), consistent with its analytic constraint. Its average delay-case relative MSE is approximately `0.00060375`. The affine amplitude result is imposed preservation, not a learned or novel capability.

## Recorded-component all-source summary

Equal nested means: observations within source/hour/method, hours within background date, then four dates within source. Designs are balanced, so the result also equals the corresponding flat mean. Gains are dB component/background ratio gains, not clean real-earthquake SNR.

| Source | Method | Ratio gain (dB) | Relative conventional-reference MSE |
|---|---|---:|---:|
| kef0131 | Neural | -0.994498 | 0.235135 |
| kef0131 | Ridge | -1.905685 | 0.220083 |
| kef0131 | Conventional | -6.131666 | 0 |
| kef0131 | Context 0.25 | 0.536003 | 0.388189 |
| kef0131 | Context 0.75 | 3.836639 | 1.009344 |
| kef0131 | LCMV context 0.25 | -0.138184 | 0.322402 |
| kef0140 | Neural | -1.869520 | 0.315185 |
| kef0140 | Ridge | -2.809921 | 0.274498 |
| kef0140 | Conventional | -7.550838 | 0 |
| kef0140 | Context 0.25 | -0.085465 | 0.515324 |
| kef0140 | Context 0.75 | 3.529833 | 1.331358 |
| kef0140 | LCMV context 0.25 | -0.839052 | 0.499098 |

Reference/input component power ratio ranges over all apertures:

- kef0131: `0.023608397784900866` to `0.3288360050352567`.
- kef0140: `0.015817509708462588` to `0.2821152120676415`.

The reference can strongly cancel the recorded component. Thus normalized reference error is comparison to a declared processing output, not an intrinsic physical fidelity ranking. Source examples contain their original ambient content. The negative neural gains must not be hidden or rewritten as a general statement that neural denoising fails.

## Independent replay closure

The original independent DFT replay stopped on a precision-semantic mismatch. `DFT_PRECISION_DIAGNOSIS.md` records the exact cause and bounded proof. The separately named v2 wrapper replaces only the matrix constructor's pre-arithmetic complex128 conversion. Frozen DFT/direct-solve code, all 384 cases, 12 metrics and the `2e-8` tolerance are unchanged.

The corrected replay passed 4,608 comparisons with maximum scaled discrepancy `2.013944566670034e-13`. The actual saved `all_cases.parquet` SHA256 is `362f161da2205ce050b520d3a6d4f201cf904329c495ab30763d3409c3ef4782`, equal to both its scientific execution receipt and v2 verification binding. Actual v1 and v2 verifier SHA256 values match the corrected receipt. The frozen v1 and `DIRECT_DFT_V1_FAILURE.json` remain retained. No scientific outcome was mutated and no tolerance was relaxed.

This closes the recorded numerical replay issue. Final manuscript-number, claim and production review remain separate integration tasks.
