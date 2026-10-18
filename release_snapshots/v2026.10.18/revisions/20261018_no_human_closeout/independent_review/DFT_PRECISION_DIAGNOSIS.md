# Independent DFT replay: precision-semantic diagnosis

The first independent full-DFT verifier stopped on the first `context_0.75` background-output power because its scaled discrepancy, `4.893634992038551e-8`, exceeded the unchanged declared `2e-8` tolerance. This failure must be preserved. It is not evidence that the saved scientific outcome needs modification.

## Exact cause

The archived context has dtype `complex64`. The scientific implementation's `forecast_adapter.context_matrix` converts it to `complex128` before multiplication. The independent verifier's original `construct_matrix` multiplied `.75 * context` in `complex64`, then upcast the already-rounded result through the double-precision identity term. The maximum scaled covariance-entry discrepancy was `5.9266959233558126e-8` for the failing context.

The same issue can affect the neural/ridge constructor through float32 diagonal powers and their products/square root; the archived neural kernel itself is complex128. Correct precision semantics therefore require converting context to complex128 at the beginning of the independent constructor, and converting a supplied kernel similarly before any arithmetic.

## Bounded exact-case diagnostic

Read-only diagnostic: source `kef0131`, background `2024-08-04_00`, cutoff 120, aperture 0, method `context_0.75`. No outcome, scientific helper, protocol, threshold or saved array was edited.

| Calculation | Background output power | Scaled difference from saved value |
|---|---:|---:|
| Saved scientific result | 1647.0539995784648 | 0 |
| Original independent DFT with float32 intermediate | 1647.053918977654 | 4.893634992038551e-8 |
| Same independent DFT/direct solve, context cast before arithmetic | 1647.0539995784588 | 3.589266388981196e-15 |

Only the dtype input to the independent constructor changed in the last diagnostic. The DFT matrix, direct solver, steering, recorded components, background, reference and saved result were identical.

## Objective closure

1. Preserve the frozen failed verifier and its failure log/receipt.
2. Add a separately named v2 independent verifier with a documented precision-semantic change only: complex128 conversion before constructor arithmetic. Do not change the `2e-8` tolerance or scientific outputs.
3. Test the independently constructed matrices against the declared double-precision formulas for complex64 inputs at both context fractions and neural/ridge scaling.
4. Re-run the original deterministic 384-case/12-metric replay once with v2 and retain both failure and corrected verification provenance.
5. If any discrepancy remains, diagnose it rather than relaxing the threshold or deleting cases.
