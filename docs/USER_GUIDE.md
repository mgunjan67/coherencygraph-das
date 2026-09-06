# Revision command reference

All commands run from the repository root. Paths inside portable caches are resolved beside the operator index before legacy absolute paths are considered. No private credentials are needed for the offline workflows.

| Stage | Inputs | Outputs and expected behaviour |
|---|---|---|
| `baseline` | historical predictions, operator caches, role tables | historical metric reconciliation, mask provenance; does not refit models |
| `raw` | original HDF5 and pick files, frozen roles | one independent audit cache per event/route, exact timing, sub-block and estimator diagnostics |
| `train` | original operator caches, Amendment 07 | matched ridge/MLP/spatial-recurrence weights and all-cell predictions; training-only tuning |
| `sensitivity` | independent audit caches | single-seed equal-duration, common-cutoff and power-pooling results |
| `internal` | 93 development events | component-fold and purged-temporal diagnostics; old test is not refitted as training |
| `projection` | direct predictions | simplex fits with duality gaps; stationary known-covariance synthetic outputs |
| `identifiability` | old targets, fixed lag/grid definitions | synthetic counterexample, real feasible ranges and synthetic abstention checks |
| `missingness` | model checkpoints, cached features | every unique 1-, 2-, 4-block mask; three endpoint definitions |
| `uncertainty` | historical per-event errors | corrected historical joint-route coverage; use `complete_revision_diagnostics.py` for revised model coverage/ranking |
| `downstream` | raw audit covariance caches, old local-state probabilities | corrected sign/frame full-window and two-way target-half comparisons |
| `summary` | machine-readable experiment products | paired intervals, figure source tables, main numerical macros |
| `figures` | summary source tables | vector PDF/SVG and raster PNG figures |
| `verify` | active source, weights, outputs | tests, numerical checks and production inventory; refuses READY when public repository is missing |
| `release` | verified local files | prepared directory and ZIP for distribution; does not publish them |

## Compute and reproducibility levels

The main analysis is based on 141 earthquakes and two routes; 30 older consistency events are retained only for provenance. CPU suffices for example, scoring, bounds and figures. Neural training used the available RTX 3070 and PyTorch CUDA runtime. The full local release is about one gigabyte before compression; raw-waveform regeneration requires additional storage. Retain at least 100 GB free when downloading raw archives.

Three levels are distinguished: (a) synthetic smoke test; (b) reconstruction of predictions and figures from checkpoints/derived inputs; (c) independent re-extraction of those inputs from original raw records. The recorded checks complete (a)--(c) locally. They are not independent human replication or proof of online causality. Full fresh neural refits may differ slightly across hardware; seeds, training receipts and scoring tolerances are supplied.

## Frozen versus retrospective

Do not run `hybrid-freeze` or alter source bins, event roles, historical mask or earlier result files to improve ranking. Amendment 07 explicitly treats prior test exposure as retrospective. Its all-cell result is a new diagnostic endpoint, not a relabelled blinded primary result.

`raw_input_checksums.csv` inventories the larger audited candidate cohort. Only the event IDs in `operator_index.parquet` and the role tables enter cached modelling. Extra candidate records are provenance, not additional test samples.
