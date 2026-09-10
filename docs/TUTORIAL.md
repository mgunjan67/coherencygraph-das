# A reproducible predict--bound--abstain example

1. Install the package in a new environment using the README.
2. Run `python scripts/demo_revision.py --output demo_output`.
3. Inspect `demo_output/result.json`. The two synthetic spectra agree at the eight supervised lags to floating-point precision, but their lag-one values are +0.45 and -0.45.
4. Inspect `demo_output/feasible_ranges.png`. A range crossing zero leads to abstention, not an inferred sign from whichever spectrum a model happened to fit.
5. Run `python scripts/verify_revision_checkpoints.py` with the full data/weights release. This reconstructs predictions from neural and ridge parameters, compares them with archived predictions and reports CPU/GPU numerical tolerance explicitly.
6. Run `summary`, the two numerical-table scripts and `figures` as in the README. Inspect `reports/critical_review/matched_paired_intervals.csv`: the source-component interval differs from ordinary earthquake bootstrap uncertainty.

The synthetic feasibility tolerance is not a probabilistic confidence interval. Real-target bounds do not establish full-matrix identification or earthquake-warning skill. Model outputs can be admissible yet insufficient for processing.
