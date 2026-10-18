CoherencyGraph DAS scientific reproducibility release v2026.10.18

Release scope
Code, configurations, author-created derived statistics, archived model assets
and numerical verification records. Private manuscript text, LaTeX/PDF sources,
cover letters, submission files and decoded upstream waveform excerpts are not
included in this public release slice. The complete scientific payload is:

CoherencyGraph_BSSA_Scientific_20261018.zip
Bytes: 1283661034
SHA256: 97fa609a70fe0c6691c0e74e0eb6c8d933cff16708efa93209eac05f1eacaa3c
ZIP members: 2460 (2459 individually hashed payload files plus the manifest).

Use the exact scientific ZIP for reproducibility. The repository/ directory is a
human-readable browse slice, not the complete model/result payload. Its explicit
copied-file inventory and hashes are in PUBLIC_SOURCE_ALLOWLIST.json. Binary
derived tables and checkpoints belong in the ZIP, not oversized Git commits.

Verification
All 53 unique tests passed. Seven production tests were a later subset rerun,
not seven additional unique tests. An initial Windows temporary-directory setup
failure is retained in dated records; the workspace-local rerun passed.
The separate direct-DFT implementation checked 384 fixed cases x 12 metrics =
4608 comparisons at unchanged tolerance 2e-8. Its maximum scaled discrepancy was
2.01394456667e-13. All 23040 recorded-component
cases also passed linearity. The initial verifier precision-order failure and
its correction remain documented; field outputs and case inventory did not change.

A freshly extracted scientific ZIP passed derived-result replay, four-date
frontier reconstruction, finite-extension scalar-table replay and a narrow
archived inference smoke. This used the retained analytical environment, not a
fresh dependency installation. It did not rerun full raw extraction/training,
new acquisition, or independent human replication. See the sanitized verification
summary and the full ZIP README for exact scope and acquisition instructions.

Reproduction entry point after downloading and extracting the complete ZIP:
python replay_all.py --package-root . --output-dir ../replay-output --with-inference-smoke
Use Python 3.12 and the archived dependency record. Choose a new output directory.
The complete ZIP includes dependency records, data/model inputs and rights records
that are deliberately omitted from this small browse slice.

Interpretation
These are bounded covariance/filtering evaluations, including unfavorable
comparisons. Added recorded components retain ambient content and are not clean
earthquake truth. Four development-exposed dates and two same-day sources do not
constitute untouched deployment validation. No early-warning, earthquake-
prediction, human-phase-label or continuous false-alarm claim is made.

Rights
Software code: BSD-3-Clause. Author-created derived assets: CC BY-NC-SA 4.0.
These are separate scopes, not a relicensing of upstream inputs. Source-provider
and third-party rights remain intact; raw/decoded upstream excerpts with
unverified redistribution permission are excluded. See LICENSE and
RIGHTS_AND_ACCESS.txt inside the scientific ZIP.

Archived receipts
Words such as local, unpublished or public=false in frozen execution receipts
describe the state when those checks ran before release. They are retained
unchanged for provenance; GitHub/Zenodo publication metadata gives the later
hosting state. Do not rewrite sealed receipts or scientific outcomes to remove
that chronology. This preparation script performs no publication or submission.
