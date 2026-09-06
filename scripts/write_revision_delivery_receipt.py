"""Seal an honest delivery report from the completed local verification receipts."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from coherencygraph_das.config import sha256
OUT=ROOT/'reports/critical_review'
verification=json.loads((OUT/'final_verification.json').read_text())
release=json.loads((OUT/'release_package_manifest.json').read_text())
checkpoints=json.loads((OUT/'checkpoint_regeneration.json').read_text())
for row in release['archives']:
    assert sha256(ROOT/row['path'])==row['sha256']
assert release['isolated_full_test_suite_passed'] and all(r['passed'] for r in checkpoints)
receipt=dict(decision='NOT READY',publicly_published=False,local_verification=verification,release=release,
    isolated_tests=(OUT/'isolated_full_test_log.txt').read_text().strip(),
    isolated_checkpoints=(OUT/'isolated_checkpoint_log.txt').read_text().strip(),
    code_only_tests=(OUT/'code_only_unit_log.txt').read_text().strip(),
    note='This receipt is outside the archives it hashes. Historical failed checks are retained as diagnostics; the isolated final run passed.')
destination=ROOT/'releases/20260906_review_revision/VERIFICATION_RECEIPTS.json'
destination.write_text(json.dumps(receipt,indent=2),encoding='utf8')
lines=['# Final critical-review revision gate', '', '**Decision: NOT READY for submission.** Local analysis, production and isolated-package tests pass. Public browsable source and final author approval remain unverified.', '',
       '## Deliverables', '', '- Clean manuscript: `manuscript/cageo_submission/output/main.pdf` (15 pages).',
       '- Line-numbered manuscript: `manuscript/cageo_submission/output/review.pdf` (15 pages).',
       '- Supplement: `manuscript/cageo_submission/output/supplement.pdf` (10 pages; 8 Text sections, 13 Tables, 2 Figures).',
       '- Cover letter for author review: `manuscript/cageo_submission/output/cover_letter.pdf` (1 page).',
       '- Point-by-point review response: `CRITICAL_REVIEW_RESPONSE.md`; claim decisions: `CLAIMS_LEDGER.md`.', '',
       '## Verified local gates', '',
       '- Re-extracted 282 raw event-route records; cached targets reproduced exactly.',
       '- '+verification['unit_tests']+'.',
       '- Seven revised model scores independently recomputed from prediction arrays.',
       '- Nineteen CPU checkpoint/ensemble predictions regenerated; maximum difference '+f"{max(r['max_prediction_difference'] for r in checkpoints):.8g}"+'.',
       '- All 41 current PDF pages rendered and visually inspected; no unresolved-reference or overflow warnings.',
       '- Source-only offline example passed; data-free test suite: '+receipt['code_only_tests'].splitlines()[-1]+'.',
       '- Extracted full release test suite: '+receipt['isolated_tests'].splitlines()[-1]+'; checkpoint regeneration passed; '+str(release['extracted_checksums_verified'])+' file checksums verified.', '',
       '## Prepared archives (not public releases)', '', '| Archive | Bytes | SHA-256 |', '|---|---:|---|']
for row in release['archives']:lines.append('| `'+row['path'].replace('\\','/')+'` | '+str(row['bytes'])+' | `'+row['sha256']+'` |')
lines.extend(['', 'Final isolated-run receipts are stored alongside the ZIP files as `VERIFICATION_RECEIPTS.json`, avoiding a circular archive checksum. Earlier failures are retained and identified; they are not the final verification outcome.', '',
    '## Scientific boundaries and author actions', '',
    'The revised paper supports retrospective sparse-coherency prediction and an identifiability/abstention analysis. It does not establish online early warning, pre-rupture earthquake prediction, full covariance reconstruction, or a learned array-processing gain. The recurrence advantage over the stronger blockwise ridge is uncertain under source-component resampling.', '',
    'Provide the intended GitHub repository, authorise the public release with its stated licences, and obtain both authors\' approval of the narrowed claims. Insert the verified public links and complete the live submission declarations before changing the decision. See `AUTHOR_ACTIONS.md`.', ''])
(ROOT/'FINAL_REVIEW_GATE_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print('Final gate and delivery receipts written; NOT READY until public/author gates are verified.')
