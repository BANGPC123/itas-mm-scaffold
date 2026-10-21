# ITAS-MM: rebuilding after loss of local-only artifacts

Status: **PARTIAL RECOVERY / NOT READY TO MERGE `main`**. This document is
for the research prototype, not a field-certified driving assistant.

## What was recovered on 2026-10-22

- All 17 tracked Git LFS model/checkpoint objects, with SHA256 and byte size
  matching `docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json`.
- Original upstream UFLD-v2 source from
  `https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2`, exact documented
  commit `c903880678454dfd9b55a63022368db05c00bc6d` (MIT license).
- Four downloadable government-hosted legal PDFs, each with SHA256 recorded in
  `data/regulations/official_20261011/sources.json`. Offline index contains
  **703 real PDF-page chunks**. Decree 168 is a scanned PDF and yielded zero
  text pages; no OCR text or current-legal-validity claim was fabricated.
- Offline legal source tests: **7 passed**.
- Full Python suite: **444 passed / 41 failed** (includes two new rebuild-audit tests) after restoring public assets
  (previously 438 / 45 on the clean clone). Failures remain unresolved.
- Prior core CI: 135 passed; frontend JS: 7 passed and production build passed.
- `scripts/check_rebuild_readiness.py --strict-core` returns PASS locally for
  asset integrity, but **historical full acceptance remains blocked**.

These numbers characterize the rebuilt environment, **not** performance
on actual Vietnamese roads. Prior human-reviewed assets and footage were not recovered.

## Rebuild on a clean Windows machine

Use a dedicated checkout/branch. Do not overwrite `main`. Install Git LFS,
Python 3.12+, `uv`, and Node. In PowerShell from the repo root:

```powershell
git lfs install
git lfs pull
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build

git clone --depth 1 https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git third_party/Ultra-Fast-Lane-Detection-v2
git -C third_party/Ultra-Fast-Lane-Detection-v2 rev-parse HEAD
# Must equal: c903880678454dfd9b55a63022368db05c00bc6d

$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe scripts/fetch_official_law_corpus_20261011.py
.\.venv\Scripts\python.exe scripts/index_official_law_pdf_20261011.py
.\.venv\Scripts\python.exe scripts/check_rebuild_readiness.py --strict-core
.\.venv\Scripts\python.exe -m pytest tests -q
```

The official PDF scripts intentionally refuse to overwrite an existing manifest.
Only run them on a clean destination, or verify the existing source hashes
and index rather than deleting files. The indexer now uses `pypdfium2` to
extract the 403-page QCVN 41 PDF efficiently; page/source provenance is retained.
The raw PDFs, index, and third-party working tree are intentionally ignored by Git.

## Artifacts not recoverable from this checkout

| Artifact class | Consequence | How to rebuild legitimately |
| --- | --- | --- |
| Original canonical source images, boxes and 30-class recognition labels | Cannot reproduce prior training sets or historic VAL/TEST metrics | Reacquire licensed images, build a *new* versioned dataset, human-review labels and split by source/video to prevent leakage |
| Vietnamese lane video + manually approved lane masks/review decisions | No independent Vietnam lane ground truth | Acquire consent/licensed clips, annotate with human reviewers and freeze a new independent holdout |
| Previous input video MP4 and derived annotated movies | Media/video route acceptance tests fail | Record/download lawful new source videos and regenerate exports from those exact files; avoid claiming the new video is the old one |
| Research output reports, QA/retraining logs, UI review packs | Historical snapshot tests fail | Re-execute experiments with new provenance and write **new** versioned reports; do not copy made-up old metrics |
| Original shared Drive folder | Folder listing returned HTTP 404 via `gdown` on 2026-10-22 | Treat as unavailable until access/ownership is independently restored |

Older tests that assert exact old file paths and reports must either be satisfied
by the true historical bytes or **superseded by reviewed tests against a new
versioned dataset**. Never insert placeholder evidence or silently skip failing
tests to make the merge gate green.

## New acceptance gates

1. **Reproducibility:** a fresh clone can restore all Git LFS weights, exact
   pinned vendor source and the verified legal PDF index.
2. **Core software:** GitHub Actions, full applicable unit/API suites, frontend
   build, and CPU inference smoke all pass. Exercise GPU separately if required;
   the installed Torch build in this check reported `CUDA=False`.
3. **New source/labels:** provenance, licenses, checksums, source-disjoint
   train/val/test, independent human review, explicit uncertainty labels.
4. **End-to-end evidence:** render new demo footage and re-run applicable
   sign/lane metrics on a newly frozen holdout. Do not reuse historical numbers.
5. **Promotion:** review all failures and newly authored acceptance tests.
   Do not merge `main` while 41 full-suite failures remain unexplained.

## Prevent a second loss

Store irreplaceable raw data, review decisions and acceptance reports in a
private, permissioned artifact store (Drive, encrypted backup, or versioned
object storage), with a recorded SHA256 manifest and verified restore drill.
Keep source code + references + manifest (no secrets, personal videos, or
restricted datasets) in Git. Continue using Git LFS for weights. At every
milestone: backup -> independently download -> validate hashes -> run smoke
tests; a successful `git push` is **not** a backup of ignored files.
