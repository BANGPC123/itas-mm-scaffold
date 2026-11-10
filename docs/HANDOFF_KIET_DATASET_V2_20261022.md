# ITAS-MM — Handoff to Kiệt: Dataset v2 & Revalidation

- **Handoff date:** 2026-10-22
- **Owner for next stage:** Kiệt (GitHub username not yet confirmed)
- **Repository:** BANGPC123/itas-mm-scaffold
- **Starting branch:** `integration/itas-mm-release-candidate-20261021`
**Target:** Continue as a research/offline-MP4 prototype; do not imply field-certified driving assistance.
**Merge rule:** Work on a feature branch / PR. Keep `main` unchanged pending reviewed acceptance.

## 0. What is already complete — DO NOT redo

- Backend/frontend, Legal RAG, existing model inference pipelines and Git LFS artifacts are in Git.
- **17/17** model files have been fetched and verified with the recorded SHA256 manifest.
- UFLD-v2 research source at commit `c903880678454dfd9b55a63022368db05c00bc6d` can be restored.
- Four government-hosted legal PDFs have versioned URL/SHA256 pins; **703 chunks** rebuilt into the local offline index. The scanned Decree 168 PDF yielded **zero searchable pages** and must not be represented as searchable OCR content.
- `scripts/check_rebuild_readiness.py --strict-core`: PASS on the rebuilt workstation.
- GitHub core checks: PASS on prior commit `9f87f43`; frontend JavaScript **7/7 PASS**. Full suite: **444 passed / 41 failed**, mostly dependent on non-versioned historical assets.
- No historical source images, user MP4, human-verified review decisions or generated experiment reports can currently be recovered from this checkout.

References:
- `docs/REBUILD_FROM_ZERO_20261022.md`
- `docs/INTEGRATION_GATE_20261021.md`
- `docs/OFFICIAL_PDF_SNAPSHOT_20261022.json`
- `data/LOCAL_SETUP.md` (historical dataset schema and counts, **not** a currently accessible data source)
- `docs/ITAS_MM_INDEPENDENT_RELEASE_20261014.md` (historical claims only; raw evidence is missing)

## 1. Immediate assignment for Kiệt (ordered)

### WP1 — Reacquire and version new dataset (`dataset_v2`)

- Inventory candidate source images/videos for **sign detection**, **sign code recognition**, **Vietnam lane geometry**, and **supplementary signs / vehicle/time context**.
- Confirm dataset access, source licenses/permissions and redistribution restrictions before copying/uploading.
- Create a **new versioned manifest** with source URI/owner/rights, immutable SHA256 for raw bytes, timestamps when relevant, frame/video IDs, model-independent IDs and labeling status.
- Build a **small representative pilot** first; do not require reconstructing the historical 15,603 localization images or 10,003 recognition samples exactly.
- Split TRAIN/VAL/TEST by **source video/road route** before using a sample for tuning. Group contiguous video frames together to prevent leakage. Freeze independent TEST inputs with SHA256.
- Store raw/private media in a **permissioned external artifact store**, not in Git. Git receives only scripts, schemas, reference manifests with non-sensitive metadata and documentation. Re-download and verify hashes on a second clean environment.

**Done when:** clean clone + authorized artifact download can reproduce the frozen pilot split and provenance reports; no sample ID or source group leaks across splits.

### WP2 — Human ground truth (do not promote AI proposals)

- For signs: human-approved boxes + sign codes; maintain unknown/occluded/out-of-vocabulary cases.
- For lanes: reviewed polylines or pixel masks, road edges versus paint, split/merge/occlusion, missing-line flags, frame/source IDs.
- For context: primary/supplementary plate association, vehicle classes, sign placement and time validity. Preserve `unknown` or `uncertain` where not visually verified.
- Store reviewer ID (non-sensitive pseudonym is sufficient), version, review timestamp, adjudication status and disagreements. Do not claim AI pseudo-labels are human GT.
- Make labeling quality visible: coverage, class balance, inter-reviewer agreement/disagreements where possible.

**Done when:** only human-approved labels are eligible for final ground-truth metrics. Freeze and independently review the held-out TEST split before measuring.

### WP3 — Rebenchmark existing checkpoints before any retraining

- Keep the 17 verified checkpoints unchanged and record their SHA256/configuration for every run.
- Run sign detection **precision/recall/mAP and small-sign recall**; recognition **per-class accuracy and confusion matrix**; lane **precision/recall/F1** with human-approved Vietnam GT.
- Benchmark video throughput/latency and temporal duplicated/stale warnings using original video source bytes; distinguish render FPS from inference FPS.
- Record true traffic-sign context uncertainty and legal citations; do not claim current-law validation or definite legal penalties.
- Export reproducible JSON reports with code SHA, model SHA, dataset manifest SHA, split, hardware and settings.
- Train/fine-tune only after comparing measured deficiencies against agreed targets. No tuning on frozen TEST.

**Done when:** current checkpoints have independently reproducible baseline results for the newly frozen pilot; limitations documented.

### WP4 — Replace historical acceptance dependencies, do not fake historical files

- Audit the **41 full-suite failures** from the 2026-10-22 rebuild and categorize: `legacy-artifact snapshot`, `missing current real input`, `real code defect`.
- Preserve the historical test contract and original reports for audit; create separately named **v2 acceptance tests** referring to versioned new evidence.
- Do not fabricate original `test_video.mp4`, review manifests, GT lane masks, 33-class training result, or the former release-gate reports.
- Restore representative end-to-end MP4 / API / overlays / TTS checks on actual new licensed video; verify audio, seek, stale-frame handling and lack of unsupported legal claims.
- Produce a new `release_gate_v2.json` with PASS/FAIL/BLOCKED, explicit provenance, CI link and remaining gaps.

**Done when:** core CI passes; full applicable v2 suite has no unexplained failures; all abandoned historical snapshot tests are explicitly archived/replaced with review rationale, never silently skipped.

## 2. First technical steps

```powershell
git clone https://github.com/BANGPC123/itas-mm-scaffold.git
cd itas-mm-scaffold
git fetch origin
git switch --track origin/integration/itas-mm-release-candidate-20261021
git switch -c work/kiet-dataset-v2
git lfs pull

uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe -m pytest tests -q --disable-warnings --tb=no
```

Expect legacy-asset failures before acquiring new data. Follow `docs/REBUILD_FROM_ZERO_20261022.md` for the pinned UFLD-v2 source and legal PDF recovery. A working GPU was detected, but the installed Torch on the recovery machine reported `CUDA=False`; GPU inference is **not verified**.

## 3. Completion & collaboration rules

- Make commits to `work/kiet-dataset-v2` or another dedicated feature branch. Open PR against the integration branch, **not `main`**.
- Each PR must include a `what changed` summary, reproducible commands, actual tests/results, and a clear list of unavailable external assets.
- No source-data backup is complete until another clean environment can restore a file and verify its SHA256.
- Do not upload personal source videos, restricted training datasets, account credentials, or private reviewer identities to this public repository.
- Present old October 2026 metrics as **historical and not independently reproducible**; do not copy them into the new dataset's benchmark report.
- This project remains a **research/offline video prototype**, not safety-certified ADAS.

**Handoff outcome expected:** one pilot `dataset_v2` with real provenance + independent human GT + fresh baseline report, then a tracked, reviewed plan to resolve the remaining historical acceptance blockers.
