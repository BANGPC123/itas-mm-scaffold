# ITAS-MM integration gate — 2026-10-21

## Scope

- Branch: `integration/itas-mm-release-candidate-20261021`
- Base: `478ad7a895e42dfa53863b6130f890f94c882928` (weights-lfs branch)
- Purpose: validate a clean clone of the integration candidate without merging or changing `main`.
- Workstation: Windows, Python 3.12.15, Node 24.21.0, RTX 4080 Laptop GPU.
- Test environment: isolated Python venv created with `uv`; packages installed from `requirements.txt`. Torch 2.14.1 reports CUDA unavailable in this environment.

## Verified checks

| Check | Result | Evidence / limitation |
| --- | --- | --- |
| Git LFS checkout | PASS | All 17 manifest-listed files downloaded; size and SHA256 matched `docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json`; `git lfs fsck` PASS. 2,589,889,919 total declared bytes. |
| Core backend CI-equivalent | PASS | 135 tests passed, run with the exact files listed in `.github/workflows/core-tests.yml`. |
| Full tracked Python tests | FAIL / BLOCKED | `python -m pytest tests -q --disable-warnings --tb=no`: **438 passed, 45 failed** after compatibility fix. |
| Frontend JS tests | PASS | `node --test frontend/src/*.test.js`: 7 passed. |
| Frontend production build | PASS | `npm --prefix frontend ci`; `npm --prefix frontend run build`. |
| Offline legal retrieval | PASS (development benchmark) | 24 cases; Recall@1=0.5417, Recall@3=0.8333, Recall@4=0.875, MRR=0.6840; benchmark explicitly declares independent field validation and legal currency verification false. |
| YOLO detector CPU smoke | PASS | `ultralytics.YOLO('models/sign_detector/best.pt').predict(np.zeros((320,320,3),dtype=np.uint8),imgsz=320,device='cpu')` returned one result. Synthetic input tests loadability/execution, **not** recognition quality. |
| GPU inference | NOT VERIFIED | `torch.cuda.is_available()` was False in installed Python environment despite an NVIDIA GPU being present. |
| Human-reviewed field accuracy | NOT VERIFIED | Missing field data/ground truth and reports; do not infer safety or driving readiness. |

## Full test blockers

The remaining 45 failures are not part of the core CI test file list and primarily refer to resources not included in a clean clone:

- Generated `output/` videos, named sign galleries, evaluation/retraining/review reports and media.
- User/offline Vietnamese road video frames, source assets, human-label review manifests and provenance packs under `data/`.
- External `third_party/Ultra-Fast-Lane-Detection-v2/utils/loss.py` dependency used by lane-training tests.
- Official current legal PDF / local source-and-index artifacts expected by `test_offline_legal_retrieval_20261011.py`.

These are **unresolved acceptance dependencies**, not test successes. Rerun all full tests with a documented, checksum-verified artifact restoration process before merge. Do not suppress or skip these tests to create an artificial green result.

## Integration changes

- Restore `THIRD_PARTY_NOTICES.md` from `wip/task6-legal-retrieval-benchmark-20261001` so adapted LuatRAG MIT material has its attribution and license notice.
- Accept both `(N,1,4)` and `(N,4)` shapes of `cv2.HoughLinesP` in `observed_paint_seeds()`. Add two explicit regression variants. Related tests: **19 passed**.

## Merge decision

**HOLD**: core path, LFS integrity and CPU smoke pass; full clean-clone acceptance remains incomplete. Do not merge `main`. Restore deterministic data dependencies, rerun the complete suite and relevant model/video workflows, verify GPU inference only if part of the acceptance target, then review an integration PR. This branch is a research prototype and is **not** certified or field-validated driving assistance.
