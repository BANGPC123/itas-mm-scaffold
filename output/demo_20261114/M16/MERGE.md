# M16 merge record (local only, not pushed)

Pre-merge hashes, to return to these states if needed:

| Branch | Hash before merge |
|---|---|
| main | df47efab3345693a45284e66bef83709fbc2a554 |
| work/demo-20261114 | b0f36237f1d57d6b568bcae0d264349cc53dc626 |
| work/audit-20261114 | 87a911d6f3c1c1a855543e0733547743fe13c0ad |

Working tree before merge: no tracked changes (untracked docs/architecture-*, docs/itas-mm-architecture.local-20260917.html, docs/superpowers/ left as they were).

Merge base main/demo: df47efab (main is an ancestor of demo; demo has 31 commits not on main).
Merge base demo/audit: db49bd8.

Step 2: merge work/audit-20261114 into work/demo-20261114 (--no-ff). Audit adds 7 commits after db49bd8; the diff from that base is 4 files: docs/audit_20261114/FORWARD_REVIEW.md, SIGN_REVIEW.md, SLIDE_NUMBERS.md, scripts/build_forward_review_20261114.py. Output M11/M15 folders are gitignored and are not part of the branch.

Step 3: merge work/demo-20261114 into main (--no-ff).

## Result

- Step 2 merge (demo <- audit): commit b9ca377 (`git merge --no-ff work/audit-20261114`), no conflict. Adds only the 4 audit files listed above.
- Step 3 merge (main <- demo): commit ef9f0404b91d1842f4236f022d18a0f869864313 (`git merge --no-ff work/demo-20261114`), no conflict. main and work/demo-20261114 now have identical trees.
- Not pushed. No branch deleted. No worktree pruned. Audit worktree at D:/DSP391m/project/wt-audit still on work/audit-20261114.

## pytest (PYTHONNOUSERSITE=1, `pytest -q tests`)

- main (ef9f040): 654 passed, 11 skipped, 38 deselected. Log: `pytest_main.log`.
- work/demo-20261114 (b9ca377 state before step 3, same tree): 654 passed, 11 skipped, 38 deselected. Log: `pytest_demo.log`.
- No earlier pytest log on the demo branch exists in output/ to compare with; the earlier figure in RESULTS.md (636 passed) predates M13 tests.

## E2E on main (start.ps1, headless Chrome 9333)

- Video tab, `live_30s_source.mp4`: 2 forward warnings, "00:12 Chú ý xe máy — Có đọc", "00:15 Chú ý xe máy — Chỉ hiển thị". Speech request "Chú ý xe máy": 200, audio/mpeg 11276 bytes.
- Image tab, `M14/demo_photo.jpg`: "1 nhận dạng · 1 phát hiện", I.423 95%. Legal lookup: QCVN 41:2024/BGTVT Điều 84.
- Console: 0 entries. Screenshots: `e2e_main/shots/01`–`05`. Report: `e2e_main/e2e_main_report.json`.
- Not covered in this short run: playing the result video, and audible output (not confirmed; headless has no audio device).

## Files on disk after checkout

- `docs/demo/itas-mm_demo_video.mp4`: present, tracked.
- `output/demo_20261114/M14/demo_photo.jpg`: present, tracked.
- `output/demo_20261114/M12/e2e_final/live_30s_source.mp4`: present, NOT tracked (output/ is gitignored).
- `output/demo_20261114/M12/live_30s_v2/render/test_video_offline_max.mp4`: present, NOT tracked.
