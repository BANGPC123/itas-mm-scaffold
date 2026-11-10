# Simplified offline lane display: acceptance result (2 Nov 2026)

Scope: uploaded MP4 only (offline render). Image tab unchanged. Setting: `configs/perception.yaml` → `offline_video_lane.display` (`simple` or `precision`; `precision` restores the previous lane path).

Clip: first 60 seconds of `docs/demo/itas-mm_demo_video.mp4` (1,800 frames at 30 FPS output, analysis 6 Hz, lane stage every 2nd sample, auxiliary review on). Same script path for both runs; one run each on the same machine.

## 1. Processing time

| Run | Lane path | Processing time (`summary.json` `elapsed_seconds`) | Wall time incl. model load |
|---|---|---|---|
| before | `precision` (previous pipeline) | 171.7 s | 230.7 s |
| after | `simple` | 77.9 s | 82.1 s |
| web (after) | `simple`, via `start.ps1` upload of a re-encoded copy of the same 60 s clip | 127.9 s | not measured |

The web run used a re-encoded copy of the clip (libx264, CRF 23), not the original file, so its sign counts differ from the script run. The cause of the slower web time was not investigated.

## 2. Sign comparison (before vs after, same clip)

Method: for every analysis sample in `predictions.jsonl`, compared the list of `(bbox_xyxy, label)` of `signs`, plus `context` and `speech`.

- Samples compared: 360 (before) / 360 (after)
- Samples with any difference in sign bbox, label, context or speech: **0**
- Result: identical.

This is a regression check on the sign output, not a measure of sign accuracy.

## 3. Lane coverage (not accuracy)

| Run | Samples with a displayed lane | Backend | Rejection reason |
|---|---|---|---|
| before | 0 / 360 | `camera_road_paint_visual` | `camera_direction_uncertain` in 360 / 360 |
| after | 8 / 360 (2.2%) | `ufld_v2` | none |

After-run lane samples (one displayed polyline each): t = 31.67, 31.83, 41.67, 41.83, 42.00, 42.17, 45.00, 45.17 s.

`camera_view` was `unknown` in all 360 samples in both runs. This is unchanged, so the car-context text still reports that the camera view is not established.

These counts show only how often a lane was displayed. They are not a measure of lane correctness; there is no human-reviewed lane ground truth.

## 4. Frames

Six before | after pairs in `output/lane_simple_20261102/frames/` (not committed):

| File | Time | Scene | Lane observation | Known issue |
|---|---|---|---|---|
| `day_lane_t31.67.jpg` | 31.67 s | Day, parked SUVs at right | Yellow line runs along the lower edge of the parked black SUV and across its body | **Yes: line crosses a vehicle** |
| `day_lane_t41.67.jpg` | 41.67 s | Day, motorbike ahead | Yellow line on the left kerb edge, not on the dashed lane markings | **Yes: follows kerb** |
| `day_lane_t45.0.jpg` | 45.0 s | Day, motorbike ahead | Yellow line on the left kerb edge, diagonal toward centre | **Yes: follows kerb** |
| `occl_car_t28.jpg` | 28.0 s | Day, parked cars at right | No lane shown in either run | No (no lane displayed) |
| `occl_car_t50.jpg` | 50.0 s | Day, red car at right occludes the kerb | No lane shown in either run | No (no lane displayed) |
| `occl_car_t58.jpg` | 58.0 s | Day, white SUV in foreground | No lane shown in either run | No (no lane displayed) |

No dashboard (táp-lô) is visible in this clip. Sign boxes and labels are identical in both runs for these frames.

## 5. Known issues

1. **Lane coverage is low.** A lane is displayed in 8 of 360 samples (2.2%) in this clip; the other samples show none.
2. **Displayed lines are not on lane markings in this clip.** In 3 of 3 frames with a line, it follows a kerb edge or a parked car. This is an accuracy problem of the UFLD-v2 output, not something the display change can fix. The "visual context only" wording must stay.
3. **The overlay header text is unchanged** and still reads "Vạch sơn quan sát được; không nối nét qua xe hoặc vùng thiếu bằng chứng". In simple mode this statement is not true for the line at 31.67 s, which crosses a vehicle. The text was not edited because the task did not cover it.
4. **`camera_view` is `unknown` in every sample**, so car-context text stays at "not established" in both runs (unchanged).
5. **Web and script runs differ** (sign counts 161 vs 173, time 127.9 s vs 77.9 s). Most likely cause: re-encoded input. Not verified.

## 6. Video integrity

- Script run: 1,800 frames written, `decoded_frames` = 1,800, `verified_full_decode` = true, `failures` = [].
- Web run: job status `complete`, `frames` = 1,800, `verified_full_decode` = true. The result file was downloaded and decoded with OpenCV: 1,800 frames at 30.0 FPS.

## 7. Web check

Started with `start.ps1` on port 8000. Uploaded the 60 s clip (`web_clip_60s.mp4`, 30.8 MB) to `/pipeline/offline-video/start` with `analysis_hz=6`, `auxiliary_review=true`. Job `041e62ce856741238dd160e4a4492b9e` reached `complete`. The server was stopped after the check.

## 8. Test results

Implementation commit: `4e662ab800629508397199e560c919e1fa4c58bb` (2 November 2026). The final report commit is recorded with the full-suite replay in `output/lane_relax_20261102/checks_final.json`.

| Check | Passed | Failed | Skipped | Deselected | Status |
|---|---:|---:|---:|---:|---|
| `tests/unit/test_lane_simple_20261102.py` | 8 | 0 | 0 | 0 | PASS |
| `tests/unit/test_lane_relax_20261102.py` | 5 | 0 | 0 | 0 | PASS |
| `python -m pytest -q tests` | 613 | 0 | 11 | 38 | PASS, exit 0 |
| `node --test frontend/src/*.test.js` | 14 | 0 | 0 | 0 | PASS, exit 0 |

The earlier 607 full-suite passes / 7 simple-lane tests were measured at `a30b4f0`, before the overlay-status test added in `dd5a3e4` raised those counts to 608 / 8; this change adds five relaxation tests, yielding 613 full-suite passes.

Logs: `output/lane_relax_20261102/pytest_focused_native.log` (16 passes: 8 simple, 5 relaxation, 3 decode), `pytest_precommit.log`, `node_precommit.log`, and the final-commit replay `pytest_final.log` / `node_final.log` in the same directory. `checks_final.json` records the exact tested final HEAD, commands, exit codes and all four counters. Tests run on Windows native with `PYTHONNOUSERSITE=1` and dedicated pytest temporary/cache directories; the earlier sandbox temp-directory PermissionError is kept separately in `pytest_focused.log`.
