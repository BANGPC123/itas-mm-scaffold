# UFLD display row relaxation experiment

Date: 2 November 2026. Implementation commit: `4e662ab800629508397199e560c919e1fa4c58bb`, on `fix/lane-simplified-demo-20261102` (base `f9454e79061212614a1897e74ceb0bd9b10a570c`). Selected `offline_video_lane.display_row_ratio`: **0.5**.

Conclusion: "giới hạn chất lượng model, nới ngưỡng không cải thiện đủ".

## Protocol and scope

Only simple offline MP4 display uses the optional ratio. The canonical `polylines`, row threshold 1/2, column threshold 1/4, minimum point count and configured geometry filter remain intact. Relaxed display curves come from the same UFLD forward, pass through `visual_polylines(mode='candidate')` and `LaneStreamPool`, and do not replace the input to `CameraLaneVisual`. Precision and image paths do not receive the ratio. Models, sign thresholds and datasets were not changed.

Six sequential `render()` calls used `analysis_hz=6`, `lane_every=2`, `output_fps_cap=30`, `include_auxiliary=True`, no high-resolution rescue and no recording-date override. Ratios were tested in order 0.5, 0.35, 0.25 for day, then night. Each call initialized its own pipeline and stream. These are one-run timings, not a controlled speed comparison.

| Clip | Source | Duration | Analysis samples | Output frames |
|---|---|---:|---:|---:|
| Day | `docs/demo/itas-mm_demo_video.mp4` | first 60 s (`max_seconds=60`) | 360 | 1,800 at 30 FPS |
| Night | `output/final_20261102/demo_clips/night_hanoi_YQl9c5gCjhg_30s_90s.mp4` | 90 s (`max_seconds=90`) | 540 | 2,700 at 30 FPS |

The sampling list was written before the first render to `output/lane_relax_20261102/frozen_samples.json`. It uses 15 equal-width interval midpoints per clip; there was no frame replacement after viewing results.

Frozen at `2026-11-02T12:54:39.455258+00:00`; manifest SHA-256: `31d136e6b6c13e301471c0b70500b77a6d0ecb2507d8752374828332bdee7cf9`.

- Day timestamps (seconds): 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58.
- Night timestamps (seconds): 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87.

At each timestamp, the same output frame from all three ratios was placed side by side in ratio order. Every one of the 30 comparison rows (90 ratio panels) was opened individually. The ambiguous night paint/kerb cases at 15, 21, 27 and 51 s were also checked against native 1920 x 1080 source frames at those same frozen timestamps. Line identifiers L1, L2, etc. follow each sample's `display_polylines` order.

**This is AI visual review, not independent human ground truth.** DUNG means a line lies on visible longitudinal paint or its painted edge; SAI means it follows kerb, vehicle, shadow or unsupported unpainted pavement; KHONG_CHAC means visible evidence does not resolve alignment. A line near one faint dash without established alignment is kept uncertain. These judgments do not measure lane accuracy, precision or recall.

## A4 measured table

"All samples with lane" counts analysis samples across the entire rendered segment. "Frozen frames with lane" and the three line counts use only the 15 frozen frames for that clip (30 frames total per ratio). Counts are displayed curves, not unique physical lane boundaries.

| Clip | Ratio | All samples with lane | Frozen frames with lane | DUNG | SAI | KHONG_CHAC | Processing s | Wall s including initialization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Day | 0.5 | 8/360 | 1/15 | 0 | 1 | 0 | 96.8 | 104.1 |
| Day | 0.35 | 50/360 | 2/15 | 0 | 4 | 0 | 84.5 | 86.1 |
| Day | 0.25 | 60/360 | 2/15 | 0 | 4 | 0 | 84.7 | 86.4 |
| Night | 0.5 | 178/540 | 5/15 | 3 | 8 | 2 | 180.7 | 182.4 |
| Night | 0.35 | 200/540 | 5/15 | 3 | 8 | 2 | 175.3 | 177.0 |
| Night | 0.25 | 212/540 | 5/15 | 3 | 9 | 2 | 176.5 | 177.9 |

Processing time is `summary.json.elapsed_seconds`, which excludes initial pipeline construction and includes the renderer's final mux/decode work. Wall time is the full `render()` call in `experiment.json`. Run files are under `output/lane_relax_20261102/<day|night>_<ratio>/`.

## A5 selection

The supplied selection rule, applied verbatim:

> Chọn ratio nhỏ nhất thỏa CẢ HAI: (a) số đường ĐÚNG tăng so với 0.5; (b) SAI / (ĐÚNG+SAI) ≤ 1/3 trên cả hai clip gộp lại.
>
> Không ratio nào đạt thì giữ 0.5 và ghi kết luận: "giới hạn chất lượng model, nới ngưỡng không cải thiện đủ".

| Ratio | DUNG pooled | SAI pooled | KHONG_CHAC pooled | SAI/(DUNG+SAI) | DUNG greater than baseline | Meets both criteria |
|---|---:|---:|---:|---:|---|---|
| 0.5 | 3 | 9 | 2 | 75.00% | no | no |
| 0.35 | 3 | 12 | 2 | 80.00% | no | no |
| 0.25 | 3 | 13 | 2 | 81.25% | no | no |

Both lower ratios FAIL the selection criteria: correct lines do not increase, and the decided-line wrong fraction exceeds 1/3. Keep 0.5. This fallback preserves the existing output; it does not certify the baseline's quality. The two uncertain lines per ratio are excluded from the specified fraction. The sample set is small, unblinded and selected on these development clips; it cannot establish general field performance.

## Per-frame AI judgments

Each cell lists L1, L2, etc. in order. D = DUNG; S = SAI; U = KHONG_CHAC; none = no displayed line. All three panels were inspected even when none appeared. Full per-line coordinates and notes are in `output/lane_relax_20261102/visual_review.json` (`human_gt: false`).

| Clip | Time s | 0.5 | 0.35 | 0.25 |
|---|---:|---|---|---|
| Day | 2 | none | none | none |
| Day | 6 | none | none | none |
| Day | 10 | none | none | none |
| Day | 14 | none | none | none |
| Day | 18 | none | none | none |
| Day | 22 | none | none | none |
| Day | 26 | none | none | none |
| Day | 30 | none | none | none |
| Day | 34 | none | none | none |
| Day | 38 | none | none | none |
| Day | 42 | L1=S | L1=S, L2=S, L3=S | L1=S, L2=S, L3=S |
| Day | 46 | none | L1=S | L1=S |
| Day | 50 | none | none | none |
| Day | 54 | none | none | none |
| Day | 58 | none | none | none |
| Night | 3 | none | none | none |
| Night | 9 | none | none | none |
| Night | 15 | L1=S, L2=S, L3=D, L4=S | L1=S, L2=S, L3=D, L4=S | L1=S, L2=S, L3=D, L4=S |
| Night | 21 | L1=D, L2=S, L3=S | L1=D, L2=S, L3=S | L1=D, L2=S, L3=S |
| Night | 27 | L1=U, L2=S, L3=D | L1=U, L2=S, L3=D | L1=U, L2=S, L3=D |
| Night | 33 | none | none | none |
| Night | 39 | none | none | none |
| Night | 45 | none | none | none |
| Night | 51 | L1=U | L1=U | L1=U |
| Night | 57 | none | none | none |
| Night | 63 | none | none | none |
| Night | 69 | L1=S, L2=S | L1=S, L2=S | L1=S, L2=S, L3=S |
| Night | 75 | none | none | none |
| Night | 81 | none | none | none |
| Night | 87 | none | none | none |

- Day 42 s: the baseline follows the left kerb; added L1 zigzags on pavement beside the dashed marking and L2 crosses the motorbike/shadow. Day 46 s: the added line zigzags beside/intermittently across dashed paint without following it.
- Night 15 s: L3 follows a white painted asphalt edge, distinct from the red-white kerb; L1 doubles back near the kerb, L2 crosses bare pavement/reflections, and L4 crosses the near scooter.
- Night 21 s: L1 follows white left-edge paint; L2 crosses the green rider and L3 crosses the riders/right shadow.
- Night 27 s: L1 is near the hatched-zone paint border but drifts laterally (U); L2 crosses bare centre pavement (S); L3 follows white right-edge paint (D).
- Night 51 s: the line is near a faint short dash, but the remaining alignment is unresolved through dark/obscured pavement (U).
- Night 69 s: baseline lines cross the red SUV and right motorbikes; 0.25 adds a zigzag below the pickup on bare pavement (S).

## A6 regression and decode

PASS. Samples were matched by both timestamp and source-frame index, with equal lengths: 360 day / 540 night. For each ratio against 0.5 on each clip, differing samples = **0** for `(signs.bbox_xyxy, signs.label)`, full sign dictionaries, context, speech, camera_view, canonical polylines and canonical lane_lines. This includes the selected 0.5 baseline; both rejected lower ratios were also checked. These are output-regression checks, not sign accuracy measurements.

All six outputs have `verified_full_decode: true`, no failures and exact decoded/written counts: 1,800/1,800 day and 2,700/2,700 night. Source hashes were verified. The new 0.5 runs additionally match the earlier simple results in `output/lane_simple_20261102/after/` (day) and `output/speed_simple_20261103/night_90s/` (night): zero differing samples for sign bbox/label, context, speech, camera_view, canonical and display geometry. `baseline_prior_comparison.json` and `night_baseline_prior_comparison.json` record those checks.

Evidence: `output/lane_relax_20261102/regression.json`, `regression.log`, `runs.json`, each run's `summary.json` and `experiment.json`, and `render.log`.

## Tests and review files

PASS: full Python suite 613 passed, 0 failed, 11 skipped, 38 deselected; Node 14 passed, 0 failed, 0 skipped, 0 deselected. Focused lane tests: 8 simple + 5 relaxation + 3 existing decode = 16 passed. Five new tests cover None, 20/72 rows at 0.25 from one forward, equivalence at 0.5, precision isolation and canonical camera input. See `LANE_SIMPLE.md` for the historical 607/7 explanation and final replay paths.

Complete comparison sheets (local, not committed):

- `output/lane_relax_20261102/day_comparison.jpg` (all 15 day rows).
- `output/lane_relax_20261102/night_comparison.jpg` (all 15 night rows).

Native-size individual three-ratio rows are in `output/lane_relax_20261102/frames/`. Reproduction helpers in the same output directory: `run_experiment.py`, `build_comparisons.py`, `verify_regression.py`, `run_checks.py`. No browser upload smoke test or retraining was run for this experiment. Existing sign/dataset/architecture work was preserved.

Because 0.5 was retained, sections 8/9 and their coverage counts remain valid. Only one sentence was added to the Markdown report's lane limitation under C; the DOCX rewrite condition (a selected ratio different from 0.5) was not triggered. Appendix A was refreshed under B. The existing DOCX was not rebuilt.

## Paper changes to apply manually

`docs/final_report/latex/main.tex` was read but not modified. These locations need synchronization with the existing simple path and this experiment:

| Line | Current text or anchor | Required edit |
|---|---|---|
| 130 | `Lane overlay\\UFLD-v2 + surface veto` | Diagram label: use `UFLD-v2 simple display`; the current upload path has no surface veto. |
| 203 | `On a 90 s night clip these changes reduced the processing time from 406 s to 165 s` | Identify 165 s as the earlier speed-step result; add the later simple-path 153.3 s single-run result from evidence/SPEED.md, without claiming a controlled speed gain. |
| 391 | `a 90 s clip is processed in 165 s, or 1.8 times its duration` | For the report's existing simple-path speed measurement, use 153.3 s, about 1.7 times duration, one run; distinguish this experiment's new 180.7 s baseline run and ordinary run variability. |
| 401 | `The lane overlay has no Vietnamese ground truth` | Append: `Lowering the UFLD row-existence threshold to 0.35 and 0.25 was tested but did not meet the selection rule on 30 preselected frames assessed by AI visual review.` Keep the absence of human GT explicit. |
| 401 | `Uploaded video is processed about 1.8 times slower than real time` | Synchronize the referenced existing simple-path measurement to about 1.7 times (153.3 s / 90 s, one run), with timing source/date. |
