# NUMBERS — source for every number in docs/final_report/ITAS-MM_Final_Report.md

Paths are relative to the repo root. "Report 3" = docs/report3/REPORT3_MODEL_RESULTS.md (section given).

## Data and training

| Number in report | Value | Source |
|---|---|---|
| Canonical localisation set | 15,603 images, 30,558 boxes; TRAIN 10,952 / VAL 2,325 / TEST 2,326 images | Report 3, IV.3; docs/DATASET_V2.md |
| Recognition manifest V1.1 | 10,003 crops, 30 codes; TRAIN 5,673 / VAL 1,272 / TEST 3,058 | Report 3, IV.3 |
| Recognizer v2 crops | TRAIN 17,445, VAL 3,933 | Report 3, IV.5 |
| Recognizer v2 classes | 87 codes + background; 55 main, 32 low-support | docs/RECOGNIZER_V2_20261028.md (lines 88–89) |
| AI-coded extension boxes | 976 used (TRAIN 553, VAL 120, TEST 303); agreement 76.9%; 555 excluded | docs/DATASET_V2.md ("Ontology extension labels") |
| Detector training time | 10 epochs, 23 min 10 s, 4 GB RTX 3050 | Report 3, IV.4 |
| Recognizer reproduction | VAL macro-F1 0.9511 original vs 0.9500 reproduced; epoch 18; threshold 0.87 | Report 3, V.6 |
| Recognizer v2 training settings | 30 epochs, seed 20261028, batch 128, 14,000 crops/epoch, lr 0.0004 | Report 3, IV.5 |
| Sign-size share | 1,502 of 4,888 TEST boxes below 16 px (31%) | Report 3, VI.1 |

## Detector and recognizer (dataset TEST)

| Number | Value | Source |
|---|---|---|
| Detector TEST precision | 0.9232 baseline; 0.9361 fine-tuned | Report 3, V.4 |
| Detector TEST recall | 0.8509 → 0.8754 | Report 3, V.4 |
| Detector TEST mAP50 | 0.9194 → 0.9357 | Report 3, V.4 |
| Detector TEST mAP50-95 | 0.7168 → 0.7360 | Report 3, V.4 |
| Bootstrap recall gain | +0.0245, 95% CI [0.0170, 0.0317] | Report 3, V.4 |
| Bootstrap mAP50 gain | +0.0163, 95% CI [0.0122, 0.0198] | Report 3, V.4 |
| Recall below 16 px | 0.651 → 0.713 | Report 3, VI.1 |
| Recall 32 px and above | about 0.96 | Report 3, VI.1 |
| Recognizer TEST ≥32 px accuracy | 0.9371 original; 0.9697 v2 | Report 3, V.5 table |
| Recognizer TEST ≥32 px macro-F1 | 0.9056 original; 0.9594 v2 | Report 3, V.5 table |
| Recognizer TEST all sizes accuracy | 0.7904 → 0.9090 | Report 3, V.5 table |
| Recognizer below 16 px accuracy | 0.5139 original; 0.7839 v2 | Report 3, V.5 table |
| Operating points (≥32 px) | original 0.63: 1,223 accepted / 1,169 correct / coverage 0.9745 / correct-over-all 0.9315; v2 0.87: 1,134 / 1,121 / 0.9036 / 0.8932; v2 T2: 1,207 / 1,186 / 0.9618 / 0.9450 | Report 3, V.5 |
| P.137 confusion | 17 of 73 correct; 39 to P.123b | Report 3, VI.3 |
| W.201a confusion | 5 of 17 to W.203c | Report 3, VI.3 |
| Field pilot precision | 0.641 → 0.729 (120 frames, 66 signs) | Report 3, VI.4 |
| Field pilot recall | 0.758 → 0.773 | Report 3, VI.4 |

## Field A/B/C precision (Report 3, V.7)

| Number | Value | Source |
|---|---|---|
| Combined precision, first reviewer's labels, B (0.87) | 0.9633 (105/109) | Report 3, V.7 table (combined population, first label set) |
| Combined precision, C (T2) | 0.8872 (118/133) | same |
| Registered T2 rule | not met (precision condition False) | Report 3, V.7 decision table |

## CP2 threshold decision

| Number | Value | Source |
|---|---|---|
| New tracks graded (N64 minus 0.87) | 52: 39 correct, 4 wrong, 9 uncertain | docs/final_report/evidence/track_grading_064.csv (same file as cp2_grading.csv; counts recomputed 31 Oct 2026) |
| Precision on decided tracks | 39/43 = 90.7% | same |
| Precision if uncertain = wrong | 39/52 = 75.0% | same |
| Chosen threshold | 0.6412901874067937 | configs/perception.yaml; commit on work/final-20261102 |
| Decision rule | lowest threshold with precision ≥ 85% on graded tracks | D:/DSP391m/FINAL_PLAN_20261102.md, section 5 (A) |

## Field segment evaluation (T8, 15 × 60 s)

| Number | Value | Source |
|---|---|---|
| Physical tracks B0 / N87 / N75 / N64 | 1,205 / 846 / 839 / 834 | recount of predictions.jsonl under output/kiet_wp_20261029/field_eval (31 Oct 2026) |
| Tracks with fewer than 3 samples B0 / N64 | 1,022 / 637 | same |
| Tracks with a code B0 / N87 / N75 / N64 | 75 / 75 / 86 / 96 | docs/final_report/evidence/eval_summary.md |
| Distinct codes | 15 / 15 / 20 / 23 | docs/final_report/evidence/eval_summary.md |
| Median confirmation | 333 ms (all configs) | docs/final_report/evidence/eval_summary.md |
| Median inference FPS | 2.82 / 3.05 / 3.00 / 3.01 | docs/final_report/evidence/eval_summary.md |
| Frame-level coded decisions in [0.75, 0.87) | 0 (B0), 0 (N87), 213 (N75), 213 (N64) | docs/final_report/evidence/eval_summary.md (wiring F) |
| Coded tracks added over B0 | +11 (N75), +21 (N64) | derived: 86−75, 96−75 |
| Three-sign frame at 33.0 s | B0: 0 coded boxes; N64: 2 (P.112, W.245a) | docs/final_report/figures/fig_three_signs_b0_vs_n64.png (script scripts/make_three_signs_figure_20261102.py) |

## Runtime

| Number | Value | Source |
|---|---|---|
| Streaming prototype (removed from the final app), lane overlay off, original clip, 115 frames | p50 78.9 ms; p95 618.1 ms | docs/final_report/evidence/t9_latency_original.json |
| Streaming prototype (removed from the final app), lane overlay on, same clip | p50 271.7 ms; p95 843.0 ms | same |
| Baseline overlay clip (T1), lane off / on p50 | 57.6 / 240.6 ms; lane on p95 271.9 ms | output/final_20261102/baseline_latency.json |
| Offline pipeline per image | 46.2 ms (21.7 FPS) | Report 3, VI.5 |
| Detector per image | 10.8 ms (92.7 FPS) | Report 3, VI.5 |
| Upload job, 90 s clip, after the speed change (01 Nov 2026): processing time | 165.0 s (171.3 s wall); 1.8× slower than real time (165.0/90 = 1.83) | docs/final_report/evidence/SPEED.md |
| Upload job, 90 s clip, after the speed change: output | 2,700 frames at 30 FPS, full decode verified; sign analysis at 6 Hz, lane at 3 Hz | docs/final_report/evidence/SPEED.md |
| Upload job, 90 s clip, before the speed change (same clip, 01 Nov 2026) | 406.3 s (job ab641b4878c6); 5,400 frames at 60 FPS; 4.5× slower than real time (406.3/90) | docs/final_report/evidence/SPEED.md |
| Upload job, 90 s clip, earlier run before the speed change (31 Oct 2026) | 301.2 s; 5,400 frames at 60 FPS; 3.3× slower than real time (301.2/90) | output/offline_video_jobs/a2b1e58883ef41e2a9b12e6c6f987d45/result/summary.json |
| Upload job, 20 s night clip through the web UI, after the speed change | 44.6 s processing (about 133 s end to end before the change) | docs/final_report/evidence/SPEED.md |
| Sign results after the speed change | 540 of 540 analysis samples have identical sign boxes and labels; 65 coded boxes in both runs; lane lines drawn 1,504 before, 1,554 after | docs/final_report/evidence/SPEED.md |
| Live p95 includes TTS | yes (endpoint default audio on) | output/final_20261102/baseline_latency.json (configs.summary.tts) |

## Legal retrieval

| Number | Value | Source |
|---|---|---|
| Index size | 703 chunks from 4 official PDFs | Report 3, IV.2 |
| Recall@1 / @3 / @4, MRR | 0.542 / 0.833 / 0.875 / 0.684 (24 cases) | Report 3, VI.5 |
| Legal currency | not verified (`legal_currency_verified: false`) | live response from POST /pipeline/legal-search, 31 Oct 2026 |

## Tests and checks (31 Oct 2026)

| Check | Value | Source |
|---|---|---|
| pytest tests/ | 600 passed, 11 skipped, 38 deselected; exit 0 (1 Nov 2026) | output/final_20261102/t15_pytest.txt |
| node --test frontend/src/*.test.js | 14 / 14 pass, exit 0 (1 Nov 2026) | output/final_20261102/t15_node_test.txt |
| npm run build | succeeded, exit 0 (1 Nov 2026) | output/final_20261102/t15_npm_build.txt |
| Earlier run, 31 Oct 2026 | pytest 597 passed; node 9 / 9 | output/final_20261102/t12_pytest.txt, t12_node_test.txt |

## Other facts used

| Fact | Source |
|---|---|
| WHO: about 1.19 million road deaths a year | Research_proposal.pdf, Abstract, citing [1] |
| Team names and planned roles | Research_proposal.pdf, Members and section 6 |
| Name coverage: 87 codes all named | output/final_20261102/missing_names.txt (0 missing) |
| Demo clip sources and hashes | docs/final_report/evidence/SOURCES.md |
| Evaluation videos are Hanoi, two day and one night | data/manifests/field_video_20261029/sources.csv (role eval) and CP2 notes |
| Simplified lane path, 60 s clip: 171.7 s to 77.9 s processing (script, one run each); 0 of 360 sign samples differ; lanes shown in 0 then 8 samples | docs/final_report/evidence/LANE_SIMPLE.md (section 1, 2, 3) |
| Simple lane path, 90 s night clip: 153.3 s processing (one run, 1.7 times real time); lane shown in 178 of 540 samples (coverage, not accuracy) | docs/final_report/evidence/SPEED.md (section 3 Nov 2026) |
