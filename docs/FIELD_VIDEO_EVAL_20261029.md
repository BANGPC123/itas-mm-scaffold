# Field video evaluation of recognizer v2 (2026-10-29)

**Status: evidence snapshot for the owner's decision; release stays HOLD.** Every label here is AI-reviewed (`provenance=ai_reviewed`, reviewer `ai-review (Claude Code)`), not human ground truth. `configs/perception.yaml`, the default model, the thresholds (0.87, T2 = 0.6412901874) and the gate configuration were **not changed**. Work package WP3 of issue #2, branch `work/kiet-recognizer-v2`. Related: [RECOGNIZER_V2_20261028.md](RECOGNIZER_V2_20261028.md), [RECOGNIZER_V2_CROSSCHECK_20261029.md](RECOGNIZER_V2_CROSSCHECK_20261029.md), [release_gate_v2.json](release_gate_v2.json).

## Summary

| Question | Result |
|---|---|
| Independent field video set | 5 YouTube videos screened and kept (2 harvest, 3 evaluation); no near-duplicate of the dataset or of V0 (pHash, Hamming <= 6) |
| Negative crops for a gate v2 | 1440 crops reviewed blind; 325 are not signs; 282 used to build the bank, 43 held out |
| Gate v2 | Built (TRAIN positives only); margin 0.24 chosen on VAL plus held-out negatives; no effect on VAL macro-F1 |
| Precision of confirmed labels, Hanoi, new model at 0.87 (B) | 0.967 [0.906, 1.000] |
| Precision, Hanoi, new model at T2 (C) | 0.863 [0.776, 0.934] |
| Precision, Hanoi, old model with old gate (A) | 0.671 [0.557, 0.779] |
| Gate v2 in the field (D vs C, E vs B) | identical results: the gate vetoed no confirmed label |
| Recommendation for T2 | **not recommended** (keep 0.87) |
| Recommendation for gate v2 | **not recommended** for both D and E (do not enable; not added to Git LFS) |

## 1. Videos

The five candidates named in the work order were all downloaded as 1080p H.264 (video stream only, no audio, no ffmpeg merge) and screened by looking at 12 frames each (5% to 95%) plus a detector pass for visible signs. None was rejected, so the replacement search was not used (the reserve candidate `zPLsq5Gbn-E` was only read for metadata). Media files are **not** committed: they sit in `output/kiet_wp_20261029/videos/` (inside the project, ignored by Git, SHA256 in `sources.csv`). The table is [sources.csv](../data/manifests/field_video_20261029/sources.csv).

| Video | Role | Channel | Uploaded | Duration (s) | Resolution / fps | City, time | Near-duplicate rate | Segments (start s) |
|---|---|---|---|---:|---|---|---:|---|
| [gMIThDmdPfU](https://www.youtube.com/watch?v=gMIThDmdPfU) | harvest | Vietnam street 4K | 20220106 | 938.7 | 1920x1080 @ 29.97 | Ho Chi Minh City (District 7), day | 0.0000 | all (0.5 s sampling) |
| [O-XWjoYscew](https://www.youtube.com/watch?v=O-XWjoYscew) | harvest | Vietnam street 4K | 20211004 | 1331.4 | 1920x1080 @ 23.976 | Ho Chi Minh City (District 1 to 7), night | 0.0000 | all (0.5 s sampling) |
| [6JuxWcAf-wM](https://www.youtube.com/watch?v=6JuxWcAf-wM) | eval | Vietnam Silent Routes | 20250113 | 3320.6 | 1920x1080 @ 60.0 | Hanoi, day | 0.0000 | 30, 830, 1630, 2430, 3230 |
| [YQl9c5gCjhg](https://www.youtube.com/watch?v=YQl9c5gCjhg) | eval | Vietnam Silent Routes | 20250124 | 2939.3 | 1920x1080 @ 60.0 | Hanoi, night | 0.0000 | 30, 734, 1439, 2144, 2849 |
| [bPFWcTjKdBU](https://www.youtube.com/watch?v=bPFWcTjKdBU) | eval | Raw Hanoi Sounds | 20260102 | 2444.7 | 1920x1080 @ 30.0 | Hanoi (West Lake), day | 0.0000 | 30, 611, 1192, 1773, 2354 |

| Video | a view | b speed | c resolution | d overlay | e signs | f duration |
|---|---|---|---|---|---|---|
| gMIThDmdPfU | pass | pass | pass | pass | pass, weakest | pass |
| O-XWjoYscew | pass | pass | pass, caveat | pass | pass | pass |
| 6JuxWcAf-wM | pass | pass | pass | pass | pass | pass |
| YQl9c5gCjhg | pass | pass | pass | pass | pass | pass |
| bPFWcTjKdBU | pass | pass | pass | pass | pass | pass |

Criteria: (a) forward view from a moving vehicle, (b) real speed, (c) real resolution at least 1080p, (d) no overlay covering the roadside, (e) real signs in at least 3 of 12 frames, (f) at least 8 minutes. Real resolution was checked with a detail ratio (Laplacian variance after a half-resolution down-up pass over the full-resolution variance): 0.104 to 0.221 for the candidates against 0.165 for V0 (real 1080p) and 0.377 for a deliberately upscaled 540p frame. Caveats: `gMIThDmdPfU` shows signs in only 3 of 12 frames; `O-XWjoYscew` is softer (24 fps, night, ratio 0.221); `bPFWcTjKdBU` is strongly colour-graded. Per-criterion reasons are in the `verdict_*` columns of `sources.csv`. Harvest and evaluation videos differ by city and channel.

**Duplicate check.** pHash (64 bit, `imagehash`) of one frame per 2 s of each video against all 15,603 dataset images (TRAIN, VAL and TEST) and against V0 frames every 2 s; a Hamming distance of at most 6 counts as near-duplicate. The limit for rejection was 1% of frames.

| Video | Frames hashed | Near-duplicates of dataset | Near-duplicates of V0 | Smallest distance to dataset | Smallest distance to V0 |
|---|---:|---:|---:|---:|---:|
| gMIThDmdPfU | 469 | 0 | 0 | 10 | 12 |
| O-XWjoYscew | 666 | 0 | 0 | 10 | 12 |
| 6JuxWcAf-wM | 1660 | 0 | 0 | 8 | 12 |
| YQl9c5gCjhg | 1470 | 0 | 0 | 10 | 14 |
| bPFWcTjKdBU | 1222 | 0 | 0 | 10 | 10 |

**Segments.** Each evaluation video uses five 60 s segments evenly spaced between 30 s after the start and 30 s before the end (the first starts at 30 s, the last ends 30 s before the end). The rule is fixed in `select_segments` and the starts are in `sources.csv`; they were fixed before any run. Harvest videos are used whole at one frame per 0.5 s.

## 2. Negative crops and reference gate v2

**Candidates.** Default detector (`finetuned_20261026.pt`, video setting 960 px, confidence >= 0.5) plus recognizer v2 without a gate (its raw output was stored but never shown to the reviewer). Video: one frame per 0.5 s of both harvest videos, repeats of the same box within 1 s dropped. TRAIN only: detector boxes that overlap no labelled box of any disposition (IoU < 0.1 and less than 30% covered). VAL, TEST and V0 were not used. All 1440 candidates were reviewed (fewer than the 1,500 allowed), 30 numbered contact sheets of 48 crops, reading only the crop against the QCVN reference table. Labels: `qcvn_sign:<code>`, `qcvn_sign_unknown_code`, `not_a_sign`, `uncertain`. Only `not_a_sign` crops are negatives. The manifest [gate_v2_negatives.jsonl](../data/manifests/field_video_20261029/gate_v2_negatives.jsonl) holds source references, boxes and labels, no pixels.

| Source | Reviewed crops | not_a_sign | qcvn_sign | qcvn_sign_unknown_code | uncertain |
|---|---:|---:|---:|---:|---:|
| video | 1037 | 268 | 325 | 250 | 194 |
| train | 403 | 57 | 103 | 103 | 140 |
| total | 1440 | 325 | 428 | 353 | 334 |

| not_a_sign crops | fit (bank) | held-out (margin) |
|---|---:|---:|
| video | 238 | 30 |
| train | 44 | 13 |
| total | 282 | 43 |

Observations: among the 403 detector boxes that overlap no label in TRAIN images, only 57 are not signs; the rest are real signs the dataset leaves unlabelled or crops too poor to read. Without a gate the recognizer v2 would accept 16 of the 325 negative crops at 0.87 and 37 at T2.

**Bank.** `scripts/build_sign_reference_gate.py` now accepts `--recognizer` (any checkpoint), `--boxes-csv`, `--negatives-jsonl` and `--video-root`; the old path is unchanged. Positives are TRAIN crops of the trained codes (<= 96 per code, >= 16 px); negatives are the fit split. The bank file is `models/sign_recognizer/reference_gate_20261029_v2.npz` (local only; SHA256 below). It was **not** added to Git LFS because section 4 does not recommend enabling it.

**Margin.** Chosen with `scripts/select_gate_margin_20261029.py` on VAL and the held-out negatives only: the margin with the highest block rate on the held-out negatives such that the VAL macro-F1 over the main classes falls by at most 0.005 at both T2 and 0.87; ties go to the larger margin.

| Item | Value |
|---|---|
| Positive references (TRAIN only) | 5639 |
| Negative references (fit split) | 282 (video 238, TRAIN 44) |
| Chosen margin | 0.24 (runtime default was 0.015) |
| VAL crops (>= 16 px, known classes plus background) | 3933 |
| VAL macro-F1 over main classes at T2, no gate / gate | 0.9484 / 0.9484 |
| VAL macro-F1 over main classes at 0.87, no gate / gate | 0.9027 / 0.9027 |
| VAL coverage of known crops at T2, no gate / gate | 0.9499 / 0.9499 |
| VAL coverage of known crops at 0.87, no gate / gate | 0.8678 / 0.8678 |
| Held-out negatives blocked | 0.372 (16 of 43) |
| Held-out negatives the recognizer accepts at T2 / blocked | 6 / 1 |
| Held-out negatives the recognizer accepts at 0.87 / blocked | 1 / 0 |

The gate vetoed no VAL crop at any margin from 0 to 0.3, so the F1 constraint never bound; the held-out block rate is flat (0.372) from margin 0 to 0.24 and drops to 0.349 above it, so the rule picks 0.24. The held-out set is small (43 crops, 6 accepted at T2, 1 at 0.87), so this selection rests on little evidence.

## 3. Field results

Configurations: A = `retrained_20261009.pt` at 0.63 with the old gate (margin 0.015); B = `retrained_20261028.pt` at 0.87, no gate; C = same model at T2; D = C plus gate v2; E = B plus gate v2 (margin 0.24). Same detector, tracker and pipeline in all runs (`scripts/benchmark_temporal_video_20261028.py`, analysis 6 Hz, three-frame confirmation, UFLD v2 lanes, offline legal retrieval). 75 runs of 60 s: 15 Hanoi segments x 5 configurations. A/B/C finished at 07:59:36 and D/E at 08:54:13 on 2026-10-29 (two runs in parallel; summed render wall time per configuration over its 15 runs: A 0.97 h, B 0.94 h, C 0.92 h, D 0.88 h, E 0.88 h).

**Blind review.** Every label confirmed after multi-frame confirmation is exported as its best crop; evidence shared by several configurations (same video, segment, track and best frame) is reviewed once and graded for each configuration, so all configurations are judged by the same reading. The reviewer saw numbered crops only (no configuration, prediction or time), read the sign, and the mapping was opened afterwards. Ambiguous crops were re-read on per-track context montages restricted to frames near the best crop. Verdicts: `correct` (same code), `wrong_code`, `not_a_sign` (including non-QCVN notices and shop signs), `uncertain` (the distinguishing feature is not visible). Precision = correct / (correct + wrong_code + not_a_sign); `uncertain` is left out and shown. Matching is by exact code, so W.207 against W.207b counts as `wrong_code`. 95% intervals resample the 60 s segments (15 for Hanoi, 5 for V0), which is optimistic because the segments are not independent signs. 3 labels were corrected after auditing the disagreements against the QCVN figures (list in `output/kiet_wp_20261029/review/label_corrections.json`; one reviewer misreading of the W.201 bend direction, one proxy-recall label made consistent with the W.207b readings).

**Hanoi EVAL, all videos**

| Config | Setting | Confirmed labels | correct | wrong_code | not_a_sign | uncertain | Precision [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| A | old 0.63 + old gate | 107 | 57 | 21 | 7 | 22 | 0.671 [0.557, 0.779] |
| B | new 0.87 | 75 | 59 | 0 | 2 | 14 | 0.967 [0.906, 1.000] |
| C | new T2 | 100 | 69 | 3 | 8 | 20 | 0.863 [0.776, 0.934] |
| D | new T2 + gate v2 | 100 | 69 | 3 | 8 | 20 | 0.863 [0.776, 0.934] |
| E | new 0.87 + gate v2 | 75 | 59 | 0 | 2 | 14 | 0.967 [0.906, 1.000] |

**Hanoi EVAL, day**

| Config | Setting | Confirmed labels | correct | wrong_code | not_a_sign | uncertain | Precision [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| A | old 0.63 + old gate | 61 | 29 | 12 | 5 | 15 | 0.630 [0.471, 0.780] |
| B | new 0.87 | 44 | 34 | 0 | 2 | 8 | 0.944 [0.833, 1.000] |
| C | new T2 | 59 | 39 | 2 | 6 | 12 | 0.830 [0.703, 0.943] |
| D | new T2 + gate v2 | 59 | 39 | 2 | 6 | 12 | 0.830 [0.703, 0.943] |
| E | new 0.87 + gate v2 | 44 | 34 | 0 | 2 | 8 | 0.944 [0.833, 1.000] |

**Hanoi EVAL, night**

| Config | Setting | Confirmed labels | correct | wrong_code | not_a_sign | uncertain | Precision [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| A | old 0.63 + old gate | 46 | 28 | 9 | 2 | 7 | 0.718 [0.529, 0.880] |
| B | new 0.87 | 31 | 25 | 0 | 0 | 6 | 1.000 [1.000, 1.000] |
| C | new T2 | 41 | 30 | 1 | 2 | 8 | 0.909 [0.818, 0.971] |
| D | new T2 + gate v2 | 41 | 30 | 1 | 2 | 8 | 0.909 [0.818, 0.971] |
| E | new 0.87 + gate v2 | 31 | 25 | 0 | 0 | 6 | 1.000 [1.000, 1.000] |

Errors on Hanoi (predicted code for blind reading): B: R.415a for not_a_sign x1, P.127 for not_a_sign x1. C: R.415a for not_a_sign x2, R.303 for not_a_sign x2, P.127 for not_a_sign x2, DP.134 for not_a_sign x1, W.207 for W.207b x1, P.103a for other_qcvn x1, P.103a for P.128 x1, P.102 for not_a_sign x1. A: 28 errors, 15 of them on signs that are not among its 30 classes (for example I.423 read as W.224 or R.303). D and E repeat C and B exactly.

**Run counts, Hanoi all**

| Config | Runs | Tracks | Confirmed pairs | Unknown samples | Alert samples | Alert episodes | Inference FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 15 | 1018 | 107 | 1747 | 541 | 116 | 2.96 |
| B | 15 | 1205 | 75 | 2116 | 444 | 71 | 3.04 |
| C | 15 | 1147 | 100 | 1894 | 471 | 81 | 3.09 |
| D | 15 | 1147 | 100 | 1894 | 471 | 81 | 3.26 |
| E | 15 | 1205 | 75 | 2116 | 444 | 71 | 3.26 |

**Run counts, day**

| Config | Runs | Tracks | Confirmed pairs | Unknown samples | Alert samples | Alert episodes | Inference FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 10 | 575 | 61 | 1003 | 381 | 63 | 3.01 |
| B | 10 | 703 | 44 | 1237 | 294 | 26 | 3.17 |
| C | 10 | 672 | 59 | 1114 | 308 | 34 | 3.21 |
| D | 10 | 672 | 59 | 1114 | 308 | 34 | 3.47 |
| E | 10 | 703 | 44 | 1237 | 294 | 26 | 3.47 |

**Run counts, night**

| Config | Runs | Tracks | Confirmed pairs | Unknown samples | Alert samples | Alert episodes | Inference FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 5 | 443 | 46 | 744 | 160 | 53 | 2.85 |
| B | 5 | 502 | 31 | 879 | 150 | 45 | 2.80 |
| C | 5 | 475 | 41 | 780 | 163 | 47 | 2.86 |
| D | 5 | 475 | 41 | 780 | 163 | 47 | 2.91 |
| E | 5 | 502 | 31 | 879 | 150 | 45 | 2.91 |

"Unknown samples" are displayed unrecognised detections per sampled frame, not unique signs; alert counts are the pipeline's alert samples and episodes. Inference FPS is samples per second inside `Pipeline.run`, with two runs sharing the GPU.

### V0 (development video, TEST group)

The 158 track-labels of the existing review pack (3 configurations, 5 segments of 60 s) were read blind first, then graded. **V0 is one TEST group of dataset_v2, and the old gate was built with 30 negative crops taken from V0, so the comparison on V0 favours A.**

**V0 (development video, TEST group)**

| Config | Setting | Confirmed labels | correct | wrong_code | not_a_sign | uncertain | Precision [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| A | old 0.63 + old gate | 46 | 31 | 6 | 0 | 9 | 0.838 [0.778, 0.955] |
| B | new 0.87 | 54 | 46 | 2 | 0 | 6 | 0.958 [0.895, 1.000] |
| C | new T2 | 58 | 49 | 3 | 1 | 5 | 0.925 [0.845, 1.000] |

V0 errors: A: P.130 for P.131a x3, P.124a for P.124b x2, W.224 for W.225 x1. B: P.130 for P.131a x2. C: P.130 for P.131a x2, P.102 for not_a_sign x1, P.124a for P.124b x1. P.124a for P.124b is a confusion inside one sign family (all vehicles against cars only).

### Proxy recall

A light proxy, not a standard recall. For each Hanoi segment, 12 frames (one per 5 s, centred in each step so a +-2 s window stays in the segment) were run through the pipeline detector; its proposals (confidence >= 0.35, short side >= 24 px, 85 proposals) were read blind and labelled with a QCVN code, `not_sign` or `unreadable`. 46 readable coded signs remain. A sign counts as caught if a track overlapping it (IoU >= 0.2) is confirmed with that code at some sample within +-2 s. Signs the detector never proposes are not counted, so this measures recognition and confirmation given a detection, and the same detector serves every configuration.

| Config | Annotated signs | Caught | Proxy recall |
|---|---:|---:|---:|
| A | 46 | 29 | 0.630 |
| B | 46 | 34 | 0.739 |
| C | 46 | 35 | 0.761 |
| D | 46 | 35 | 0.761 |
| E | 46 | 34 | 0.739 |

The difference between B and C is one sign out of 46; it carries no weight.

## 4. Recommendations under the pre-registered rules

Rules fixed in the work order (applied by `decide_recommendations`, minimum 30 decidable labels per configuration, otherwise "not enough evidence"):

* **T2 instead of 0.87** if precision(C) >= precision(B) - 0.02, and the wrong labels (wrong_code + not_a_sign) of C do not exceed those of A, and proxy recall(C) > proxy recall(B).
* **Enable gate v2** if, against the same threshold without the gate, wrong_code + not_a_sign fall by at least 30%, correct labels fall by at most 5%, and VAL macro-F1 falls by at most 0.005.

**Hanoi EVAL + V0 combined (rule input)**

| Config | Setting | Confirmed labels | correct | wrong_code | not_a_sign | uncertain | Precision [95% CI] |
|---|---|---:|---:|---:|---:|---:|---|
| A | old 0.63 + old gate | 153 | 88 | 27 | 7 | 31 | 0.721 [0.626, 0.803] |
| B | new 0.87 | 129 | 105 | 2 | 2 | 20 | 0.963 [0.920, 1.000] |
| C | new T2 | 158 | 118 | 6 | 9 | 25 | 0.887 [0.825, 0.942] |

| Rule | Inputs | Checks | Decision |
|---|---|---|---|
| T2, Hanoi + V0 combined | precision B 0.963, C 0.887; wrong labels A 34, C 15; proxy recall B 0.739, C 0.761 | precision within 0.02: NO, errors not above A: yes, proxy recall above B: yes | **not recommended** |
| T2, Hanoi only | precision B 0.967, C 0.863; wrong labels A 28, C 11 | precision within 0.02: NO, errors not above A: yes, proxy recall above B: yes | **not recommended** |
| Gate v2, D against C (Hanoi) | wrong labels 11 to 11; correct 69 to 69; VAL F1 drop 0.0000 | errors drop 30pct: NO, correct loss within 5pct: yes, val f1 drop within 0.005: yes | **not recommended** |
| Gate v2, E against B (Hanoi) | wrong labels 2 to 2; correct 59 to 59; VAL F1 drop 0.0000 | errors drop 30pct: NO, correct loss within 5pct: yes, val f1 drop within 0.005: yes | **not recommended** |

Reading for the owner (measured results first, then inference):

* **Measured.** At 0.87 the new model confirmed fewer labels than at T2 (129 against 158 on Hanoi + V0) and was clearly more precise (0.963 against 0.887); C has 15 wrong labels against 4 for B, and 9 of the 15 are codes read on objects the review judged not to be signs (for example R.415a, R.303, P.127, DP.134 or P.102). Both new-model settings are far more precise than the old model with its gate on these videos, but that gap is inflated by signs outside the old 30 classes. D and E are identical to C and B: the gate vetoed nothing.
* **Why the gate did nothing (measured).** The 11 non-sign crops confirmed in the field have a highest similarity of 0.689 to the negative bank, below the gate's floor of 0.80, so it abstains whatever the margin (0 vetoed at 0.24, 0 at 0.015). The bank comes from Ho Chi Minh City videos and TRAIN images; the Hanoi false confirmations are different objects.
* **Inference.** Keeping 0.87 is the safer setting for precision; T2 trades precision for coverage that this evidence does not show to be worth it. A gate needs negatives that resemble the false confirmations of the target scenes; this bank did not transfer across cities. Neither statement is a safety claim.

## 5. Limitations

* **AI-reviewed labels, one reviewer.** The same model family labelled training data and reviewed these results, so shared blind spots are possible; there is no second reader and no human check. Agreement with the recognizer is not accuracy.
* The earlier crosscheck's 100-crop control by another AI reviewer failed (29% agreement) because crops were judged in isolation; here ambiguous tracks were read on context montages, but this reviewer has no such control, and 14 B labels and 20 C labels stayed `uncertain` and are excluded from precision.
* One physical sign appears in many frames and in several configurations; the intervals resample segments and are optimistic. The Hanoi set is 15 segments of three videos from one city and similar cameras; one video is heavily colour-graded.
* Exact-code matching counts W.207 against W.207b and P.124a against P.124b as wrong, which is strict for sign families that differ by a letter; all configurations are treated alike.
* Proxy recall rests on 46 signs and on detector proposals; it is not a recall.
* V0 belongs to the TEST group and its frames fed the old gate, so V0 numbers are reporting only and favour A. Gate configurations D and E were run on Hanoi only (the V0 pack has A, B and C), so the gate rule uses Hanoi.
* The gate margin was selected on 43 held-out negatives; the held-out and fit splits are by video segment or TRAIN image.
* The old model's errors include signs outside its 30 classes (15 of 28 on Hanoi), so A against B/C mixes vocabulary and calibration.
* Process notes: two jobs were stopped and rerun from scratch when the videos were moved into the project (their outputs were deleted first, no result was lost); `benchmark_temporal_video_20261028.py` gained `--reference-gate-margin` after the first A/B/C runs (defaults unchanged), so the runner hash differs between runs. Hanoi night B has an all-correct sample, so its interval is degenerate.
* No human ground truth, no legal-currency check and no safety claim; release remains HOLD.

## 6. Reproduce

```powershell
$env:PYTHONNOUSERSITE=1; $env:PYTHONPATH="<repo root>"; $env:YOLO_OFFLINE="true"
python scripts/field_video_eval_20261029.py make-sources ...            # sources.csv from yt-dlp metadata, screening verdicts, SHA256, pHash
python scripts/harvest_gate_negatives_20261029.py scan-video|scan-train|build-review|finalize ...
python scripts/build_sign_reference_gate.py --recognizer models/sign_recognizer/retrained_20261028.pt --boxes-csv ... --negatives-jsonl ... --video-root ... --output ...
python scripts/select_gate_margin_20261029.py --gate ... --negatives-jsonl ... --video-root ... --out ...
python scripts/run_field_eval_20261029.py --jobs 2 [--configs D E --gate-v2 ...]
python scripts/field_review_pack_20261029.py build|montage|merge ...      # blind review packs and evidence-keyed label store
python scripts/field_proxy_pack_20261029.py build ...
python scripts/field_video_report_20261029.py --field-eval ... --store ... --v0-graded ... --proxy-items ... --margin ... --out-json ... --out-md ...
```

Videos, frames, crops, contact sheets, run folders and labels stay under `output/` (not tracked); all temporary files were kept on drive D.

## 7. Provenance

| Artifact | SHA256 |
|---|---|
| data/manifests/field_video_20261029/sources.csv | `fb4e1837638cb9c001ecc5e51a5073cf789d1856a3c8175c5458426fbe4b8bd5` |
| data/manifests/field_video_20261029/gate_v2_negatives.jsonl | `02d039ee71552ca3867b6ee66814ce2f639f10c342a2114b4c543a596813f720` |
| models/sign_recognizer/reference_gate_20261029_v2.npz (local) | `a54d79a99dc008ca422b44908c48af04a056299cd296f4a2345e6d09ddd2a23e` |
| models/sign_recognizer/retrained_20261028.pt | `01d5c677af749ae14170b8008d304fb852f8c0baa7a606a3101c69e2a6b80d30` |
| output/.../review/label_store.json (local) | `4372edad6fcdd7644110d34e17208f0b25fea1d2262325cae1447ed28bd36144` |
| output/.../v0_review/blind_labels.json (local) | `916235804d0654dc07dcd7739c7b45791c648eac9c0e21584ed38bfd2e342773` |
| output/.../proxy_recall/blind_labels.json (local) | `3d2c7543b9e3ec6821cc3acafc680294c1147aee2427700133313be17c77a48b` |
| output/.../gate_v2/margin_selection.json (local) | `82c669373355c0a18a8462616153fca747a3873ce4b7af7d9ce70253de266794` |
| output/.../summary_all.json (local) | `678bca26f05223b287c1e21cfb32d7a214df439cb85ba13b48371c78a7979264` |
