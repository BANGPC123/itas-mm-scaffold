# Recognizer v2 independent crosscheck

Run started 2026-10-28 (Asia/Saigon). The requested `20261029` suffix and sampling seed are identifiers.

Evidence snapshot; NOT an approved release or human ground truth. TEST is for reporting only. The model, default thresholds and gate configuration were not changed.

## Part 0: source and baseline

- Evidence HEAD: `8d04c6771078d429240944e1a17d51a2487d7171`. Model SHA256: `01d5c677af749ae14170b8008d304fb852f8c0baa7a606a3101c69e2a6b80d30`; LFS and manifest agree.
- Default pytest: 498 passed, 11 skipped, 38 deselected. Local acceptance 11/11, Node 7/7 and frontend build pass.
- After the code changes, full pytest: 506 passed, 11 skipped, 38 deselected; local acceptance: 11/11 passed.
- Manifest `--verify` passes, but this invocation checks zero canonical files and does not rehash or verify the pilot.
- Current training code loads the entire boxes CSV, including TEST metadata, before selecting TRAIN and VAL crops. Epoch selection and threshold calibration use VAL in this code; that is different from saying the script never reads TEST metadata. See [loading](../scripts/train_recognizer_v2_20261028.py#L88), [checkpoint selection](../scripts/train_recognizer_v2_20261028.py#L115), and [calibration](../scripts/train_recognizer_v2_20261028.py#L123).
- The training log records HEAD `8534843`, at which the training script was not tracked. No script SHA was captured, so byte-identical source and actual training inputs are not independently proven.
- Current background logic allows small intersections: 14 random crops intersect OUT_OF_ONTOLOGY boxes and 22 intersect UNREADABLE boxes; 35 AI-not-sign crops coincide with OUT_OF_ONTOLOGY, two of which also intersect UNREADABLE. A positive intersection does not itself prove an incorrect background label. See [overlap thresholds](../scripts/build_recognizer_v2_dataset_20261028.py#L22) and [AI-not-sign inclusion](../scripts/build_recognizer_v2_dataset_20261028.py#L66).
- The 30 prior code indices are unchanged; the background index moved from 30 to 87. The current script copies matching head rows by label at [initialisation](../scripts/train_recognizer_v2_20261028.py#L101).
- The checkpoint threshold overrides the YAML threshold at load time. The current video path permits 12 px crops, but merged two-scale decisions for crops below 16 px also require agreeing labels, confidence >=0.95 and detector confidence >=0.5.
- The two-pass extension has 1,531 rows: agreed codes TRAIN 553, VAL 120, TEST 303; 35 both not-sign, 166 both uncertain, 354 disagree, raw agreement 76.9%. Both passes used the same model.
- No identical image SHA or split-constraint group was found between TRAIN and TEST. No shared IMG-named clip was found across all recorded occurrence paths, but other source naming schemes are not covered.
- All 75 R.301c TEST crops visibly show the same arrow sign and construction-site context; 43 come from IMG_4032 with 19 distinct source-frame indices. Whether the single TRAIN crop is the same physical sign remains UNVERIFIED.
- Rerunning the evaluator reproduced the recorded raw-argmax metrics and CIs exactly. It reused the existing cache; 100 random TEST cached crops matched independently reconstructed source crops.

| TEST old-code size | n | Old argmax accuracy | New argmax accuracy | New - old 95% CI |
|---|---:|---:|---:|---:|
| <16 | 1078 | 0.5139 | 0.7839 | [+0.2379, +0.3017] |
| 16-32 | 811 | 0.9309 | 0.9815 | [+0.0336, +0.0665] |
| 32-64 | 753 | 0.9150 | 0.9628 | [+0.0328, +0.0664] |
| >=64 | 502 | 0.9701 | 0.9801 | [-0.0058, +0.0262] |
| >=32 | 1255 | 0.9371 | 0.9697 | [+0.0209, +0.0452] |
| all | 3144 | 0.7904 | 0.9090 | [+0.1051, +0.1325] |

## Part 1: blinded AI crosscheck - STOP

The predeclared 100-crop old-code control produced 29 exact agreements and 71 uncertain decisions: 29.0% agreement over all controls, below the 85% stop threshold. No independent accuracy, class-removal recommendation or video precision is inferred from this review. The 631 blinded decisions are kept only as local, explicitly unvalidated evidence; they are not a published dataset manifest and are not human GT.

## Part 2: thresholded crop operating points

Cells show coverage / accepted accuracy / end-to-end accuracy / wrong-label rate. All are TEST reports against AI-assisted labels; no runtime selection uses these results. The old gate is the 2026-10-11 bank, margin 0.015. These are GT crops, and do not include all runtime filters. The old model cannot output the new class codes; its coverage for those rows counts returned old codes, which are necessarily incorrect. Untrained TEST codes are I.401: 7, W.205c: 2 crops, and are outside these three reported categories. The old-gate replay used batch 256 and reproduced all 4,077 saved old argmax labels; an initial batch-64 replay differed on one argmax.

| Class group | Crop size | n | Old 0.63 | Old 0.63 + gate | New 0.87 | New T2 |
|---|---|---:|---|---|---|---|
| old30 | <16 | 1078 | 0.5492 / 0.8615 / 0.4731 / 0.0761 | 0.5473 / 0.8610 / 0.4712 / 0.0761 | 0.5455 / 0.9677 / 0.5278 / 0.0176 | 0.7208 / 0.9292 / 0.6698 / 0.0510 |
| old30 | 16-32 | 811 | 0.9531 / 0.9677 / 0.9223 / 0.0308 | 0.9531 / 0.9677 / 0.9223 / 0.0308 | 0.9063 / 0.9973 / 0.9038 / 0.0025 | 0.9704 / 0.9936 / 0.9642 / 0.0062 |
| old30 | 32-64 | 753 | 0.9695 / 0.9370 / 0.9084 / 0.0611 | 0.9681 / 0.9369 / 0.9070 / 0.0611 | 0.8765 / 0.9848 / 0.8632 / 0.0133 | 0.9548 / 0.9791 / 0.9349 / 0.0199 |
| old30 | >=64 | 502 | 0.9821 / 0.9838 / 0.9661 / 0.0159 | 0.9821 / 0.9838 / 0.9661 / 0.0159 | 0.9442 / 0.9937 / 0.9382 / 0.0060 | 0.9721 / 0.9877 / 0.9602 / 0.0120 |
| old30 | >=32 | 1255 | 0.9745 / 0.9558 / 0.9315 / 0.0430 | 0.9737 / 0.9558 / 0.9307 / 0.0430 | 0.9036 / 0.9885 / 0.8932 / 0.0104 | 0.9618 / 0.9826 / 0.9450 / 0.0167 |
| new_main | <16 | 2 | 0.5000 / 0.0000 / 0.0000 / 0.5000 | 0.5000 / 0.0000 / 0.0000 / 0.5000 | 1.0000 / 1.0000 / 1.0000 / 0.0000 | 1.0000 / 1.0000 / 1.0000 / 0.0000 |
| new_main | 16-32 | 106 | 0.5849 / 0.0000 / 0.0000 / 0.5849 | 0.5566 / 0.0000 / 0.0000 / 0.5566 | 0.7736 / 0.9268 / 0.7170 / 0.0566 | 0.9151 / 0.8866 / 0.8113 / 0.1038 |
| new_main | 32-64 | 208 | 0.5000 / 0.0000 / 0.0000 / 0.5000 | 0.4375 / 0.0000 / 0.0000 / 0.4375 | 0.7404 / 0.8961 / 0.6635 / 0.0769 | 0.9279 / 0.8601 / 0.7981 / 0.1298 |
| new_main | >=64 | 194 | 0.5464 / 0.0000 / 0.0000 / 0.5464 | 0.5361 / 0.0000 / 0.0000 / 0.5361 | 0.6289 / 0.7705 / 0.4845 / 0.1443 | 0.8144 / 0.7975 / 0.6495 / 0.1649 |
| new_main | >=32 | 402 | 0.5224 / 0.0000 / 0.0000 / 0.5224 | 0.4851 / 0.0000 / 0.0000 / 0.4851 | 0.6866 / 0.8406 / 0.5771 / 0.1095 | 0.8731 / 0.8319 / 0.7264 / 0.1468 |
| low_support | <16 | 12 | 0.5833 / 0.0000 / 0.0000 / 0.5833 | 0.5833 / 0.0000 / 0.0000 / 0.5833 | 0.0000 / - / 0.0000 / 0.0000 | 0.0000 / - / 0.0000 / 0.0000 |
| low_support | 16-32 | 102 | 0.5686 / 0.0000 / 0.0000 / 0.5686 | 0.5686 / 0.0000 / 0.0000 / 0.5686 | 0.5588 / 0.9825 / 0.5490 / 0.0098 | 0.6961 / 0.9577 / 0.6667 / 0.0294 |
| low_support | 32-64 | 137 | 0.5401 / 0.0000 / 0.0000 / 0.5401 | 0.5401 / 0.0000 / 0.0000 / 0.5401 | 0.5620 / 0.8442 / 0.4745 / 0.0876 | 0.6569 / 0.8667 / 0.5693 / 0.0876 |
| low_support | >=64 | 163 | 0.3804 / 0.0000 / 0.0000 / 0.3804 | 0.3742 / 0.0000 / 0.0000 / 0.3742 | 0.4663 / 0.8947 / 0.4172 / 0.0491 | 0.5583 / 0.8462 / 0.4724 / 0.0859 |
| low_support | >=32 | 300 | 0.4533 / 0.0000 / 0.0000 / 0.4533 | 0.4500 / 0.0000 / 0.0000 / 0.4500 | 0.5100 / 0.8693 / 0.4433 / 0.0667 | 0.6033 / 0.8564 / 0.5167 / 0.0867 |

VAL-only selection uses known crops with short side >=16 px. T2 = 0.6412901874; criterion: coverage_target.

| VAL threshold | Accepted / known | Coverage | Accepted accuracy | Wrong | Background accepted / total |
|---|---:|---:|---:|---:|---:|
| 0.870000 | 2716 / 3131 | 0.8675 | 0.9901 | 27 | 0 / 802 |
| 0.641290 | 2975 / 3131 | 0.9502 | 0.9792 | 62 | 0 / 802 |

## Part 3: reference gate v2 - BLOCKED

Every prescribed 60-second harvest segment contains six pilot TEST frames. The instruction prohibits any harvest segment containing a TEST frame, so no approved V0 harvest segment is available. A new independent video or a revised sampling rule requires owner direction. No v2 bank, margin, or model selection was produced.

## Part 4: video measurements

The selected V0 evaluation segments overlap the development pilot TEST video and are descriptive only. No event GT exists; precision and recall are BLOCKED. A and B/C use identical instrumented protocols; C uses T2. D is BLOCKED because gate v2 could not be constructed. Counts of unknown are sampled detection displays, not unique signs.

| Config | Completed segments | Tracks | Confirmed track-label pairs | Unknown samples | Alert episodes | Inference FPS | Bus matches |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 5/5 | 326 | 46 | 634 | 104 | 3.014 | 0 |
| B | 5/5 | 323 | 54 | 586 | 101 | 3.317 | 9 |
| C | 5/5 | 303 | 58 | 507 | 102 | 3.313 | 9 |
| D | 0/5 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

Config B pre-temporal 12-16 px: 0 / 42 returned a code (rate 0.0000); repeated samples are not independent. The multiscale tiny-crop rule also requires agreement and confidence >=0.95. Original training crops were >=16 px, but training augmentation simulated lower resolutions, including 10 and 14 px.

## Part 5: legal locator

- I.423: Phụ lục E, mục E.23a (biển số I.423) (`qcvn-41-2024-bgtvt:259:duefbo`).
- P.103c: Phụ lục B, mục B.3 (biển số P.103c) (`qcvn-41-2024-bgtvt:175:e1z61f`).
- P.127c: Phụ lục B, mục B.27c (biển số P.127c) (`qcvn-41-2024-bgtvt:192:yhixy2`).

The regression comparison retained all 84 previously resolved locators. The corpus and legal currency still require verification; release remains HOLD.

## Decision for the owner

- Keep 0.87 pending an explicit tradeoff decision. On VAL, it accepts fewer known crops but makes fewer wrong accepted assignments than T2; both reject all 802 sampled VAL background crops. The background set is not independently human-reviewed: three random VAL crops overlap excluded boxes, and two AI-not-sign VAL crops come from excluded boxes. This does not establish field safety.
- Do not enable a new gate: the permitted negative-harvest protocol is blocked and its effectiveness has not been measured.
- Do not remove or merge extension classes based on the failed blinded control. Obtain a qualified independent review and source-disjoint evaluation before making class decisions.

## Recorded execution time

- Baseline full pytest: 21.61 seconds. Final full pytest: 22.69 seconds; final separate acceptance: 2.71 seconds.
- Blinded decisions were written over approximately 4 minutes 28 seconds, based on local file timestamps; this is an approximate wall interval, not a per-crop timing.
- Old gate replay: approximately 8.24 seconds for the successful batch-256 run. A failed preliminary batch-64 replay and a corrected local script attempt are excluded.
- Video A: 5/5 segments, 928.7 seconds total including process startup, render and verification.
- Video B: 5/5 segments, 846.9 seconds total including process startup, render and verification.
- Video C: 5/5 segments, 847.0 seconds total including process startup, render and verification.
- Rerun evaluation, locator investigation and documentation time were not recorded separately.

## Local evidence hashes

These files are local ignored artifacts and are not on GitHub.

| Artifact | SHA256 |
|---|---|
| output/kiet_wp_20261028/codex_crosscheck_20261029/phase0_results.json | 49f6b2ac97fc486536347f5ceb0eadf47eb53a7a964f88309ff45698ea4330bc |
| output/kiet_wp_20261028/codex_crosscheck_20261029/final_checks.json | e5c0addec400de4feab7e3c8c9bcaffbe0d38ad03c27d5b5ba9437dd176669f8 |
| output/kiet_wp_20261028/codex_crosscheck_20261029/blind_control_check.json | 6a4b047295f81d7d492b05ce40ca298c53aad146b48d837d303dff607400d399 |
| output/kiet_wp_20261028/codex_crosscheck_20261029/operating_points_final.json | c52aa7d1f887ab5db51c1f6c2142382e95f5bf7df8984c1da5456814a3698789 |
| output/kiet_wp_20261028/codex_crosscheck_20261029/val_operating_points.json | ccb4266f91250e32a732f9521cbfe7781fa99b49870ca69d566c61244192696c |
| output/kiet_wp_20261028/codex_crosscheck_20261029/legal_after.json | fb01b5d7dd59d7408fa0f95ad42d073420d740882015bb2f3ff4f75c70087355 |
| output/kiet_wp_20261028/wp3/codex_20261029/summary_raw_counts.json | 7be0f12d14ef329b6b1923e41c7fddb52d65c246c154ae806b811230affc8b13 |

All artifacts under `output/` are local and ignored by Git; no images or videos are committed. The local evidence uses AI-reviewed labels and does not establish human ground truth.
