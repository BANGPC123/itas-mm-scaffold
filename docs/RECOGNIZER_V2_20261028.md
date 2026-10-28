# Recognizer v2 (2026-10-28)

Retrained sign recognizer with the V1.2 ontology extended by AI-reviewed labels, small-crop training (16 px and up) and 87 QCVN codes plus a background class. Work package of issue #2, branch `work/kiet-recognizer-v2` (stacked on `work/kiet-pipeline-fixes`, PR #4).

**Status: accepted as the default recognizer under the owner's acceptance rule, with the limits below.** `configs/perception.yaml` now points to `models/sign_recognizer/retrained_20261028.pt` (SHA256 `01d5c677af749ae14170b8008d304fb852f8c0baa7a606a3101c69e2a6b80d30`, Git LFS, listed in `docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json`). The previous `retrained_20261009.pt` and its reference gate stay on disk for rollback.

## What these numbers are not

- **No human ground truth.** Every label used here is AI-assisted: released V1.2 codes come from earlier AI work, and the extension labels come from two AI review passes by the same model (`ai-review (Claude Code)`, `provenance=ai_reviewed`). TEST labels are of the same kind. Agreement between passes measures consistency, not correctness.
- TEST crops are ground-truth boxes, not detector output: these are classifier numbers, not end-to-end sign recognition.
- TEST is a frozen evaluation split used in earlier phases, not a pristine holdout. Nothing in this work (checkpoint, threshold, class list, data) was selected on TEST.
- Many crops are consecutive video frames of the same physical sign. Confidence intervals resample images, not physical signs, so they are optimistic.

## 1. Label source: AI review of boxes outside the ontology

Input: the 1,531 boxes with `semantic_final_disposition = EXCLUDED_OUT_OF_ONTOLOGY_V1_2` (TRAIN 898, VAL 142, TEST 491). The 3,703 `EXCLUDED_UNREADABLE_V1_2` boxes were not used for recognition or as background.

Process (`scripts/build_qcvn_reference_table_20261028.py`, `scripts/build_ontology_review_sheets_20261028.py`, `scripts/record_ontology_ai_labels_20261028.py`):

1. Reference table: every sample-sign figure ("Hình B.30 - Biển số P.130") was cropped from the SHA-tracked QCVN 41:2024 PDF with pypdfium2: 188 of 188 captioned figures, none missing, yielding 349 code strings (a few are artefacts of figure labels, for example `W.245h`).
2. Crops with a 15% margin were tiled 20 per numbered contact sheet (77 sheets). Neither the recognizer, the detector nor any earlier label was shown or used.
3. Pass A: sheets in image order. Pass B: relabelled from scratch with a shuffled box order (seed 20261028) and different sheet grouping, without opening pass A's file. Each box got a QCVN code, `not_qcvn_sign` or `uncertain`.
4. A box is kept only if both passes give the same QCVN code. Output: `data/manifests/dataset_v2/ai_ontology_extension_labels.jsonl` (SHA256 `090f17b4e1a7c3fa90adf450c4aeb5fe97acbbd3087f6e9fdb41c1c101058cd9`; one record per box with both passes, `agreed` and `qcvn_ref`; no image bytes). `scripts/build_dataset_v2_manifest.py --verify` still passes.

| Split | Boxes | Agreed QCVN code | Both not_qcvn_sign | Both uncertain | Disagree | Raw agreement |
|---|---|---|---|---|---|---|
| TRAIN | 898 | 553 | 26 | 106 | 213 | 76.3% |
| VAL | 142 | 120 | 2 | 4 | 16 | 88.7% |
| TEST | 491 | 303 | 7 | 56 | 125 | 74.5% |
| ALL | 1531 | 976 | 35 | 166 | 354 | 76.9% |

Of the disagreements, 78 are two different codes (most often R.412 vs R.415a, W.235 vs W.246a, P.127c vs R.415a, R.411 vs R.415a) and 266 are a code in one pass against `uncertain` or `not_qcvn_sign` in the other; 10 are `not_qcvn_sign` against `uncertain`. Disagreeing boxes are dropped, not adjudicated.

Agreed boxes by code:

| Code | TRAIN | VAL | TEST | Total |
|---|---|---|---|---|
| P.105 | 105 | 39 | 40 | 184 |
| R.420 | 97 | 27 | 37 | 161 |
| R.421 | 68 | 19 | 26 | 113 |
| W.203a | 49 | 15 | 43 | 107 |
| R.301c | 1 | 0 | 75 | 76 |
| R.415a | 26 | 0 | 27 | 53 |
| P.124e | 36 | 9 | 0 | 45 |
| P.127c | 33 | 0 | 4 | 37 |
| I.442 | 20 | 8 | 8 | 36 |
| R.412 | 20 | 0 | 9 | 29 |
| W.246a | 11 | 2 | 7 | 20 |
| W.205b | 13 | 1 | 1 | 15 |
| DP.135 | 4 | 0 | 8 | 12 |
| R.411 | 10 | 0 | 0 | 10 |
| R.302a | 9 | 0 | 0 | 9 |
| P.131a | 8 | 0 | 0 | 8 |
| I.401 | 0 | 0 | 7 | 7 |
| W.205a | 4 | 0 | 3 | 7 |
| P.103b | 3 | 0 | 3 | 6 |
| W.239b | 4 | 0 | 0 | 4 |
| W.219 | 4 | 0 | 0 | 4 |
| I.409 | 4 | 0 | 0 | 4 |
| W.239a | 3 | 0 | 0 | 3 |
| P.112 | 3 | 0 | 0 | 3 |
| P.103c | 3 | 0 | 0 | 3 |
| W.244 | 2 | 0 | 1 | 3 |
| I.402 | 2 | 0 | 0 | 2 |
| P.110a | 2 | 0 | 0 | 2 |
| P.104 | 2 | 0 | 0 | 2 |
| W.205c | 0 | 0 | 2 | 2 |
| I.423 | 0 | 0 | 1 | 1 |
| R.303 | 1 | 0 | 0 | 1 |
| R.301a | 1 | 0 | 0 | 1 |
| P.107a | 1 | 0 | 0 | 1 |
| P.124a | 1 | 0 | 0 | 1 |
| W.201b | 0 | 0 | 1 | 1 |
| W.233 | 1 | 0 | 0 | 1 |
| W.208 | 1 | 0 | 0 | 1 |
| W.222a | 1 | 0 | 0 | 1 |

Names in the manifest come from the QCVN heading that covers the code. For a code inside a group heading (for example W.203 a,b,c) `qcvn_ref.name_scope` is `group_heading`: the name is the group's, not the variant's.

## 2. Classes

Codes are merged from released V1.2 boxes (all sizes) and the agreed extension. A crop counts when its short side is at least 16 px.

- **Main** (at least 30 TRAIN and 5 VAL crops): 55 classes.
- **Low-support** (at least 1 TRAIN crop, otherwise below the main rule): 32 classes. They are trained but reported separately.
- **Not trained** (no TRAIN crop of 16 px or more): 2 codes, `W.205c` (2 TEST boxes) and `I.401` (7 TEST boxes).
- Output classes: 88 (87 codes plus `unknown` background). The 30 previous codes keep their position, so 31 head rows (30 codes and `unknown`) are copied from the old checkpoint.

Columns are box counts: TRAIN, VAL and TEST at 16 px and up, then all sizes.

### Main classes

| Code | Origin | TRAIN >=16px | VAL >=16px | TEST >=16px | TRAIN all | VAL all | TEST all |
|---|---|---|---|---|---|---|---|
| P.102 | old 30 | 331 | 68 | 47 | 912 | 229 | 159 |
| P.103a | new | 144 | 9 | 71 | 144 | 9 | 71 |
| P.105 | new | 105 | 39 | 40 | 105 | 39 | 40 |
| P.106a | new | 400 | 35 | 98 | 403 | 35 | 98 |
| P.107a | new | 53 | 5 | 7 | 53 | 5 | 7 |
| P.117 | old 30 | 257 | 49 | 26 | 310 | 70 | 51 |
| P.123a | old 30 | 230 | 34 | 34 | 280 | 47 | 45 |
| P.123b | old 30 | 217 | 53 | 71 | 316 | 71 | 97 |
| P.124a | old 30 | 153 | 77 | 18 | 211 | 82 | 24 |
| P.124b | new | 65 | 12 | 8 | 65 | 12 | 8 |
| P.124c | new | 96 | 16 | 6 | 96 | 16 | 6 |
| P.124d | old 30 | 120 | 19 | 17 | 120 | 19 | 17 |
| P.124e | new | 36 | 9 | 0 | 36 | 9 | 0 |
| P.125 | old 30 | 88 | 29 | 23 | 88 | 29 | 23 |
| P.127 | old 30 | 1915 | 529 | 566 | 2918 | 816 | 920 |
| P.128 | new | 88 | 6 | 14 | 89 | 6 | 14 |
| P.130 | old 30 | 1350 | 328 | 251 | 1973 | 469 | 405 |
| P.131a | old 30 | 1178 | 302 | 302 | 1543 | 351 | 363 |
| P.131b | new | 41 | 6 | 1 | 41 | 6 | 1 |
| P.131c | new | 61 | 23 | 14 | 61 | 23 | 14 |
| P.137 | old 30 | 120 | 27 | 80 | 155 | 36 | 87 |
| P.139 | new | 48 | 6 | 0 | 48 | 6 | 0 |
| DP.134 | old 30 | 95 | 22 | 24 | 95 | 22 | 24 |
| R.122 | old 30 | 75 | 25 | 27 | 75 | 25 | 27 |
| R.302a | old 30 | 214 | 66 | 48 | 644 | 192 | 169 |
| R.303 | old 30 | 256 | 28 | 20 | 325 | 36 | 35 |
| R.420 | new | 97 | 27 | 37 | 97 | 27 | 37 |
| R.421 | new | 68 | 19 | 26 | 68 | 19 | 26 |
| W.201a | old 30 | 151 | 163 | 30 | 199 | 205 | 49 |
| W.201b | new | 163 | 22 | 5 | 256 | 72 | 5 |
| W.202 | new | 136 | 25 | 14 | 136 | 25 | 14 |
| W.203a | new | 49 | 15 | 43 | 49 | 15 | 43 |
| W.203c | old 30 | 153 | 19 | 13 | 173 | 21 | 16 |
| W.205a | new | 112 | 58 | 34 | 112 | 58 | 35 |
| W.205d | new | 44 | 8 | 0 | 49 | 9 | 0 |
| W.207 | old 30 | 458 | 113 | 89 | 567 | 136 | 112 |
| W.207b | old 30 | 159 | 48 | 22 | 168 | 76 | 55 |
| W.208 | old 30 | 148 | 17 | 13 | 168 | 17 | 20 |
| W.209 | new | 106 | 13 | 3 | 111 | 14 | 3 |
| W.210 | old 30 | 195 | 21 | 19 | 217 | 22 | 21 |
| W.212 | old 30 | 80 | 22 | 19 | 80 | 22 | 19 |
| W.220 | old 30 | 96 | 33 | 22 | 96 | 33 | 22 |
| W.221b | new | 108 | 35 | 25 | 157 | 49 | 25 |
| W.222a | old 30 | 69 | 16 | 15 | 69 | 16 | 15 |
| W.224 | old 30 | 581 | 74 | 69 | 766 | 113 | 95 |
| W.225 | old 30 | 325 | 30 | 25 | 374 | 31 | 29 |
| W.235 | new | 60 | 18 | 0 | 81 | 23 | 0 |
| W.244 | old 30 | 116 | 26 | 22 | 116 | 26 | 22 |
| W.245a | old 30 | 428 | 51 | 134 | 590 | 74 | 188 |
| I.407a | old 30 | 210 | 58 | 8 | 338 | 76 | 22 |
| I.408 | new | 47 | 16 | 12 | 48 | 17 | 12 |
| I.409 | old 30 | 113 | 50 | 12 | 169 | 51 | 13 |
| I.423 | new | 152 | 25 | 24 | 171 | 29 | 25 |
| I.425 | new | 78 | 5 | 3 | 80 | 6 | 3 |
| I.434a | new | 227 | 20 | 23 | 344 | 42 | 23 |

### Low-support classes

| Code | Origin | TRAIN >=16px | VAL >=16px | TEST >=16px | TRAIN all | VAL all | TEST all |
|---|---|---|---|---|---|---|---|
| P.103b | new | 48 | 0 | 16 | 48 | 0 | 16 |
| P.103c | new | 79 | 4 | 119 | 79 | 4 | 119 |
| P.104 | new | 108 | 3 | 3 | 108 | 3 | 3 |
| P.106b | new | 67 | 0 | 44 | 67 | 0 | 44 |
| P.107 | new | 34 | 4 | 1 | 34 | 4 | 1 |
| P.110a | new | 2 | 0 | 0 | 2 | 0 | 0 |
| P.112 | new | 26 | 0 | 0 | 26 | 0 | 0 |
| P.115 | new | 19 | 6 | 0 | 19 | 6 | 0 |
| P.127c | new | 33 | 0 | 4 | 33 | 0 | 4 |
| DP.135 | new | 64 | 0 | 9 | 64 | 0 | 9 |
| R.301a | new | 1 | 0 | 0 | 1 | 0 | 0 |
| R.301c | new | 1 | 0 | 63 | 1 | 0 | 75 |
| R.301d | new | 50 | 0 | 16 | 50 | 0 | 16 |
| R.301e | new | 12 | 3 | 42 | 14 | 5 | 42 |
| R.302b | new | 91 | 0 | 0 | 101 | 5 | 0 |
| R.305 | new | 3 | 0 | 0 | 4 | 3 | 0 |
| R.411 | new | 27 | 13 | 0 | 28 | 19 | 0 |
| R.412 | new | 18 | 0 | 9 | 20 | 0 | 9 |
| R.415a | new | 26 | 0 | 27 | 26 | 0 | 27 |
| W.202a | new | 28 | 73 | 2 | 28 | 73 | 2 |
| W.205b | new | 13 | 1 | 1 | 13 | 1 | 1 |
| W.207a | new | 41 | 4 | 0 | 41 | 4 | 0 |
| W.219 | new | 4 | 0 | 0 | 4 | 0 | 0 |
| W.221a | new | 13 | 2 | 0 | 13 | 2 | 0 |
| W.227 | new | 47 | 0 | 26 | 53 | 0 | 26 |
| W.233 | new | 35 | 2 | 5 | 56 | 2 | 5 |
| W.239a | new | 3 | 0 | 0 | 3 | 0 | 0 |
| W.239b | new | 14 | 139 | 0 | 14 | 139 | 0 |
| W.246a | new | 11 | 2 | 7 | 11 | 2 | 7 |
| I.402 | new | 2 | 0 | 0 | 2 | 0 | 0 |
| I.407 | new | 15 | 0 | 0 | 21 | 0 | 0 |
| I.442 | new | 19 | 6 | 8 | 20 | 8 | 8 |

### Codes not trained

| Code | Origin | TRAIN >=16px | VAL >=16px | TEST >=16px | TRAIN all | VAL all | TEST all |
|---|---|---|---|---|---|---|---|
| W.205c | new | 0 | 0 | 2 | 0 | 0 | 2 |
| I.401 | new | 0 | 0 | 7 | 0 | 0 | 7 |

Full table: `output/kiet_wp_20261028/recognizer_v2/class_counts.csv` (local, ignored by Git); the class order is in the checkpoint's `classes`.

## 3. Training

- MobileNetV3-small, 128 px input, initialised from `retrained_20261009.pt` (backbone and the head rows of the 30 old codes and background; new head rows use the default initialisation).
- TRAIN data: 17445 crops: released V1.2 boxes and agreed extension boxes of 16 px and up, 4,000 random background crops and 26 boxes both passes called `not_qcvn_sign`. Background crops are random regions of TRAIN images with IoU below 0.1 against every labelled box and under 30% of any box covered, counting unreadable and out-of-ontology boxes. Unreadable boxes were never used as labels.
- Augmentation: box jitter (-10% to +10% per edge), brightness/contrast, light blur, rotation of +-5 degrees with 0.9-1.1 zoom, down-then-up resolution to imitate small signs, JPEG. No horizontal flip.
- Sampling: `WeightedRandomSampler`, weight = class count^-0.7 (near class-balanced without unbounded repetition of rare codes); 14000 samples per epoch.
- Optimiser: AdamW (lr 0.0004, weight decay 1e-4), OneCycle cosine schedule, label smoothing 0.05, batch 128, 30 epochs, seed 20261028.
- Checkpoint: epoch 18, chosen by macro-F1 on VAL over the 55 main classes (VAL crops of 16 px and up, raw argmax): **0.9511**. This is the maximum over 30 epochs, so it is slightly optimistic; epochs after the peak sit near 0.94.
- Confidence threshold, calibrated on VAL only: **0.87** (smallest threshold with accepted accuracy at least 0.99 and coverage at least 0.80 on known-class VAL crops; accepted accuracy 0.9901, coverage 0.8675; VAL background accepted 0.0).
- Command: `scripts/train_recognizer_v2_20261028.py --output output/kiet_wp_20261028/recognizer_v2/train --epochs 30`; torch 2.11.0+cu128, NVIDIA GeForce RTX 3050 Laptop GPU, 590 s, git head when the run started `8534843`.

### Reference gate

The old `reference_gate_20261011_final.npz` stores embeddings from the old encoder (2,299 positive and 30 negative references) and cannot be used with the new model. Its 30 negatives are crops from the development video, and the manifest that lists their timestamps and boxes (the `--negative-manifest` input of `scripts/build_sign_reference_gate.py`) is not in the repository, so the gate could not be rebuilt for the new encoder. A gate with positives only would never veto anything. The new configuration therefore has **no reference gate**. This is a missing input, not a measured effect on VAL macro-F1; the regression check on segment 0 of the development video is in section 5. A v2 gate needs new AI-reviewed negative crops from video.

## 4. TEST results

Identical ground-truth crops, raw argmax (no confidence threshold); old model is `retrained_20261009.pt`. Macro-F1 is the unweighted mean over classes with TEST support in that size group.

### (a) The 30 old codes: old vs new

| Short side (px) | Crops | Classes | Old acc. | New acc. | Old macro-F1 | New macro-F1 |
|---|---|---|---|---|---|---|
| <16 | 1078 | 22 | 0.5139 | 0.7839 | 0.5151 | 0.7912 |
| 16-32 | 811 | 25 | 0.9309 | 0.9815 | 0.9266 | 0.9874 |
| 32-64 | 753 | 30 | 0.9150 | 0.9628 | 0.9002 | 0.9562 |
| >=64 | 502 | 27 | 0.9701 | 0.9801 | 0.9057 | 0.9499 |
| >=32 | 1255 | 30 | 0.9371 | 0.9697 | 0.9056 | 0.9594 |
| all | 3144 | 30 | 0.7904 | 0.9090 | 0.8194 | 0.9274 |

Paired image-level bootstrap (1,000 resamples, seed 20261028) of new minus old accuracy:

| Short side (px) | Crops | Images | New minus old accuracy | 95% CI (image bootstrap) |
|---|---|---|---|---|
| <16 | 1078 | 548 | +0.2699 | [+0.2379, +0.3017] |
| 16-32 | 811 | 555 | +0.0506 | [+0.0336, +0.0665] |
| 32-64 | 753 | 566 | +0.0478 | [+0.0328, +0.0664] |
| >=64 | 502 | 452 | +0.0100 | [-0.0058, +0.0262] |
| >=32 | 1255 | 986 | +0.0327 | [+0.0209, +0.0452] |
| all | 3144 | 1740 | +0.1186 | [+0.1051, +0.1325] |

The old model's earlier published figures (0.9398 at 32 px and up, 0.5135 below 16 px) used V1.1 labels; on the V1.2 labels used here the same old model scores 0.9371 and 0.5139. Per old code (TEST crops of all sizes):

| Code | TEST support | Old correct | New correct |
|---|---|---|---|
| DP.134 | 24 | 24 | 24 |
| I.407a | 22 | 12 | 20 |
| I.409 | 13 | 13 | 13 |
| P.102 | 159 | 107 | 140 |
| P.117 | 51 | 33 | 44 |
| P.123a | 45 | 34 | 41 |
| P.123b | 97 | 67 | 89 |
| P.124a | 24 | 15 | 15 |
| P.124d | 17 | 17 | 17 |
| P.125 | 23 | 23 | 23 |
| P.127 | 920 | 776 | 877 |
| P.130 | 405 | 332 | 354 |
| P.131a | 363 | 326 | 338 |
| P.137 | 87 | 30 | 78 |
| R.122 | 27 | 27 | 27 |
| R.302a | 169 | 83 | 131 |
| R.303 | 35 | 30 | 32 |
| W.201a | 49 | 26 | 42 |
| W.203c | 16 | 14 | 14 |
| W.207 | 112 | 87 | 106 |
| W.207b | 55 | 38 | 39 |
| W.208 | 20 | 14 | 14 |
| W.210 | 21 | 21 | 21 |
| W.212 | 19 | 19 | 19 |
| W.220 | 22 | 22 | 22 |
| W.222a | 15 | 15 | 15 |
| W.224 | 95 | 77 | 77 |
| W.225 | 29 | 25 | 29 |
| W.244 | 22 | 22 | 22 |
| W.245a | 188 | 156 | 175 |

### (b) New main classes (new model only)

510 crops, 21 classes with TEST support:

| Short side (px) | Crops | Classes | Accuracy | Macro-F1 |
|---|---|---|---|---|
| <16 | 2 | 2 | 1 | 1 |
| 16-32 | 106 | 14 | 0.8585 | 0.8570 |
| 32-64 | 208 | 19 | 0.8510 | 0.8654 |
| >=64 | 194 | 15 | 0.7680 | 0.7927 |
| >=32 | 402 | 20 | 0.8109 | 0.8534 |
| all | 510 | 21 | 0.8216 | 0.8736 |

Weak codes among them: `W.205a` (5 of 35), `W.203a` (19 of 43), `P.124b` (0 of 8), `W.221b` (17 of 25). `W.203a` and `W.205a` are AI-extension codes, so their training and their TEST labels are both AI-reviewed and part of the error may be label noise. Per-class results: `output/kiet_wp_20261028/recognizer_v2/eval/per_class.csv`.

### (c) Low-support classes (new model only)

Classes with TEST support below 5 are listed but not reliably evaluated.

| Code | TEST support | Correct | Accuracy | Status |
|---|---|---|---|---|
| P.103b | 16 | 13 | 0.8125 | reported |
| P.103c | 119 | 119 | 1 | reported |
| P.104 | 3 | 3 | 1 | not reliably evaluated (support < 5) |
| P.106b | 44 | 23 | 0.5227 | reported |
| P.107 | 1 | 1 | 1 | not reliably evaluated (support < 5) |
| P.110a | 0 | 0 |  | not reliably evaluated (support < 5) |
| P.112 | 0 | 0 |  | not reliably evaluated (support < 5) |
| P.115 | 0 | 0 |  | not reliably evaluated (support < 5) |
| P.127c | 4 | 0 | 0 | not reliably evaluated (support < 5) |
| DP.135 | 9 | 7 | 0.7778 | reported |
| R.301a | 0 | 0 |  | not reliably evaluated (support < 5) |
| R.301c | 75 | 21 | 0.2800 | reported |
| R.301d | 16 | 15 | 0.9375 | reported |
| R.301e | 42 | 24 | 0.5714 | reported |
| R.302b | 0 | 0 |  | not reliably evaluated (support < 5) |
| R.305 | 0 | 0 |  | not reliably evaluated (support < 5) |
| R.411 | 0 | 0 |  | not reliably evaluated (support < 5) |
| R.412 | 9 | 9 | 1 | reported |
| R.415a | 27 | 25 | 0.9259 | reported |
| W.202a | 2 | 0 | 0 | not reliably evaluated (support < 5) |
| W.205b | 1 | 1 | 1 | not reliably evaluated (support < 5) |
| W.207a | 0 | 0 |  | not reliably evaluated (support < 5) |
| W.219 | 0 | 0 |  | not reliably evaluated (support < 5) |
| W.221a | 0 | 0 |  | not reliably evaluated (support < 5) |
| W.227 | 26 | 15 | 0.5769 | reported |
| W.233 | 5 | 5 | 1 | reported |
| W.239a | 0 | 0 |  | not reliably evaluated (support < 5) |
| W.239b | 0 | 0 |  | not reliably evaluated (support < 5) |
| W.246a | 7 | 7 | 1 | reported |
| I.402 | 0 | 0 |  | not reliably evaluated (support < 5) |
| I.407 | 0 | 0 |  | not reliably evaluated (support < 5) |
| I.442 | 8 | 8 | 1 | reported |

`R.301c` has 75 TEST crops but only one TRAIN crop; the TEST crops come from one physical sign seen in many frames, so its 0.280 is one sign's failure, not a class estimate. Untrained codes with TEST boxes: `I.401` 7, `W.205c` 2.

## 5. Video check (development video V0, 0-60 s)

`scripts/benchmark_temporal_video_20261028.py --starts 0 --recognizer models/sign_recognizer/retrained_20261028.pt` on `D:\DSP391m\datasets\test_video.mp4` (SHA256 `eb109f8c...`), output `output/kiet_wp_20261028/wp3/recognizer_v2_20261028/` (local). The script now accepts `--recognizer` and `--reference-gate`, so a candidate can be run without editing the config.

- The blue board at the right edge is track 26: `I.434a` (Bến xe buýt) from 14.67 s to 16.17 s, ten consecutive analysis samples, source `temporal_best_crop` after the three-frame confirmation; it is lost at 16.33 s when it leaves the view. In the earlier run (`temporal_fix_20261028`) the same interval only showed `unknown` / `temporal_pending`. Two frames: `bus_stop_frame_14.9s.png` and `bus_stop_frame_15.8s.png` in the output folder.
- The overlay's bottom line still says the evidence is not enough for a legal reading ("I.434a: Ô tô: chưa đủ bằng chứng nhận dạng biển."). Recognising the code does not by itself make the legal text applicable.
- Segment 0, old run vs new: tracks 88 vs 94; confirmed labels P.130 9 vs 8, P.131a 3 vs 4, I.434a 0 vs 1, and no other confirmed label; alert samples 121 vs 121; alert episodes 52 vs 53; stale alerts 0 vs 0; inference FPS 3.25 vs 2.91 (the runs are not matched for warm-up). There is no event ground truth, so this shows that no unexpected code was confirmed without the gate in this segment; it does not measure false alarms.

## 6. Legal lookup for the new codes

`OfflineLegalRetriever().lookup_sign(code)` for the 87 trained codes: 84 return an appendix locator ("Phụ lục X, mục X.n"), 3 return only the generic corpus heading "Điều 84. Tổ chức thực hiện" (`I.423, P.103c, P.127c`), 0 are missing from the corpus. That heading is the one the appendix pages inherit from the last article, so those three locators are wrong for the sign; they are not fixed here. Details: `output/kiet_wp_20261028/recognizer_v2/legal_lookup.json`.

## 7. Acceptance decision

Rule (owner, 2026-10-28): switch the config only if, on the 30 old codes, TEST accuracy at 32 px and up does not fall more than 1 point below 0.940, and 16-32 px or below 16 px improves, and VAL macro-F1 over the main classes is at least 0.95.

| Condition | Result | Met |
|---|---|---|
| TEST >=32 px accuracy, 30 old codes, not below 0.930 | 0.9697 (old model on the same crops: 0.9371) | yes |
| TEST 16-32 px or <16 px improves | 16-32: 0.9309 to 0.9815; <16: 0.5139 to 0.7839 | yes |
| VAL macro-F1, 55 main classes, at least 0.95 | 0.9511 (epoch maximum, see section 3) | yes |

The config, the LFS manifest and the two model contract tests (`tests/unit/test_release_gate_20261011.py`, `tests/acceptance_v2/test_legacy_replacements.py`; recognizer line only) were updated. `pytest tests` exits 0 (498 passed, 11 skipped, 38 deselected), `tests/acceptance_v2` passes 11/11 with `ITASMM_ACCEPTANCE_DATASET_ROOT` set, `node --test frontend/src/*.test.js` passes 7/7 and the frontend build passes.

## 8. Limits

- The VAL figure clears 0.95 by 0.0011 and is a maximum over epochs; treat it as roughly 0.94 to 0.95.
- Labels are AI-assisted throughout. The extension codes were assigned from small, often blurry crops by one model twice, with no human check and no field holdout.
- Several extension classes are weak on TEST (`W.205a`, `W.203a`, `P.124b`, `R.301c`); the new low-support classes have little data and several have no TEST crop.
- No reference gate (section 3). The runtime still skips crops under 32 px in image mode (`min_short_side: 32`) and 12 px in video (`video_min_short_side: 12`). The model was trained on crops of 16 px and up, so 12-16 px video crops lie outside its training range; these settings were left unchanged because they were not evaluated here.
- Background crops are random image regions assumed unlabelled; unreadable boxes were excluded from them, but real signs missing from the annotations may remain.
- `I.401` and `W.205c` have no TRAIN crop and are not recognised.
- Rollback: set `sign_recognizer.weights_path` back to `models/sign_recognizer/retrained_20261009.pt` with `confidence_threshold: 0.63`, `reference_gate_path: models/sign_recognizer/reference_gate_20261011_final.npz` and `reference_gate_margin: 0.015`, and revert the two contract-test lines.

## Reproduce

```powershell
$env:PYTHONNOUSERSITE=1; $env:PYTHONPATH="<repo root>"; $env:YOLO_OFFLINE="true"
python scripts/build_qcvn_reference_table_20261028.py
python scripts/build_ontology_review_sheets_20261028.py refs
python scripts/build_ontology_review_sheets_20261028.py crops --name A
python scripts/build_ontology_review_sheets_20261028.py crops --name B --seed 20261028
python scripts/record_ontology_ai_labels_20261028.py merge        # needs the local pass A and B label files
python scripts/build_recognizer_v2_dataset_20261028.py
python scripts/train_recognizer_v2_20261028.py --epochs 30
python scripts/evaluate_recognizer_v2_20261028.py --candidate models/sign_recognizer/retrained_20261028.pt
python scripts/check_recognizer_v2_legal_lookup_20261028.py
```

Contact sheets, crops, caches, logs and the video outputs stay under `output/` and are not tracked. The pass A and pass B label files (`output/kiet_wp_20261028/recognizer_v2/labels_A.jsonl`, `labels_B.jsonl`) are local; the tracked manifest keeps both passes for every box.
