# Benchmark 20261112 — Stage A results

Status: Stage A partially complete. The yolo11s run was stopped at epoch 12/40 (not finished), and its evaluation, the current-detector VAL mAP50-95 under the same protocol, per-model timing, and the one-stage end-to-end comparison were not produced. Every number below is copied from a file under `output/bench_20261112/`, listed in each table.

Frozen protocol for all detector numbers: confidence 0.35, NMS IoU 0.45, match IoU 0.5 (one-to-one), AP with 101-point interpolation. Bootstrap: 1000 paired image-level resamples, seed 20261112, candidate minus current detector on TEST.

## A2 — Detector candidates (TEST, 1 class, sign localization)

Current detector: `models/sign_detector/finetuned_20261026.pt`.

| Candidate | VAL mAP50-95 | TEST recall | TEST mAP50 | Δ recall 95% CI | Δ mAP50 95% CI | Source |
|---|---|---|---|---|---|---|
| current (reference) | — | 0.8754 | 0.9357 | — | — | `eval_candidates/*.json` (`current_TEST`) |
| yolo11n | 0.7716 | 0.8818 | 0.9353 | [-0.0008, 0.0133] | [-0.0039, 0.0033] | `eval_candidates/yolo11n.json` |
| yolov8n | 0.7650 | 0.8662 | 0.9299 | [-0.0179, -0.0008] | [-0.0103, -0.0012] | `eval_candidates/yolov8n.json` |
| yolo12n | 0.7602 | 0.8445 | 0.9208 | [-0.0398, -0.0216] | [-0.0201, -0.0094] | `eval_candidates/yolo12n.json` |
| yolo26n | 0.7674 | 0.8693 | 0.9370 | [-0.0147, 0.0023] | [-0.0029, 0.0057] | `eval_candidates/yolo26n.json` |
| yolov8n-p2 | 0.7635 | 0.8648 | 0.9306 | [-0.0190, -0.0014] | [-0.0100, -0.0001] | `eval_candidates/yolov8n-p2.json` |
| current_finetune30 | 0.7642 | 0.8805 | 0.9299 | [-0.0014, 0.0119] | [-0.0093, -0.0024] | `eval_candidates/current_finetune30.json` |

Not run: yolo26n-p2, yolo11s (stopped at epoch 12/40, `A2/yolo11s/train.log`), rtdetr-l (deferred by owner).

No candidate has a TEST mAP50 gain whose CI excludes zero. Gate thresholds from plan section 2 were not re-checked in this session; the gate decision is not stated here.

## A3 — Inference modes on the current detector (`A3/ablation_summary.json`)

| Mode | TEST recall | TEST mAP50 | TEST precision | FIELD recall | FIELD mAP50 |
|---|---|---|---|---|---|
| plain 640 | 0.8754 | 0.9357 | 0.9361 | 0.7727 | 0.8471 |
| plain 960 | 0.8781 | 0.9255 | 0.9084 | 0.7576 | 0.7640 |
| multi 960+640 | 0.8820 | 0.9275 | 0.9047 | 0.7727 | 0.7786 |
| SAHI 512, overlap 0.2 | 0.7803 | 0.8351 | 0.8057 | 0.7727 | 0.5255 |

FIELD is the 239-frame pack (66 reference boxes). Timing per mode was not measured.

## A4 — Recognizer backbones (`A4/*/results.json`, 87 codes + unknown)

Test set: old 30-class set (`test.old30.all`).

| Backbone | Epoch (VAL) | VAL macro-F1 | Test accuracy | Test macro-F1 | Calibrated threshold | Source |
|---|---|---|---|---|---|---|
| mobilenet_v3_small | 20 | 0.9442 | 0.8998 | 0.9083 | 0.84 | `A4/mobilenet_v3_small/results.json` |
| mobilenet_v3_large | 14 | 0.9468 | 0.9160 | 0.9183 | 0.87 | `A4/mobilenet_v3_large/results.json` |
| efficientnet_b0 | 13 | 0.9516 | 0.9160 | 0.9297 | 0.92 | `A4/efficientnet_b0/results.json` |
| resnet18 | 9 | 0.9551 | 0.9024 | 0.9232 | 0.88 | `A4/resnet18/results.json` |

Reference row (`A4/reference_retrained_20261028/results.json`) reproduces the 2026-10-28 numbers. Recognizer inference time per crop was not measured.

## A5 — One-stage 88-class detector

Dataset: `output/bench_20261112/A5/dataset_summary.json` (TRAIN 20,923 / VAL 4,747 / TEST 4,888 boxes; 88 classes including `other_sign`, which holds 5,518 boxes; 3 codes outside the 87-class list mapped to `other_sign`).

Training `yolo11n` 88 classes, 40 epochs: best VAL mAP50-95 0.1879 at epoch 31, VAL mAP50 0.7160, VAL precision 0.6722, VAL recall 0.4762 (`A5/onestage_yolo11n_88/results.json`). Training took 243 s, which is short for 40 epochs; the dataset size has not been checked against this.

Not produced: end-to-end TEST comparison against the two-stage pipeline (the evaluation script was not written).

## A6 — Sign-aware legal retrieval (`A6/retrieval_eval.json`)

Dev: 33 cases. Holdout: 34 cases. Original regression set: 24 cases.

| Variant | Holdout recall@1 | Holdout MRR | Original-24 recall@1 |
|---|---|---|---|
| R0 (baseline) | 0.9706 | 0.9853 | 0.5417 |

Other variants (R1–R3) are in the same file. The file reports R2 and R3 fail the 24-case non-regression check. Defaults are off in the application. The set is saturated on holdout, so a variant cannot show a +0.10 gain here.

## A7 — Blind double labeling of the 239-frame field pack (`A7/analysis.json`)

Two passes by the same reviewer with rule-based procedure, not independent human labels.

- Boxes labeled by both passes: 298. Exact label agreement: 0.9094. Cohen's kappa by size: 16–32 px 0.708, 32–64 px 0.826, ≥64 px 0.967.
- Reference code boxes (both passes agree on a code): 65.
- Static pipeline (SignDetector + SignRecognizer, production thresholds): detection recall against the 327 pack boxes 0.352, precision 0.810. End-to-end code accuracy on the 65 reference boxes: 0.738 (48 of 65 correct; 58 matched by the detector).

The pack boxes are AI proposals from the detector-era review, so the detection numbers are circular for that detector family.

## Not done (see `output/bench_20261112/CP1.md`)

- yolo11s evaluation, yolo26n-p2, rtdetr-l, one-stage end-to-end comparison.
- Current-detector VAL mAP50-95 under the frozen protocol.
- Timing (ms per image or per crop) for any model. GPU-shared runs make it unreliable.
- GFLOPs: unavailable from the installed Ultralytics model.
