# ITAS-MM — Import comment.zip + yolo11_optimized.zip (2026-10-14)

## Source identity & secure extraction

ZIPs from local `Downloads` were inspected before extraction, with zip-slip/symlink/size checks and a SHA256 manifest. Both archives were extracted locally to `data/raw/user_archives_20261014/{comment,yolo11_optimized}`, **not committed** because model binaries, screenshots, third-party data and personal inputs remain excluded by `.gitignore`.

- `comment.zip`: SHA256 `47a13309423cf81b859bfbf1353014784dcd13976a686b69dcea7df029382797`, 23 files, screenshots/report/scripts. Contains **no new lane checkpoint**.
- `yolo11_optimized.zip`: SHA256 `74fafac25c82eb82634ef988232444dbdb053c7b365440dc67dd78930ea08d43`, 24 files including `best.pt` and `last.pt`.
- YOLO11 `best.pt`: SHA256 `bdef54fc5b9c24136d1f71579d868a029c8f7b5e0ec79bb81b5fe01678c87da9`. 59 localized Vietnamese sign class names extracted from checkpoint pickle metadata **without unpickling during label discovery**, pinned in `data/reference/yolo11_optimized_classes_20261014.json`.
- **Checkpoint loading executes serialized PyTorch model objects**; use only trusted checkpoint sources. The supplied model was inspected for embedded class globals (PyTorch and Ultralytics modules) before test inference.
- Checkpoint metadata notes **Ultralytics AGPL-3.0 licensing**. We do not commit or redistribute model binaries, and anyone distributing/rehosting the checkpoint must check applicable license obligations.

## Imported UFLD-v2 comparison (comment.zip)

The package reports an old **CPU** comparison on one 1260-frame video (`example.mp4`, 25 FPS, 1280×720) and cites paper F1, but does **not** include annotated independent field ground truth or model weights. Its CULane ResNet18 vs ResNet34: ~5.43 vs 4.80 FPS; reported detection rate 100% vs 99.9%, and average 3.52 vs 3.94 lanes/frame. These are *coverage/performance measurements*, not lane correctness. Claims like "94.8% frames with four lanes = 94.8% accuracy" are **not valid**. TuSimple models are tested on a different road domain, and CurveLanes may overpredict. We keep the ITAS-MM UFLD-v2 ResNet18 current checkpoint. No unverified ResNet34 replacement or invented Vietnam lane F1.

## Holdout candidate test: YOLO11 versus current panel detector

Run: `scripts/evaluate_imported_yolo11_val_20261014.py`

- Independent **canonical VAL** from this project, first 240 images by SHA256 order, **449 GT boxes**, IoU ≥ 0.5. This is *not* a Vietnam field test.
- Detector in use at image size 960: TP 416/449, recall **92.65%**, precision **87.58%** (416/475).
- Imported YOLO11 at image size 416, confidence 0.55: TP 120/449, recall **26.73%**, precision **78.95%** (120/152).
- Merging original + auxiliary candidates at size 416: TP 416/449, recall **92.65%**, precision **83.03%** (416/501).
- Merging with YOLO11 size 640: TP 417/449, recall **92.87%**, precision **81.76%** (417/510).
- Mean per-image localization-only inference: current 960 ~17.1 ms; YOLO11 ~9.6 ms at 416; these timings exclude all other stages.
- This **does not support replacing** the primary sign detector. Additional code-recognition precision/recall was not established for YOLO11 classes on unseen Vietnamese scenes. Training metrics in its ZIP were recorded on its own validation distribution and are **not directly comparable**.

## Safe integrated use

- New optional `auxiliary_sign_review` config under `configs/perception.yaml`.
- Model adapter: `src/perception/auxiliary_sign_review.py`. Lazy-load the checkpoint only after a user checks **YOLO11 optimized (research preview)** in image/video UI. SHA256 must match the pinned manifest; missing or mismatched checkpoint fails closed.
- New API opt-in form field `auxiliary_sign_review=true` on existing `POST /pipeline/run`, result list `perception.auxiliary_sign_candidates`, and `GET /pipeline/auxiliary-sign-status` for non-regulatory provenance/metrics. Rendered candidates get purple boxes.
- Threshold 0.65, minimum 16 px, IoU deduplication against primary boxes, top 12. These boxes are **review hypotheses only**. They never enter `perception.signs`, `SignApplicability`, `car_context`, verified lane/scope, legal RAG, temporal sign tracking or TTS.
- No changes to selected sign/lane checkpoint, and no change to the previously verified complete video. To see the auxiliary review overlay, enable the option and upload a video or image through the UI.

## Reproduction / limitations

A fresh Git clone intentionally excludes the binary weights. Copy the trusted 59-class `best.pt` to `data/raw/user_archives_20261014/yolo11_optimized/weights/best.pt`, verify its SHA256, and retain `data/reference/yolo11_optimized_classes_20261014.json`. If absent, the feature is unavailable while the primary system remains operational.

Run `python -m pytest -q tests`, `npm --prefix frontend run build`, and the VAL benchmark after supplying pretrained weights and canonical labels. This is a research prototype, **not human-verified field safety acceptance**, and the work does not certify time or vehicle applicability of any detected sign.

## Live integration acceptance on the real `test_video.mp4`

The backend was restarted after code updates. The new `GET /pipeline/auxiliary-sign-status` endpoint responds HTTP 200 and `POST /pipeline/run` accepts `auxiliary_sign_review=true` on the current executable app. The same four frames were each run with the option OFF and ON; **all 8 calls returned HTTP 200** and primary `signs` (labels, boxes, car context), `car_summary`, and `speech_text` were **bytewise equivalent within the returned structures**. This is direct evidence the imported detector does not change legal conclusions.

| Second in input video | Main model | New review-only YOLO11 candidate |
|---|---|---|
| 149.333 | P.131a | Supplementary time panel (0.7752) |
| 299.2 | no released sign | **False positive**: advertising sign mistakenly labeled “Cấm đỗ xe” (0.6841) |
| 660.7 | no released sign | Blue lane vehicle-use board (0.8485) — a useful detection missed by the main system |
| 899.3 | P.130 | Supplementary time panel (0.9011) |

The detected false positive at 299.2 s proves **why direct merging into legal traffic-sign results is unsafe**. Source frame overlays are in `output/import_yolo11_comment_20261014/auxiliary_four_frames.jpg`. Matching live API JSON is `output/import_yolo11_comment_20261014/api_optin_smoke.json`.

Production checkpoint and previously verified 19m53s rendered video were kept unchanged; they do not contain the new opt-in YOLO11 overlay. Use the video/MP4 panel and check the new **YOLO11 optimized** research toggle to see the secondary candidates.
