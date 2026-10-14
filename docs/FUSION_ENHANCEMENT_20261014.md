# ITAS-MM — Fused model enhancement, independent VAL and in-video regression (2026-10-14)

## Status & provenance

Integrated original YOLO11n 1-class panel detector + pretrained MobileNetV3-Small 30 released codes + user-supplied 59-class YOLO11 optimized as a **separate review-only candidate channel**. Existing UFLD-v2 ResNet18 + road/paint/occlusion camera-aware lane fusion are unchanged at weight level. No new fine-tuned checkpoint or human-verified independent Vietnam lane GT was produced. Existing primary legal decisions are intentionally kept unchanged; neither YOLO11 predictions nor UFLD high-recall visual proposals may establish a legally applicable rule for car.

New model/logic files:
- `src/perception/sign_concordance.py`: YOLO code-to-index mapping only for exact, non-ambiguous QCVN-supported sign codes; MobileNetV3 agreement requires `yolo_conf>=0.80`, `MobileNet code confidence>=0.90`, `min crop side>=24px`; a consensus is *image-classifier agreement*, never legal verification.
- `src/perception/auxiliary_sign_review.py`: imported class names and weight SHA256 pinned as before, optional only.
- `src/perception/lane_display_quality.py`: **opt-in** `candidate` display mode exposes in-bounds raw UFLD proposals, without using them to infer sign-to-lane scope. Default `precision` retains strict legacy visual suppression. `src/pipeline/orchestrator.py` clears `ego_boundaries` when candidate mode is used on UFLD output.
- `src/perception/road_edge_veto.py`: additionally considers narrow semantic-road strips **only for long/densely sampled contours** (a median adjacent to cars can produce a very narrow mask). Requires 78% near-margin support on narrow mask, does not fabricate points, and retains shorter isolated dashes.
- `backend/routers/pipeline.py`: `POST /pipeline/run` now accepts `lane_display_mode=precision|candidate`; `auxiliary_sign_review=true` continues opt-in. Frontend exposes both controls in image and MP4 panels. Candidate lane lines are dashed/yellow in the video UI; detected supplementary boxes are purple and remain tentative.
- `tests/unit/test_concordance_lane_modes_20261014.py`: regression tests for narrow road edge, model agreement, legal abstention and candidate lane display.

## Frozen CULane VAL sample: 240 images with labeled lanes

Benchmark script `scripts/compare_lane_filters_val_20261014.py`; `output/evaluation/lane_filter_val_20261014/results.json`. Same 240 hash-selected frames, 809 GT lanes, official-like spline and 30 px IoU-matched F1:

| Visual hypothesis policy | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Raw UFLD-v2 unfiltered proposals | 637 | 131 | 172 | 82.94% | 78.74% | **80.79%** |
| Legacy deployed plausible geometry screen | 346 | 61 | 463 | 85.01% | 42.77% | 56.91% |
| Strict display screen | 129 | 5 | 680 | 96.27% | 15.95% | 27.36% |

This does NOT prove that raw candidates are accurate on dashboard-heavy Vietnam scenes. We preserve the user-selectable precision policy and restrict high-recall to a **visually marked, opt-in, legally non-binding comparison**. No baseline checkpoint changed. This is a compromise between omission and false-positive density, not a claimed field-F1 improvement.

## YOLO11 secondary + MobileNet model agreement on frozen canonical VAL

Benchmark `scripts/evaluate_yolo11_concordance_val_20261014.py`; `output/evaluation/yolo11_concordance_val_20261014/results.json`. Across 240 SHA256-selected VAL images with 309 released-code targets and 475 primary detector boxes: **17 additional auxiliary YOLO candidates** after original box overlap and confidence screening, **0** meet strict independent classifier/code-agreement gate. This is NOT evidence for promoting new automatic legal signs. Existing earlier 240-image localization validation showed direct union **reduces precision** from 87.58% to 81.76% with 640px YOLO auxiliary. Hence no automatic union of sign outputs, no legal confidence escalation.

## Visual lane hard case & direct live API

`lane_0020` (approx. 42.7s of user's MP4) previously drew a *median white road edge* as a lane for 36 points. SegFormer road strip only 60 px wide by the nearby vehicle, which defeated the old wide-mask veto. Narrow-edge policy now removes this **1/1 demonstrated incorrect contour**. On 6 additional hand-reviewed AI image frames, 26/26 previous visual lane pieces were retained. This is a small curated *AI-reviewed regression*, NOT a representative pixel-level GT or human-validated accuracy metric. Evidence `output/evaluation/road_edge_narrow_veto_20261014.json`.

After restarting the API, `scripts/verify_live_fusion_20261014.py` ran 5 real frames with secondary OFF/ON (**10 HTTP 200** responses): all primary sign labels, car context statuses, legal speech texts, car summaries and lane line counts were identical across opt-in settings. In `lane_0020`, lane line count was 0 after the upgraded edge veto; the 4 other scenes (149.333, 299.2, 660.7, 899.3s) produced tentative secondary hypotheses, all with `legal_warning_allowed=false`; advertising sign 299.2s remains **uncertain** rather than a legal parking prohibition. Evidence `output/evaluation/fusion_online_20261014.json`.

## Deployment acceptance and limitations

- `python -m pytest -q tests`: **465 passed**, 6 warnings at time of comprehensive suite run.
- Frontend `npm --prefix frontend run build`: **PASS**.
- **Complete video VERIFIED:** `output/video_fusion_20261014/test_video_annotated.mp4`, 445,360,104 bytes, 35,793/35,793 decoded frames, 1280×720 at 30 FPS, 3,580 previously verified analysis samples, **29/29** audio events audible, **12/12** verification checks passed. Output SHA256 `0e6c79be7e6d1d0ea137056461aeccb9c4739215ade7302402880d8f73e18abb`. The replay removed 6,016 additional original lane points without introducing new points. It uses verified October 13 ML candidate cache + updated edge veto; does **not** run the added YOLO11 on 35,793 frames, so it is not a new 59-class annotated video. The API serves it via `/pipeline/capstone-video/user-fusion`, HTTP 206 range request for bytes 0–1023 confirmed after completed verification. Original video artifact remains untouched.
- Research only. No human-reviewed, source-disjoint Vietnam lane and traffic-sign benchmark; no validated legal knowledge base (`validated_corpus=false`); no real driving safety certification.
- User-supplied YOLO checkpoint contains AGPL-3.0 license metadata, weight never pushed to GitHub.

## Reproduce

```powershell
python scripts/compare_lane_filters_val_20261014.py
python scripts/evaluate_yolo11_concordance_val_20261014.py
python -m pytest -q tests
npm --prefix frontend run build
python scripts/verify_live_fusion_20261014.py
```

Do not tune models on CULane TEST or user video while subsequently claiming blind independent performance.
