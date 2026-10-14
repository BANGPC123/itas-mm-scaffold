# ITAS-MM — Accuracy-first offline MP4 simulation (2026-10-14)

## What was actually implemented

The user runs **recorded MP4 clips**, not a live camera. An offline mode can spend more time on each frame, sample frames at 3/6/10 Hz, wait for model results instead of dropping them during playback, and produce a native-FPS **rendered MP4**. It is available in the local Web UI on the Video MP4 panel under **Phân tích MP4 chất lượng cao**. Choose the file, sampling Hz, optional video start datetime with timezone and optional preview duration; the backend uploads an MP4, queues one job, displays progress, verifies exported video before enabling download.

Code: `src/perception/offline_sign_quality.py`, `src/pipeline/orchestrator.py` (`offline_video_quality`), `scripts/render_offline_video_max_20261014.py`, `backend/routers/offline_video.py`, `frontend/src/VideoPanel.jsx`, `frontend/src/DemoVideos.jsx`.

- Sign **detector**: primary YOLO11n in three sequential resolutions 960, 640 and 1280, with MobileNetV3 and an independent reference gate for known-code classifications. The high-res supplement is admitted only for clear candidate boxes with detector confidence ≥ 0.75, recognizer confidence ≥ 0.93 and at least 16 px in its shortest direction; it cannot replace existing known-code detections.
- **Video track**: `TemporalSigns` requires two sightings of the same label within the track window before it releases it. A missing detection never draws a ghost sign. Audio is **not synthesized** during offline rendering; original audio is retained, if present, via FFmpeg.
- **Lane**: existing UFLD-v2, YOLOPv2, CLRNet (when local weights are available), SegFormer road/sidewalk masks, paint-color evidence, vehicle occlusion, road-edge rejection. Visualization does not interpolate through occluding cars, and unverified lane hypotheses do not establish sign-to-lane legal scope.
- **YOLO11 optimized 59-class** is a separate optional review channel (purple labels `gợi ý (chưa xác minh)`), enabled by default for offline rendering in the local Web UI; its guesses never enter the main sign list, car applicability, speech or legal reasoning. A static advertising logo has been observed erroneously classified `Cấm đỗ xe`; shown purple explicitly as a false-candidate example.
- **Recorded time** is used only if the user explicitly supplies a timezone-aware start datetime. The video's playback clock or current local time is not substituted.
- Source input bytes remain local under `output/offline_video_jobs/<uuid>/source.mp4`; output and reports are under the same job folder. Git excludes all images/video/weights/checkpoints/user files.

## Static labeled benchmark — frozen VAL and TEST

Script: `scripts/evaluate_offline_sign_quality_val_20261014.py`. Computes one-to-one IoU≥0.5 code correctness against canonical released labels. It does NOT measure time-series/video accuracy; the TEST split was not used to select the thresholds (.75/.93 were selected on VAL).

| Split | Mode | Correct code | Precision | Recall | Code F1 |
|---|---|---:|---:|---:|---:|
| VAL, 400 images / 540 released GT | Original 960/640 | 396 | 92.31% | 73.33% | 81.73% |
| VAL | +1280 cautious rescue | **399** | 92.15% | **73.89%** | **82.01%** |
| TEST, 2,326 images / 3,058 released GT | Original 960/640 | 2032 | **90.84%** | 66.45% | **76.75%** |
| TEST | +1280 cautious rescue | **2037** | 90.21% | **66.61%** | 76.64% |

**IMPORTANT: static TEST F1 decreased slightly. Do not advertise the third pass as a proven global accuracy improvement.** It is an optional, more exhaustive offline video mode with temporal confirmation, which needs an independently annotated VIDEO TEST set to measure real gain or loss. For lower false alarms, leave the original mode in use.

## Live verified evidence

- Local Web API smoke with 4-second user clip: accepted HTTP 202, processed **120/120** native 30 FPS frames and 24 AI samples at 6 Hz, returned HTTP 206 for byte-range video, and served the JSON summary. Evidence `output/offline_job_api_smoke_20261014.json`.
- Secondary YOLO on 4 seconds near the real video 299-second advertising false-positive: again **120/120** frames verified, 24 inference samples, **12 YOLO11 review proposals**, optional simulated timezone-aware recording start accepted, original video soundtrack retained when present. Evidence `output/offline_max_sign_job_smoke_20261014.json`; image `output/offline_max_sign_overlay_best_20261014.jpg`. Purple boxes are **not** traffic-sign legal confirmations.
- Standalone 30-second clip from original video seconds 38–68: **900/900** exported/decoded video frames at 1280×720 and 30 FPS, **180 analysis samples at 6 Hz**, **verified_full_decode=true**, SHA256 `81eb5af599aa927468ca029b167ea27ec4eb49c47be42eff972526bee8e94dac`. `output/offline_video_max_preview_20261014/test_video_offline_max.mp4`, `summary.json`, `predictions.jsonl`.
- Entire TEST suite: **471 passed, 6 warnings**, frontend JS tests **7 passed**, production build **PASS**. Logs `output/offline_max_final_pytest_20261014.log`, `output/offline_max_final_build_20261014.log`.

## Running it

Web (backend running at `http://127.0.0.1:8000`):
1. Open Video MP4, select a local MP4.
2. Choose sample rate (3/6/10 Hz). Six Hz is a reasonable start; 10 Hz means more compute and temporal samples, not a guarantee of higher precision.
3. Optionally enter recording start date/time (only if known) and a preview length in seconds.
4. Click **Bắt đầu phân tích MP4 offline chất lượng cao**. Leave the app backend running; processing takes longer than video duration. Poll progress in the page, then download the verified MP4 and report.

CLI:
```powershell
cd D:\itas-mm
python scripts/render_offline_video_max_20261014.py "C:\path\to\video.mp4" --output output/my_offline_run_001 --analysis-hz 6 --auxiliary-review
```

Each run must use a NEW output directory, to preserve provenance. For only the first minute, append `--max-seconds 60`. To analyze a specific interval, append `--start-second 120 --max-seconds 60`. To make simulated time constraints explicit, use `--recorded-at 2026-10-14T08:30:00+07:00` only for a user-provided or deliberately chosen simulation time.

Constraints for Web uploads: local `.mp4` only, ≤850 MiB, ≤100,000 frames and max 3840×2160. One offline job at a time, sharing one GPU inference worker with normal preview. Other per-frame inference requests may return HTTP 429 until the offline job finishes; Web server restart interrupts the job. Videos and checkpoints are NOT committed to Git.

## Remaining accuracy blockers

- No exhaustive human-verified source-disjoint Vietnam dashcam GT for lane, sign and car applicability, so cannot claim a Vietnam field F1 or that processing more frames increases accuracy.
- The imported YOLO11 59-class checkpoint loses localization precision when blindly merged; leave it as a visible review-only channel.
- Offline 3-pass can recover tiny signs but can increase false classifications, even with a temporal 2-vote gate. Full video should be spot-checked before use in a report.
- Simulation mode remains incapable of inferring the actual date/time of an unlabeled recording or legally valid car-specific restrictions without evidence.

**This is a research simulation / prerecorded-video analysis system. It is not designed for vehicle actuation or real-time camera operation.**
