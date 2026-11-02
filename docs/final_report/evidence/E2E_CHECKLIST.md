# E2E checklist (T13) — run 2026-10-31, branch work/final-20261102

Server: `start.ps1` on 127.0.0.1:8000, Python `D:\miniconda\envs\itas-mm\python.exe`, frontend `frontend/dist` (built with `npm run build`).
Browser: built-in browser pane (cannot select a local file, cannot use a webcam). Items that need file selection or a camera were not run in the UI.
Evidence files: `t13_upload_start.json`, `t13_upload_status.json`, `t13_server.log`, `t13_server_restart.log` in this folder; job folder `output/offline_video_jobs/a2b1e58883ef41e2a9b12e6c6f987d45/`.

| # | Item | Result | Evidence / note |
|---|---|---|---|
| 1 | Run `start.ps1`, open web | PASS | Server up on port 8000; page "ITAS-MM — Traffic assistance demo" shows "backend connected". |
| 2 | Live, day clip: boxes, code + Vietnamese name, lane, FPS/latency, voice | Removed from final scope (01/11) | API only: `/pipeline/run` with stream_id on the night clip (frames 1.0–4.5 s, 0.5 s step) returned P.112 at 2.5 s (0.74) and 3.0 s (0.66), 81–225 ms per request. Day clip not run. UI boxes, lane drawing, FPS display and voice NOT VERIFIED in UI (no file selection in the pane). |
| 3 | Live, night clip: boxes and count of signs read | Removed from final scope (01/11) | Same API run as item 2. Count over the full 60 s not taken. |
| 4 | Live: seek mid-run clears old boxes; new clip starts clean | Removed from final scope (01/11) | Backend stream isolation covered by `tests/unit/test_temporal_*` (pass in T12). UI behaviour not exercised. |
| 5 | Live: camera | Removed from final scope (01/11) | No webcam access from this session. Error path (`Không mở được camera...`) is in code, not exercised. |
| 6 | Upload, day clip: progress, result video with overlay and voice by timestamp | PARTIAL | Job `a2b1e588…` (night clip 30–120 s, 90 s) status `complete`, 5400/5400 frames, `verified_full_decode` true (summary.json). Result video served (`video/mp4`, HTTP 206 range), report served (HTTP 200). Status carries 4 cues: P.112 @3.167 s, W.245a @3.667 s, P.117 @13.167 s, P.127 @40.333 s. Overlay and cue playback in browser NOT VERIFIED. Day clip upload not run. |
| 7 | Upload: non-MP4 and over-long file give the right message | PARTIAL | Non-MP4 (`.avi`): HTTP 415 with `Hiện chỉ hỗ trợ MP4.` PASS. Over-long: message covered by `tests/unit/test_offline_length_error.py` (PASS); not uploaded through the server (a >100k-frame file was not created). |
| 8 | Two uploads in a row: second refused or queued | PASS (refusal) | While job 1 was running, a second upload returned HTTP 409 `Đang có một video được phân tích offline.` Queueing was not tested. |
| 9 | Legal search for one sign code: results and source | PASS (API) | `POST /pipeline/legal-search` for "P.131a cấm đỗ xe" returned 3 hits from QCVN 41:2024/BGTVT with source URL, source sha256 and `legal_currency_verified: False`. UI not clicked. |
| 10 | Stop server, start again, web still works | PASS | Process on :8000 stopped (`taskkill`), `start.ps1` restarted it; `/` returns 200 and `POST /pipeline/speech` returns 200 with audio_url. |

Screens (`screens/`): the upload-only interface was captured on 01/11/2026 (`v2*` and `v3*` files). Those files show the result list, the legal lookup and the 390 px mobile layout.

Summary: 4 PASS (1, 8, 9, 10), 4 PARTIAL (2, 3, 6, 7). Items 2 to 5 (live boxes, live seek, camera) were removed from the final scope on 01/11/2026. Speech playback in the browser has not been checked by ear.
