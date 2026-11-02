# Upload processing speed (01/11/2026)

Same source: night_hanoi_YQl9c5gCjhg_30s_90s.mp4 (90 s, 5,400 frames at 60 fps, sha256 prefix c22caa280981).

| Setting | Processing time | vs real time | Output |
|---|---:|---:|---|
| Before: 60 fps output, lane on every 6 Hz sample (job ab641b4878c6) | 406.3 s (an earlier run of the same clip on 31/10: 301.2 s) | 3.3-4.5x slower | 5,400 frames |
| After: 30 fps output, lane on every second sample (3 Hz), encoder threads unrestricted | 165.0 s (171.3 s wall) | 1.8x slower | 2,700 frames, full decode verified |

- Sign results identical: 540/540 analysis samples have the same sign boxes and labels; 65 coded boxes in both runs.
- Speech cues identical: P.112 + W.245a at 3.167 s, P.117 at 13.167 s, P.127 at 40.333 s.
- Lane lines drawn over the clip: 1,504 before, 1,554 after.
- Through the web UI (server restarted with the change): 20 s night clip processed in 44.6 s (before: about 133 s end to end), output 30 fps, cues P.112 + W.245a at 3.167 s and P.117 at 13.167 s.

## 3 Nov 2026: simplified lane path, same 90 s night clip (one run)

Same source as above (sha256 prefix c22caa280981, 5,400 frames at 60 fps). Rendered with `render()` using the web parameters (analysis 6 Hz, `lane_every=2`, `output_fps_cap=30`, `include_auxiliary=True`, no `max_seconds`), with `offline_video_lane.display: simple`. Summary file: `output/speed_simple_20261103/night_90s/summary.json` (not committed).

| Setting | Processing time | vs real time | Output |
|---|---:|---:|---|
| Simple lane path (UFLD-v2 polylines, no YOLOPv2 / surface veto), one run | 153.3 s elapsed (157.4 s wall) | 1.7x slower (153.3 / 90) | 2,700 frames at 30 fps, full decode verified (2,700 of 2,700) |

- Lane coverage: a lane was displayed in 178 of 540 analysis samples (33%). This is coverage, not accuracy.
- Sign output: 65 known signs and 159 uncertain signs, the same 65 coded boxes as the earlier runs above. This was not a sample-by-sample comparison on this run.
- Speech cues: P.112 + W.245a at 3.167 s, P.117 at 13.167 s, P.127 at 40.333 s.
- Compared with the 1 Nov run (165.0 s, precision lane path): the difference of about 12 s is within what single runs can separate. The lane change alone was not measured on this clip.
- The result video was rendered before the overlay header text change; the header text does not affect processing time.
