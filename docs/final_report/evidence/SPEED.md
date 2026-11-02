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
