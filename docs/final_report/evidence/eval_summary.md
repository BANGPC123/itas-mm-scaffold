# Field segment evaluation, configurations B0 / N87 / N75 / N64 (T8)

Same 15 Hanoi segments (3 eval videos x 5 x 60 s, original field videos, unannotated), 6 Hz, same detector and lanes.
B0 is the 2026-10-29 run of the old code at 0.87 (`output/kiet_wp_20261029/field_eval/B_*`), reused, not re-run.
N87/N75/N64 are new code (T5-T7) at 0.87 / 0.75 / T2 = 0.6413 (`output/kiet_wp_20261029/field_eval/N*_*`).
No correctness grading here; review sheets are in `output/final_20261102/review/`.

| Config | Runs | Physical tracks with a code | Distinct codes | Median confirmation (ms) | Inference FPS (median) | Render wall time (s, sum) |
|---|---:|---:|---:|---:|---:|---:|
| B0 (old code, 0.87) | 15 | 75 | 15 | 333 | 2.82 | 3397 |
| N87 (new code, 0.87) | 15 | 75 | 15 | 333 | 3.05 | 3099 |
| N75 (new code, 0.75) | 15 | 86 | 20 | 333 | 3.00 | 3121 |
| N64 (new code, T2) | 15 | 96 | 23 | 333 | 3.01 | 3076 |

Missing runs: B: 0, N87: 0, N75: 0, N64: 0

## Wiring check (F)

| Config | Frame-level coded decisions with recognition_confidence in [0.75, 0.87) |
|---|---:|
| B0 (old code, 0.87) | 0 |
| N87 (new code, 0.87) | 0 |
| N75 (new code, 0.75) | 213 |
| N64 (new code, T2) | 213 |

N87 must be 0 (threshold 0.87 rejects that range); N75 must be > 0, otherwise the config did not take effect.

- N75 (new code, 0.75): 40 coded episodes not matched by B0 (same code, within 2 s, IoU >= 0.2).
- N64 (new code, T2): 52 coded episodes not matched by B0 (same code, within 2 s, IoU >= 0.2).
