# Demo clips (T11)

Cut on 2026-10-31 from the field-evaluation source videos in `data/manifests/field_video_20261029/sources.csv`
(`role = eval`). Each clip is 90 s, re-encoded with H.264 (CRF 22) and AAC audio using the imageio-ffmpeg
binary bundled in `D:\miniconda\envs\itas-mm\Lib\site-packages\imageio_ffmpeg\binaries\` (ffmpeg 7.1).
Clip time 0 = source time given below. All clips are 1920x1080 as sources.

| Clip file | Source video (channel, upload date) | Source sha256 (manifest) | Cut (source time) | Clip sha256 | Why this clip |
|---|---|---|---|---|---|
| `day_west_lake_bPFWcTjKdBU_600s_90s.mp4` | `bPFWcTjKdBU` (Raw Hanoi Sounds, 2026-01-02) | `e85b6c4042dbc2b31e1fea251edb3eaff33b772267fd3ab3b0df0a2b9da9dd3e` | 600 s to 690 s, daytime, West Lake road | `804cbe68014b86ce49557f07b564d5cd8ca9308771f62b3ca2405baa5dc51fb0` | Daytime Hồ Tây road; the AI labels include P.131a and P.103a around source 611-669 s |
| `night_hanoi_YQl9c5gCjhg_30s_90s.mp4` | `YQl9c5gCjhg` (Vietnam Silent Routes, 2025-01-24) | `1892465ad1438e2a626d34e770a56e895777193319fa04937c0ce2e8bc012d05` | 30 s to 120 s, night, Hanoi | `c22caa280981dbba4b738e6296a5d70550bce227143e89c3c57a2ced96791b21` | Night clip with the three-sign overlap at source 31.7-34.8 s (clip 1.7-4.8 s); P.117 near clip 13 s |
| `day_hanoi_many_signs_6JuxWcAf-wM_825s_90s.mp4` | `6JuxWcAf-wM` (Vietnam Silent Routes, 2025-01-13) | `fd06b48e245b77b95af3e11d8d27cde4279c3eec68b3bc4e09165b63ee70316d` | 825 s to 915 s, daytime, Hanoi | `164c51e2e994b6cad70c82e69042d1cfb41a110540bc72d2c2003c1818ffb4e5` | Dense prohibition and danger signs (AI labels P.130, P.106a, P.104, W.207, W.245a, P.105) around source 830-884 s |

Notes:
- Source videos are third-party YouTube uploads, used for research evaluation only, not redistributed in the repo (`output/` is gitignored).
- Labels named above are AI-assisted and unverified; the review sheets in `output/final_20261102/review/` record the CP2 grading.
- Field-test sources are not annotated with ground truth; use these clips to demonstrate the pipeline, not to claim accuracy.
