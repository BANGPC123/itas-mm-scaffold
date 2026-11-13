# Recut check for live_30s_source.mp4 (M16)

Command (CPU; output is a temp file, the existing file was not touched):

    ffmpeg -hide_banner -y -ss 30 -i docs/demo/itas-mm_demo_video.mp4 -t 30 -an -c:v libx264 -preset medium -b:v 5800k -pix_fmt yuv420p -r 30 output/demo_20261114/M16/tmp_live_30s_recut.mp4

ffmpeg binary: imageio_ffmpeg ffmpeg-win-x86_64-v7.1.exe (environment itas-mm).

| File | Frames (decoded) | Duration | Size | Resolution |
|---|---:|---:|---:|---|
| output/demo_20261114/M12/e2e_final/live_30s_source.mp4 (existing) | 900 | 30.00 s | 21,665,180 B | 1280x720 |
| output/demo_20261114/M16/tmp_live_30s_recut.mp4 (temp) | 900 | 30.00 s | 22,894,874 B | 1280x720 |

Mean absolute pixel difference between the two (0–255): first frame 2.45, middle frame 1.97, last frame 2.52. Same content from 30 s, differences from re-encoding only.

The temp file is kept (no deletion). It is under output/ and not in git.
