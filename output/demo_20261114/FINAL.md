# FINAL - demo 14/11 (M6)

Nhánh: `work/demo-20261114` (chưa push, chưa merge `main`, chưa PR).
Commit cuối: 68d3f9c. Commit chính theo milestone: 84b2157 (M0), 3cca60b (M1), d04a6dd (M1.5), dd76c7d + 8534eac (M2), 18689c4 (merge audit), 6a77bd4 (M5 README/docs).
Nhánh Codex: `work/audit-20261114` (worktree D:/DSP391m/project/wt-audit), đã merge vào nhánh demo ở 18689c4.

## Bảng gate

| Gate | Kết quả | Nguồn |
|---|---|---|
| M0 baseline (render, test, build) | Đo xong. pytest 623 pass; node 14/14; build OK | output/demo_20261114/M0/BASELINE.md |
| M0.4 profile lane tắt | Đo xong bằng cProfile (có overhead). Detector ~29 s, make_anchors ~20 s | output/demo_20261114/M0/PROFILE.md |
| M1 (a) biển/nhãn/speech giữ nguyên khi tắt lane | PASS trên 2 clip (100% mẫu) | output/demo_20261114/M1/gate_night.json, gate_demo.json |
| M1 latency 90 s ≤ 90 s | FAIL: 113.57 s (lane tắt), 81.84 s chỉ khi dùng detector 640 (không qua gate) | output/demo_20261114/M1/LATENCY.md, M1_5.md |
| M1.5 (e) detector 640 | FAIL: clip đêm mất 65/65 mã, demo mất 190/420 | output/demo_20261114/M1e/codes_*.json |
| M1.5 (d) FP16 | FAIL: 146/410 mẫu lệch boxes; clip đêm mất 14 mã | output/demo_20261114/M1d/*.json |
| M1.5 (b),(c),(a) | Không áp dụng/không cần: encoder đã veryfast; TTS đã cache; recognizer chiếm ~2 ms | output/demo_20261114/M1/M1_5.md |
| M2 (a) biển không đổi | PASS: 0 lệch boxes/nhãn; mã 420/420 | output/demo_20261114/M2v2/gate_demo.json, codes_demo.json |
| M2 (b) latency ≤ 15 ms/sample | FAIL: p50 42.04 ms | output/demo_20261114/M2v2/latency_probe_night.json |
| M2 (c) ≤ 6 cảnh báo/phút | FAIL: demo 9.87/phút (sau chỉnh ngưỡng trên clip đêm); đêm 5.34 | output/demo_20261114/M5/clips/*/render/predictions.jsonl |
| M2 (d) không crash | PASS | output/demo_20261114/M5/clips/*/render/summary.json |
| M3 (recognizer) | FAIL (theo mục 6-7 plan, không tích hợp) | docs/audit_20261114/RECOGNIZER_PROMOTION.md, MODEL_REVIEW.md (nhánh audit) |
| M4 (audit nhánh) | Hoàn thành, không port file nào | docs/audit_20261114/BRANCH_AUDIT.md |
| M5 tests | 636 pass, 11 skip, 0 fail; node 14/14; build OK | M5 log trong terminal (chưa lưu file riêng) |

## Bản cuối (cấu hình đã commit)
- Clip đêm 90 s: 179.83 s (cảnh báo bật), 8 cảnh báo (5.34/phút). Biển không đổi.
- Video demo: 146.21 s, 23 cảnh báo (9.87/phút). Biển không đổi.
- Đoạn demo 30 s: 35.52 s, 4 cảnh báo.
- Ghi chú: bản cuối chậm hơn bản M1 (114 s) vì bật cảnh báo. Không đạt mục tiêu 90 s.

## Việc bỏ dở / chưa làm
- E2E trên trình duyệt: đã mở giao diện ở 1440 px (screenshot, chỉ xem), **chưa** upload clip hay ảnh qua giao diện, chưa nghe giọng nói. Backend và frontend đã tắt.
- Ảnh chụp 1440 px vào `output/demo_20261114/M5/screens/`: chưa lưu file.
- Bảng review cảnh báo (`output/demo_20261114/M2/review/`, cột hop_ly): chưa tạo.
- Chưa đo ffmpeg encode riêng trong profile.
- A5 one-stage: không hợp lệ (theo plan mục 7), không đưa lên slide.

## Cách chạy demo
```bash
"D:/miniconda/envs/itas-mm/python.exe" -m uvicorn backend.main:app --port 8000
npm --prefix frontend run dev
```
Clip dự phòng: output/demo_20261114/M5/backup/demo_video_result.mp4 và demo_short_30s_result.mp4. Kịch bản: docs/demo_20261114/DEMO_SCRIPT.md. Số liệu: docs/demo_20261114/RESULTS.md.
Lưu ý: chạy pytest và uvicorn với `PYTHONNOUSERSITE=1` (gói user-site `sniffio` gây lỗi import; không cài gói vào env itas-mm).

## Cờ và cách rollback (configs/perception.yaml, không xóa code)
- `offline_video_lane.enabled`: false (lane tắt). Đặt true để bật lại lane.
- `forward_warning.enabled`: true (đang bật, gate b/c không đạt). Đặt false để tắt cảnh báo vật cản.
- `sign_detector.video_multiscale`: true; `video_image_size`: 960 (giữ). Đặt lại nếu đã đổi.
- `sign_detector.half`: false (FP16 đã thử, không qua gate).
- Rollback cả nhánh: `git checkout 84b2157` (trạng thái trước M1) hoặc đặt các cờ trên.

## Dừng
Theo plan, dừng ở M6. Không merge `main`, không push, không PR. Không xóa file hay nhánh nào. GPU lock đã trả (không còn `GPU_LOCK`).

## Vòng 2 (M8, M9, M10) - kết quả đo

Nguồn số liệu: `output/demo_20261114/M8/M8_TABLE.md` (sinh từ các `summary.json`), `output/demo_20261114/M10/clips/*/render/summary.json`, `output/demo_20261114/M10/e2e/e2e_report.json`.

### Latency (wall-clock render, giây)

| Clip | M1 (lane tắt, không cảnh báo) | Trước M8 (HEAD) | Sau M8/M9 (M10) | Gate |
|---|---|---|---|---|
| Đêm 90 s | 113.57 (`M1/LATENCY.md`) | 119.0 và 133.2 (2 lần, `M8/before_night*/`) | 106.7 (`M10/clips/night_90s/`) | <= 131 s: PASS |
| Video demo | 128.70 (`M1/LATENCY.md`) | 153.1 (`M8/m8_demo/`, trước M8.3c) | 144.0 (`M10/clips/demo_video/`) | <= 148 s: PASS (cách ngưỡng dưới 5 s, chênh lệch đo lặp trên cùng mã là khoảng 15 s) |
| 30 s đầu video demo | chưa đo | - | 34.2 | chưa đặt gate |

Thời gian tính trong 'summary.json' là `elapsed_seconds`. Các lần đo lặp cùng mã cho thấy nhiễu khoảng ±10 s, nên ghi nhận số trên là một lần đo, không phải trung bình.

### Thay đổi từng bước (M8)

- M8.0: hai instance detector theo kích thước ảnh. Micro-benchmark (`M8/anchor/bench_anchor.json`): 14.31 và 13.78 ms/lần một instance xen kẽ, 13.75 ms/lần hai instance, không khác biệt đáng kể, hộp trùng nhau. Đã hoàn tác.
- M8.1: overlay chỉ pha màu trong hình chữ nhật bao vùng (commit `7920203`). Thời gian overlay 7.4-8.3 s -> 1.5 s (`M8_TABLE.md`).
- M8.2: FP16 cho COCO trên CUDA chậm hơn (detect 15.2 s so với 10.7-12.6 s). Cờ `half` giữ `false`. Micro-benchmark (`M8/detect/bench_detect.json`): fp32 16.5 và 15.7 ms/khung, fp16 17.3 ms/khung.
- M8.2b: resize cạnh dài 640 trước khi track chậm hơn (18.0 ms/khung). Không áp dụng.
- M8.3c: COCO chạy trên mỗi 2 mẫu phân tích (3 Hz). Ngưỡng persist và `ttc_k` đổi để giữ nguyên thời gian thực: `chu_y_persist` 6 -> 3, `nguy_co_persist` 2 -> 1, `ttc_k` 3 -> 2. Detect đêm 10.7 s -> 6.2 s; demo 16.4 s -> 8.4 s. Số cảnh báo demo giảm từ 23 xuống 15.

Lưu ý: commit `0aaf4c8` gộp M8.2 (cờ FP16, giữ tắt) và M8.3c vì cùng một file cấu hình và mã. Quy tắc một thay đổi một commit không được giữ đúng ở bước này.

### Gate (a) biển, nhãn, applicability so với M1

| Clip | Mẫu so sánh | Lệch boxes/nhãn | Nguồn |
|---|---|---|---|
| Đêm 90 s (mọi bước M8, M9, M10) | 540 | 0 | `M10/clips/night_90s/gate_vs_M1.json` |
| Video demo | 840 | 0 | `M10/clips/demo_video/gate_vs_M1.json` |
| 30 s đầu | 180 | 0 | `M10/clips/demo_short_30s/gate_vs_M1_prefix.json` (so với 180 mẫu đầu của M1 demo) |

`speech_equal` là `false` trên clip đêm và demo vì M9 đã tắt giọng biển có chủ đích. Số liệu này không dùng làm gate giọng nói. Với clip 30 s, baseline giọng được đặt rỗng để so sánh, nên kết quả speech không có ý nghĩa.

### Cảnh báo và giọng đọc (M9)

| Clip | Cảnh báo | Câu được đọc | Câu được đọc / phút | Nguồn |
|---|---|---|---|---|
| Đêm 90 s | 9 (5 nguy_co) | 6 | 4.0 | `M10/clips/night_90s/render/summary.json` |
| Video demo | 15 (1 nguy_co) | 7 | 3.0 (mục tiêu <= 4) | `M10/clips/demo_video/render/summary.json` |
| 30 s đầu | 0 | 0 | 0 | `M10/clips/demo_short_30s/render/summary.json` |

Giọng biển đã tắt (`interaction.sign_speech.enabled: false`). Danh sách biển vẫn có: 6 biển trên demo, 3 trên đêm.

### E2E trên giao diện (headless Chrome qua DevTools)

Script: `output/demo_20261114/M10/e2e/e2e_cdp_20261114.mjs`. Log: `e2e_run.log`, `network_log.json`, `console_log.json`, `e2e_report.json`.

Các bước đã chạy:
1. Tải clip 30 s (48-78 s của video demo, `e2e/window_48s_78s.mp4`) qua tab Video. Phân tích xong, video kết quả phát đủ 30 s.
2. Danh sách "Cảnh báo phía trước" hiện 4 cảnh báo, trong đó 2 có "Có đọc" (`07_forward_list_seek.png`). Bấm mốc 00:12 tua đúng vị trí.
3. Yêu cầu giọng đọc qua mạng: `Chú ý người đi bộ`, `Cẩn thận, xe phía trước tiến gần`, và một yêu cầu thứ ba không lấy được nội dung (postData không có trong log). Không có yêu cầu giọng cho biển báo.
4. Tab Ảnh: ảnh khung 61 s của video demo. Có 4 khung vật cản (3 trong vùng). Không có nút "Đọc kết quả" khi giọng biển tắt. Danh sách biển có 11 dòng.
5. Tra cứu luật từ dòng biển đầu tiên (`09_legal_lookup.png`).

Console: chỉ có lỗi 404 của `favicon.ico`. Không có lỗi JavaScript và không có lỗi mạng ngoài 404 đó.

Ảnh chụp 1440 px: `output/demo_20261114/M10/screens/01_home_video_tab.png` ... `09_legal_lookup.png`.

Chưa làm:
- Chưa đo thời gian xử lý end-to-end qua giao diện (log upload không ghi đồng hồ riêng; 30 s clip render khoảng 34 s theo `summary.json`).
- Chưa kiểm tra giọng đọc có phát ra âm thanh không. Trình duyệt headless chạy với `--autoplay-policy`, nhưng log không có sự kiện phát âm.
- Trình duyệt headless đầu tiên bị dừng khi lỗi script. Tiến trình Chrome đó còn sống, không kill được vì lệnh bị từ chối tự động. Cần đóng tay nếu còn thấy trong Task Manager.
- `RESULTS.md`, `DEMO_SCRIPT.md` và bảng review cảnh báo (`M11`) chưa cập nhật theo số mới ngoài mục này.

## Vòng 3 (M13, M12) - 14/11

Nhánh: `work/demo-20261114`. Chưa push, chưa merge main. Commit cục bộ, mã milestone trong message.

### Commit

| Mã | Hash | Nội dung |
|---|---|---|
| M13.1 | 062ab3f | Box cảnh báo được scale từ khung nguồn 1920x1080 sang canvas 1280x720; bbox nguồn giữ trong metadata |
| M13.2 | bf0aa8e | Câu nguy_co đọc theo label_vi (không còn "xe phía trước" cho người đi bộ hoặc xe máy); prewarm lại |
| M13.3 | fab8fc6 | Tối đa 1 câu nguy_co mỗi mẫu, cách nhau tối thiểu 2 s |
| M13.4 | e6c0c87 | `M13/ERROR_ANALYSIS.md`: phân loại 14 ca khong_hop_ly của M11 v2, viết trước khi sửa lọc |
| M13.5 | 868e3e4 | Bộ lọc, mỗi bộ một cờ; chỉnh ngưỡng trên clip đêm; `M13/THRESHOLDS.md` |
| M13.6 | b02663d | Render demo và đêm với cấu hình M13; review v3; quyết định nguy_co |
| M13 READY | d67a491 | Ghi hash vào `M13/READY_FOR_REVIEW.txt` |
| M12 | 09d4a42 | Trung vị latency 3 lần; clip live; cập nhật RESULTS và DEMO_SCRIPT |

### Gate và quyết định

- Biển: nhãn và box trùng M10 100% theo mẫu (đêm 540/540, demo 840/840).
- Box: trên clip đêm, viền trong video trùng bbox đã scale ở 5/5 cảnh báo (vị trí chưa scale 0.00–0.06).
- `nguy_co_enabled`: giữ `true`. Không có cảnh báo nguy_co nào trên video demo để tắt. Nguồn `M13/NGUY_CO_DECISION.md`.
- Không chỉnh ngưỡng sau khi xem video demo. Clip đêm chỉ dùng để chỉnh ngưỡng (in-sample).

### Số liệu

- Latency (trung vị 3 lần): video demo 140 s, elapsed 119.3 s, wall 122.7 s. Đêm 90 s, elapsed 97.5 s, wall 101.5 s. Đêm 90 s chưa đạt mục tiêu 90 s.
- Holdout video demo, theo review độc lập `M11/v3` (commit `665310f`, nhánh audit): 15 → 10 cảnh báo; khong_hop_ly 8 → 5; hop_ly 6 → 5; không có nguy_co. AI-reviewed, chưa đo recall.
- Đêm 90 s (in-sample, dùng để chỉnh ngưỡng): 9 → 5 cảnh báo; khong_hop_ly 6 → 2; hop_ly 2 → 2.
- Giọng cảnh báo: 14/14 câu prewarm có audio, 0.91–1.87 s (`M12/PREWARM.md`).
- Clip live: `M12/live_118s`, start 118 s, 22 s. Hai câu đọc, cả hai là ô tô trong làn phía trước (hợp lý). Một câu (0.67 s) là chấm bổ sung của tôi, không có trong review v3. Clip 110 s bị bỏ vì câu 0.67 s là ô tô làn trái.

### Rollback (configs/perception.yaml, không xóa code)

- `forward_warning.enabled: false`: tắt toàn bộ cảnh báo vật cản.
- `forward_warning.nguy_co_enabled: false`: chỉ tắt nguy_co, giữ chu_y.
- `rider_merge_enabled`, `nguy_zone_enabled`, `lateral_filter_enabled`: tắt từng bộ lọc (mặc định bật).
- `track_class_reset_enabled`, `track_min_age_samples`: đang tắt, đã thử trên clip đêm.
- `voice_nguy_gap_s`: khoảng cách tối thiểu giữa hai câu nguy_co (mặc định 2 s).

### Chưa làm

- Latency clip đêm 90 s chưa đạt (97.5 s).
- Không có nhãn người nên chưa đo độ chính xác hay recall.
- Bộ lọc chỉnh trên clip đêm; độ nhạy lateral đã ghi trong `M13/THRESHOLDS.md`.
- Nhịp câu nói trên clip live 22 s là 5.45 câu/phút, vì clip ngắn; không đại diện cho cả video.
