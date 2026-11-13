
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
