# Kết quả demo 14/11 (dùng cho slide)

## 0. Cập nhật sau M13 và M12 (14/11, mới nhất)

Nguồn chính: `output/demo_20261114/M13/` và `output/demo_20261114/M12/`. Cảnh báo vật cản luôn ghi kèm số ca và "AI-reviewed (không phải nhãn người), chưa đo recall".

### 0.1 Độ trễ (trung vị 3 lần chạy, cấu hình M13)

| Clip | Trung vị elapsed | Trung vị wall | Nguồn |
|---|---:|---:|---|
| Video demo (140 s) | 119.3 s | 122.7 s | `M13/clips/demo_video/`, `M12/latency/demo_video_run{2,3}/` |
| Đêm 90 s | 97.5 s | 101.5 s | `M13/clips/night_90s/`, `M12/latency/night_90s_run{2,3}/` |

Chạy tuần tự, một GPU job tại một thời điểm. Đêm 90 s chưa đạt mục tiêu ≤ 90 s.

### 0.2 Biển báo (không đổi)

- Gate (a): nhãn và box biển trùng 100% theo từng mẫu với M10. Đêm 540/540 mẫu, video demo 840/840 mẫu.

### 0.3 Cảnh báo vật cản (holdout = video demo)

Số liệu holdout lấy từ review độc lập `M11/v3` (commit `665310f` ở nhánh `work/audit-20261114`, phiếu `output/demo_20261114/M11/v3/demo_video/review_sheet.csv`). Đây là AI-reviewed, chưa đo recall.

| Bản | Cảnh báo | hop_ly | khong_hop_ly | nguy_co |
|---|---:|---:|---:|---:|
| v2 (trước M13) | 15 | 6 | 8 | 1 |
| v3 (M13) | 10 | 5 | 5 | 0 |

Kiểm tra chéo của M13 (cùng luật, phiếu `M13/review/v3/demo_video/`) cũng ra 5 hợp lý và 5 không hợp lý trên 10 cảnh báo.

Clip đêm không dùng để trình bày trực tiếp (còn 1 cảnh báo nguy_co khong_hop_ly ở 60.7 s). Số đêm chỉ dùng cho chỉnh ngưỡng:

- Đêm 90 s, **in-sample (dùng để chỉnh ngưỡng)**: cảnh báo 9 (M10) → 5 (M13), khong_hop_ly 6 → 2, hop_ly 2 → 2, khong_ro 1 → 1. Nguồn `M13/replay_night_final_config.json`.

Câu nói: video demo 6 câu trong 140 s (2.57/phút, dưới mục tiêu 4/phút). Đêm 3 câu trong 90 s (2.0/phút).

Quyết định `nguy_co_enabled`: giữ `true`, vì video demo không có cảnh báo nguy_co nào để tắt. Nguồn: `M13/NGUY_CO_DECISION.md`.

### 0.4 Giọng cảnh báo (prewarm)

14/14 câu (7 nguy_co theo label_vi, 7 chu_y) có audio, độ dài 0.91–1.87 s. Nguồn `M12/PREWARM.md`.

### 0.5 Kiểm tra box

Clip đêm (nguồn 1920x1080): viền box trong video trùng bbox đã scale ở 5/5 cảnh báo; vị trí chưa scale chỉ 0.00–0.06. Đây là bằng chứng cho M13.1.

### 0.6 Clip live 22 s (M12)

- File: `output/demo_20261114/M12/live_118s/render/test_video_offline_max.mp4`, `start_sec = 118`, dài 22 s, xử lý 19.1 s.
- Hai câu được đọc, cả hai là ô tô trong làn xe mình phía trước (hop_ly):
  - 0:00.67 (118.7 s của video gốc): chấm bổ sung của tôi, không có trong review v3.
  - 0:21.0 (139 s của video gốc): hop_ly theo review v3.
- Cảnh báo 133.0 s không được đọc (gate giọng). Cảnh báo ở 133 s và 139 s là hop_ly theo review v3.
- Nhịp: 2 câu trong 22 s = 5.45 câu/phút. Con số này cao vì clip ngắn, không đại diện cho cả video.
- Clip live cũ 110 s (`M12/live_110s/`) có câu ở 0:00.67 là ô tô làn trái (sai) nên không dùng.

### 0.7 Chưa làm / chưa đo

- Latency đêm 90 s chưa đạt (97.5 s).
- Không có nhãn người nên chưa đo độ chính xác hay recall của cảnh báo.
- Bộ lọc chỉnh trên clip đêm (in-sample). Độ nhạy lateral: xem `M13/THRESHOLDS.md`.
- Không có số so sánh latency với M10 (144 s) trong mục này. Số M10 chỉ có ở phần cũ bên dưới.

---

## Các mục cũ (trước M13), giữ để đối chiếu

Mọi số có file nguồn trong `output/demo_20261114/`. Số nào chưa có nguồn thì ghi "chưa đo".

## 1. Độ trễ (xử lý offline, máy RTX 3050 4 GB)

| Clip | Baseline (làn bật) | Làn tắt (M1) | Làn tắt + cảnh báo vật cản (bản cuối) | Nguồn |
|---|---|---|---|---|
| Đêm 90 s | 178.36 s | 113.57 s | 179.83 s | M0/night_90s/baseline_meta.json; M1/night_90s/baseline_meta.json; M5/clips/night_90s/baseline_meta.json |
| Video demo 2 min 20 s | 169.66 s | 128.70 s | 146.21 s | M0/demo_video/...; M1/demo_video/...; M5/clips/demo_video/... |

Ghi chú: trên bản cuối, cảnh báo vật cản tốn thêm thời gian. Latency của bộ phát hiện vật cản đo riêng bằng probe: p50 42.04 ms/sample (`output/demo_20261114/M2v2/latency_probe_night.json`). Mục tiêu "clip 90 s ≤ 90 s" **chưa đạt**.

## 2. Biển báo (không đổi)

- Lane tắt: boxes, nhãn và applicability của biển trùng 100% với baseline trên clip đêm (65 mã) và video demo (420 mã). Nguồn: M1/gate_night.json, M1/gate_demo.json; bản cuối M5/clips/gate_*.json, M5/clips/codes_*.json.
- Cue giọng nói biển: clip đêm 3/3 cue giữ nguyên; video demo 15/15 cue giữ nguyên văn bản, 2 cue bị lùi 2 s vì cảnh báo nguy_co chen vào. Nguồn: M2/FORWARD_WARNING.md.

## 3. Cảnh báo phía trước (chưa có nhãn người)

| Clip | Số cảnh báo | Mỗi phút | Chú ý / nguy cơ | Nguồn |
|---|---|---|---|---|
| Đêm 90 s | 8 | 5.34 | 6 / 2 | M5/clips/night_90s/render/predictions.jsonl |
| Video demo | 23 | 9.87 | 16 / 7 | M5/clips/demo_video/render/predictions.jsonl |

Ngưỡng cuối (chu_y bottom 0.80, persist 6) được chỉnh trên clip đêm; clip demo không dùng để chọn. Gate số cảnh báo ≤ 6/phút **không đạt** trên clip demo. Không có nhãn người nên chưa đo độ chính xác của cảnh báo.

## 4. Nhận dạng và detector (so sánh model)

Xem `docs/audit_20261114/MODEL_REVIEW.md` (nhánh `work/audit-20261114`, đã merge). Kết luận: không thay model detector hiện tại; không thay recognizer MobileNetV3-small (M3 không đạt gate thăng hạng). A5 one-stage: không hợp lệ, không đưa lên slide (xem MODEL_REVIEW.md).

## 5. Kiểm thử

- `pytest -q tests`: 636 passed, 11 skipped, 0 fail (sau merge). Baseline M0 là 623 passed.
- `node --test frontend/src/*.test.js`: 14/14 pass.
- `npm --prefix frontend run build`: OK.

## 6. Cái chưa làm / chưa đo

- Latency mục tiêu 90 s: chưa đạt.
- Gate cảnh báo vật cản (b) latency ≤ 15 ms, (c) ≤ 6/phút: không đạt, báo số thật.
- Đánh giá độ đúng của cảnh báo: chưa đo (chưa có nhãn người). Bảng review `output/demo_20261114/M2/review/` chưa tạo.
- E2E trên trình duyệt và ảnh chụp 1440 px: chưa làm (xem báo cáo cuối).
