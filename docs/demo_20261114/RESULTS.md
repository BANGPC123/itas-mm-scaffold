# Kết quả demo 14/11 (dùng cho slide)

## 0. Cập nhật sau M13 và M12 (14/11, mới nhất)

Nguồn chính: `output/demo_20261114/M13/` và `output/demo_20261114/M12/`. Cảnh báo vật cản luôn ghi kèm số ca và "AI-reviewed (không phải nhãn người)".

### 0.1 Độ trễ (cấu hình M13, 3 lần chạy mỗi clip, trung vị)

| Clip | Elapsed 3 lần (s) | Trung vị elapsed | Trung vị wall | Mục tiêu | Nguồn |
|---|---|---:|---:|---|---|
| Đêm 90 s | 97.27 / 98.35 / 97.48 | 97.48 s | 101.53 s | ≤ 90 s: **chưa đạt** | M13/clips/night_90s/, M12/latency/night_90s_run{2,3}/ |
| Video demo 2 min 20 s | 115.44 / 123.10 / 119.29 | 119.29 s | 122.67 s | ≤ 148 s: đạt | M13/clips/demo_video/, M12/latency/demo_video_run{2,3}/ |

Wall lấy từ `baseline_meta.json`, elapsed từ `render/summary.json`. Chạy tuần tự, một GPU job tại một thời điểm (`GPU_LOCK`). Profile bật bằng `ITAS_PROFILE=1`.

### 0.2 Biển báo (không đổi)

- Gate (a): nhãn và box biển trùng 100% theo từng mẫu với M10. Đêm 540/540 mẫu, video demo 840/840 mẫu. Nguồn: so sánh `predictions.jsonl` (trường `signs`) M13 với M10.

### 0.3 Cảnh báo vật cản trước và sau M13

| Clip | Bản | Cảnh báo | nguy_co | chu_y | Câu nói | Câu/phút | Nguồn |
|---|---|---:|---:|---:|---:|---:|---|
| Đêm 90 s | M10 | 9 | 5 | 4 | 6 | 4.0 | M10/clips/night_90s/render/summary.json |
| Đêm 90 s | M13 | 5 | 1 | 4 | 3 | 2.0 | M13/clips/night_90s/render/summary.json |
| Video demo | M10 | 15 | 1 | 14 | 7 | 3.0 | M10/clips/demo_video/render/summary.json |
| Video demo | M13 | 10 | 0 | 10 | 6 | 2.57 | M13/clips/demo_video/render/summary.json |

- Clip đêm ở bảng này là **in-sample (dùng để chỉnh ngưỡng)**. Số này không dùng để đánh giá.
- Video demo là holdout: không chỉnh ngưỡng sau khi xem.
- Mục tiêu "chu_y đọc ≤ 4 câu/phút" trên video demo: đạt (2.57, 6 câu, AI-reviewed).

### 0.4 Chất lượng cảnh báo (AI-reviewed, không phải nhãn người)

| Clip | Vai trò | Mức | Số ca | hop_ly | khong_hop_ly | khong_ro | Nguồn |
|---|---|---|---:|---:|---:|---:|---|
| Video demo | holdout, review v3 | chu_y | 10 | 5 (50%) | 5 | 0 | M13/review/v3/demo_video/review_sheet.csv |
| Video demo | holdout, review v3 | nguy_co | 0 | - | - | - | như trên |
| Đêm 90 s | in-sample, verdict v2 | nguy_co | 1 | 0 | 1 | 0 | M13/review/v3/night_90s/review_sheet.csv |
| Đêm 90 s | in-sample, verdict v2 | chu_y | 4 | 2 | 1 | 1 | như trên |

Quyết định `nguy_co_enabled`: giữ `true`, vì không có cảnh báo nguy_co nào trên video demo để tắt. Nguồn và lý do: `M13/NGUY_CO_DECISION.md`.

### 0.5 Bộ lọc và phát lại (đêm, in-sample)

Cảnh báo đêm sau khi lọc (rider_merge + nguy_zone + lateral 0.25) khớp với phiếu v2 theo track, lớp, mức và thời điểm ±1.5 s: khong_hop 6 → 2, hop 2 → 2, khong_ro 1 → 1. Cấu hình lateral 0.15 hoặc 0.35 cho kết quả khác (xem `M13/THRESHOLDS.md`). **In-sample (dùng để chỉnh ngưỡng).**

### 0.6 Kiểm tra box

Trên clip đêm (nguồn 1920x1080, canvas 1280x720), viền box trong video trùng vị trí đã scale ở 5/5 cảnh báo. Vị trí chưa scale chỉ có 0.00 đến 0.06. Đây là bằng chứng cho M13.1.

### 0.7 Clip live 30 s (M12)

- File: `output/demo_20261114/M12/live_110s/render/test_video_offline_max.mp4`, bắt đầu từ `start_sec = 110`, dài 30 s, xử lý 27.6 s.
- Cảnh báo trong clip (4 ca, 2 đã nói; AI-reviewed, không phải nhãn người):
  - 0:00.67, ô tô bên trái làn xe mình, **đã nói**, theo luật là không hợp lý.
  - 0:06.67, ô tô làn trái (116.7 s của video gốc), không nói.
  - 0:23.0, ô tô trong làn xe mình phía trước, **đã nói**, hợp lý.
  - 0:29.0, ô tô (139 s của video gốc), không nói, hợp lý theo review v3.
  Có ít nhất một cảnh báo hợp lý, nhưng có một cảnh báo không hợp lý được nói ra. Chưa chấm bằng review v3 cho clip này.
- Clip 0–30 s cũ (`demo_short_30s`) không có cảnh báo nào.

### 0.8 Chưa làm / chưa đo

- Latency đêm 90 s chưa đạt (97.5 s).
- Không có nhãn người nên chưa đo độ chính xác của cảnh báo; review v3 là AI-reviewed.
- Bộ lọc được chỉnh trên clip đêm (in-sample); độ nhạy lateral đã được ghi trong THRESHOLDS.md.
- Phần E2E trình duyệt và ảnh chụp 1440 px (M10) không nằm trong cập nhật này.

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
