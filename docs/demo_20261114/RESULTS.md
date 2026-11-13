# Kết quả demo 14/11 (dùng cho slide)

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
