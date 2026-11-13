# M13.6 - Quyết định nguy_co_enabled và review v3

## Quy tắc áp dụng

Theo FORWARD_REVIEW.md (luật chấm cố định): nếu trên video demo còn cảnh báo nguy_co rõ ràng vô lý thì đặt `forward_warning.nguy_co_enabled: false` và giữ chu_y. Không chỉnh ngưỡng nào sau khi xem video demo.

## Quyết định

- **`nguy_co_enabled: true` (giữ nguyên).**
- Lý do: trên video demo, bản render cuối (M13.6) không có cảnh báo nguy_co nào (`output/demo_20261114/M13/clips/demo_video/render/summary.json`, 10 cảnh báo, đều chu_y). Không có cảnh báo nguy_co vô lý trên demo để tắt.
- Ghi chú: cảnh báo nguy_co duy nhất trên video demo của M11 v2 (track 281, ô tô sát bó vỉa, 60,667 s) đã biến mất ở cấu hình M13.5. Nguyên nhân là bộ lọc vùng hẹp và lateral, không phải chỉnh ngưỡng sau khi xem demo.
- Lưu ý cho chủ dự án: trên clip đêm (dùng để chỉnh ngưỡng, in-sample) vẫn còn 1 cảnh báo nguy_co, ở 60,667 s, track 342, lớp `person`, và phiếu v3 chấm là `khong_hop_ly` (người lái xe máy bị gán người đi bộ). Quyết định này không dựa trên clip đêm. Nếu muốn tắt nguy_co trong bản trình diễn, đó là quyết định của chủ dự án.

## Review v3 (AI-reviewed, không phải nhãn người)

Nguồn: `output/demo_20261114/M13/review/v3/{demo_video,night_90s}/review_sheet.csv` (ảnh sự kiện và ảnh trước 0,5 s, `sources.json`).

| Clip | Vai trò | Mức | Tổng | hop_ly | khong_hop_ly | khong_ro |
|---|---|---|---:|---:|---:|---:|
| demo_video | holdout | nguy_co | 0 | 0 | 0 | 0 |
| demo_video | holdout | chu_y | 10 | 5 | 5 | 0 |
| night_90s | in-sample (dùng để chỉnh ngưỡng) | nguy_co | 1 | 0 | 1 | 0 |
| night_90s | in-sample (dùng để chỉnh ngưỡng) | chu_y | 4 | 2 | 1 | 1 |

Ghi chú về cách chấm:
- Video demo: mỗi cảnh báo được xem ảnh sự kiện và chấm theo luật ở đầu FORWARD_REVIEW.md.
- Clip đêm: các cảnh báo vẫn còn sau M13.5 khớp với phiếu M11 v2 theo track, lớp, mức và thời điểm ±0,5 s, nên lấy lại verdict v2 (ghi trong cột `ai_reason`). Đây không phải chấm mới.
- Ghi chú các ca khong_hop_ly trên demo: 003 (xe máy xa, bên trái), 004 (xe máy sát mép phải, xa), 005 (người lái xe máy bị gán người đi bộ), 007 và 008 (ô tô làn trái, chưa cắt vào).

## Kiểm tra render (M13.6)

- Gate (a), biển: nhãn và box biển trùng 100% theo từng mẫu với M10. Clip đêm 540/540 mẫu, video demo 840/840 mẫu (`predictions.jsonl`, trường `signs`).
- Box cam và đỏ: trên clip đêm, viền màu trong video trùng bbox đã scale (1,0 trên 5/5 cảnh báo), trong khi vị trí chưa scale chỉ 0,00 đến 0,06. Đây là bằng chứng cho M13.1.
- Số câu nói: clip đêm 3 câu, 2,0 câu/phút; video demo 6 câu, 2,57 câu/phút (mục tiêu ≤ 4 câu/phút đạt).
- Thời gian (lần chạy 1, chưa phải trung vị): clip đêm elapsed 97,3 s, wall 100,8 s; video demo elapsed 115,4 s, wall 119,9 s. Trung vị 3 lần sẽ đo ở M12.
