# Kịch bản demo 5 phút (14/11)

Chuẩn bị trước: mở ứng dụng ở chế độ video, có sẵn 3 file trong `output/demo_20261114/M5/clips/`: `demo_short_30s/`, `demo_video/`, `night_90s/` (mỗi thư mục có `render/test_video_offline_max.mp4`).

1. (0:00–0:40) Giới thiệu: hệ thống nhận biển báo giao thông từ video hoặc ảnh, đọc nội dung và giải thích phạm vi áp dụng. Nói rõ: lane đã tắt mặc định; hệ thống ra "chưa xác định" khi thiếu bằng chứng làn.
2. (0:40–2:00) Video ngắn `demo_short_30s`: bật video, nghe giọng nói biển. Chỉ vào mục "Cảnh báo phía trước" khi xuất hiện. Nói: "Chú ý" là cảnh báo nhẹ, "Cẩn thận" là nguy cơ tiếp cận nhanh theo chỉ báo box; đây không phải thời gian va chạm.
3. (2:00–3:00) Ảnh: tải một ảnh biển, cho xem kết quả và lý do áp dụng.
4. (3:00–4:00) Tra cứu luật: mở tab tra cứu, tìm một mã biển, cho xem nguồn.
5. (4:00–5:00) Số liệu: mở `docs/demo_20261114/RESULTS.md`. Nói đúng: latency 114–180 s cho 90 s (chưa đạt mục tiêu 90 s); cảnh báo 5–10 lần/phút (chưa đạt ngưỡng 6/phút); không có nhãn người.

Phương án khi lỗi:
- Không chạy được video trực tiếp: mở `output/demo_20261114/M5/backup/` (video dự phòng).
- Cảnh báo phía trước gây nhiễu: tắt `forward_warning.enabled` (config) rồi chạy lại; biển báo không bị ảnh hưởng.
- Chậm: không xử lý video dài trên máy; dùng clip 30 s.
- Không được nói "va chạm" hoặc "sắp đâm".
