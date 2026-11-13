# Kịch bản demo 5 phút (14/11)

Chuẩn bị trước: mở ứng dụng ở chế độ video. Có sẵn các file trong `output/demo_20261114/`: clip live 30 s `M12/live_110s/render/test_video_offline_max.mp4` (bắt đầu từ 110 s của video demo), `M13/clips/demo_video/render/` và `M13/clips/night_90s/render/` (mỗi thư mục có `test_video_offline_max.mp4`).

1. (0:00–0:40) Giới thiệu: hệ thống nhận biển báo giao thông từ video hoặc ảnh, đọc nội dung và giải thích phạm vi áp dụng. Nói rõ: lane đã tắt mặc định; hệ thống ra "chưa xác định" khi thiếu bằng chứng làn.
2. (0:40–2:00) Clip live 30 s `M12/live_110s`: bật video. Chỉ vào mục "Cảnh báo phía trước" khi xuất hiện, ở khoảng 0:23 (ô tô trong làn xe mình) và 0:01 (ô tô bên trái làn). Nói: "Chú ý" là cảnh báo nhẹ, "Cẩn thận, <loại xe> phía trước" là cảnh báo mức cao; đây không phải thời gian va chạm. Nói thật: cảnh báo 0:01 là sai (xe ở làn bên trái), hệ thống đang ở giai đoạn thử nghiệm.
3. (2:00–3:00) Ảnh: tải một ảnh biển, cho xem kết quả và lý do áp dụng.
4. (3:00–4:00) Tra cứu luật: mở tab tra cứu, tìm một mã biển, cho xem nguồn.
5. (4:00–5:00) Số liệu: mở `docs/demo_20261114/RESULTS.md`, mục 0. Nói đúng: latency trung vị 97.5 s cho clip đêm 90 s (chưa đạt 90 s) và 119.3 s cho video demo 140 s; video demo có 10 cảnh báo (AI-reviewed, 5 hợp lý, 5 không hợp lý, không có nhãn người); clip đêm dùng để chỉnh ngưỡng nên số đó chỉ mang tính minh họa.

Phương án khi lỗi:
- Không chạy được video trực tiếp: mở `output/demo_20261114/M5/backup/` (video dự phòng).
- Cảnh báo phía trước gây nhiễu: tắt `forward_warning.enabled` (config) rồi chạy lại; biển báo không bị ảnh hưởng. Nếu chỉ muốn bỏ cảnh báo nguy_co: đặt `forward_warning.nguy_co_enabled: false`.
- Chậm: không xử lý video dài trên máy; dùng clip 30 s.
- Không được nói "va chạm" hoặc "sắp đâm".
