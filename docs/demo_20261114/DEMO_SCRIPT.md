# Kịch bản demo 5 phút (14/11)

Chuẩn bị trước: mở ứng dụng ở chế độ video. Có sẵn:
- Clip live 22 s: `output/demo_20261114/M12/live_118s/render/test_video_offline_max.mp4` (bắt đầu từ giây 118 của video demo; đã có kết quả xử lý sẵn).
- Video demo đầy đủ: `output/demo_20261114/M13/clips/demo_video/render/`.
- Không dùng clip đêm trong phần trình diễn trực tiếp (còn 1 cảnh báo nguy_co không hợp lý ở 60.7 s).

1. (0:00–0:40) Giới thiệu: hệ thống nhận biển báo giao thông từ video hoặc ảnh, đọc nội dung và giải thích phạm vi áp dụng. Nói rõ: làn đường đã tắt mặc định; hệ thống ra "chưa xác định" khi thiếu bằng chứng làn.
2. (0:40–1:40) Clip live 22 s `M12/live_118s`: bật video. Chỉ vào mục "Cảnh báo phía trước" khi xuất hiện, ở khoảng 0:00.7 (ô tô trong làn xe mình phía trước) và 0:21 (ô tô trong làn xe mình rất gần). Nói: "Chú ý ô tô" là cảnh báo nhẹ; "Cẩn thận, <loại xe> phía trước" là cảnh báo mức cao. Đây không phải thời gian va chạm, và hệ thống chỉ dựa trên khung hình.
3. (1:40–2:40) Ảnh: tải một ảnh biển, cho xem kết quả và lý do áp dụng.
4. (2:40–3:40) Tra cứu luật: mở tab tra cứu, tìm một mã biển, cho xem nguồn.
5. (3:40–5:00) Số liệu: mở `docs/demo_20261114/RESULTS.md`, mục 0. Nói đúng:
   - Độ trễ (trung vị 3 lần): video demo 140 s xử lý trong 119.3 s; clip đêm 90 s xử lý trong 97.5 s. Clip đêm chưa đạt mục tiêu 90 s.
   - Cảnh báo trên video demo (holdout, AI-reviewed, không có nhãn người): 10 cảnh báo, 5 hợp lý và 5 không hợp lý, không có mức nguy_co. Chưa đo recall.
   - Giọng cảnh báo: 6 câu trong 140 s trên video demo (2.57 câu/phút).

Phương án khi lỗi:
- Không chạy được video trực tiếp: mở `output/demo_20261114/M5/backup/` (video dự phòng).
- Cảnh báo phía trước gây nhiễu: đặt `forward_warning.nguy_co_enabled: false` hoặc `forward_warning.enabled: false` (config) rồi chạy lại; biển báo không bị ảnh hưởng.
- Chậm: không xử lý video dài trên máy; dùng clip 22 s.
- Không được nói "va chạm" hoặc "sắp đâm".
