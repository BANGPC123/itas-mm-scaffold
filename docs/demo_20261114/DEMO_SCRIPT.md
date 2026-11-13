# Kịch bản demo 5 phút (14/11)

Chuẩn bị trước: mở ứng dụng ở chế độ video. Có sẵn:
- Clip live 30 s: `output/demo_20261114/M12/live_30s/render/test_video_offline_max.mp4` (bắt đầu từ giây 30 của video demo; đã có kết quả xử lý sẵn). Chọn vì mọi câu được đọc khớp một cảnh báo hop_ly trong review v3.
- Video demo đầy đủ: `output/demo_20261114/M13/clips/demo_video/render/`.
- Không dùng clip đêm trong phần trình diễn trực tiếp (còn 1 cảnh báo nguy_co không hợp lý ở 60.7 s).

1. (0:00–0:40) Giới thiệu: hệ thống nhận biển báo giao thông từ video hoặc ảnh, đọc nội dung và giải thích phạm vi áp dụng. Nói rõ: làn đường đã tắt mặc định; hệ thống ra "chưa xác định" khi thiếu bằng chứng làn.
2. (0:40–1:40) Clip live 30 s `M12/live_30s`: bật video. Chỉ vào mục "Cảnh báo phía trước" khi xuất hiện, ở khoảng 0:12 (xe máy trong làn xe mình phía trước, được đọc "Chú ý xe máy"). Nói: "Chú ý" là cảnh báo nhẹ; đây không phải thời gian va chạm, và hệ thống chỉ dựa trên khung hình.
3. (1:40–2:40) Ảnh: tải một ảnh biển, cho xem kết quả và lý do áp dụng.
4. (2:40–3:40) Tra cứu luật: mở tab tra cứu, tìm một mã biển, cho xem nguồn.
5. (3:40–5:00) Số liệu: mở `docs/demo_20261114/RESULTS.md`, mục 0. Nói đúng:
   - Độ trễ (trung vị 3 lần): video demo 140 s xử lý trong 119.3 s; clip đêm 90 s xử lý trong 97.5 s. Clip đêm chưa đạt mục tiêu 90 s.
   - Cảnh báo trên video demo (holdout, AI-reviewed, không có nhãn người): 10 cảnh báo, 5 hợp lý và 5 không hợp lý, không có mức nguy_co. Chưa đo recall.
   - Giọng cảnh báo: 6 câu trong 140 s trên video demo (2.57 câu/phút).
   - Clip live 30 s: 1 câu được đọc, 2 cảnh báo, cả hai khớp v3 hop_ly.

Phương án khi lỗi:
- Không chạy được video trực tiếp: mở `output/demo_20261114/M5/backup/` (video dự phòng).
- Cảnh báo phía trước gây nhiễu: đặt `forward_warning.nguy_co_enabled: false` hoặc `forward_warning.enabled: false` (config) rồi chạy lại; biển báo không bị ảnh hưởng.
- Chậm: không xử lý video dài trên máy; dùng clip 30 s.
- Không được nói "va chạm" hoặc "sắp đâm".
