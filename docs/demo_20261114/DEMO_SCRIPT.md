# Kịch bản demo 5 phút (14/11)

Chuẩn bị trước: mở ứng dụng ở chế độ video. Có sẵn:
- Clip live 30 s: `output/demo_20261114/M12/live_30s_v2/render/test_video_offline_max.mp4` (bắt đầu từ giây 30 của video demo; đã có kết quả xử lý sẵn; header không còn dòng làn). Câu được đọc khớp một cảnh báo hop_ly trong review v3.
- Để upload trực tiếp trong tab Video: `output/demo_20261114/M12/e2e_final/live_30s_source.mp4` (cùng đoạn 30 s, mã hóa lại).
- Video demo đầy đủ: `output/demo_20261114/M13/clips/demo_video/render/`.
- Không dùng clip đêm trong phần trình diễn trực tiếp (còn 1 cảnh báo nguy_co không hợp lý ở 60.7 s).

1. (0:00–0:40) Giới thiệu: hệ thống nhận biển báo giao thông từ video hoặc ảnh, đọc nội dung và giải thích phạm vi áp dụng. Nói rõ: làn đường đã tắt mặc định; hệ thống ra "chưa xác định" khi thiếu bằng chứng làn.
2. (0:40–1:40) Clip live 30 s `M12/live_30s_v2`: bật video. Chỉ vào mục "Cảnh báo phía trước" khi xuất hiện, ở khoảng 0:12 (xe máy trong làn xe mình phía trước, được đọc "Chú ý xe máy"). Nói: "Chú ý" là cảnh báo nhẹ; đây không phải thời gian va chạm, và hệ thống chỉ dựa trên khung hình.
3. (1:40–2:40) Ảnh: tải `output/demo_20261114/M14/demo_photo.jpg` (frame 105.667 s của video demo, 1 biển I.423 "Vị trí người đi bộ sang ngang", nhận dạng 95%, cạnh ngắn 38 px). Cho xem biển và khung vật cản (cam trong vùng đường đi, xám ngoài vùng). Bấm chip "Quy định" của biển để mở tra cứu: QCVN 41:2024/BGTVT Điều 84 (`M14/e2e_image/shots/03_image_legal_lookup.png`). Lần upload đầu nhanh nhờ `ITAS_WARMUP=1` trong `start.ps1` (M14: 0.1 s). Ảnh `photo_61s.jpg` cũ không nhận dạng biển (1 biển, cạnh 16 px dưới ngưỡng 32 px): không dùng trong demo.
4. (2:40–3:40) Tra cứu luật: trong tab Video, bấm mốc biển P.102 "Cấm đi ngược chiều" ở danh sách biển, cho xem nguồn QCVN 41:2024/BGTVT Điều 74 (đã kiểm tra trên trình duyệt, xem RESULTS.md mục 0.8).
5. (3:40–5:00) Số liệu: mở `docs/demo_20261114/RESULTS.md`, mục 0. Nói đúng:
   - Độ trễ (trung vị 3 lần): video demo 140 s xử lý trong 119.3 s; clip đêm 90 s xử lý trong 97.5 s. Clip đêm chưa đạt mục tiêu 90 s.
   - Cảnh báo trên video demo (holdout, AI-reviewed, không có nhãn người): 10 cảnh báo, 5 hợp lý và 5 không hợp lý, không có mức nguy_co. Chưa đo recall.
   - Giọng cảnh báo: 6 câu trong 140 s trên video demo (2.57 câu/phút).
   - Clip live 30 s: 1 câu được đọc, 2 cảnh báo, cả hai khớp v3 hop_ly.

Phương án khi lỗi:
- Không chạy được video trực tiếp: mở `output/demo_20261114/M5/backup/` (video dự phòng).
- Cảnh báo phía trước gây nhiễu: đặt `forward_warning.nguy_co_enabled: false` hoặc `forward_warning.enabled: false` (config) rồi chạy lại; biển báo không bị ảnh hưởng.
- Chậm: không xử lý video dài trên máy; dùng clip 30 s.
- Không nghe thấy giọng: kiểm tra loa và âm lượng. E2E trên trình duyệt headless không xác nhận được âm thanh phát ra loa (xem RESULTS.md mục 0.8).
- Không được nói "va chạm" hoặc "sắp đâm".

## Tạo lại file demo nếu thiếu

Clip live 30 s (`output/demo_20261114/M12/e2e_final/live_30s_source.mp4`) được cắt từ `docs/demo/itas-mm_demo_video.mp4` (bắt đầu giây 30, dài 30 s). File này không nằm trong git (`output/` bị gitignore), nên clone mới sẽ thiếu. Lệnh tái tạo (CPU, không dùng GPU). Lệnh này dựng lại theo thuộc tính của file hiện có (H.264 High, yuv420p, 30 fps, 1280x720, không âm thanh, khoảng 5.8 Mb/s); lệnh gốc không được ghi lại:

```bash
ffmpeg -hide_banner -ss 30 -i docs/demo/itas-mm_demo_video.mp4 -t 30 -an -c:v libx264 -preset medium -b:v 5800k -pix_fmt yuv420p -r 30 output/demo_20261114/M12/e2e_final/live_30s_source_recut.mp4
```

Tên đầu ra có `_recut` để không ghi đè file hiện có. Kiểm tra đã chạy (M16, `output/demo_20261114/M16/RECUT_CHECK.md`):

- File hiện có: 900 khung, 30.00 s, 1280x720, 21.7 MB.
- File tạo lại: 900 khung, 30.00 s, 1280x720, 22.9 MB.
- Sai khác điểm ảnh trung bình ở khung đầu, giữa và cuối: 2.45, 1.97, 2.52 (thang 0–255, do mã hóa lại).

Nếu không có `ffmpeg` trên PATH, dùng binary đi kèm môi trường `itas-mm`: `D:\miniconda\envs\itas-mm\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe`.
