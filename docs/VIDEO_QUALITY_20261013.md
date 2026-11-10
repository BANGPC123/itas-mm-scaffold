# ITAS-MM — Sửa và kiểm tra lại video, 13/10/2026

Đã cập nhật bản chạy trên máy và xuất toàn bộ test_video.mp4: 35.793 frame, 1.193,1 giây, 30 FPS, 1280×720; AI phân tích 3.580 thời điểm ở 3 Hz. Video có 29 câu đọc và lớp phủ theo chuyển động ở từng frame.

[Mở ứng dụng](http://127.0.0.1:8000/) · [Video cuối](http://127.0.0.1:8000/pipeline/capstone-video/user-complete) · [Video ô tô nguồn khác](http://127.0.0.1:8000/pipeline/capstone-video/independent-car)

## Thay đổi có bằng chứng

- Giữ nguyên các checkpoint; bổ sung lượt dò 960/640 để cứu crop chưa rõ. Không thay nhãn chính đã rõ hoặc vượt qua phủ định hình biển/tham chiếu. Biển dưới 16 px cần hai lượt đọc cùng mã với confidence ≥0,95; dưới 12 px giữ chưa xác định. Cấu hình chọn trên 2.325 ảnh VAL rồi chốt trước TEST.
- Giữ thêm vạch vàng nhìn thấy rõ khi đầu paint BDD bỏ sót; vạch trắng mờ cần bằng chứng màu/độ tương phản và đầu paint mạnh. Vẫn kiểm tra mặt đường, vỉa hè và xe che khuất; không tạo biên làn pháp lý.
- Rà hình phát hiện bản thử vẽ một vạch qua đường thành làn ở giây 660. Bước loại trừ mới tìm hàng vạch trắng rộng trên mặt đường, chỉ xóa các điểm đã có. Đã giữ các đoạn ở cảnh 540/740 và sửa cảnh 660.
- Khi chỉ có dải đường thấp, lệch bên qua cửa kính, hướng camera giữ chưa xác định; không vẽ làn. Sửa chữ để thiếu bằng chứng hướng nhìn không bị mô tả chắc chắn là camera quay ngang.
- Chú thích đặt tránh box biển và vạch; biển sát mép trên được đặt tên bên cạnh. Ảnh 150 giây giữ được vòng biển và hai khoảng giờ để xem lại.

## TEST có nhãn đã phát hành

2.326 ảnh TEST, 3.058 biển có nhãn semantic đã phát hành, ghép một-một ở IoU ≥0,5. Precision mã biển tính trên nhãn có thể đánh giá và dự đoán không khớp GT; không coi nhãn semantic đã che/mơ hồ là một mã đúng hay sai. Đây là đo nhận dạng trên ảnh, không phải F1 làn hoặc tỷ lệ cảnh báo sai của video.

| Chỉ số | Bản trước b2634dd | Bản hai độ phân giải |
| --- | ---: | ---: |
| Precision mã biển | 91.19% | 90.84% |
| Recall mọi kích thước | 61.58% | 66.45% |
| Recall GT cạnh ngắn ≥32 px | 90.59% | 92.37% |
| Recall GT cạnh ngắn <16 px | 7.24% | 18.76% |
| Số mã đúng | 1883 | 2032 |

Nhận đúng thêm 149 biển, đổi lại thêm 10 mã sai trong phần GT đã phát hành và 13 dự đoán biết mã không khớp GT. Precision giảm 0,35 điểm %, recall tăng 4,87 điểm %. Tiêu chí TEST đã chốt trước: recall tăng ≥2 điểm %, precision giảm ≤0,5 điểm %, recall ≥32 px ≥89%; bản được chọn đạt cả ba. Không chỉnh ngưỡng bằng TEST.

## Chạy video và HTTP

| Phép kiểm tra | Kết quả |
| --- | --- |
| Video chính | 35.793 frame giải mã; 12/12 kiểm tra kỹ thuật đạt |
| Âm thanh | 29 câu đọc có tín hiệu; 0 lỗi suy luận/xuất/TTS |
| HTTP | 72 frame của bạn + 60 frame đường phố từ nguồn khác; 0 lỗi |
| HTTP video của bạn | trung bình 247.3 ms; P95 290.8 ms |
| Pipeline video của bạn | trung bình 225.5 ms; P95 259.2 ms |
| HTTP 60 frame đường phố | trung bình 150.0 ms; P95 290.3 ms |
| TTS mới / cache | 1.42–1.55 giây / 15–31 ms; 4/4 thành công |
| Video + TTS đồng thời | 7/7 kiểm tra đạt; P95 video 303.1 ms; TTS mới 1.16 giây |
| Phần mềm | pytest tests: 439 đạt; JavaScript 7 đạt; build production đạt |

30 FPS là tốc độ phát, AI 3 Hz là nhịp phân tích. Thời gian dựng lại từ kết quả đã lưu không phải tốc độ suy luận ML. Độ trễ gọi OCR trong video chỉ là gửi/nhận bất đồng bộ; chưa phải thời gian hoàn tất đọc cả biển phụ. Các phép đo HTTP thực hiện sau khi kết thúc xuất/benchmark mô hình. 60 frame ngoài nguồn là clip ven đường/trên cầu, chỉ dùng thử vận hành.

## Điều kiện cho ô tô

Frame 4480 (149,333 giây) được đưa lại qua HTTP với JPEG chất lượng 95. P.131a và các giờ 09:00–16:00, 20:00–06:00 được đọc trên ảnh thật; 6/6 kiểm tra logic đạt:

| Thời gian đưa vào để thử | Trạng thái |
| --- | --- |
| Không biết ngày/giờ ghi hình | conditional, cần thời điểm ghi hình |
| 10:00, UTC+7 | applies_candidate |
| 18:00, UTC+7 | excluded_candidate |

Ngày 13/10/2026 và các giờ trên là đầu vào thử tổng hợp, không phải ngày quay video. Giờ phát/đồng hồ máy không được thay thế ngày ghi hình. API ảnh đơn đọc OCR đồng bộ: lượt đầu khoảng 0,74 giây, các lượt sau 0,45–0,46 giây. Trong bản xuất có 36 lượt quan sát điều kiện giờ, không phải 36 biển độc lập.

car_context giữ vehicle=car, điều kiện đọc được, phần thiếu và trạng thái uncertain/information/conditional/applies_candidate/excluded_candidate. human_verified và legal_effect_established vẫn false; không tự bật label_verified/context_verified/validated_corpus. Giá trị tốc độ/chiều cao, vị trí làn/hướng hoặc biển phụ chưa đọc đủ được giữ chưa xác định.

## Video ô tô nguồn khác và giới hạn kiểm tra

Đã chạy toàn bộ 3.628 frame của video Driving in Kim Boi and Luong Son Districts (Obakeneko, CC BY 3.0), 363 thời điểm phân tích. Sáu kiểm tra kỹ thuật đạt, không nhận tên biển rõ. Trước sửa, 1/11 ảnh âm tính do AI rà soát có nét làn giả ở giây 84; sau sửa là 0/11.

Clip có cảnh quay ngang, phản chiếu kính và camera từ hàng ghế sau; các ảnh được rà soát không thấy vạch sơn dọc rõ. Một cảnh lỗi đã dùng sửa bộ lọc làn, nên đây là kiểm tra phát triển/vận hành từ nguồn khác, không phải TEST làn người gán nhãn độc lập. Không lấy toàn bộ clip làm nhãn âm tính chỉ từ 11 ảnh. Video phụ không có giọng đọc.

[Nguồn và tác giả](https://commons.wikimedia.org/wiki/File:Driving_in_Kim_Boi_and_Luong_Son_Districts.ogv) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0). Đã thêm lớp phủ ITAS-MM; không suy ra hiệu lực pháp lý hiện tại từ clip lịch sử.

## Phần chưa được chứng nhận

- Checkpoint nhận dạng vẫn có 30 mã. Biển ngoài các lớp, biển rất nhỏ/nhòe và sơ đồ làn phức tạp có thể chưa được đọc. Mẫu hình xe có kiểm thử phần mềm, chưa có benchmark thực địa riêng.
- Chưa có GT người độc lập cho video để công bố F1 làn, IDF1 tracking hoặc tỷ lệ cảnh báo sai thực địa. Số đoạn vạch và số lượt dự đoán không thay thế accuracy.
- Retrieval hiện có Recall@4 87.50% trên 24 ca hồi quy phát triển, còn 3 ca bỏ sót. Hiệu lực văn bản, ngoại lệ và điều kiện áp dụng chưa được kiểm chứng đầy đủ; validated_corpus vẫn tắt.

Bản này đã hoàn tất lượt sửa, xuất video và đo lại trong phạm vi trên; chưa đủ bằng chứng để gọi toàn hệ thống được chứng nhận hoàn chỉnh ngoài thực địa.

## Dữ liệu tái lập

Video gốc SHA256: eb109f8c852df580611c236e2b313b01cf4a62662586ce71fa35525bdda05ac4.
Video cuối SHA256: edc1e3bd52de3ee1e3505916b22948efaecef21fe7c796ae10dccadc2549ebe1; 476663174 byte.

full_v1 là quét ML mới toàn bộ nguồn ở cấu hình hai độ phân giải. full_v2 tính lại mặt đường/hướng camera, loại điểm vạch qua đường và ngữ cảnh ô tô; kiểm tra AST chứng minh seed/khớp vạch và luật xe/giờ không đổi ngoài bước loại trừ và chữ mô tả hướng. full_v4 dựng lại mọi frame từ các dự đoán đã xác minh, giữ tọa độ/chuyển động và âm thanh; bố trí chữ tránh che biển. Các trường thời gian ML được dùng lại được ghi rõ, không dùng để đo FPS mới.

[Bằng chứng hiện hành](evidence/video_quality_20261013/LATEST.json) · [TEST](evidence/video_quality_20261013/sign_test.json) · [HTTP](evidence/video_quality_20261013/live_api.json) · [Ngữ cảnh ô tô](evidence/video_quality_20261013/car_http.json) · [Checksums](evidence/video_quality_20261013/FILES_SHA256.json)

![Biển và giờ ở 150 giây](evidence/video_quality_20261013/frame_150.jpg)
![Cảnh giao lộ 660 giây sau sửa](evidence/video_quality_20261013/frame_660.jpg)
