# ITAS-MM — Kiểm tra lại video và ngữ cảnh ô tô, 12/10/2026

Đã sửa và chạy lại toàn bộ test_video.mp4: 35.793 frame, 1.193,1 giây, 30 FPS, 1280×720. AI phân tích 3.580 thời điểm ở 3 Hz; lớp phủ theo dõi chuyển động giữa các frame. Bản đã kiểm tra được phục vụ tại [ứng dụng](http://127.0.0.1:8000/) và [video đầy đủ](http://127.0.0.1:8000/pipeline/capstone-video/user-complete).

## Thay đổi đã kiểm tra

- Bỏ nhánh UFLD trả sớm trước kiểm tra vạch sơn, mặt đường và xe che khuất. Mọi đề xuất làn phải có bằng chứng trên mặt đường; vẽ từng đoạn sơn nhìn thấy, không nối qua tap-lô, xe hoặc khoảng trống. Tách vạch qua đường khỏi vạch làn và giữ vạch đứt ngắn.
- Chặn logo ngân hàng bị nhận thành I.409 bằng hình mũi tên trong QCVN; bằng chứng trái hình biển xóa nhãn tracking cũ.
- Hướng camera dùng cả phân bố mặt đường và dấu hiệu nội thất. Đã sửa hai cảnh 207,333/1073 giây bị gán nhầm quay vào cabin; cảnh biển trong gương ở 960 giây vẫn không được gán cho hướng đi.
- API/giao diện trả car_context gồm phạm vi loại xe, điều kiện đọc được và phần còn thiếu. Phân biệt ô tô con/xe tải, biển chỉ xe tải và biển “trừ xe tải”.
- OCR ghép chữ đúng dòng, đọc lại cả dòng bị tách và kiểm tra ít nhất hai biến thể ảnh. Không thay đổi hai giá trị giờ đã quan sát. Điều kiện đi theo mã tracking; đổi thứ tự detections/mất biển khác không xóa kết quả của biển đang theo dõi. Vị trí biển phụ được kiểm tra trên frame hiện tại.
- Làm nóng OCR trước khi API sẵn sàng và giới hạn BLAS một luồng cho các phép khớp đường nhỏ.

## Chạy và kiểm thử

| Phép kiểm tra | Kết quả |
| --- | --- |
| Video | 35793 frame giải mã; 12/12 kiểm tra kỹ thuật đạt |
| Âm thanh | 29 câu đọc có âm thanh; không lỗi xuất/giọng đọc |
| HTTP vận hành | 72 frame của bạn + 60 frame hành trình khác; 0 lỗi |
| HTTP video của bạn | Trung bình 213.0 ms; P95 242.6 ms |
| Pipeline | Trung bình 192.3 ms; P95 216.7 ms |
| HTTP hành trình khác | Trung bình 150.4 ms; P95 198.0 ms |
| TTS | Tạo mới 1,78–1,98 giây; đã có nội dung 29–31 ms; 4/4 thành công |
| Điều kiện giờ trong bản xuất | 19 lượt quan sát; không phải số biển độc lập |
| Phần mềm | pytest tests: 425 đạt; 7 kiểm thử JavaScript và build production đạt |

Số đoạn vạch, số lượt nhận dạng và các ca kiểm thử phần mềm không thay thế độ chính xác thực địa. 30 FPS là tốc độ phát; 3 Hz là nhịp phân tích bản xuất. Thời gian dựng dùng kết quả đã lưu không phải tốc độ suy luận mô hình.

## Ngữ cảnh ô tô và giờ

Ở frame 4480 (149,333 giây), HTTP đọc P.131a — Cấm đỗ xe và 09:00–16:00, 20:00–06:00. Ba trường hợp trên ảnh thực đạt 6/6 tiêu chí:

| Đầu vào thời gian dùng để thử | Kết quả |
| --- | --- |
| Không có thời điểm ghi hình | conditional: cần xét giờ ghi hình |
| 10:00, múi giờ Việt Nam | applies_candidate: có thể thuộc phạm vi/giờ đã đọc |
| 18:00, múi giờ Việt Nam | excluded_candidate: ngoài các giờ đã đọc |

Ngày giờ trên là đầu vào thử logic, không được suy ra là ngày giờ thật của video. Video gốc chưa cung cấp thời điểm ghi hình; đồng hồ phát lại/ngày hiện tại không được dùng thay thế. Ảnh nén ở 148,667 giây có điểm OCR dưới ngưỡng nên giữ trạng thái chưa xác định; frame rõ hơn được đọc tự động.

car_context có các trạng thái applies_candidate, excluded_candidate, conditional, information, uncertain. Các kết quả này không tự đổi human_verified, label_verified, context_verified, legal_effect_established hoặc validated_corpus thành đúng. Biển phụ không rõ, giá trị tốc độ/chiều cao chưa đọc được và phạm vi làn/hướng thiếu bằng chứng vẫn được ghi rõ.

Mẫu biểu tượng xe tải/ô tô phía trước có kiểm thử phần mềm, chưa có benchmark thực địa độc lập cho phần nhận dạng biểu tượng.

## TEST độc lập và giới hạn

TEST gồm 2.326 ảnh, nhãn semantic đã phát hành, ghép một-một ở IoU 0,5. VAL 2.325 ảnh được kiểm tra trước TEST. Mẫu hình đến từ QCVN, không lấy từ TEST; tham chiếu TRAIN/âm tính phát triển giữ nguyên nguồn.

| Chỉ số, detector 960 | Bản trước fd23f88 | Bản kiểm tra hình biển |
| --- | ---: | ---: |
| Precision mã biển | 91,16% | 91.19% |
| Recall mọi kích thước | 61,74% | 61.58% |
| Recall khi GT có cạnh ngắn ≥32 px | 91,01% | 90.59% |

Kiểm tra hình chặn logo sai trong video nhưng bỏ sót thêm 5 biển thật ở TEST; không coi đây là cải thiện toàn bộ mô hình nhận dạng. Vạch mờ/che khuất và biển nhỏ vẫn có thể bị bỏ sót. Chưa có nhãn chuẩn độc lập cho video để công bố F1 làn, tracking hoặc tỷ lệ cảnh báo sai thực địa. 60 frame hành trình khác là kiểm thử vận hành.

validated_corpus vẫn tắt; chưa chứng nhận đầy đủ hiệu lực/ngoại lệ của suy luận pháp lý. Bản này hoàn thành sửa và đánh giá video/ngữ cảnh ô tô trong phạm vi trên, không tuyên bố chứng nhận hoàn chỉnh ngoài thực địa.

## Nguồn và tái lập

Video gốc SHA256: eb109f8c852df580611c236e2b313b01cf4a62662586ce71fa35525bdda05ac4.
Video xuất SHA256: 57583a86de5b776fc09d3790b58a7fd14cb267d27e80c60f460be89ca53f5a55, 443103793 byte.

Mốc trước: fd23f8847cee30bdeb4c781dd7e9e3f0e8de8ac3, cùng byte video/timestamp. full_v2 là lần quét ML đầy đủ đã được xác minh. full_v4 dựng lại mọi frame, dùng lại hình học làn/vị trí detections; kiểm tra AST hình học không đổi. Nhận dạng, tracking, hướng camera và ngữ cảnh được tính lại ở mọi thời điểm phân tích. Mã nguồn/hashes nguồn ML được lưu trong các JSON bằng chứng.

[Bằng chứng hiện hành](evidence/car_video_recheck_20261012/LATEST.json), [đối chiếu 5 cảnh](evidence/car_video_recheck_20261012/before_after_lane.jpg), [TEST](evidence/car_video_recheck_20261012/sign_test.json), [cấu hình 8 bộ trọng số](evidence/car_video_recheck_20261012/run_configuration.json).

Nguồn chính thức: [Thông tư 51/2024/TT-BGTVT](https://vanban.chinhphu.vn/?classid=1&docid=211908&orggroupid=4&pageid=27160), [QCVN 41:2024/BGTVT](https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf), các phần B.30/B.31, E.9, F.3/F.4/F.5 và F.10. PDF có 403 trang, SHA256 b6b99d0a0aeb13fe79072831857ece15bb254d8fb7817e80e4a2739a0d2145ce. Mẫu hình có trang/vùng trích/hash trong data/reference/qcvn2024_symbols.json; đối chiếu không tự xác nhận hiệu lực mọi văn bản ở mọi thời điểm.
