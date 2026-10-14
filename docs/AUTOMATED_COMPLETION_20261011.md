# Kết quả hoàn thiện tự động — 11/10/2026

Đã triển khai và chạy trên máy Windows tại `D:\itas-mm`. Giao diện đang phục vụ ở http://127.0.0.1:8000, có bảng kết quả nghiên cứu tự động. Không yêu cầu người dùng duyệt nhãn để chạy các thí nghiệm này. Các nhãn máy vẫn giữ trạng thái chưa được người xác minh; kết quả dưới đây không chứng nhận độ chính xác khi lái xe.

## 1. Nhãn làn Việt Nam và dữ liệu mới

Đã xử lý đủ **600 frame video người dùng và 59 frame từ hai nguồn video khác** bằng UFLD và CLRNet cố định. Chỉ giữ đoạn làn đồng thuận về hình học và có bằng chứng vạch sơn. Có **91 frame chứa nhãn dương một phần**: TRAIN 54, VAL_INTERNAL 24, TEST cùng video 13. Không có frame nguồn ngoài được đưa vào train. Không biến frame không tìm thấy làn thành nhãn âm.

Mask dùng lớp 0 cho nền trong hành lang cục bộ, 1 cho vạch đồng thuận, 255 cho vùng chưa biết và bỏ qua khi học. Đây là **nhãn yếu**, chưa phải nhãn chuẩn Việt Nam. Mọi `human_confirmed` vẫn false.

Kết quả: `data/processed/lane_auto_consensus_20261011/report.json` và `manifest.jsonl` trên máy Windows.

## 2. Fine-tune đủ ba mô hình và so sánh

Đã chạy thật ba epoch thích nghi trên 54 frame TRAIN có nhãn yếu cho UFLD-v2, CLRNet và SegFormer-B0. UFLD/CLRNet chỉ cập nhật các lớp chiếu cuối bằng các điểm dương đã chọn; không tự tạo nhãn làn thiếu. SegFormer bỏ qua pixel 255. CLRNet dùng mã tác giả ghim commit, checkpoint chính thức và NMS CPU tham chiếu, trong môi trường nghiên cứu riêng `.clr-runtime`.

So sánh cùng **100 ảnh CULane VAL có nhãn gốc**, raster vạch rộng 16 pixel ở kích thước ảnh gốc. Chỉ số này là **pixel F1**, không phải lane-level F1 chính thức CULane, cũng không phải F1 Việt Nam.

| Mô hình | Trước thích nghi | Sau thích nghi |
|---|---:|---:|
| UFLD-v2 | 0,627451 | 0,627468 |
| CLRNet | 0,664739 | 0,659937 |
| SegFormer-B0 | 0,262280 | 0,233390 |

UFLD thay đổi không đáng kể; hai mô hình còn lại giảm điểm. **Giữ checkpoint đang triển khai**, không chọn mô hình từ TEST hay kết quả không có nhãn chuẩn. Checkpoint thử nghiệm vẫn được lưu để tái lập. Thời gian báo cáo huấn luyện là phiên tiếp tục cuối; tổng thời gian qua các lần tiếp tục không được đo.

Kết quả: `output/retraining_auto_20261011/report.json`, `output/evaluation/auto_lane_comparison_20261011/report.json`. Cả sáu biến thể cũng chạy trên 59 frame nguồn mới để đo khả năng thực thi, không suy ra accuracy từ chúng.

## 3. Đọc biển phụ tự động

Đã tích hợp worker RapidOCR và bộ phân tích tiếng Việt vào pipeline và giao diện. Tìm vùng ứng viên phía dưới biển chính đã nhận diện, đọc qua hai biến thể tiền xử lý, phân tích giờ, loại xe và từ loại trừ. Cache gắn với stream và nhận dạng/vị trí biển chính, tránh tái dùng điều kiện cho biển khác.

Kết quả OCR hiển thị thành **ứng viên máy đọc**, không tự thay điều kiện pháp lý đã xác minh. Kiểm thử thực tế tám vùng nhỏ: không lỗi worker sau khi sửa thời gian khởi động; chỉ một vùng đọc được đầy đủ hai khoảng `06:00–08:00; 16:00–18:00`. Văn bản thiếu như `6:00-;22:00` được từ chối. Chưa có bằng chứng đủ để coi đọc loại xe/giờ trên biển nhỏ là đáng tin cậy.

## 4. Kho luật và retrieval

Đã thêm PDF nguồn chính thức và OCR 42 trang của:

- [Nghị định 238/2026/NĐ-CP](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/7/238-ndcp.signed.pdf), hiệu lực 15/08/2026, sửa đổi Nghị định 168/2024/NĐ-CP.
- [Luật 118/2025/QH15](https://datafiles.chinhphu.vn/cpp/files/vbpq/2026/01/luatso118.2025.pdf), hiệu lực 01/07/2026, trong đó có sửa đổi Luật 36/2024/QH15.

Kho hiện có **5 nguồn, 974 đoạn**. Kiểm tra SHA256 nguồn đạt; khôi phục đúng byte CRLF của HTML cũ khi và chỉ khi hash khớp hash nguồn đã lưu. PDF và nội dung OCR giữ xuất xứ riêng, cảnh báo chưa hợp nhất văn bản và chưa xác minh OCR. Không sửa nội dung gốc để giả thành bản hợp nhất.

BM25 thêm bằng chứng cụm hai/ba từ, không dùng đáp án benchmark để định tuyến. Trên **24 câu hồi quy giữ nguyên**, Recall@4 tăng từ **83,33% lên 87,50%**; Recall@1 54,17%, Recall@3 83,33%, MRR 0,684028. Vẫn còn ba câu không đạt Recall@4. Bộ này nhỏ, chưa phải TEST độc lập pháp lý.

`validated_corpus` vẫn false. Chưa xác minh hết sửa đổi, ngoại lệ, hiệu lực và điều kiện không gian; OCR nguồn không được người kiểm từng trang. Không mở luồng hướng dẫn pháp lý tự động dựa trên kho chưa kiểm chứng.

Nguồn và dấu vết: `data/regulations/legal_updates_20261011.json`, `data/regulations/current_raw/`, `output/evaluation/auto_legal_20261011/audit.json`.

## 5. Benchmark toàn pipeline trên nguồn hành trình khác

Đã chạy **240 frame**: 30 giây đầu từ mỗi trong hai video nguồn khác, lấy mẫu 4 Hz, ảnh 1280×720. GPS mô phỏng. Luồng gồm phát hiện, nhận dạng, tracking, làn, ngữ cảnh, retrieval và OCR khi có ứng viên. Không lỗi pipeline.

| Số đo phiên cuối | Kết quả |
|---|---:|
| Khởi tạo pipeline | 5.944,6 ms |
| Trung bình xử lý, sau bỏ frame khởi động đầu tiên | 159,47 ms |
| Trung vị | 68,04 ms |
| P95 | 237,90 ms |
| Thông lượng suy luận tuần tự | 6,27 frame/giây |

Đây là thông lượng suy luận trên frame lấy mẫu, không phải FPS phát/capture video. OCR khởi động nguội có một lần khoảng 8,65 giây; các lần OCR tiếp theo khoảng 0,58–2,26 giây. TTS đo riêng: ba nội dung chưa cache mất khoảng **3,00–4,20 giây** với dịch vụ mạng; nội dung cache nhanh hơn nhiều. Không tính TTS vào FPS suy luận.

Video mới chưa có nhãn tham chiếu độc lập nên **F1 làn Việt Nam, biển nhận nhầm, cảnh báo sai và accuracy tracking vẫn chưa đo được**. Chúng giữ giá trị chưa có dữ liệu, không điền 0 hay suy ra từ số lần chương trình chạy thành công. Không có cảnh báo pháp lý phát sinh trong 240 frame không chứng minh tỷ lệ cảnh báo sai bằng 0.

Kết quả đầy đủ: `output/evaluation/auto_system_20261011_v2/report.json`; phiên đầu chứa số đo TTS chưa cache tại `output/evaluation/auto_system_20261011/report.json`.

## Kiểm tra và triển khai

- `pytest tests -q`: **373 passed**, sáu cảnh báo thư viện.
- Hai bộ kiểm thử JavaScript video: **7 passed**.
- Frontend build đạt. Sau bổ sung bảng nghiên cứu: 13 kiểm thử API/OCR/retrieval đạt và build lại đạt.
- Backend khởi động lại; HTTP `/pipeline/automatic-research-status` xác nhận đủ báo cáo và `/legal/query` xác nhận năm nguồn. Dấu vết tại `output/evaluation/auto_system_20261011_v2/live_status.json`.
- Chạy pytest từ toàn thư mục còn thu thập bộ audit dữ liệu cũ trong `data/audit/`: 27 thất bại vì thiếu các artifact đóng băng của bộ đó, 392 đạt. Không xóa hay sửa bộ audit để che lỗi; bộ kiểm thử phần mềm `tests/` đạt đầy đủ.

## Tái lập

Các script mới: `setup_clrnet_research.py`, `build_auto_lane_corpus.py`, `train_auto_lane_candidates.py`, `compare_auto_lane_candidates.py`, `audit_legal_currency.py`, `benchmark_auto_system.py`. Chạy từ gốc repo với Python 3.11 và runtime hiện có; nghiên cứu CLRNet cần `.clr-runtime` trong PYTHONPATH, OCR dùng `.ocr-venv`. Không thay bản Torch CUDA chính. Báo cáo/checkpoint đã tồn tại được bảo vệ khỏi ghi đè; benchmark nhận `--output` để tạo một phiên mới.

Phần tự động, huấn luyện, so sánh và đo vận hành đã chạy xong. Phần còn thiếu để công bố độ chính xác thực địa là dữ liệu tham chiếu độc lập và kiểm chứng nội dung pháp lý/OCR; công cụ không thể tự biến kết quả máy dự đoán thành đáp án chuẩn độc lập.
