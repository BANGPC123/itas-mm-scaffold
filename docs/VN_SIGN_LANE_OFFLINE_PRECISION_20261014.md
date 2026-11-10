# ITAS-MM — Tối ưu nhận diện biển Việt Nam và lane trên video offline (14/10/2026)

## Mục tiêu và phạm vi

Phân tích video MP4 đã ghi, **không dùng camera trực tiếp**, ưu tiên độ tin cậy hơn tốc độ thời gian thực. Tận dụng các mẫu canonical có bounding box và mã biển đã gán nhãn của dự án; đối chiếu YOLO11 optimized 59 lớp với YOLO gốc + MobileNetV3 trước khi hợp nhất. Với lane, chống việc bộ lọc hậu xử lý xóa nhiều vạch thật nhưng không tái hiển thị đường sai đi qua xe/táp-lô.

Không có ground truth người gán nhãn đầy đủ cho `test_video.mp4`; **không tuyên bố accuracy trên toàn video**. TEST có nhãn của dự án dành riêng để kiểm định đóng băng, không dùng để chọn ngưỡng.

## 1. Benchmark biển báo Việt Nam có nhãn

Bộ mã biển canonical của dự án: 400 ảnh VAL, 540 biển có mã được RELEASED.

Đối chiếu: `scripts/evaluate_yolo11_primary_rescue_val_20261014.py`; log/results tại `output/evaluation/yolo11_primary_rescue_val_20261014/result.json`.

| Giải pháp | Mã đúng | Precision mã | Recall mã | F1 |
|---|---:|---:|---:|---:|
| Gốc 960/640 + MobileNetV3 | 396 | **92,31%** | 73,33% | 81,73% |
| YOLO11 59 lớp cứu box gốc không nhận được (IoU≥0,60, conf≥0,50) | **398** | 91,92% | **73,70%** | **81,81%** |
| Chế độ lọc cứu chặt hơn (IoU≥0,60, conf≥0,80) | 396 | 92,31% | 73,33% | 81,73% |

Mức cải thiện ở cấu hình nới ngưỡng chỉ **+0,07 điểm F1** trên VAL và giảm precision. **Không đủ bằng chứng để tự động đưa mã YOLO11 vào biển chính**. Giữ YOLO11 như một lớp gợi ý màu tím, yêu cầu người rà. Mọi nhãn/biển phụ/loại xe/giờ đều chỉ được quyết định bởi logic chính có bảo vệ. Kết quả TEST tĩnh đã đóng băng cho chế độ phát hiện: 2 lượt 960/640 F1 mã **76,75%**; 3 lượt thêm 1280 F1 mã **76,64%** (precision giảm, recall tăng nhẹ).

Vì vậy **mặc định offline hiện là hai lượt 960/640**, có thể bật `--highres-rescue` / tick chọn 1280 như thử nghiệm tùy chọn, không tự nhận tốt hơn.

## 2. Kiểm tra lọc UFLD trên CULane VAL có nhãn

Script `scripts/evaluate_lane_visual_policy_val_20261014.py`, kết quả `output/evaluation/lane_visual_policy_val_20261014/results.json`; 240 ảnh CULane VAL, 809 ground-truth lane:

| Cấu hình | Precision | Recall | F1 |
|---|---:|---:|---:|
| UFLD-v2 raw | 82,94% | 78,74% | **80,79%** |
| Cũ — lọc hiển thị chặt | **96,27%** | 15,95% | 27,36% |
| Lọc geometry mềm | 86,94% | 71,57% | 78,51% |
| Bỏ nét có độ nghiêng quá lớn (slope 4) | 83,99% | 63,54% | 72,34% |

Các bộ lọc gắt làm mất nhiều vạch thật. Tuy nhiên **raw không có nghĩa tự động đúng trên video xe máy/ô tô Việt Nam**. Hệ thống giữ UFLD raw riêng để so sánh theo thời gian, tối đa năm ứng viên khi đủ điều kiện. **Nếu camera-aware road/paint/occlusion đã bác bỏ nét**, UFLD không được hồi sinh nét đó làm overlay; đặc biệt chặn nét xuyên xe/táp-lô đã thấy tại video giây 149.

Module `src/perception/lane_temporal.py` thêm `OfflineLaneEvidence`: yêu cầu ít nhất hai frame liên tục hỗ trợ cùng đường, reset theo hard-cut/seek, không vẽ lane khi frame hiện tại không có. `offline_visual_fallback_allowed` chỉ cho phép ứng viên khi backend vẫn là UFLD nhìn phía trước và không có kết quả đáng tin hơn. `LaneInfo.candidate_polylines` là trường đánh dấu **chưa xác minh**, không được dùng làm ego lane hay quyết định phạm vi biển phụ.

## 3. Kết hợp đa frame để giảm nhầm biển báo

`src/perception/temporal_signs.py` bổ sung chế độ offline **3 phiếu nhận dạng cùng mã** (so với 2 phiếu trong chế độ xem video cũ), có thể giữ trạng thái nội bộ qua một frame không phát hiện, trong tối đa 0,36 giây. **Không vẽ ghost boxes**; cắt cảnh, tua ngược, đổi kích thước frame, background rejection đều xóa bằng chứng cũ. Các ca kiểm thử độc lập dùng video giả lập có dấu thời gian xác minh các điều kiện.

`Pipeline` có StreamEvidence riêng cho offline; nhận biết chế độ qua `offline_video_quality`, không thay đổi hành vi chế độ mặc định ảnh tĩnh/video xem trước.

## 4. Video thử trên file thực

- **18 giây, giây 38–56**: 540/540 frame, 108 lần suy luận (6 Hz), SHA256 video `8517468996b4598da14203ad50e97feebd79234d752799f36cf71d0a37ed6c5d`, 12 bước kiểm định chưa áp dụng vì chỉ có full decode riêng. Có checkpoint cũ, biểu hiện dấu hiệu không đủ tin cậy và vạch candidate xấu; đây là bản trước sửa sau kiểm tra trực quan.
- **9 giây, giây 146–155 trước sửa**: 270/270 frame, 54 lần phân tích, 5 ứng viên UFLD chưa kiểm chứng trong khoảng bị xe/táp-lô che, không nên dùng.
- **9 giây, giây 146–155 đã sửa**: 270/270 frame, 54 lần phân tích, 20 lượt vạch paint-backed, **0 UFLD giả qua xe**, `verified_full_decode=true`, dùng đúng mặc định 960/640 và 3 phiếu biển. SHA256 `c6e9fada874d2aea4afe12436ca29dc5653b43d9c89bbefa8f4fb0b61f5f43af`, output `output/offline_vn_precision_corrected_20261014/test_video_offline_max.mp4`.

Chỉ số về số đường **không phải accuracy**. Đây là các đoạn được kiểm tra trực quan, không đại diện toàn bộ 19 phút 53 giây. Video full trước đó vẫn được giữ; chức năng MP4 Offline trên Web có thể xử lý toàn bộ 35.793 frame bằng mã mới.

## 5. Dùng trên Web hoặc CLI

Trên máy đã kết nối: `http://127.0.0.1:8000/` → Video MP4 → **Phân tích MP4 chất lượng cao**.

- Chọn 6 Hz (mặc định) hoặc 10 Hz nếu muốn thêm nhiều thời điểm xác nhận, xử lý chậm hơn tốc độ video.
- Mặc định **960/640 + xác nhận 3 frame**. Không bật 1280 chỉ vì muốn thêm nhiều box; tùy chọn 1280 có ghi cảnh báo F1 TEST thấp hơn một chút.
- Có thể bật YOLO11 59 lớp màu tím để nhìn gợi ý mới. Biển này chưa được xác minh luật.
- Nhập ngày giờ ghi hình nếu biết; không tự suy từ ngày phát lại.
- Xuất MP4 và file JSON kết quả khi giải mã đầy đủ. Video và model weights nằm cục bộ, không push GitHub.

CLI:

```powershell
cd D:\itas-mm
python scripts/render_offline_video_max_20261014.py "C:\path\to\test_video.mp4" --output output/offline_new_run --analysis-hz 6 --auxiliary-review
```

Tùy chọn `--highres-rescue` được giữ cho nghiên cứu. Mỗi lần chạy phải dùng thư mục output mới.

## Giới hạn còn lại

- Bộ dữ liệu Việt Nam có nhãn biển đã có, nhưng nhận dạng trên video chưa có ground truth độc lập gán toàn bộ frame cho phân tích thời gian.
- Không có nhãn lane người xác nhận đầy đủ của video; chưa thể chấm một F1 lane đường Việt Nam đáng tin cậy.
- Chưa fine-tune weights. Cải thiện thực tế trong lần này là **bộ chọn phương án dựa trên nhãn VAL/TEST, xác nhận thời gian nghiêm ngặt hơn, tránh hồi sinh lane sai, và cấu hình offline tốt hơn với bằng chứng**.
- Demo mô phỏng, không dùng để quyết định lái xe ngoài thực tế.
