# ITAS-MM — Rà soát nhãn vạch sơn và biển báo bằng thị giác AI (14/10/2026)

## Phạm vi và nguồn gốc người đánh giá

**Người tạo nhãn: ChatGPT/GPT-6 (AI đọc hình trực quan, tự chọn tọa độ), KHÔNG PHẢI con người kiểm tra độc lập.**
`human_verified=false`, `field_accuracy_validated=false`. Không gộp với ground truth chuẩn của người; tuyệt đối không dùng các chỉ số ở đây để chứng nhận sản phẩm hoặc để chọn checkpoint mà sau đó chấm lại trên cùng ảnh.

Lần rà này dùng 36 frame từ ba nguồn nội dung: 12 frame `test_video.mp4`, 12 frame hình đường phố Hà Nội và cầu vượt Gò Vấp (hai video riêng), và 12 frame video ô tô Hòa Bình. Mỗi frame được kiểm tra qua ảnh nguyên gốc (không dùng kết quả model để quyết định nội dung nhãn). Tọa độ sau đó được so sánh với inference chạy lại **trên chính frame gán nhãn** qua HTTP API.

## Định nghĩa và số lượng nhãn

| Trạng thái | Frame | Có tính alignment? |
|---|---:|---|
| `annotated`: nhìn rõ ít nhất một đoạn vạch dọc đã vẽ | **6** | Có |
| `no_visible_lane`: mặt đường quan sát đủ rõ, không thấy vạch dọc | **2** | Có (kiểm tra false-alarm) |
| `uncertain`: che khuất hoặc quá ít thông tin để xác định làn | **11** | Không |
| `not_assessable`: camera không hướng dọc theo đường | **17** | Không |
| Tổng | **36** | **8** |

Đã nhập polyline cụ thể của các đoạn vạch dọc (vạch trắng đứt, vàng đứt), road-edge riêng, crosswalk riêng và ba đối tượng biển/tấm biển ở các cảnh liên quan. Nhãn đánh dấu **chỉ phần vạch quan sát được**, không nối qua xe hoặc vùng khuất.

## Chỉ số so với nhãn AI vừa tạo — exploratory, không phải F1 được người xác minh

- Trên 8 frame khả dụng, so trùng nét vạch theo pixel với tolerance 6 px: **precision 68,46%**, **recall 65,59%**, **F1 67,00%**. Mẫu quá nhỏ, không phải diện rộng hay kiểm định độc lập.
- 6/6 frame có vạch nhìn thấy model vẽ ít nhất một polyline; điều này **không chứng minh** model vẽ đúng hoàn toàn.
- 2/2 frame không thấy vạch ở video Hòa Bình model trả 0 polyline; số lượng này quá nhỏ để khái quát.
- Đặc biệt, tại **42,7s** (`lane_0020`), model trực tiếp trả một nét dọc **nằm đúng trên vạch TRẮNG SÁT DẢI PHÂN CÁCH**, không phải vạch phân làn bên trong. Trùng 619/623 pixel của nhãn road-edge ở tolerance 6px. Đây là một **false lane có bằng chứng ảnh**.
- Tại **299,2s**, một biển trang trí/quảng cáo xanh có hình mũi tên trông giống biển đường không bị model phát hiện thành mã biển. Ở **660,7s**, bảng chỉ dẫn làn màu xanh không được nhận biết thành mã biển đã biết (open-set gap).
- Tại **899,3s**, biển P.130 có thể nhìn thấy rõ, mô hình nhận P.130 và trả ngữ cảnh ô tô **`conditional`**. Biển phụ giờ 06:00–08:00 và 16:00–18:00 có thể đọc nhưng chưa biết thời điểm ghi hình, không suy đoán hiệu lực cụ thể.

**Không đủ số lượng biển có nhãn đầy đủ/exhaustive để tính mAP, precision, recall phát hiện biển trên bộ mới.** Không dùng số cảnh mẫu được phát hiện để khẳng định recall.

## File trên máy tính của bạn

- Nhãn từ thị giác AI: `output/ai_visual_gt_20261014/ai_visual_labels.json`
- Danh sách ảnh, gốc SHA256 và provenance: `output/ai_visual_gt_20261014/selection.json`
- Dự đoán JSONL gọi trên ảnh đúng nguồn: `output/ai_visual_gt_20261014/live_model_predictions.jsonl`
- Báo cáo số học chi tiết: `output/ai_visual_gt_20261014/visual_agreement_report.json`
- Bảng từng frame: `output/ai_visual_gt_20261014/frame_review_summary.csv`
- Từng ảnh overlay: `output/ai_visual_gt_20261014/comparisons/`
- Ảnh tổng hợp so sánh: `output/ai_visual_gt_20261014/model_vs_ai_visual_review_9.jpg`
- Trang người kiểm tra rà, sửa và xác nhận: **`output/ai_visual_gt_20261014/review_ai_visual_36.html`**
- Mã tái tạo và chạy đối chiếu: `scripts/audit_ai_visual_labels_20261014.py`, `scripts/build_visual_gt_review_20261014.py`

Ảnh nguồn và bản ghi nhãn đều lưu **cục bộ**, `output/` được `.gitignore` loại khỏi GitHub, không tự động công khai ảnh từ video của người dùng. Tập nhãn này KHÔNG được dùng làm TEST chính thức cho fine-tune mô hình.

## Cần làm để chuyển thành human-reviewed GT đúng chuẩn

1. Người thật mở `review_ai_visual_36.html` và đối chiếu từng frame với nhãn màu xanh (vạch dọc), đỏ (mép), tím (biển). Nếu sai, người rà sửa bằng nhấp hai điểm.
2. Ghi tên reviewer thật, nhấn Xác nhận frame. Xuất JSON `itasmm_human_review_response.json` bằng nút trên trang. **Chỉ xác nhận những frame đã xem**. Chưa xác nhận hết vẫn được tiếp tục sau.
3. Người đánh giá thứ hai nên rà độc lập với nhãn AI bị ẩn trước khi so khác biệt (blind review). Hai video nguồn độc lập nhiều đoạn không hướng camera về mặt đường nên cần bổ sung **video dashcam nhìn trước** có vạch rõ làm TEST.
4. Chỉ sau khi có số frame GT được người xác nhận, khóa split theo `original video source`, chấm F1 lane và detection sign trên TEST riêng; không tune vào TEST.

Kết luận: Đã thực hiện audit bằng AI thị giác và phát hiện lỗi còn thật trong pipeline, nhưng **chưa thể tuyên bố đã có human ground truth** vì hệ thống không được mạo danh người gán nhãn.
