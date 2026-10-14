# ITAS-MM | Kiểm thử độc lập và hoàn thiện bản prototype (14/10/2026)

## Mục đích và nguyên tắc nghiệm thu

Đồ án là Web/Desktop nghiên cứu xử lý video, **không phải ADAS đã được chứng nhận để lái xe thực tế**. Không dùng nhãn AI hoặc bộ lọc tự sinh làm human ground truth. Không lấy số đường vẽ / số biển đọc được để suy ra accuracy. Khi điều kiện xe, hướng, làn, ngày giờ ghi hình hoặc biển phụ không rõ, hiển thị `conditional/uncertain`, không phát âm lệnh pháp lý chắc chắn.

## Kết quả chạy độc lập, 14/10

| Công việc | Bằng chứng | Kết quả |
|---|---|---|
| Python toàn bộ `tests/` | `output/tests_20261014_full.log` | **451 passed**, 6 warnings, 0 failed |
| Frontend JS | `output/frontend_tests_20261014.log` | **7 passed**, 0 failed |
| Frontend production | `output/frontend_build_20261014.log` | **PASS** |
| UFLD hiện hành, **600 CULane TEST** chọn hash trên 34.680 ảnh | `output/evaluation/culane_holdout_20261014/results.json` | Raw TP=1317, FP=417, FN=453; F1=**75,17%**. Bộ lọc hình học cũ TP=762, FP=199, FN=1008; F1=**55,80%** |
| Nhận dạng biển báo, **400 ảnh TEST có nhãn** | `output/evaluation/sign_test_recheck_20261014/report.json` | Recall mã=**66,86%** (339/507), precision mã=**90,64%** (339/374); biển dưới 16 px=**18,71%** (32/171) |
| Kiểm thử vận hành trên ảnh **khác nguồn** ở Hà Nội/Gò Vấp | `output/evaluation/source_disjoint_smoke_20261014.json` | **20/20** không lỗi; không phát ra khẳng định luật chưa xác minh; **không có GT để tính accuracy** |
| Audit checksum corpus | `output/evaluation/independent_holdout_audit_20261014.json` | **659/659** ảnh trong hai review pack khớp SHA256; **0 frame** có xác nhận ground truth lane của người |
| Đối chiếu hình trực quan | `output/video_recheck_20261014/before_after_5_scenes.jpg` | Nét sai ở giây 43/58 cạnh dải phân cách đã bị lọc; nét làn đứt giây 540/740 được giữ |
| Full video mới | `output/video_recheck_full_20261014/` | **ĐÃ XÁC MINH: 35.793/35.793 frame, 30 FPS, 3.580 thời điểm AI, 29 sự kiện âm thanh, 12/12 checks đạt** |

Lưu ý: CULane TEST là ảnh có nhãn từ miền CULane, không phải miền Việt Nam. Đợt đánh giá mới phải tách khỏi báo cáo cũ: bản cũ 34.680 ảnh dùng phiên bản decoder khác; phát hiện bằng đối chiếu mã nguồn SHA256. Bản 600 ảnh này được chạy lại trên mã hiện hành. `source_disjoint_smoke` độc lập về **nguồn video**, nhưng **không độc lập theo chuẩn GT có người gán nhãn**.

## Sửa chữa kỹ thuật

- Thêm `src/perception/road_edge_veto.py`, tích hợp `camera_lane.py`: loại các đường dài dự đoán bám sát bờ mặt đường, giữ nét đứt giữa đường; chỉ sửa **visual display**, không tự tạo cặp ego lane cho lập luận pháp lý.
- Cải thiện `sign_scope_principles.py` và `sign_applicability.py`: S.505a chỉ `ô tô` chung mở rộng sang ô tô con/xe tải/xe buýt/van; `ô tô con` chỉ riêng car; các điều kiện không biết luôn `uncertain`.
- `car_context.py` đổi thông báo chưa kiểm chứng thành **có thể áp dụng**, phân biệt ngày giờ giả lập với timestamp ghi hình thực.
- Thêm `tests/unit/test_video_car_lane_recheck_20261014.py` và những kịch bản ngữ cảnh ô tô có biển phụ.
- Không chỉnh checkpoint hoặc thay đổi model chỉ để tối ưu TEST. Các thử nghiệm tăng độ phân giải/tile trên VAL đã làm tăng dự đoán sai; được giữ ở mức thử nghiệm.

## Phần chưa thể tuyên bố hoàn thành

1. **Không có ground truth lane độc lập được người xác nhận cho đường Việt Nam**. Hai pack 600 và 59 frame mới chỉ có AI proposals, không phải nhãn người.
2. **Recall biển nhỏ rất thấp**, hệ thống còn giới hạn bộ từ vựng 30 mã và có thể bỏ sót biển ngoài danh mục.
3. **Phạm vi pháp lý, luật sửa đổi, dữ liệu ngày giờ/xe/làn** chưa kiểm chứng đầy đủ; `validated_corpus=false`. Không thể xác nhận mọi trường hợp áp dụng cho car trong video nếu thiếu nhìn rõ biển phụ, chiều đường và ngày giờ.
4. Cần người rà dữ liệu phân đoạn làn, tập TEST không cùng route train, fine-tune và báo cáo precision/recall/F1 của pipeline cuối trên đường Việt Nam. Cần đánh giá tỷ lệ cảnh báo sai theo thời gian video.

**Trạng thái nghiệm thu:** Prototype phần mềm chạy được và có kiểm thử độc lập theo phạm vi mô tả; **NOT FIELD VALIDATED**. Không phát hành chức năng ra quyết định lái xe ngoài thực tế.

## Video cuối đã phát hành và mã kiểm chứng

- File: `D:\itas-mm\output\video_recheck_full_20261014\test_video_annotated.mp4`.
- Kích thước: **445.369.225 bytes**; SHA256: `0d3813e5621a3c54fbe1bdee5dd07c3024c26b7b3b7a1815d85fb92fb7002ec1`.
- Source gốc SHA256: `eb109f8c852df580611c236e2b313b01cf4a62662586ce71fa35525bdda05ac4`.
- `verification.json`: **12/12 checks PASS**, giải mã toàn bộ 35.793 frame, audio AAC giải mã có âm, 29/29 sự kiện giọng đọc có tín hiệu, 0 phát ngôn luật chưa xác minh.
- 5.841 điểm vạch dự đoán bị lọc thêm trên toàn video (đếm số điểm, không phải số đường đúng/sai), không có điểm dự đoán mới được sinh từ bộ lọc.
- Video giữ các kết quả nhận diện ML đã quét trước đó; tính lại mask mặt đường và điều kiện car, không phải một lượt fine-tune hay đo FPS AI mới.
- Đường dẫn phục vụ: `http://127.0.0.1:8000/pipeline/capstone-video/user-complete` khi backend đang chạy.

## HTTP kiểm thử trên backend đã khởi động

- `/pipeline/video-completion-status`: `complete=true`, `verification.passed=true`, 35.793 frame và SHA256 đúng bản 14/10.
- `GET /pipeline/capstone-video/user-complete` với Range 0–1023: HTTP **206**, trả đúng **1.024 byte**, Content-Type `video/mp4`.
- `/pipeline/status`: `mode=research_preview`, `field_validated=false`, `legal_corpus_validated=false` và các checkpoint sẵn sàng.
- `/pipeline/run` trên các frame 149,333s/540s/740s: **3/3 HTTP 200**, không phát thông báo luật thiếu xác minh. Frame 149,333s phát hiện P.131a và trả `car_context.status=conditional` khi không biết thời điểm ghi hình.
- Manifest nghiệm thu cập nhật: `output/evaluation/ITAS_MM_RELEASE_GATE_20261014.json`: `software_operational=true`, `full_video_verified=true`, `field_validation_complete=false`.
