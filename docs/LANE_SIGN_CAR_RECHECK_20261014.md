# ITAS-MM — kiểm tra lại lane, biển báo và ngữ cảnh ô tô (14/10/2026)

## Phạm vi
- Video gốc: `C:\Users\YOUR_USERNAME\Downloads\test_video.mp4`, 35.793 frame, 30 FPS, 1.193,1 giây.
- Đã rà lại 10 mốc hình ảnh (28, 33, 43, 58, 150, 540, 660, 740, 900, 1073 giây); lane ở 43 và 58 giây có nét hiển thị sai sát rào/mép đường.
- Chưa có ground truth lane do người độc lập gán nhãn cho video này. Không công bố F1/accuracy lane.
- Video cuối 13/10 ở `output/video_quality_20261013/full_v4/test_video_annotated.mp4` **chưa** được thay thế bằng bản hôm nay.

## Code được sửa
1. `src/perception/road_edge_veto.py`: loại nét dài bám sát mép vùng road segmentation dựa trên nhiều điểm trong nét; không sinh nét mới; vạch ngắn giữ để rà.
2. `src/perception/camera_lane.py`: nối bộ lọc mép đường sau kiểm tra vạch qua đường ở nhánh camera-aware (visual only).
3. `src/context/sign_scope_principles.py`: phân biệt ô tô nói chung với ô tô con; xe tải/bus/van có thể thuộc nhóm chung ô tô, nhưng không nằm trong nhóm riêng ô tô con.
4. `src/context/sign_applicability.py`: phép giao điều kiện đã xác minh của biển S.505a với điều kiện biển chính trên các lớp xe thực, không gộp ô tô chung thành ô tô con.
5. `src/context/car_context.py`: giữ kết luận *có thể áp dụng* khi chưa đủ điều kiện, thay vì khẳng định đã áp dụng.
6. `tests/unit/test_video_car_lane_recheck_20261014.py`: ca hồi quy mới.
7. `scripts/recheck_car_lane_video_20261014.py`: xuất preview từ kết quả suy luận đã kiểm chứng, có tính lại mặt đường và ngữ cảnh; không train hoặc tính lại mô hình sign/lane.

## Kiểm thử
- Toàn bộ pytest: **451 passed**, 6 warnings, 0 failed. Log `output/video_quality_20261013/pytest_car_lane_20261014.txt`.
- VAL thử detector/recognizer: 400 ảnh cùng thứ tự SHA256, TEST không dùng để chọn cấu hình hôm nay.
  - Cấu hình rescue_12 (hai lượt 960/640): recall 396/540 = **73,33%**, precision 396/429 = **92,31%**.
  - full1280_12: recall 385/540 = **71,30%**, precision 385/440 = **87,50%**.
  - tiles_12: recall 420/540 = **77,78%**, precision 420/500 = **84,00%**.
  - focus960_12: recall 408/540 = **75,56%**, precision 408/459 = **88,89%**.
  - Không có phương án thay thế ngay đạt đồng thời recall và precision tốt hơn baseline, vì vậy giữ cấu hình đã phát hành.
  - Log: `output/video_recheck_20261014_sign_val400.log`, `output/video_recheck_20261014_sign_loc.log`.
- TEST cũ (chưa bị chỉnh/tái sử dụng để tune): 2.326 ảnh, recall mã **66,45%**, precision mã **90,84%**, recall dưới 16px **18,76%**; không phản ánh accuracy trực tiếp trên video người dùng.

## Ngữ cảnh car
- Biển P.130/P.131a: xét phạm vi phương tiện theo ý nghĩa biển chính trước, biển phụ S.505a chỉ thu hẹp phạm vi khi đọc và xác minh được. Điều kiện giờ/ngày, biển trừ và điều kiện hướng/làn xét riêng.
- Biển với S.505a chỉ định ô tô con: `car` có thể thuộc phạm vi, `truck` không; S.505a chỉ định ô tô nói chung: `car`, `truck`, `bus`, `van` thuộc nhóm (nhưng không có nghĩa tất cả loại ô tô đều được khẳng định có hiệu lực ngay).
- Khi biển phụ không rõ hoặc hướng camera không xác định, kết quả `conditional` / `uncertain`, tuyệt đối không ra lời chỉ dẫn đã xác minh.
- Frame 4480 (149,333s) phát hiện P.131a với các khoảng 09:00–16:00 và 20:00–06:00. Ngày/giờ thực tế ghi hình không có, nên `conditional`. Kiểm thử 10:00 và 18:00 là dữ liệu giờ giả lập, không phải thời điểm video.
- `human_verified=false`, `legal_effect_established=false` trong nhận diện từ máy.

## Kết luận giới hạn
- Sửa mép đường chỉ là bộ lọc visual precision-first; chưa phải fine-tune model lane, và **không** xác nhận làn pháp lý. Một vài vạch thật có thể bị loại.
- Chưa đạt yêu cầu nâng chất lượng biển báo vượt baseline với độ chính xác đủ tin cậy; cần dữ liệu biển nhỏ/che khuất của Việt Nam và nhãn người.
- Preview đã hoàn tất ở `output/video_recheck_20261014/lane_car_scope_preview.mp4`: sáu đoạn tổng cộng 48 giây, 1.440 frame, 30 FPS, đã loại 27 lượt nét dài nghi là mép đường; vùng giây 42–43 và 55–62 đã được rà theo bộ lọc mới. Đây là lượt lọc dự đoán theo frame, không phải 27 lỗi được người xác minh.
- `output/video_recheck_20261014/preview_summary.json` lưu phạm vi, số frame, và timestamp từng lần lọc. Không được ghi là đã xử lý lại toàn bộ 35.793 frame.
