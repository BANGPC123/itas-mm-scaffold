# M15 - Rà soát biển báo trên video demo

## Luật chấm cố định trước khi xem crop

- Đối chiếu màu, hình dạng, ký hiệu và hướng mũi tên trong crop với danh mục mã ở `data/reference/recognizer_v2_classes.json`, các mẫu có sẵn ở `data/reference/qcvn2024_symbols.json` và hình/tên trong bản QCVN lưu cục bộ. Không dùng mã dự đoán để nhận diện lúc xem crop.
- `dung_ma` khi đọc được mã tham chiếu và mã đối chiếu trùng. `sai_ma` khi đọc được mã tham chiếu và mã đối chiếu khác, ghi cả hai mã. `khong_phai_bien` khi có bằng chứng đó là vật khác. `khong_ro` khi hình chưa đủ rõ để xác định, hoặc khi nhìn rõ biển nhưng không có mã để đối chiếu. Nếu nhận ra biển nhưng chưa xác định mã, ghi đặc điểm nhìn được.
- Với known, mã đối chiếu là mã đã xác nhận của nhóm quan sát. Với uncertain, hệ thống không phát mã. Chỉ dùng `raw_label` khác `unknown` của chính crop được chọn để chấm ứng viên thô. Nếu crop đọc được nhưng thiếu mã thô, vẫn lưu mã quan sát và ghi rõ lý do `khong_ro`. Kết quả uncertain không tính vào tỷ lệ known.
- Ghép một-một theo thời gian và IoU, bỏ qua nhãn khi ghép. Hai quan sát cách tối đa 3 mẫu phân tích, tương đương 0,5 s ở 6 Hz, có thể vào cùng nhóm nếu IoU đạt ít nhất 0,30 trên box gần nhất hoặc box ngoại suy tuyến tính từ hai mẫu trước. Với cùng `track_id`, cho phép ghép khi box có tỷ lệ diện tích từ 0,5 đến 2 và tâm cách không quá đường chéo box trước, theo quy tắc nối track của hệ thống. Ưu tiên cùng ID, sau đó IoU cao hơn. Khi hai nhóm cùng có mặt trong một mẫu, giữ tách. Mỗi lần xuất hiện liên tục được đếm một lần. Khi biển ra khỏi khung rồi xuất hiện lại, hai lần xuất hiện có thể là hai nhóm.
- Một nhóm vào known nếu có ít nhất một mẫu có mã xác nhận. Lấy mã xác nhận xuất hiện nhiều nhất. Khi hòa, lấy mã xuất hiện trước và ghi lại tất cả mã từng phát. Nhóm còn lại vào uncertain. Chấm toàn bộ known. Lấy mẫu đều, không hoàn lại, tối đa 30 nhóm uncertain bằng Python `random.Random(20261114)` trên danh sách ID tăng dần. Nếu quần thể dưới 30, chấm toàn bộ.
- Giữ một crop từ ảnh gốc cho mỗi nhóm. Xếp hạng bằng cạnh ngắn và độ nét Laplacian đo trên crop gốc, ưu tiên diện tích hữu dụng và phạt crop sát mép khung. Khóa thuật toán trước khi xem. Crop xem có viền ngữ cảnh, phần bên trong box quyết định nhãn. Ảnh phóng để đọc, không phục hồi chi tiết.
- Tỷ lệ known dùng `dung_ma` trên toàn bộ nhóm known đã lấy mẫu, kể cả `khong_ro` và `khong_phai_bien`. Đếm cặp nhầm có hướng mã dự đoán sang mã quan sát, tách mã phát và mã thô. AI-reviewed, không phải nhãn người. Chỉ một video, chưa đo recall.

Nguồn thiết kế: Mục 10 của `D:/DSP391m/PLAN_DEMO_20261114.md`, tần số trong `M13/clips/demo_video/render/summary.json` và IoU trong `src/perception/temporal_signs.py`.

## Nguồn và đơn vị đếm

Nguồn chính `output/demo_20261114/M13/clips/demo_video/render/predictions.jsonl` trong repo chính. Ảnh lấy từ video gốc theo `source_frame`, kiểm SHA256 với `summary.json`. Danh mục mã lấy từ `data/reference/recognizer_v2_classes.json`. Hình đối chiếu trích từ `data/regulations/official_20261011/QCVN41_2024.pdf` tại trang 80 (P.102), 117 (R.302a, R.303), 137 (nhóm I.423). Mẫu I.409 đối chiếu thêm với `data/reference/qcvn2024_symbols.json::templates.I.409`. Kết quả ghi mã họ I.423 theo danh mục của hệ thống. [Hình tham chiếu](../../output/demo_20261114/M15/scored/qcvn_reference.png)

Điểm crop là cạnh ngắn nhân `log1p` phương sai Laplacian của ảnh xám resize 64x64, theo `src/perception/image_quality.py::crop_quality`. Crop cách mép khung tối đa 1 px nhận nửa điểm. Khi điểm hòa, giữ crop sớm hơn. Khung nhìn chấm thêm biên bằng 22% cạnh ngắn. [Manifest crop và nguồn](../../output/demo_20261114/M15/scored/manifest.json)

Sau khi ghép theo thời gian và hình học, có 28 nhóm known và 29 nhóm uncertain. Seed **20261114**. Quần thể uncertain có 29 nhóm, thấp hơn mức lấy mẫu 30, nên đã chấm toàn bộ. Các nhóm là lần xuất hiện liên tục trong video. Số biển vật lý duy nhất trên tuyến chưa đo. [Manifest](../../output/demo_20261114/M15/scored/manifest.json)

## Kết quả

| Nhóm | n | dung_ma | sai_ma | khong_phai_bien | khong_ro |
| --- | ---: | ---: | ---: | ---: | ---: |
| Known | 28 | 27 | 0 | 0 | 1 |
| Uncertain | 29 | 11 | 1 | 3 | 14 |

**Known đúng mã 27/28 (96,4%).** Nhóm known chưa rõ là S024, box bao cùng lúc một biển mũi tên chéo và một biển vòng xuyến. Trong nhóm uncertain, cột `dung_ma` là so với ứng viên thô, dù hệ thống không phát mã. [Phiếu chấm](../../output/demo_20261114/M15/scored/review_sheet.csv) · [Bảng đếm](../../output/demo_20261114/M15/scored/counts.json)

### Cặp nhầm quan sát được

| Nhóm | Mã đối chiếu | Mã quan sát | Số ca | ID |
| --- | --- | --- | ---: | --- |
| Known | Không thấy cặp sai xác định được |  | 0 |  |
| Uncertain, mã thô | I.409 | I.423 | 1 | S036 |

Cặp I.409 sang I.423 xuất hiện một lần trong tập này. S036 có biển vuông xanh, tam giác trắng và dáng người, khác mẫu mũi tên quay đầu I.409. Các ca box chứa nhiều biển và ca thiếu ký hiệu rõ giữ `khong_ro`. [Crop S036](../../output/demo_20261114/M15/scored/S036.png) · [Bảng cặp nhầm](../../output/demo_20261114/M15/scored/counts.json)

**AI-reviewed, không phải nhãn người.** Kết quả giới hạn ở một video demo và các box hệ thống đã phát. Recall chưa đo.
