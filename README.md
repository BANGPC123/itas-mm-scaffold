# Nâng cấp kết hợp model và kiểm chứng (14/10/2026)

Đã thử nghiệm cơ chế đối chiếu mã biển của YOLO11 59 lớp với MobileNetV3 gốc, bổ sung chế độ UFLD-v2 hiển thị nhiều ứng viên hơn và sửa bộ lọc mép đường khi bị xe che khuất. **Không có bằng chứng đủ để tự động gộp YOLO vào biển đã xác minh cho ô tô**: trên 240 ảnh VAL chỉ có 17 ứng viên ngoài vùng biển gốc và không ứng viên nào đạt đồng thuận mã ở ngưỡng nghiêm ngặt. Kiểm tra camera thực đã chứng minh một nét mép đường sai được loại, 26 nét khác trên sáu ảnh được giữ. Xem [báo cáo fusion, benchmark và giới hạn](docs/FUSION_ENHANCEMENT_20261014.md). Các tính năng thử nghiệm không được dùng làm quyết định lái xe.

# Cập nhật 14/10/2026 — YOLO11 optimized và benchmark UFLD-v2 từ ZIP

Đã giải nén an toàn hai gói `comment.zip` và `yolo11_optimized.zip`; nguồn gốc, SHA256, danh sách 59 lớp biển và kết quả VAL được ghi trong [báo cáo nhập mô hình](docs/IMPORTED_COMMENT_YOLO11_20261014.md). Có thể bật **YOLO11 optimized — gợi ý phụ trợ màu tím** khi phân tích ảnh/video. Đây là chế độ rà soát tùy chọn và không thay thế detector chính hoặc tự suy luận hiệu lực biển với ô tô, vì thử nghiệm trên VAL chưa chứng minh cải thiện precision/recall. Trọng số gốc nằm cục bộ trong `data/raw/user_archives_20261014/yolo11_optimized/weights/best.pt` và không được push lên GitHub.

Video có đủ 35.793 frame đã được xác minh ngày 14/10 tại `output/video_recheck_full_20261014/test_video_annotated.mp4`; bản video đó **không** chứa lớp phủ YOLO11 mới vì YOLO11 chỉ chạy khi bật tùy chọn trực tiếp. Lane Việt Nam và điều kiện biển phụ chưa được human-certified. Xem thêm [báo cáo kiểm thử độc lập](docs/ITAS_MM_INDEPENDENT_RELEASE_20261014.md).

# Lịch sử tích hợp — 13/10/2026

Đã chạy mới toàn bộ `test_video.mp4`, sửa vạch qua đường/camera bên hông và xuất 35.793 frame cùng 29 câu đọc. TEST 2.326 ảnh đạt precision mã biển 90,84%, recall 66,45% và recall với biển ≥32 px 92,37%. Ngữ cảnh ô tô đọc điều kiện giờ tự động và giữ trạng thái chưa xác định khi thiếu ngày ghi hình hoặc chi tiết biển. [Báo cáo mới và bằng chứng](docs/VIDEO_QUALITY_20261013.md).

Video cuối: `output/video_quality_20261013/full_v4/test_video_annotated.mp4`; ứng dụng phục vụ tại http://127.0.0.1:8000. 30 FPS là tốc độ phát, AI phân tích 3 Hz. Video ô tô nguồn khác dùng thử vận hành và sửa bộ lọc làn; chưa có GT người độc lập để công bố F1 làn.

## Các mốc nghiên cứu trước

Đã chạy bổ sung tạo nhãn làn yếu tự động, thích nghi và so sánh đủ UFLD-v2/CLRNet/SegFormer-B0, OCR biển phụ, cập nhật nguồn luật và benchmark 240 frame từ hai nguồn mới. [Kết quả và giới hạn của lượt tự động](docs/AUTOMATED_COMPLETION_20261011.md). Bảng nghiên cứu tự động trong giao diện hiển thị trực tiếp báo cáo trên máy; các kết quả OCR chưa xác minh không tự mở điều kiện pháp lý.

Đã kế thừa bản làm việc Windows và nhánh Task 6. [Chi tiết thay đổi, kiểm thử và phần còn thiếu](docs/COMPLETION_20261011.md). Runtime dùng weights đã có ở máy bạn; repository không phân phối lại dữ liệu hay weights.

# ITAS-MM

Ứng dụng học thuật nhận diện biển báo và làn đường, kiểm tra ngữ cảnh cho ô tô và đọc kết quả bằng tiếng Việt.

## Chạy sản phẩm

Yêu cầu Python 3.11+, Node.js và dữ liệu đã nhập theo [hướng dẫn dữ liệu](data/LOCAL_SETUP.md).

```powershell
python -m pip install -r requirements.txt
npm --prefix frontend install
npm --prefix frontend run build
.\start.ps1
```

Mở http://127.0.0.1:8000. Backend phục vụ cả giao diện và API. Trong môi trường Codex hiện tại, các thư viện bổ sung được đặt ở `.runtime-packages`; `start.ps1` đã thêm đường dẫn này.

- Chọn ảnh, nhập GPS mô phỏng và chạy phân tích.
- Hoặc chọn video/mở camera, rồi bắt đầu xử lý từng khung hình. Dừng phân tích/đóng nguồn video để giải phóng camera.
- Xem khung hình có lớp phủ, mã biển, độ tin cậy phát hiện/nhận dạng và trạng thái áp dụng.
- Nếu cần thử logic ngữ cảnh, xác nhận biển chính/phụ, loại xe và vị trí treo trong phần review. Các dữ kiện này là thông tin do người dùng xác nhận.
- Bật giọng đọc để tạo MP3 tiếng Việt. Video đọc theo quan sát đã xác nhận qua nhiều frame; tua video vô hiệu hóa cảnh báo cũ. VieNeu ONNX chạy offline, nội dung đã tạo được lưu cache.

## Mô hình và phạm vi

| Thành phần | Hiện thực |
|---|---|
| Định vị biển báo | YOLOv11n, một lớp `traffic_sign`, canonical TRAIN; video đọc bổ sung ở 960/640 khi crop chính chưa rõ |
| Nhận dạng mã biển | MobileNetV3-Small, 30 mã biển; bản retrain thêm lớp nền để từ chối vật thể khác, crop 128 x 128, chọn checkpoint/ngưỡng trên VAL |
| Làn đường | UFLD-v2/CLRNet/YOLOPv2 đề xuất; SegFormer Cityscapes kiểm tra mặt đường, vạch sơn và xe che khuất; loại vạch qua đường và hướng camera thiếu bằng chứng. Chưa có F1 làn Việt Nam theo GT người độc lập |
| Biển chính/phụ | Gộp theo vai trò đã xác nhận, chồng lấn ngang và khoảng cách dọc; không gộp mọi vật thể gần nhau |
| Lọc xe | Ngữ cảnh `car`, OCR giờ/loại xe và mẫu hình xe; ứng viên tự động giữ cờ pháp lý/chủ thể xác minh ở trạng thái chưa xác nhận |
| Khớp làn | Chiếu dọc biển overhead đã xác nhận vào biên làn hiện tại; thiếu/biên mơ hồ trả `uncertain` |
| Văn bản tham khảo | Tra cứu đúng mã biển trong QCVN 41:2024 đã ghim hash, có trang và URL nguồn |
| Giọng nói | VieNeu v3 ONNX INT8, giọng Hải Đăng có sẵn, chạy offline; MP3 cache theo văn bản/giọng/tốc độ |

Đường dẫn trọng số đang dùng được ghi trong `configs/perception.yaml`. Bản gốc được giữ để đối chiếu. Lượt huấn luyện lại lưu checkpoint, SHA256 và quyết định chọn model tại `output/retraining_20261009/deployment.json`; xem [báo cáo lượt retrain](output/retraining_20261009/REPORT.md). Các manifest/báo cáo cũ mô tả checkpoint ở thời điểm tạo, không đại diện cho mọi bản cập nhật sau đó.

Lượt retrain dùng TRAIN gốc cho YOLO, TRAIN crop biển và crop nền cho classifier, 4.096 ảnh CULane TRAIN cho UFLD. Không đưa video `test_video.mp4` vào train. Video đã dùng phân tích lỗi nên vẫn là tập phát triển, không phải holdout độc lập. Trang kết quả trong ứng dụng hiển thị riêng số đo mới và baseline; video xuất để rà soát lấy mẫu 3 FPS, còn trình phát video đầu vào giữ FPS gốc.

Checkpoint nhận dạng vẫn chỉ có 30 mã; không tự thêm mã ngoài lớp đã huấn luyện. Ảnh đơn dùng crop ≥32 px. Video cho phép crop ≥12 px; crop dưới 16 px phải được hai lượt dò đọc cùng mã với confidence ≥0,95. OCR/hình xe chỉ tạo điều kiện máy gợi ý; không tự xác nhận vị trí overhead, biên làn, hiệu lực văn bản hoặc nhãn người. Kiểm tra tham chiếu/hình biển chặn một số logo sai, chưa chứng nhận khả năng từ chối OOD chung.

GPS/vùng đường hiện được mô phỏng. Kho QCVN là tài liệu tham khảo theo phiên bản, chưa được xác minh hiệu lực pháp lý hiện tại. `validated_corpus: false` chặn luồng sinh hướng dẫn pháp lý tự động. Không suy đoán mức phạt hoặc điều kiện áp dụng từ confidence. Mẫu quy định cũ không được dùng làm kho luật hoàn chỉnh.

## Huấn luyện và đánh giá

Kết quả đã đo ngày 08/10/2026:

| Chỉ số TEST | Kết quả |
|---|---:|
| Detector mAP50 / mAP50-95 | 0,8952 / 0,6808 |
| Recognition accuracy / macro-F1 trên crop chuẩn | 78,78% / 0,7549 |
| Recognition accuracy với cạnh ngắn >=32 px | 93,72% |
| Pipeline phục hồi đúng mã: toàn bộ biển đã phát hành nhãn | 36,23% |
| Pipeline phục hồi đúng mã: biển chuẩn >=32 px | 91,43% |
| Xử lý trung bình, 10 khung hình sau warm-up, không TTS | 67,1 ms |

Baseline detector được giữ vì VAL mAP50-95 cao hơn pha fine-tuning (0,6952 so với 0,6918). Recall toàn pipeline thấp trên toàn TEST chủ yếu cần được xem cùng quy tắc từ chối crop nhỏ; đây không phải độ chính xác lái xe. Báo cáo đầy đủ: [Report 3](output/pdf/DSP391m_FA26_Group2_Report3.pdf). Giao diện cũng có bảng kết quả và liên kết PDF.

Giữ nguyên split canonical; TEST không tham gia chọn mô hình.

```powershell
python scripts/verify_dataset.py
python scripts/train_detector.py
python scripts/train_recognizer.py
python scripts/build_sign_reference.py
python scripts/evaluate_product.py
python scripts/build_model_report.py
```

Huấn luyện detector gồm baseline và một pha tinh chỉnh nhẹ, chọn bằng VAL mAP50-95. Recognition gồm baseline và pha cân bằng lớp/tăng giảm sáng, chọn bằng VAL macro-F1. `train_detector.py --resume-baseline <last.pt>` tiếp tục baseline với hai worker trên máy Windows hạn chế RAM; thêm `--amp-on-resume` để bật mixed precision ở mốc checkpoint; `--baseline <best.pt>` dùng baseline đã hoàn tất để chạy pha tinh chỉnh.

Thử nghiệm báo cáo lần này giữ 7 epoch baseline đã hoàn tất và chạy `--baseline output/training/detector_baseline2/weights/best.pt --fine-epochs 3`. Lịch mặc định 20/10 epoch là cấu hình cho lượt mới, không phải số epoch đã chạy trong báo cáo. Đây là thử nghiệm giới hạn tài nguyên; chưa kết luận mô hình hội tụ.

Kết quả và biểu đồ nằm trong `output/evaluation/`. Crop TEST có kích thước nhỏ và >=32 px được báo riêng. `evaluate_product.py` đo thêm khả năng phục hồi đúng mã biển từ ảnh đầy đủ, với IoU >=0.50 và cơ chế từ chối thực tế. Đây là chỉ số biển báo, không phải độ chính xác hướng dẫn lái xe.

Report PDF nằm trong `output/pdf/`. Báo cáo phân biệt số liệu chạy thật, nhãn AI, luật hình học và phần chưa đánh giá. Chưa thực hiện thí nghiệm contrastive learning, SAHI hay AFPCNN; không gắn tên B1-B3 của proposal cho các thí nghiệm chưa chạy.

## Kiểm thử

Kiểm thử một video ngoài tập dữ liệu, giữ nguyên checkpoint:

```powershell
python scripts/test_video.py "C:\Users\YOUR_USERNAME\Downloads\test_video.mp4" --sample-fps 3
```

Lệnh duyệt toàn bộ video, suy luận mỗi 10 frame đối với nguồn 30 fps, xuất MP4 H.264 720p không âm thanh cùng `frames.csv`, `predictions.jsonl` và `summary.json` trong `output/video_tests/test_video/`. TTS tắt và GPS mô phỏng. Số lần dự đoán lặp theo frame không phải số biển thực, và không phải accuracy khi video chưa có nhãn chuẩn.

Lượt test thực tế đã phát hiện đồng hồ tap-lô bị gán P.127 và đường làn lấn vào nội thất xe. Camera cũng xoay sang bên ở một số đoạn. Xem [báo cáo kiểm thử video](output/video_tests/test_video/TEST_REPORT.md) và MP4 có lớp phủ trong cùng thư mục.

### Chế độ video nhẹ

Video được phát trực tiếp ở FPS nguồn; canvas chỉ cập nhật khi kết quả thay đổi. Nhận diện chạy tối đa 10 lượt/giây, chỉ có một yêu cầu đang xử lý, không xếp hàng các frame cũ. Khi dừng, tua hoặc đổi nguồn, yêu cầu cũ bị hủy và lớp phủ được xóa. Kết quả quá 300 ms bị ẩn; yêu cầu backend quá 20 giây kết thúc với lỗi thay vì chờ vô hạn. Video không gọi TTS.

Trước khi chạy, chỉnh **Ranh dưới vùng đường** ngay trên tap-lô. Giá trị 65% dành cho góc quay ban đầu của `test_video.mp4`, không phải hiệu chỉnh chung cho mọi camera. Chọn 100% khi không cần loại vùng dưới. Camera phải hướng trước; khi quay ngang hoặc bị che, dừng phân tích và hiệu chỉnh lại. Không dùng hình học làn để tự khẳng định hướng camera.

UFLD-v2 dùng FP16 trên CUDA, FP32 trên CPU, giải mã tọa độ theo lô và loại các đường quá ngắn/ngang/ngoằn ngoèo theo hình học. Nạp trọng số qua meta tensors tránh khởi tạo thêm một bản trọng số CPU; checkpoint zip hỗ trợ memory mapping, checkpoint gốc định dạng cũ vẫn được hỗ trợ. Recognition gộp các crop trong một lần forward; video nhận JSON hình học thay vì ảnh JPEG base64 từ server. Checkpoint huấn luyện gốc được giữ nguyên.

```powershell
python scripts/benchmark_runtime.py "C:\Users\YOUR_USERNAME\Downloads\test_video.mp4" --output output/performance/recheck.json --road-bottom .65 --check-dashboard
python scripts/test_video.py "C:\Users\YOUR_USERNAME\Downloads\test_video.mp4" --output output/video_tests/recheck --road-bottom .65
node --test frontend/src/videoRuntime.test.js
```

Chạy benchmark và demo lần lượt trên máy 16 GB RAM, tránh nạp hai bản mô hình cùng lúc. Báo cáo tối ưu và giới hạn: [PERFORMANCE.md](output/performance/PERFORMANCE.md). Report 3 giữ số đo baseline cũ; các thay đổi ROI/FP16 chưa được coi là một lần đánh giá accuracy mới trên TEST. Không dùng confidence nhận diện để kết luận chất lượng vật lý hoặc khả năng phản quang của biển.

```powershell
python -m pytest tests -q
npm --prefix frontend run build
```

API:

- `GET /health`: trạng thái dịch vụ.
- `GET /pipeline/status`: file trọng số và phiên bản lane backend.
- `GET /pipeline/evaluation`: các chỉ số TEST đã đo.
- `GET /pipeline/report`: Report 3 dạng PDF.
- `POST /pipeline/run`: multipart ảnh, latitude, longitude, synthesize_audio; tùy chọn `road_bottom` (0.3–1, mặc định 1), `annotate` (mặc định true; video dùng false).
- `POST /pipeline/interpret`: chạy luật với metadata được người dùng xác nhận.
- `GET /pipeline/audio/{hash}.mp3`: phát âm thanh đã tạo.
- `POST /context/zone`: phân loại vùng GPS mô phỏng.

## Dữ liệu và nguồn

### Cải thiện và rà soát nhãn (09/10/2026)

[Báo cáo thay đổi và phép đo VAL](output/evaluation/improvements_20261009/REPORT.md) lưu riêng với TEST cũ. Runtime dùng classifier `models/sign_recognizer/robust_val.pt`; checkpoint `best.pt` gốc được giữ nguyên. Crop video từ 16px cần hai quan sát cùng mã, chỉ dùng bằng chứng tối đa 0,75 giây trong cùng phiên. Ảnh đơn vẫn yêu cầu 32px. Đây là phiên bản phát triển đang chờ người duyệt và holdout mới, không phải xác nhận accuracy ngoài đường.

[Trang rà soát nhãn video](http://127.0.0.1:8000/review/test_video_development/) có 24 thời điểm kèm ảnh trước/sau, vẽ box và hai biên làn, sửa mã biển và metadata bằng JSON. Nhấn áp dụng sau khi sửa, nhập tên, xác nhận từng ảnh rồi xuất `reviewed_labels.json`. Không đánh dấu nhãn máy là nhãn người; giữ `unknown`/`uncertain` khi không đủ thông tin.

```powershell
python scripts/evaluate_reviewed_context.py "C:\duong-dan\reviewed_labels.json"
python scripts/prepare_video_review.py "D:\video-chuyen-di-moi.mp4" --holdout --output output/review/independent_holdout
```

Video `test_video.mp4` đã dùng để phát triển nên không phải holdout độc lập. Video mới cần từ chuyến đi/camera khác; công cụ lưu SHA nguồn và phiên bản hệ thống trước khi tạo bộ ảnh rà soát. Kết quả từ `/pipeline/evaluation` và PDF Report 3 vẫn là baseline cũ; xem báo cáo mới để biết cấu hình hiện tại.

### Đánh giá lại từng model (08/10/2026)

Kết quả đo trên checkpoint cố định, toàn bộ TEST biển báo và CULane: [báo cáo accuracy và test case](output/evaluation/revalidation_20261008/TEST_REPORT.md). Báo cáo phân biệt mAP của detector, accuracy của classifier và F1 của làn; có số liệu trước/sau lọc hình học, ca lỗi và giới hạn của nhãn. Các số liệu này được lưu riêng với baseline trong Report 3.

CULane TEST nằm ở `data/raw/culane/test/`: 34.680 cặp ảnh/nhãn, kiểm tra tại `data/raw/culane/verification.json`. Nguồn tải theo [INSTALL.md của UFLD-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/INSTALL.md); [điều kiện sử dụng CULane](https://xingangpan.github.io/projects/CULane.html) dành cho nghiên cứu phi thương mại/cá nhân. Không đưa ảnh CULane vào Git hoặc phát hành lại.

Chạy lần lượt để tránh nạp nhiều model trên máy 16 GB RAM:

```powershell
$env:PYTHONPATH='D:\itas-mm\.runtime-packages;D:\itas-mm'
python scripts/evaluate_all_models.py
python scripts/evaluate_detector_conditions.py
python scripts/evaluate_product.py --output output/evaluation/revalidation_20261008/product --skip-benchmark
python scripts/evaluate_culane.py --output output/evaluation/revalidation_20261008/culane_full
python -m pytest tests -q --junitxml=output/evaluation/revalidation_20261008/software-tests.xml
node --test --test-reporter=junit --test-reporter-destination=output/evaluation/revalidation_20261008/video-tests.xml frontend/src/videoRuntime.test.js
python scripts/build_test_report.py
```

Lượt CULane có thể tiếp tục từ `frames.jsonl` khi mã và checkpoint không đổi; nếu thay đổi, dùng thư mục kết quả mới. Metric hiện là bản Python chuyển từ evaluator gốc, chưa đối chiếu binary C++ chính thức. TEST dùng để báo cáo; hiệu chỉnh tiếp theo cần VAL và holdout độc lập.

15.603 ảnh localization / 30.558 boxes; nhận dạng gồm 5.673 TRAIN, 1.272 VAL và 3.058 TEST metadata rows. Nhãn ngữ nghĩa là AI-verified, chưa có human verification độc lập. Dữ liệu masked không phải nhãn âm. Chi tiết nguồn, giấy phép và tính toàn vẹn trong `data/sources.json`, `data/audit/` và Report 2.

- [UFLD-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2), adapter theo MIT; bản giấy phép giữ trong `third_party/UFLD-v2-LICENSE`.
- [Ultralytics](https://github.com/ultralytics/ultralytics), xem giấy phép của thư viện khi phân phối.
- [edge-tts](https://github.com/rany2/edge-tts), sử dụng dịch vụ tổng hợp trực tuyến.

Giấy phép nguồn dữ liệu chưa thống nhất; chưa có quyết định phát hành lại toàn bộ dữ liệu/trọng số. Sản phẩm hiện phục vụ thử nghiệm học thuật có giám sát.
# Cập nhật video và TTS — 10/10/2026

- Video trực tiếp giữ ảnh tối đa 1920 px để đọc crop biển nhỏ; phát video độc lập với suy luận tối đa 10 Hz.
- Sửa hết hạn crop tốt nhất làm mất tên biển dù các quan sát mới vẫn nhận đúng. Vẫn cần hai quan sát đồng thuận và xóa bằng chứng khi tua, mất detection hoặc gặp nhãn xung đột.
- Edge TTS chạy qua `/pipeline/speech`, không giữ khóa GPU. Dừng/tua hủy lời đọc đang chờ; bỏ lời đọc quá cũ, không xếp hàng cảnh báo và chống đọc lặp 20 giây.
- Nội dung nói lấy sau ghép biển chính/phụ, lọc xe và khớp làn. Biển bị `ignored` không đọc; thiếu bằng chứng áp dụng thì chỉ thông báo tên kèm trạng thái chưa xác định.
- Model chỉ có 30 lớp biển chính. Chưa tự đọc biển phụ loại xe/khung giờ hoặc nhận diện vị trí treo; ba quy tắc dùng được khi có metadata đáng tin cậy, không phải đã giải quyết nhận biết ngữ cảnh tự động.
- `scripts/render_product_video.py` xuất đủ khung hình nguồn 30 FPS, chữ Việt và Edge TTS. Âm thanh gốc được thay bằng TTS; thời điểm ghép âm offline không phải độ trễ TTS trực tiếp.
- Kết quả phát triển: `output/product_20261010/`. Đối chiếu 28 crop được AI rà soát: ảnh 1280 khớp 26, ảnh nguồn 1920 khớp 28; cả hai nhận nhầm 2/22 crop nền. Đây không phải human-verified accuracy.


## Cập nhật vận hành 11/10/2026 — chế độ nghiên cứu an toàn

**Không được sử dụng ITAS-MM như hệ thống lái xe, hỗ trợ quyết định pháp lý, hoặc cảnh báo ADAS đã chứng nhận.** Chưa có TEST độc lập có nhãn người xác nhận cho giao thông Việt Nam; `validated_corpus: false`.

### 2 model + 3 logic + Edge-TTS

- YOLO11n xác định ứng viên biển, MobileNetV3-Small đọc mã; nhận biển dùng phần ảnh **phía trên ranh ROI** do camera chỉnh. Nếu khung hình có bảng điều khiển, sử dụng nút **Khoanh vùng loại trừ cabin** trên tab video: tạm dừng, kéo để đánh dấu đồng hồ, gương hoặc vật thể cố định, rồi bấm **Xong khoanh vùng**. Tối đa 8 vùng; tọa độ chuẩn hoá được kiểm tra ở API. Mặt trái: mọi biển thật nằm trong vùng này cũng sẽ bị bỏ qua. Vùng loại trừ đặt lại khi đổi nguồn video/camera.
- UFLD-v2 nhận **toàn khung hình** và trả đường làn dưới dạng toạ độ gốc, không còn nhận ảnh bị cắt 65% như trước. Với video, lớp phủ được làm mượt nhẹ theo thời gian chỉ khi có làn hiện tại; không suy diễn các đường làn thiếu và không dùng đường được làm mượt cho kết luận pháp lý.
- Geometry Matching yêu cầu vị trí biển đã xác minh cùng hình học hai biên làn đáng tin. Vehicle Filtering chỉ dùng điều kiện phương tiện có xác minh. Primary & Supplementary Grouping tự đề xuất **ứng viên** từ bố trí hình học ngay bên dưới, không đọc nội dung OCR và không tạo kết luận khi biển phụ chưa rõ.
- Edge-TTS mặc định tắt trên video và chỉ đọc khi mã biển và phạm vi áp dụng đã được xác minh. Các họ biển cần giá trị số như P.127/P.117 không được phát thành thông báo thiếu giá trị.

### Chạy ứng dụng Windows

Mở hai cửa sổ PowerShell riêng:

```powershell
cd D:\itas-mm
.\start.ps1
```

```powershell
cd D:\itas-mm
npm --prefix frontend run dev
```

### Kiểm thử nhanh

```powershell
cd D:\itas-mm
$env:PYTHONPATH='D:\itas-mm\.runtime-packages;D:\itas-mm'
python -m pytest tests -q
node --test frontend/src/*.test.js
npm --prefix frontend run build
python scripts/compare_lane_roi_20261011.py
python scripts/compare_lane_roi_vietnam_20261011.py
python scripts/smoke_video_stream_20261011.py
```

Các kết quả chi tiết, các giới hạn về nhãn/độ chính xác và ứng viên huấn luyện thử nằm ở `output/ITAS_MM_COMPLETION_STAGE_20261011.md`. Không tự động thay trọng số checkpoint nếu chưa qua đánh giá trên video TEST giao thông Việt Nam độc lập.


## Tạo bộ ground truth từ video Việt Nam — cập nhật 11/10/2026

**Tình trạng:** Hệ thống có công cụ gán nhãn, nhưng hiện chưa có bộ field TEST được người duyệt xác nhận. Các chỉ số accuracy/mAP từ bộ dữ liệu cũ không chứng nhận tính chính xác thực địa.

### A. Duyệt 24 ảnh video phát triển hiện có

Mở `D:\itas-mm\output\review\field_review_pack_20261011\index.html` bằng Edge hoặc Chrome. Bộ này gồm 24 khung hình và 82 box do AI gợi ý (**59 box AI giữ lại + 23 box AI từng loại bỏ**). Tất cả khởi tạo trạng thái `unreviewed`. Trong giao diện:
1. Nhập tên/mã người duyệt; chọn từng box và phân loại `sign / not_sign / uncertain`; sửa mã biển, vai trò chính/phụ. Bổ sung những biển YOLO bỏ sót bằng công cụ vẽ box.
2. Chọn chế độ vẽ lane và gán `ego_left / ego_right / other` cho các vạch nhìn thấy. Không tự bịa đoạn bị phương tiện che.
3. Chọn trạng thái làn; chỉ đánh dấu **đã gán nhãn đầy đủ mọi biển** khi thực sự duyệt toàn khung hình; bấm xác nhận.
4. Bấm **Xuất file JSON**. Kiểm tra bằng:

```powershell
cd D:\itas-mm
$env:PYTHONPATH='D:\itas-mm\.runtime-packages;D:\itas-mm'
python scripts/validate_human_review_20261011.py "C:\Users\<USER>\Downloads\itasmm_human_review_2026-10-11.json"
```

Ảnh cũ thuộc cùng một video **đã dùng để phát triển**; sau khi người duyệt xác nhận, chúng hỗ trợ sửa lỗi/huấn luyện nhưng vẫn **không được gọi là independent field TEST**.

### B. Gán nhãn video mới thực sự độc lập

Giữ video nguyên bản ở vị trí do bạn chọn. Ví dụ:

```powershell
cd D:\itas-mm
$env:PYTHONPATH='D:\itas-mm\.runtime-packages;D:\itas-mm'
python scripts/prepare_new_video_review_20261011.py --video "D:\Videos\route_hcm_001.mp4" --source-group vn_hcm_route_001_20261011 --output "D:\itas-mm\output\review\hcm_route_001" --interval-seconds 5 --max-frames 120 --proposals
```

Mở `output\review\hcm_route_001\index.html`; sửa và xác nhận tương tự. `--proposals` chỉ tạo bbox YOLO **chưa xác minh** để người duyệt kiểm tra, không tự biến dự đoán thành ground truth. Nên lấy nhiều **chuyến đi độc lập** đủ trời sáng, tối, mưa, giao lộ, xe máy đông, bảng quảng cáo/cột đèn và biển phụ.

### C. Chia tập theo CHUYẾN ĐI, không theo ảnh

Đặt một file JSON danh sách các cặp `pack` + `review` + `split` (TRAIN/VAL/TEST) cho từng chuyến đi. Ví dụ:

```json
[
  {"pack":"D:/itas-mm/output/review/hcm_route_001","review":"D:/Labels/route001_review.json","split":"TRAIN"},
  {"pack":"D:/itas-mm/output/review/hcm_route_002","review":"D:/Labels/route002_review.json","split":"VAL"},
  {"pack":"D:/itas-mm/output/review/hcm_route_003","review":"D:/Labels/route003_review.json","split":"TEST"}
]
```

Sau đó chạy `python scripts/import_human_field_corpus_20261011.py --entries "D:\Labels\field_splits.json" --output "D:\itas-mm\output\review\field_corpus_v1"`. Công cụ từ chối ảnh trùng SHA256, thay đổi metadata nguồn, gán cùng `source_group` vào hai split và không bao giờ ghi đè corpus đã lập. Chỉ những khung hình được người duyệt đánh dấu đầy đủ mới dùng để đo recall của detector; bbox chưa duyệt **không** được biến thành hard negative.

### D. Thử nghiệm đọc số P.127 (chưa triển khai)

`scripts/probe_speed_attribute_20261011.py` dùng 639 crop TRAIN và 149 crop VAL để thử đầu phân loại tốc độ trên đặc trưng MobileNet đóng băng. Macro-F1 chỉ **66,27%** trên VAL, nên không đủ tin cậy và **không được đưa vào TTS/áp dụng luật**. Chi tiết `output/evaluation/speed_attribute_probe_20261011/results.json`.
