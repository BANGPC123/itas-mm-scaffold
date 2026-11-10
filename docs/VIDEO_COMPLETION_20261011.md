# ITAS-MM: video đầy đủ và đánh giá ngày 12/10/2026

Đã sửa pipeline, xuất lại toàn bộ `test_video.mp4`, chạy TEST ảnh độc lập và đo API thực tế trên máy Windows tại `D:\itas-mm`. Luồng trình diễn chạy được từ nhận diện đến giọng đọc tiếng Việt. **Chưa đủ bằng chứng để công bố hệ thống hoàn chỉnh về độ chính xác thực địa hoặc suy luận pháp lý.** Recall mã biển trên toàn TEST vẫn chỉ đạt 61,74%; nhãn làn Việt Nam và điều kiện biển phụ chưa được xác minh độc lập.

- Mở ứng dụng: <http://127.0.0.1:8000>.
- Xem video đầy đủ: <http://127.0.0.1:8000/pipeline/capstone-video/user-complete>.
- Video trên máy: `output/video_completion_20261011/release_20261012/test_video_annotated.mp4`.
- Trạng thái và kết quả đo: <http://127.0.0.1:8000/pipeline/video-completion-status>.

Các URL trên dùng trên chính máy đang chạy ứng dụng.

## Video đã chạy đủ và được kiểm tra

| Nội dung | Kết quả |
|---|---:|
| Độ dài nguồn | 1.193,1 giây — 19 phút 53,1 giây |
| Frame nguồn / frame xuất | 35.793 / 35.793 |
| FPS video xuất | 30 |
| Kích thước nguồn / video xuất | 1920×1080 / 1280×720 |
| Tần suất phân tích / số thời điểm | 3 Hz / 3.580 |
| Ảnh so sánh trước/sau lấy đều trên video | 41 cặp |
| Lượt giọng đọc có âm thanh trong video cuối | 30 |
| Lỗi suy luận / lỗi TTS trong bản xuất cuối | 0 / 0 |
| Kiểm tra kỹ thuật bản xuất | 12/12 đạt |

Giải mã kiểm tra **mọi frame của video xuất**, kiểm tra số frame, FPS, kích thước, tọa độ làn, sự khớp thời điểm trước/sau, thời lượng WAV và giải mã track AAC thật. Mọi sự kiện đọc có tín hiệu âm thanh; phép kiểm tra này không chấm độ tự nhiên hoặc độ chính xác phát âm.

Nguồn SHA256: `eb109f8c852df580611c236e2b313b01cf4a62662586ce71fa35525bdda05ac4`.

Video cuối SHA256: `bbcf1f109e4aa7e3742aeef89074250db7bbfe876a8292e21fa398e0683c78b1`, 444.341.266 byte.

Đây là video phát triển sau khi đã dùng nó để sửa lỗi. Không gọi đây là TEST thực địa độc lập. Video giữ 30 FPS; AI chạy 3 Hz, kết quả hiển thị được giữ giữa các thời điểm phân tích. Không có phép suy luận 30 frame/giây trong báo cáo này.

## Những lỗi đã sửa

1. **Vạch làn bị mất vì dashboard chiếm khung hình:** dùng toàn khung cho làn, ước lượng vùng mặt đường, kết hợp ứng viên UFLD-v2, CLRNet và YOLOPv2 với bằng chứng vạch sơn thật. Xử lý pixel tối đa rộng 1280, rồi ánh xạ tọa độ về ảnh nguồn. Không kéo dài đường ra ngoài khoảng có bằng chứng.
2. **Bóng xe, vỉa hè và vạch ngang thành đường làn:** lọc che khuất, tương phản hai phía, vạch ngang và vùng vỉa hè SegFormer. Bản hiện tại loại thêm 123 nét so với ứng viên trước bước veto. Các đường này chỉ dùng hiển thị, không tạo cặp ranh làn có hiệu lực pháp lý.
3. **Đường làn bị kéo cong khi camera thay đổi:** bỏ làm mượt vị trí từ frame cũ đối với backend vạch sơn theo camera. Bước này từng gây nét lệch rõ ở cuối video; bản hiện tại dùng đường khớp trên frame đang quan sát.
4. **Biển ngoài bộ mã và quảng cáo bị ép thành mã đã biết:** thêm bộ từ chối dựa trên embedding đóng băng. Dùng 2.299 tham chiếu từ TRAIN và 30 crop âm của video phát triển do AI quan sát. Không lấy VAL/TEST làm tham chiếu; không gán thêm lớp mới. Crop không được người xác nhận vẫn giữ `human_verified=false`.
5. **OCR chặn xử lý video hoặc trả kết quả cũ sau tua:** tách worker, giới hạn hàng đợi, gắn kết quả với phiên, track, vị trí và tuổi quan sát. Trạng thái mới sau tua loại kết quả OCR của phiên cũ. Điều kiện đọc được vẫn là ứng viên, chưa tự coi biển phụ áp dụng.
6. **Giọng đọc trực tuyến lỗi ngẫu nhiên:** chuyển luồng hiện tại sang VieNeu 3.8.3, ONNX INT8 trên CPU, giọng dựng sẵn Hải Đăng. Worker nằm trong `.tts-venv`, không thay Torch CUDA của nhận diện. Sửa mã hóa UTF-8 giữa các tiến trình Windows, xuất MP3 nguyên tử và dùng bộ nhớ đệm riêng cho giọng mới.
7. **Lượt nhận diện đầu mất hơn 33 giây:** launcher làm nóng mô hình và âm thanh trước khi báo API sẵn sàng, qua cùng đường JPEG của API. Mọi frame và bước làm nóng chạy trên một luồng suy luận cố định, tránh khởi động CUDA lại khi API đổi luồng lúc tạo giọng. Không giữ tracking của phiên làm nóng. Kiểm tra sau khởi động đạt 191,68 ms ở yêu cầu frame đầu; phép đo 132 frame sau đó đạt 170,27 ms ở yêu cầu đầu.

Giọng đọc chỉ phát biểu một **dự đoán tham khảo**. Video đòi hỏi xác nhận qua nhiều frame, confidence đủ và cạnh ngắn bbox ít nhất 32 pixel trước khi đọc. Không phát hướng dẫn pháp lý từ các điều kiện chưa xác minh.

## TEST biển báo độc lập có nhãn

Chạy trên 2.326 ảnh của split TEST đã đóng băng, chứa 3.058 bbox có mã biển được phát hành và 4.888 bbox định vị. Ghép một-một IoU≥0,5; crop tối thiểu 16 pixel, ngưỡng recognizer 0,63. Bộ tham chiếu không dùng ảnh TEST. Detections ghép vào bbox chưa có semantic được phát hành được loại khỏi mẫu số precision mã biển, không mặc định chúng là đúng hoặc sai.

| Chỉ số | Detector 640 | Detector 960 — video hiện tại |
|---|---:|---:|
| Precision mã biển biết được | 92,06% — 1.878/2.040 | **91,16% — 1.888/2.071** |
| Recall mã biển, toàn bộ GT được phát hành | 61,41% — 1.878/3.058 | **61,74% — 1.888/3.058** |
| Recall mã biển, bbox GT cạnh ngắn ≥32 px | 92,88% — 1.095/1.179 | 91,01% — 1.073/1.179 |
| Precision định vị | 92,97% | 90,41% |
| Recall định vị | 78,74% | 81,75% |

Ở 960 vẫn có 69 dự đoán sai mã đã biết và 114 dự đoán mã đã biết không ghép được với GT. Recall ≥32 pixel là số đo trên một nhóm kích thước, không thay thế recall toàn TEST hoặc chứng minh các biển nhỏ đã được giải quyết.

Kích thước 640 cho ảnh tĩnh và 960 cho video được giữ theo kiểm tra VAL và hành vi trên video phát triển. VAL 400 ảnh không đổi kết quả sau thêm bộ từ chối. Trên TEST 960, bộ từ chối giảm một TP và một FP so với trước nó; cải thiện precision rất nhỏ. Việc bỏ nhầm quảng cáo hoặc biển ngoài bộ mã trong video phát triển không chứng minh cải thiện lớn trên dữ liệu mới.

Đây là TEST ảnh có nhãn, **không phải F1/accuracy thực địa trên video của người dùng**.

## Đo API trên video và nguồn khác

Đo riêng khi không còn tác vụ xuất video hay benchmark mô hình chạy đồng thời. Gửi JPEG chất lượng 95 qua HTTP đến `/pipeline/run`, không tạo TTS bên trong yêu cầu frame. 72 frame 1920×1080 từ sáu đoạn của video người dùng và 60 frame 1280×720 từ hai nguồn hành trình khác. GPS mô phỏng.

| Số đo | Trung bình | P95 |
|---|---:|---:|
| HTTP video người dùng, bỏ lượt đầu | **188,93 ms** | **246,58 ms** |
| Pipeline nội bộ, cùng các frame | 164,33 ms | 219,31 ms |
| HTTP hai nguồn hành trình khác | 92,99 ms | 157,28 ms |
| Chi phí gọi OCR bất đồng bộ | 0,19 ms | 1,44 ms |

132/132 frame trả kết quả, không lỗi API. Không có lời hướng dẫn pháp lý chưa xác minh trong mẫu đo. Điều này không chứng minh tỷ lệ cảnh báo sai bằng 0 hoặc accuracy tracking đạt yêu cầu. Hai nguồn khác chưa có ground truth video.

Chi phí gọi OCR không phải thời gian hoàn tất đọc biển phụ: công việc đó chạy phía sau. Âm thanh được đo riêng qua HTTP và tải MP3 trả về, đạt 4/4 yêu cầu:

| Nội dung TTS | Chưa có cache | Đã có cache |
|---|---:|---:|
| Mẫu 1 | 1.127,03 ms | 15,89 ms |
| Mẫu 2 | 1.315,83 ms | 14,53 ms |

Các mẫu chưa cache được xác nhận bằng việc kiểm tra file trước yêu cầu, không suy ra từ số lần thử. Đây là hai nội dung ngắn, không phải benchmark mọi độ dài văn bản. Endpoint video trả Range HTTP 206 đúng 1.024 byte, phục vụ tua/phát video trực tiếp.

Kiểm tra bổ sung **video và giọng đọc đồng thời** trên 12 frame đầu: HTTP trung bình 129,57 ms, P95 233,11 ms; một nội dung TTS mới chưa cache trả âm thanh trong 1.443,34 ms. Frontend hiện tại, asset JS, báo cáo và video đều trả HTTP đúng. Mẫu đồng thời này ngắn, không thay thế đánh giá tải kéo dài.

Bản render cuối tái dùng dự đoán lõi đã quét đủ nguồn và SHA khớp, rồi chạy lại bộ lọc vỉa hè và chính sách đọc. Thời gian lõi trong summary cũ và thời gian replay bộ lọc không được cộng lại để giả thành latency HTTP đo thật; dùng bảng đo API ở trên.

## Đối chiếu cùng các thời điểm trước/sau

Mốc trước là commit `167647521558ed6d5bdef2bb77156515ff7a7552`. Bản trước tắt OCR để không chặn replay; bản sau dùng OCR bất đồng bộ, vì vậy đối chiếu tốc độ không phải hai cấu hình OCR giống nhau.

| Số lượng trên 3.580 thời điểm | Trước | Sau |
|---|---:|---:|
| Detections có mã biết được | 956 | 1.238 |
| Detections chưa nhận dạng | 577 | 1.141 |
| Thời điểm có nét làn hiển thị | 772 | 1.132 |
| Tổng nét làn hiển thị | 786 | 1.323 |
| Track ID xuất hiện | 493 | 786 |
| Thời điểm có mô tả quan sát | 668 | 550 |
| Ứng viên OCR trả về | 0 | 21 |

Đây là **độ phủ dự đoán và số lượng**, không phải accuracy, F1 hoặc số biển riêng biệt. AI quan sát 41 cặp ảnh lấy mỗi 30 giây và cuối video; thấy nét cuối video bám lại vạch sơn, đồng thời còn bỏ sót vạch ngắn, cảnh đông xe, cảnh quay ngang và cảnh bị tay che. Không dùng quan sát này làm nhãn chuẩn độc lập.

## Năm hạng mục yêu cầu ban đầu

| Hạng mục | Công việc đã thực hiện | Giới hạn còn tồn tại |
|---|---|---|
| Nhãn làn Việt Nam | 600 frame người dùng + 59 frame nguồn khác đã xử lý tự động; giữ 91 frame dương một phần, có dấu vết nhãn yếu | Chưa có nhãn chuẩn độc lập; không chuyển `human_confirmed` thành true |
| Fine-tune UFLD-v2, CLRNet, SegFormer-B0 | Đã chạy ba epoch trên nhãn yếu, so sánh sáu biến thể trên cùng 100 ảnh CULane có GT; giữ checkpoint triển khai vì cải thiện không đáng kể hoặc giảm | Không gọi đây là so sánh trên GT Việt Nam đã duyệt |
| Tự đọc biển phụ | Có đề xuất vùng, RapidOCR, hai biến thể đọc, phân tích giờ/loại xe/ngoại lệ, worker video bất đồng bộ | Biển nhỏ, mờ hoặc không đồng thuận chưa đọc đáng tin cậy; điều kiện chưa tự được xác minh |
| Suy luận pháp lý | Kho 5 nguồn chính thức, 974 đoạn; hồi quy 24 câu Recall@4 tăng 83,33%→87,50% | Còn 3 câu bỏ sót; hiệu lực, ngoại lệ, OCR văn bản và phạm vi áp dụng chưa kiểm chứng hết; `validated_corpus=false` |
| TEST toàn hệ thống | Video gốc đầy đủ, TEST ảnh 2.326 ảnh có GT, hai nguồn video khác, đo HTTP/OCR/TTS, kiểm tra xuất và frontend | Chưa có GT video độc lập để công bố F1 làn Việt Nam, biển nhận nhầm theo thời gian, cảnh báo sai hoặc accuracy tracking |

Chi tiết nhãn yếu, fine-tune và kiểm tra nguồn luật nằm trong [báo cáo giai đoạn tự động](AUTOMATED_COMPLETION_20261011.md). Không cần người dùng duyệt nhãn để chạy các bước đã hoàn thành; công cụ không tạo được chứng nhận độc lập chỉ bằng việc tự gán nhãn rồi tự chấm nó.

## Kiểm thử và tái lập

- `python -m pytest tests -q`: **395 passed**, sáu cảnh báo thư viện, 17,91 giây.
- `node --test src/video*.test.js`: **7 passed**, không lỗi.
- `npm run build`: đạt, 5,69 giây.
- Kiểm tra video: 12/12 đạt; mọi frame và AAC được giải mã kiểm tra.
- Bộ audit dữ liệu cũ được thu thập khi chạy pytest từ toàn repo vẫn thiếu 27 artifact đóng băng trong phiên trước. Không xóa hoặc sửa chúng để che lỗi; số 395 ở đây là bộ phần mềm `tests/`.

Mã, cấu hình và các kiểm thử mới nằm trong repo. Dấu vết sao lưu tại `docs/evidence/video_completion_20261012/`; các predictions JSONL và video lớn giữ trong thư mục output trên máy. Bộ tham chiếu 8.085.682 byte có SHA256 `e9f543aba6df389015c537b8f456aeb1d2ec91a2bb5fcf8e3c7528a6ad023cb9`; binary hiện có trên máy, không nằm trong bản source checkout. Manifest ghi toàn bộ positive box ID và negative bbox/thời điểm để tái tạo khi có lại dữ liệu nguồn.

`LATEST.json` trỏ đến số liệu cuối trong `live_api_after_worker.json`, `delivery_after_worker.json` và `software_after_worker.json`. Các phiên đo trước vẫn được giữ nguyên. Trường `reference_margin` bị thiếu trong report TEST gốc; cấu hình thực tế dùng margin 0,015 và min similarity 0,8, được ghi riêng trong `run_configuration.json`.

Các lệnh chính, chạy từ gốc repo với runtime đã cài:

```powershell
python scripts/setup_road_surface_veto.py
python scripts/setup_offline_speech.py
python scripts/build_sign_reference_gate.py --negative-manifest docs/evidence/video_completion_20261012/reference_bank.json --source <test_video.mp4> --output <new-reference-bank.npz>
python scripts/complete_test_video.py <source.mp4> --output <new-core-output> --analysis-hz 3 --baseline-predictions <verified-baseline/predictions.jsonl>
python scripts/render_surface_completion.py --base <verified-core-output> --output <new-overlay-output>
python scripts/release_completed_video.py --base <verified-overlay-output> --output <new-release-output>
python scripts/verify_complete_video.py <new-release-output>
python scripts/evaluate_completed_video_live.py --output <new-http-evaluation-output>
python scripts/verify_product_delivery.py --output <new-delivery-evaluation-output>
powershell -ExecutionPolicy Bypass -File scripts/start_itas_mm_capstone.ps1
```

Giữ `.runtime-packages` trong PYTHONPATH như launcher; CLRNet dùng `.clr-runtime`, OCR dùng `.ocr-venv`, giọng đọc dùng `.tts-venv`. Các script xuất/đánh giá yêu cầu thư mục mới để không ghi đè kết quả cũ. Muốn checkout source trên máy khác phải khôi phục checkpoint và dữ liệu đã ghi provenance trước khi tái lập.

## Nguồn mô hình mới

- [SegFormer-B0 Cityscapes của NVIDIA](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024), ghim revision `21b3847fae21ddee674abd31129307b6a1235bd9` và SHA của ba file trong helper setup. Chỉ dùng veto vùng vỉa hè, không coi output phân đoạn là nhãn chuẩn Việt Nam. Model card ghi giấy phép `other`; xem điều khoản nguồn trước khi phân phối trọng số.
- [VieNeu-TTS của tác giả](https://github.com/pnnbao97/VieNeu-TTS/blob/main/README_PYPI.md) và [v3 Turbo model card](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo). SDK được ghim 3.8.3; inference hiện tại chạy ONNX CPU, dùng giọng dựng sẵn Hải Đăng. Model card v3 Turbo ghi Apache-2.0. Giữ cache mô hình đã tải khi chạy offline; không dùng bản sửa Torch CUDA để tạo giọng.
