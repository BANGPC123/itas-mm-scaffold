# M7 - Bằng chứng mô hình cho slide

**Giữ detector hiện tại và MobileNetV3-small. A5 KHÔNG HỢP LỆ, loại khỏi bảng kết quả mô hình.**

Các đường dẫn A2 đến A7 và `eval_candidates/` dưới đây tính từ `output/bench_20261112/` trong repo chính. Kết quả M3 nằm trong `output/demo_20261114/M3/` của repo chính. Audit A5 đọc artifact có sẵn trên CPU. Bản đối chiếu log, cache và SHA256 lưu tại `output/demo_20261114/M7/a5_audit.json` trong worktree audit.

## A2 - So sánh detector định vị biển

Cùng conf 0.35, NMS IoU 0.45 và IoU ghép 0.5. CI 95% là chênh lệch ứng viên trừ detector hiện tại trên TEST, từ 1.000 lượt bootstrap theo cặp ảnh, seed 20261112. Nguồn giao thức `scripts/eval_candidates_20261112.py` và `docs/bench_20261112/RESULTS.md`.

| Detector | VAL mAP50-95 | TEST recall | TEST mAP50 | CI delta recall | CI delta mAP50 | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| Detector hiện tại | chưa đo | 0.8754 | 0.9357 | tham chiếu | tham chiếu | eval_candidates/yolo11n.json (current_TEST) |
| yolo11n | 0.7716 | 0.8818 | 0.9353 | [-0.0008, 0.0133] | [-0.0039, 0.0033] | eval_candidates/yolo11n.json |
| yolov8n | 0.7650 | 0.8662 | 0.9299 | [-0.0179, -0.0008] | [-0.0103, -0.0012] | eval_candidates/yolov8n.json |
| yolo12n | 0.7602 | 0.8445 | 0.9208 | [-0.0398, -0.0216] | [-0.0201, -0.0094] | eval_candidates/yolo12n.json |
| yolo26n | 0.7674 | 0.8693 | 0.9370 | [-0.0147, 0.0023] | [-0.0029, 0.0057] | eval_candidates/yolo26n.json |
| yolov8n-p2 | 0.7635 | 0.8648 | 0.9306 | [-0.0190, -0.0014] | [-0.0100, -0.0001] | eval_candidates/yolov8n-p2.json |
| current_finetune30 | 0.7642 | 0.8805 | 0.9299 | [-0.0014, 0.0119] | [-0.0093, -0.0024] | eval_candidates/current_finetune30.json |

Giữ detector hiện tại `models/sign_detector/finetuned_20261026.pt`. Cận dưới CI mAP50 của từng ứng viên đều dưới 0. Detector hiện tại chưa có VAL mAP50-95 theo cùng giao thức. Nguồn bảng và `docs/bench_20261112/RESULTS.md`.

## A3 - Chế độ suy luận của detector hiện tại

| Chế độ | TEST recall | TEST mAP50 | TEST precision | FIELD recall | FIELD mAP50 |
| --- | --- | --- | --- | --- | --- |
| Plain 640 | 0.8754 | 0.9357 | 0.9361 | 0.7727 | 0.8471 |
| Plain 960 | 0.8781 | 0.9255 | 0.9084 | 0.7576 | 0.7640 |
| Multiscale 960+640 | 0.8820 | 0.9275 | 0.9047 | 0.7727 | 0.7786 |
| SAHI 512, overlap 0.2 | 0.7803 | 0.8351 | 0.8057 | 0.7727 | 0.5255 |

Nguồn `A3/ablation_summary.json`. FIELD là pack 239 frame với 66 box tham chiếu A3, có thiên lệch từ box đề xuất. Latency theo chế độ chưa đo trong A3. Gate video M1.5e **đang đo**, chưa có artifact kết quả tại thời điểm rà soát. Nguồn quy trình `scripts/ablation_inference_20261112.py`, `docs/bench_20261112/RESULTS.md` và đường dẫn dự kiến `output/demo_20261114/M1/LATENCY.md`.

## A4 và M3 - Recognizer

A4 báo cáo raw argmax trên `test.old30.all` với 3144 crop. Epoch và ngưỡng được chọn trên VAL. Ngưỡng trong bảng là ngưỡng lưu ở checkpoint hoặc hiệu chỉnh A4. Nguồn theo từng dòng.

| Backbone | Epoch VAL | VAL macro-F1 | TEST accuracy | TEST macro-F1 | Ngưỡng | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| mobilenet_v3_small | 20 | 0.9442 | 0.8998 | 0.9083 | 0.84 | A4/mobilenet_v3_small/results.json |
| mobilenet_v3_large | 14 | 0.9468 | 0.9160 | 0.9183 | 0.87 | A4/mobilenet_v3_large/results.json |
| efficientnet_b0 | 13 | 0.9516 | 0.9160 | 0.9297 | 0.92 | A4/efficientnet_b0/results.json |
| resnet18 | 9 | 0.9551 | 0.9024 | 0.9232 | 0.88 | A4/resnet18/results.json |
| MobileNetV3-small đang dùng | chưa ghi | chưa ghi | 0.9090 | 0.9274 | 0.87 | A4/reference_retrained_20261028/results.json |

A4 cho EfficientNet-B0 tăng **+1.62 điểm phần trăm** accuracy so với MobileNetV3-small huấn luyện lại trong A4. M3 đối chiếu đúng checkpoint MobileNet đang dùng trên TEST old30 chính thức của báo cáo, nên mức tăng còn **+0.70 điểm phần trăm**, CI 95% **[-0.13, +1.50]**. Cả hai phép so sánh dùng cùng tập crop old30. Mốc tham chiếu MobileNet khác nhau. Nguồn `A4/*/results.json`, `M3/test.json` và `docs/final_report/evidence/NUMBERS.md`.

| Chỉ số TEST M3 | MobileNet đang dùng | EfficientNet-B0 | Delta [CI 95%], điểm phần trăm | Gate |
| --- | --- | --- | --- | --- |
| Accuracy | 0.9090 | 0.9160 | +0.70 [-0.13, +1.50] | FAIL |
| Macro-F1 | 0.9274 | 0.9297 | +0.23 [-0.89, +1.46] | FAIL |

M3 **FAIL**. Ở ngưỡng ảnh tĩnh A7, MobileNet đạt 48/65, EfficientNet đạt 44/65. Với ngưỡng video khóa trên VAL (MobileNet 0.6413, EfficientNet 0.840640426), cả hai đạt 49/65 khi áp lên ảnh tĩnh. Latency M3 chưa đo. Nguồn `M3/field.json`, `M3/val.json` và [RECOGNIZER_PROMOTION.md](RECOGNIZER_PROMOTION.md).

## A5 - Detector one-stage 88 lớp

**KHÔNG HỢP LỆ cho benchmark A5.** Run hoàn thành 40 epoch, nhưng lỗi chuẩn hóa nhãn làm loader loại phần lớn ảnh TRAIN và VAL. Nguồn số ảnh và box dự kiến `A5/dataset_manifest.csv` và `A5/dataset_summary.json`. Nguồn thực nạp `A5/onestage_yolo11n_88/train.log` và `labels/train.cache`, `labels/val.cache` bên dưới dataset path trong `args.yaml`.

| Split | Ảnh dự kiến | Box dự kiến | Ảnh loader giữ | Ảnh bị loại | Box được giữ | Lớp có box được giữ |
| --- | --- | --- | --- | --- | --- | --- |
| TRAIN | 10952 | 20923 | 365 | 10587 | 365 | 17 |
| VAL | 2325 | 4747 | 77 | 2248 | 77 | 4 |
| TEST | 2326 | 4888 | chưa đo | chưa đo | chưa đo | chưa đo |

- `attempt_b16/args.yaml` và dataset YAML `D:/DSP391m/output/bench_onestage_20261112/data.yaml` xác nhận `fraction: 1.0`, `cache: false`, batch size 16 và 88 lớp cấu hình. `cache: false` tắt cache ảnh. Log ghi cache nhãn được tạo khi quét TRAIN và VAL.
- Log ghi `non-normalized or out of bounds coordinates` cho toàn bộ 10587 ảnh TRAIN và 2248 ảnh VAL bị loại. Trong mã xuất `scripts/build_onestage_dataset_20261112.py`, mẫu số là `width_px` và `height_px` của **box**. Ví dụ `000e1972a9649df16036beabb3e7793cc10b782f304233b8cc38126da854144e.jpg` có ảnh 640x640 px, box đầu 19.846x20.954 px, và tâm YOLO đã ghi (13.100726, 10.724587). Nguồn là dòng tương ứng trong `data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv`, ảnh và file nhãn cùng stem tại dataset path, và cảnh báo trong log.
- Thời lượng 243.45 s là số tổng từ `results.json`. `attempt_b16/results.csv` có 40 dòng epoch và mốc cuối 228.814 s. Log ghi 23 batch TRAIN mỗi epoch (920 batch tổng), median 3.9 s cho vòng batch TRAIN, và 3 batch VAL trên 77 ảnh. Khối lượng thực mỗi epoch nhỏ do ảnh bị loại.
- CSV ghi epoch 31 có VAL mAP50-95 0.18787, mAP50 0.71604, precision 0.67215 và recall 0.47619. Các số chỉ mô tả subset VAL đã lọc lỗi và không dùng để so mô hình trên slide. So sánh end-to-end trên TEST với pipeline hai stage chưa đo theo `docs/bench_20261112/RESULTS.md`.

## A6 - Truy xuất luật theo mã biển

DEV 33 ca dùng để chọn boost. Holdout 34 ca dùng để báo cáo. Bộ hồi quy gốc 24 ca. Nguồn toàn bộ bảng `A6/retrieval_eval.json`.

| Biến thể | Holdout recall@1 | Holdout MRR | Hồi quy recall@1 | Hồi quy MRR |
| --- | --- | --- | --- | --- |
| R0 | 0.9706 | 0.9853 | 0.5417 | 0.6840 |
| R1 | 0.9706 | 0.9853 | 0.5417 | 0.6840 |
| R2 | 0.9706 | 0.9853 | 0.4583 | 0.6389 |
| R3 | 0.9706 | 0.9853 | 0.4583 | 0.6389 |

Holdout đã gần bão hòa tại baseline, nên không còn dư địa cho mức tăng recall@1 +0.10. R2 và R3 giảm trên bộ hồi quy gốc. Các biến thể thử nghiệm tắt mặc định. Nguồn `A6/retrieval_eval.json` và `docs/bench_20261112/RESULTS.md`.

## A7 - Field pack

Cùng một hệ thống thực hiện hai lượt chấm mù trên pack 239 frame. Có 298 box được chấm cả hai lượt, tỷ lệ đồng thuận nhãn 0.9094. Cohen's kappa ở các nhóm 16-32, 32-64 và >=64 px lần lượt là 0.708, 0.826, 0.967. Nguồn `A7/analysis.json` và `docs/bench_20261112/RESULTS.md`.

Pipeline ảnh tĩnh có recall 0.352 và precision 0.810 đối chiếu 327 box đề xuất. Trên 65 box có mã đồng thuận, detector ghép 58 box, pipeline trả đúng mã 48/65 (0.738). Nguồn `A7/analysis.json`. Pack thiên lệch về họ detector từng đề xuất box. Nhãn tham chiếu đến từ hai lượt chấm bằng mô hình. Nhãn người độc lập chưa có.

## Câu nói trên slide

1. Giữ detector hiện tại vì cận dưới CI của cả sáu chênh lệch mAP50 đều dưới 0. Nguồn bảng A2.
2. EfficientNet tăng +0.70 điểm accuracy so với MobileNet đang dùng, nhưng CI chứa 0 và macro-F1 trượt ngưỡng cận dưới. Nguồn `M3/test.json`.
3. A5 chạy nhanh vì chỉ nạp 365 ảnh TRAIN mỗi epoch sau lỗi chuẩn hóa nhãn, nên loại số liệu A5 khỏi so sánh. Nguồn log train và cache nhãn A5.
4. Truy xuất luật cần bộ hồi quy gốc để thấy suy giảm của R2 và R3 khi holdout gần bão hòa. Nguồn `A6/retrieval_eval.json`.
5. Mốc 48/65 là accuracy mã trên box FIELD được chấm đồng thuận, với thiên lệch từ quá trình đề xuất box. Nguồn `A7/analysis.json`.

Ghi chú kiểm thử cho bàn giao. Sáu test fail của lượt M4 phát sinh do tài nguyên gitignored thiếu trong `wt-audit`. Báo cáo số test chính thức theo M0.5 ở repo chính. Nguồn Mục 6 của `D:/DSP391m/PLAN_DEMO_20261114.md` và `output/demo_20261114/M4/pytest-cpu.log`.
