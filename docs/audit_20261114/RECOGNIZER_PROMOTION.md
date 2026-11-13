# M3 - Đánh giá thăng hạng EfficientNet-B0

**Verdict: FAIL.** Giữ MobileNetV3-small làm recognizer mặc định. Gate accuracy TEST và macro-F1 đều fail. FIELD theo ngưỡng ảnh tĩnh cũng fail. Latency CPU và GPU chưa đo.

## TEST

Tái lập đúng tập old30 của Report 3 V.5 qua `docs/final_report/evidence/NUMBERS.md` và `scripts/evaluate_recognizer_v2_20261028.py`. MobileNet đạt 2858/3144 (0.9090) trên mọi kích thước và 1217/1255 (0.9697) trên crop >=32 px. Cả hai trong dung sai 0.002. Nguồn `baseline.json`.

Bảng dùng raw argmax và macro-F1 theo các mã có nhãn thật trong từng bucket, giống evaluator gốc. Bootstrap percentile 95% dùng 1000 lần lấy mẫu ảnh nguồn có hoàn lại trong mỗi bucket, seed 20261112. Hai model dùng cùng trọng số resample trên từng ảnh. Giữ cố định tập mã của từng bucket trong bootstrap, dùng F1 bằng 0 khi mẫu bootstrap không có mẫu thật lẫn dự đoán của một mã. Hai cột delta là EfficientNet trừ MobileNet, đơn vị điểm phần trăm. Nguồn `test.json` và `test-predictions.csv`.

| Short side (px) | Crop | Accuracy MobileNet [CI] | Accuracy EfficientNet [CI] | Delta accuracy [CI] | Macro-F1 MobileNet [CI] | Macro-F1 EfficientNet [CI] | Delta macro-F1 [CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 3144 | 90.90% [89.77, 92.02] | 91.60% [90.53, 92.63] | +0.70 [-0.13, +1.50] | 92.74% [91.21, 93.85] | 92.97% [91.61, 94.07] | +0.23 [-0.89, +1.46] |
| <16 | 1078 | 78.39% [75.57, 81.10] | 79.96% [77.49, 82.55] | +1.58 [-0.64, +3.71] | 79.12% [68.82, 81.89] | 78.58% [70.57, 82.65] | -0.54 [-4.71, +5.94] |
| 16-32 | 811 | 98.15% [97.12, 99.02] | 98.27% [97.37, 99.04] | +0.12 [-0.61, +0.76] | 98.74% [86.79, 99.22] | 98.22% [86.25, 98.97] | -0.52 [-1.86, +0.28] |
| 32-64 | 753 | 96.28% [94.96, 97.62] | 96.68% [95.37, 97.94] | +0.40 [-0.39, +1.18] | 95.62% [84.96, 96.86] | 95.95% [85.17, 97.00] | +0.33 [-0.69, +1.37] |
| >=64 | 502 | 98.01% [96.72, 99.20] | 98.21% [96.98, 99.21] | +0.20 [-0.61, +1.00] | 94.99% [79.92, 95.53] | 94.99% [79.86, 95.63] | +0.01 [-0.77, +0.72] |
| >=32 | 1255 | 96.97% [96.05, 97.88] | 97.29% [96.35, 98.18] | +0.32 [-0.24, +0.87] | 95.94% [89.17, 97.05] | 96.21% [89.36, 97.29] | +0.27 [-0.59, +1.05] |

## VAL và FIELD

- VAL dùng 3933 crop có nhãn, theo tập VAL A4 gồm biển đủ điều kiện và background. Precision là tỷ lệ đúng trong các dự đoán có mã qua ngưỡng, bao gồm dự đoán sai mã trên background trong mẫu số. MobileNet tại 0.6413 đạt 2913/2974 = 0.97948890. EfficientNet tại 0.840640426 đạt 2865/2925 = 0.97948718. Không có ngưỡng với precision đúng bằng mục tiêu. Chọn điểm gần nhất, ưu tiên coverage cao hơn khi hòa. Nguồn `val.json`.
- FIELD dùng cùng detector, thứ tự ghép IoU >= 0.5 và crop policy ảnh tĩnh như A7. Có 58/65 box được detector ghép. MobileNet tái lập 48/65 ở ngưỡng checkpoint 0.87. EfficientNet đạt 44/65 ở ngưỡng checkpoint 0.92 đã chọn trên VAL ở A4.
- Chạy thêm phép đối chiếu với ngưỡng video đã khóa trên cùng box FIELD và cùng crop policy ảnh tĩnh. MobileNet 49/65, EfficientNet 49/65. Phép này dùng ảnh tĩnh và không có xác nhận theo thời gian. Nguồn `field.json`, quy trình `scripts/analyze_field_a7_20261112.py` và mốc `output/bench_20261112/A7/analysis.json`.

## Gate

| Gate | Kết quả | Evidence |
| --- | --- | --- |
| Accuracy TEST mọi kích thước tăng, CI loại 0 | FAIL | Cận dưới delta âm trong bảng TEST |
| Macro-F1 TEST mọi kích thước, cận dưới CI >= -0.005 | FAIL | Cận dưới delta thấp hơn -0.50 điểm phần trăm |
| FIELD >=48/65 theo A7 | FAIL ở operating point ảnh tĩnh | 44/65. Phép đối chiếu dùng ngưỡng video riêng đạt 49/65 |
| Latency thêm <=3 ms/sample | CHƯA ĐO | `latency-lock.log` ghi nhận khóa M0 còn tồn tại |

## Tái lập

Chạy `scripts/eval_recognizer_promotion_20261114.py` từ worktree audit bằng Python trong kế hoạch. Chạy `crops` trước `field`, có thể truyền cùng `--run-name` cho một lần chạy mới. Đầu ra trong `output/demo_20261114/M3/` của repo chính. Kết quả chính tái lập bằng `--run-name verify-final` và được kiểm lại bằng cách tính metric từ CSV với scikit-learn và đếm từ từng dòng FIELD trong `verification.log`. SHA256 checkpoint và manifest/cache ghi ở `test.json`.

Khi GPU rảnh, người chạy phải lấy `GPU_LOCK` độc quyền, ghi tên và giờ, đặt `ITAS_M3_GPU_LOCK_TOKEN` bằng nội dung khóa rồi chạy `latency-cpu` và `latency-gpu` trên cùng thư mục kết quả. Script đo batch 1 và batch 8, 200 lượt sau 20 lượt warmup, gồm crop/preprocess, model, softmax, hậu xử lý và đồng bộ GPU. Đổi tên khóa khi kết thúc theo kế hoạch. Số đo CPU/GPU chưa được dùng trong verdict.

Nhãn TEST và FIELD là nhãn hỗ trợ bằng máy. FIELD dùng box đề xuất và chỉ có reference khi hai lượt chấm đồng thuận. Kết quả giới hạn ở tập crop và FIELD đã chốt.
