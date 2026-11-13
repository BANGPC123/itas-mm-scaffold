# Số liệu cho slide demo

Đường dẫn tương đối tính từ root repo. Artifact M11 và M7 nằm trong worktree audit. Artifact benchmark và render còn lại đọc từ repo chính.

## Detector định vị biển

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| Detector hiện tại - TEST recall / mAP50 | 0,8754 / 0,9357 | 2.326 ảnh TEST | [RESULTS.md:13](../../docs/bench_20261112/RESULTS.md#L13) | khớp nguồn |
| Detector hiện tại - VAL mAP50-95 | chưa đo | chưa đo | [RESULTS.md:13](../../docs/bench_20261112/RESULTS.md#L13) | chưa đo |
| yolo11n - VAL mAP50-95 | 0,7716 | 2.325 ảnh VAL | [RESULTS.md:14](../../docs/bench_20261112/RESULTS.md#L14) | khớp nguồn |
| yolo11n - TEST recall / mAP50 | 0,8818 / 0,9353 | 2.326 ảnh TEST | [RESULTS.md:14](../../docs/bench_20261112/RESULTS.md#L14) | khớp nguồn |
| yolo11n - CI 95% delta TEST recall / mAP50 | [-0,0008; +0,0133] / [-0,0039; +0,0033] | 2.326 ảnh TEST ghép cặp | [RESULTS.md:14](../../docs/bench_20261112/RESULTS.md#L14) | khớp nguồn |
| yolov8n - VAL mAP50-95 | 0,7650 | 2.325 ảnh VAL | [RESULTS.md:15](../../docs/bench_20261112/RESULTS.md#L15) | khớp nguồn |
| yolov8n - TEST recall / mAP50 | 0,8662 / 0,9299 | 2.326 ảnh TEST | [RESULTS.md:15](../../docs/bench_20261112/RESULTS.md#L15) | khớp nguồn |
| yolov8n - CI 95% delta TEST recall / mAP50 | [-0,0179; -0,0008] / [-0,0103; -0,0012] | 2.326 ảnh TEST ghép cặp | [RESULTS.md:15](../../docs/bench_20261112/RESULTS.md#L15) | khớp nguồn |
| yolo12n - VAL mAP50-95 | 0,7602 | 2.325 ảnh VAL | [RESULTS.md:16](../../docs/bench_20261112/RESULTS.md#L16) | khớp nguồn |
| yolo12n - TEST recall / mAP50 | 0,8445 / 0,9208 | 2.326 ảnh TEST | [RESULTS.md:16](../../docs/bench_20261112/RESULTS.md#L16) | khớp nguồn |
| yolo12n - CI 95% delta TEST recall / mAP50 | [-0,0398; -0,0216] / [-0,0201; -0,0094] | 2.326 ảnh TEST ghép cặp | [RESULTS.md:16](../../docs/bench_20261112/RESULTS.md#L16) | khớp nguồn |
| yolo26n - VAL mAP50-95 | 0,7674 | 2.325 ảnh VAL | [RESULTS.md:17](../../docs/bench_20261112/RESULTS.md#L17) | khớp nguồn |
| yolo26n - TEST recall / mAP50 | 0,8693 / 0,9370 | 2.326 ảnh TEST | [RESULTS.md:17](../../docs/bench_20261112/RESULTS.md#L17) | khớp nguồn |
| yolo26n - CI 95% delta TEST recall / mAP50 | [-0,0147; +0,0023] / [-0,0029; +0,0057] | 2.326 ảnh TEST ghép cặp | [RESULTS.md:17](../../docs/bench_20261112/RESULTS.md#L17) | khớp nguồn |
| yolov8n-p2 - VAL mAP50-95 | 0,7635 | 2.325 ảnh VAL | [RESULTS.md:18](../../docs/bench_20261112/RESULTS.md#L18) | khớp nguồn |
| yolov8n-p2 - TEST recall / mAP50 | 0,8648 / 0,9306 | 2.326 ảnh TEST | [RESULTS.md:18](../../docs/bench_20261112/RESULTS.md#L18) | khớp nguồn |
| yolov8n-p2 - CI 95% delta TEST recall / mAP50 | [-0,0190; -0,0014] / [-0,0100; -0,0001] | 2.326 ảnh TEST ghép cặp | [RESULTS.md:18](../../docs/bench_20261112/RESULTS.md#L18) | khớp nguồn |
| current_finetune30 - VAL mAP50-95 | 0,7642 | 2.325 ảnh VAL | [RESULTS.md:19](../../docs/bench_20261112/RESULTS.md#L19) | khớp nguồn |
| current_finetune30 - TEST recall / mAP50 | 0,8805 / 0,9299 | 2.326 ảnh TEST | [RESULTS.md:19](../../docs/bench_20261112/RESULTS.md#L19) | khớp nguồn |
| current_finetune30 - CI 95% delta TEST recall / mAP50 | [-0,0014; +0,0119] / [-0,0093; -0,0024] | 2.326 ảnh TEST ghép cặp | [RESULTS.md:19](../../docs/bench_20261112/RESULTS.md#L19) | khớp nguồn |
| Quyết định detector | Giữ detector hiện tại. Cận dưới CI mAP50 của từng ứng viên đều dưới 0. | 6 ứng viên; 2.326 ảnh TEST | [RESULTS.md:23](../../docs/bench_20261112/RESULTS.md#L23) | khớp nguồn |

Số ảnh lấy từ `output/bench_20261112/eval_candidates/records_<model>_<split>.pkl`. Đã đối chiếu ID mẫu giữa các ứng viên. TEST có 4.888 box tham chiếu, VAL có 4.747 box tham chiếu, đếm từ `gts` trong các file đó. CI tính trên chênh lệch ứng viên trừ detector hiện tại theo [giao thức A2](../../docs/bench_20261112/RESULTS.md#L5).

## Recognizer

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| MobileNetV3-small - Accuracy TEST | 90,90% | 3.144 crop từ 1.740 ảnh | `output/demo_20261114/M3/test.json::buckets.all.mobilenet.accuracy.point` | khớp nguồn |
| EfficientNet-B0 - Accuracy TEST | 91,60% | 3.144 crop từ 1.740 ảnh | `output/demo_20261114/M3/test.json::buckets.all.efficientnet.accuracy.point` | khớp nguồn |
| EfficientNet-B0 trừ MobileNet - Accuracy | +0,70 điểm phần trăm, CI 95% [-0,13; +1,50] | 3.144 crop từ 1.740 ảnh | `output/demo_20261114/M3/test.json::buckets.all.delta.accuracy.point / ci95` | khớp nguồn |
| MobileNetV3-small - Macro-F1 TEST | 92,74% | 3.144 crop từ 1.740 ảnh | `output/demo_20261114/M3/test.json::buckets.all.mobilenet.macro_f1.point` | khớp nguồn |
| EfficientNet-B0 - Macro-F1 TEST | 92,97% | 3.144 crop từ 1.740 ảnh | `output/demo_20261114/M3/test.json::buckets.all.efficientnet.macro_f1.point` | khớp nguồn |
| EfficientNet-B0 trừ MobileNet - Macro-F1 | +0,23 điểm phần trăm, CI 95% [-0,89; +1,46] | 3.144 crop từ 1.740 ảnh | `output/demo_20261114/M3/test.json::buckets.all.delta.macro_f1.point / ci95` | khớp nguồn |
| Quyết định recognizer | Giữ MobileNetV3-small. CI accuracy chứa 0. | 3.144 crop từ 1.740 ảnh | `output/demo_20261114/M3/test.json::buckets.all.delta.accuracy.ci95` và [kết luận M3](RECOGNIZER_PROMOTION.md) | khớp nguồn |

## Thời gian xử lý offline

| Nội dung | 02/11 | M10 | M13/M12 trung vị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- |
| night_90s - elapsed 02/11 | 153,3 s |  |  | 1 lượt; 540 mẫu phân tích | `output/speed_simple_20261103/night_90s/summary.json::elapsed_seconds` | khớp nguồn |
| night_90s - elapsed M10 |  | 106,69 s |  | 1 lượt; 540 mẫu phân tích | `output/demo_20261114/M10/clips/night_90s/render/summary.json::elapsed_seconds` | khớp nguồn |
| night_90s - elapsed trung vị |  |  | 97,48 s | 3 lượt; 540 mẫu/lượt | `output/demo_20261114/M13/clips/night_90s/render/summary.json::elapsed_seconds`; `output/demo_20261114/M12/latency/night_90s_run2/render/summary.json::elapsed_seconds`; `output/demo_20261114/M12/latency/night_90s_run3/render/summary.json::elapsed_seconds` | khớp nguồn |
| night_90s - wall 02/11 | 157,4 s |  |  | 1 lượt; 540 mẫu phân tích | [SPEED.md:21](../../docs/final_report/evidence/SPEED.md#L21) | khớp nguồn |
| night_90s - wall M10 |  | 110,77 s |  | 1 lượt; 540 mẫu phân tích | `output/demo_20261114/M10/clips/night_90s/baseline_meta.json::wall_seconds` | khớp nguồn |
| night_90s - wall trung vị |  |  | 101,53 s | 3 lượt; 540 mẫu/lượt | `output/demo_20261114/M13/clips/night_90s/baseline_meta.json::wall_seconds`; `output/demo_20261114/M12/latency/night_90s_run2/baseline_meta.json::wall_seconds`; `output/demo_20261114/M12/latency/night_90s_run3/baseline_meta.json::wall_seconds` | khớp nguồn |
| demo_video - elapsed 02/11 | chưa đo |  |  | chưa đo | [SPEED.md:15](../../docs/final_report/evidence/SPEED.md#L15) | chưa đo |
| demo_video - elapsed M10 |  | 144,01 s |  | 1 lượt; 840 mẫu phân tích | `output/demo_20261114/M10/clips/demo_video/render/summary.json::elapsed_seconds` | khớp nguồn |
| demo_video - elapsed trung vị |  |  | 119,29 s | 3 lượt; 840 mẫu/lượt | `output/demo_20261114/M13/clips/demo_video/render/summary.json::elapsed_seconds`; `output/demo_20261114/M12/latency/demo_video_run2/render/summary.json::elapsed_seconds`; `output/demo_20261114/M12/latency/demo_video_run3/render/summary.json::elapsed_seconds` | khớp nguồn |
| demo_video - wall 02/11 | chưa đo |  |  | chưa đo | [SPEED.md:15](../../docs/final_report/evidence/SPEED.md#L15) | chưa đo |
| demo_video - wall M10 |  | 147,95 s |  | 1 lượt; 840 mẫu phân tích | `output/demo_20261114/M10/clips/demo_video/baseline_meta.json::wall_seconds` | khớp nguồn |
| demo_video - wall trung vị |  |  | 122,67 s | 3 lượt; 840 mẫu/lượt | `output/demo_20261114/M13/clips/demo_video/baseline_meta.json::wall_seconds`; `output/demo_20261114/M12/latency/demo_video_run2/baseline_meta.json::wall_seconds`; `output/demo_20261114/M12/latency/demo_video_run3/baseline_meta.json::wall_seconds` | khớp nguồn |
| demo_short_30s - elapsed 02/11 | chưa đo |  |  | chưa đo | [SPEED.md:15](../../docs/final_report/evidence/SPEED.md#L15) | chưa đo |
| demo_short_30s - elapsed M10 |  | 34,18 s |  | 1 lượt; 180 mẫu phân tích | `output/demo_20261114/M10/clips/demo_short_30s/render/summary.json::elapsed_seconds` | khớp nguồn |
| demo_short_30s - wall 02/11 | chưa đo |  |  | chưa đo | [SPEED.md:15](../../docs/final_report/evidence/SPEED.md#L15) | chưa đo |
| demo_short_30s - wall M10 |  | 38,29 s |  | 1 lượt; 180 mẫu phân tích | `output/demo_20261114/M10/clips/demo_short_30s/baseline_meta.json::wall_seconds` | khớp nguồn |
| demo_short_30s - trung vị elapsed / wall |  |  | chưa đo | 0 lượt trong nhóm đo lặp | `output/demo_20261114/M13/clips/` và `output/demo_20261114/M12/latency/` | chưa đo |

Trung vị dùng lượt M13 và hai lượt M12 (run2, run3) của từng clip. Số 02/11 và M10 là từng lượt render. Điều kiện máy khác nhau giữa lượt đơn M10 và nhóm đo M13/M12, nên chênh lệch thời gian chỉ mang tính mô tả. Hai lượt clip đêm cùng SHA256 nguồn, đối chiếu key `source_sha256` trong hai summary. `elapsed_seconds` là thời gian render ghi trong summary. `wall_seconds` là thời gian toàn lệnh bọc render trong `baseline_meta.json`. Latency end-to-end qua trình duyệt chưa đo.

## Lane trong prototype streaming

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| Prototype streaming - p50 khi lane tắt | 78,9 ms | 115 frame | [NUMBERS.md:77](../../docs/final_report/evidence/NUMBERS.md#L77) và `docs/final_report/evidence/t9_latency_original.json::configs.lane_visual_roi_false.summary.total.p50_ms / frames_measured` | khớp nguồn |
| Prototype streaming - p50 khi lane bật | 271,7 ms | 115 frame | [NUMBERS.md:78](../../docs/final_report/evidence/NUMBERS.md#L78) và `docs/final_report/evidence/t9_latency_original.json::configs.lane_visual_roi_true.summary.total.p50_ms / frames_measured` | khớp nguồn |

Khi trình bày, giữ bối cảnh prototype streaming của báo cáo cuối và đơn vị theo frame.

## Cảnh báo phía trước

**AI-reviewed, chưa đo recall.** Số câu/phút đếm cue được lên lịch trong render. Tỷ lệ lấy `hop_ly` trên toàn bộ cảnh báo đã phát, gồm cả `khong_ro`. Clip ngắn có mẫu số bằng 0 nên tỷ lệ không xác định.

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| M10 / night_90s - cue giọng cảnh báo | 4,0 câu/phút | 6 cue / 9 cảnh báo, 90 s | `output/demo_20261114/M10/clips/night_90s/render/summary.json::forward_warning.spoken_per_minute / spoken_cues / alerts; output_frames / output_fps` | khớp nguồn |
| M10 / demo_video - cue giọng cảnh báo | 3,0 câu/phút | 7 cue / 15 cảnh báo, 140 s | `output/demo_20261114/M10/clips/demo_video/render/summary.json::forward_warning.spoken_per_minute / spoken_cues / alerts; output_frames / output_fps` | khớp nguồn |
| M10 / demo_short_30s - cue giọng cảnh báo | 0,0 câu/phút | 0 cue / 0 cảnh báo, 30 s | `output/demo_20261114/M10/clips/demo_short_30s/render/summary.json::forward_warning.spoken_per_minute / spoken_cues / alerts; output_frames / output_fps` | khớp nguồn |

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| M11 v2 / night_90s - chu_y | 2/4 (50,0%) | 4 cảnh báo | `output/demo_20261114/M11/v2/counts.json::night_90s.chu_y` | khớp nguồn |
| M11 v2 / night_90s - nguy_co | 0/5 (0,0%) | 5 cảnh báo | `output/demo_20261114/M11/v2/counts.json::night_90s.nguy_co` | khớp nguồn |
| M11 v2 / night_90s - mọi mức | 2/9 (22,2%) | 9 cảnh báo | `output/demo_20261114/M11/v2/counts.json::night_90s.all` | khớp nguồn |
| M11 v2 / demo_video - chu_y | 6/14 (42,9%) | 14 cảnh báo | `output/demo_20261114/M11/v2/counts.json::demo_video.chu_y` | khớp nguồn |
| M11 v2 / demo_video - nguy_co | 0/1 (0,0%) | 1 cảnh báo | `output/demo_20261114/M11/v2/counts.json::demo_video.nguy_co` | khớp nguồn |
| M11 v2 / demo_video - mọi mức | 6/15 (40,0%) | 15 cảnh báo | `output/demo_20261114/M11/v2/counts.json::demo_video.all` | khớp nguồn |
| M11 v2 / demo_short_30s - chu_y | Không xác định | 0 cảnh báo | `output/demo_20261114/M11/v2/counts.json::demo_short_30s.chu_y` | khớp nguồn |
| M11 v2 / demo_short_30s - nguy_co | Không xác định | 0 cảnh báo | `output/demo_20261114/M11/v2/counts.json::demo_short_30s.nguy_co` | khớp nguồn |
| M11 v2 / demo_short_30s - mọi mức | Không xác định | 0 cảnh báo | `output/demo_20261114/M11/v2/counts.json::demo_short_30s.all` | khớp nguồn |
| M11 v2 / gộp - chu_y | 8/18 (44,4%) | 18 cảnh báo | `output/demo_20261114/M11/v2/counts.json::all.chu_y` | khớp nguồn |
| M11 v2 / gộp - nguy_co | 0/6 (0,0%) | 6 cảnh báo | `output/demo_20261114/M11/v2/counts.json::all.nguy_co` | khớp nguồn |
| M11 v2 / gộp - mọi mức | 8/24 (33,3%) | 24 cảnh báo | `output/demo_20261114/M11/v2/counts.json::all.all` | khớp nguồn |

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| M13 / night_90s (dev) - cue giọng cảnh báo | 2,00 câu/phút | 3 cue / 5 cảnh báo, 90 s | `output/demo_20261114/M13/clips/night_90s/render/summary.json::forward_warning.spoken_per_minute / spoken_cues / alerts; output_frames / output_fps` | khớp nguồn |
| M11 v3 / night_90s (dev) - chu_y | 2/4 (50,0%) | 4 cảnh báo | `output/demo_20261114/M11/v3/counts.json::night_90s.chu_y` | khớp nguồn |
| M11 v3 / night_90s (dev) - nguy_co | 0/1 (0,0%) | 1 cảnh báo | `output/demo_20261114/M11/v3/counts.json::night_90s.nguy_co` | khớp nguồn |
| M11 v3 / night_90s (dev) - mọi mức | 2/5 (40,0%) | 5 cảnh báo | `output/demo_20261114/M11/v3/counts.json::night_90s.all` | khớp nguồn |
| M13 / demo_video (holdout) - cue giọng cảnh báo | 2,57 câu/phút | 6 cue / 10 cảnh báo, 140 s | `output/demo_20261114/M13/clips/demo_video/render/summary.json::forward_warning.spoken_per_minute / spoken_cues / alerts; output_frames / output_fps` | khớp nguồn |
| M11 v3 / demo_video (holdout) - chu_y | 5/10 (50,0%) | 10 cảnh báo | `output/demo_20261114/M11/v3/counts.json::demo_video.chu_y` | khớp nguồn |
| M11 v3 / demo_video (holdout) - nguy_co | 0/0 (không xác định) | 0 cảnh báo | `output/demo_20261114/M11/v3/counts.json::demo_video.nguy_co` | khớp nguồn |
| M11 v3 / demo_video (holdout) - mọi mức | 5/10 (50,0%) | 10 cảnh báo | `output/demo_20261114/M11/v3/counts.json::demo_video.all` | khớp nguồn |
| M11 v2 sang v3 / demo_video (holdout) | Tổng cảnh báo 15 xuống 10, `khong_hop_ly` 8 xuống 5, `hop_ly` 6 xuống 5. V3 không có mẫu `nguy_co`. | 15 ca v2, 10 ca v3 | `output/demo_20261114/M11/v2/counts.json::demo_video.all`; `output/demo_20261114/M11/v3/counts.json::demo_video.all / nguy_co` | khớp nguồn |

Kết luận so sánh v2 và v3 dựa trên video demo holdout. Clip đêm là dev đã dùng để chỉnh ngưỡng. Kết quả v3 là **AI-reviewed, chưa đo recall**. [Rà soát M11 v3](FORWARD_REVIEW.md#v3---render-m13).

### M17 - Hai clip ban ngày và tổng ba holdout

**AI-reviewed, chưa đo recall.** Tổng ba holdout gồm demo_video v3 và hai clip M17. Cả 3 ca `nguy_co` trong nhóm này đến từ clip Hà Nội.

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| M17 / day_hanoi_many_signs - cue giọng cảnh báo | 2,67 câu/phút | 4 cue / 5 cảnh báo, 90 s | `output/demo_20261114/M17/clips/day_hanoi_many_signs/render/summary.json::forward_warning.spoken_per_minute / spoken_cues / alerts` | khớp nguồn |
| M17 / day_hanoi_many_signs - chu_y | 1/2 (50,0%) | 2 cảnh báo | `output/demo_20261114/M11/M17/counts.json::day_hanoi_many_signs.chu_y` | khớp nguồn |
| M17 / day_hanoi_many_signs - nguy_co | 1/3 (33,3%) | 3 cảnh báo | `output/demo_20261114/M11/M17/counts.json::day_hanoi_many_signs.nguy_co` | khớp nguồn |
| M17 / day_hanoi_many_signs - mọi mức | 2/5 (40,0%) | 5 cảnh báo | `output/demo_20261114/M11/M17/counts.json::day_hanoi_many_signs.all` | khớp nguồn |
| M17 / day_west_lake - cue giọng cảnh báo | 2,67 câu/phút | 4 cue / 11 cảnh báo, 90 s | `output/demo_20261114/M17/clips/day_west_lake/render/summary.json::forward_warning.spoken_per_minute / spoken_cues / alerts` | khớp nguồn |
| M17 / day_west_lake - chu_y | 6/11 (54,5%) | 11 cảnh báo | `output/demo_20261114/M11/M17/counts.json::day_west_lake.chu_y` | khớp nguồn |
| M17 / day_west_lake - nguy_co | 0/0 (Không xác định) | 0 cảnh báo | `output/demo_20261114/M11/M17/counts.json::day_west_lake.nguy_co` | khớp nguồn |
| M17 / day_west_lake - mọi mức | 6/11 (54,5%) | 11 cảnh báo | `output/demo_20261114/M11/M17/counts.json::day_west_lake.all` | khớp nguồn |
| M17 / gộp hai clip - mọi mức | 8/16 (50,0%), `khong_hop_ly` 4, `khong_ro` 4 | 16 cảnh báo | `output/demo_20261114/M11/M17/counts.json::m17.all` | khớp nguồn |
| Gộp ba holdout - mọi mức | 13/26 (50,0%), `khong_hop_ly` 9, `khong_ro` 4 | 26 cảnh báo | `output/demo_20261114/M11/M17/counts.json::three_holdouts.all` | khớp nguồn |

[Rà soát M17](FORWARD_REVIEW.md#m17---holdout-mới-2-clip-đông-xe-ban-ngày).

## Biển báo trên video demo M15

**AI-reviewed, không phải nhãn người.** Chỉ một video demo, chưa đo recall. Nhóm uncertain dùng mã ứng viên thô để đối chiếu.

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| Known - đúng mã | 27/28 (96,4%) | 28 nhóm | `output/demo_20261114/M15/scored/counts.json::verdicts.known.dung_ma / known_n` | khớp nguồn |
| Uncertain - phân bố | `dung_ma` 11, `sai_ma` 1, `khong_phai_bien` 3, `khong_ro` 14 | 29 nhóm | `output/demo_20261114/M15/scored/counts.json::verdicts.uncertain / uncertain_n` | khớp nguồn |
| Uncertain - cặp nhầm mã thô | I.409 sang I.423, 1 ca | 29 nhóm | `output/demo_20261114/M15/scored/counts.json::confusion_pairs` | khớp nguồn |

[Phương pháp và phiếu M15](SIGN_REVIEW.md)

## Không đưa lên slide

| Nội dung | Giá trị | n | File nguồn và dòng hoặc key | Trạng thái |
| --- | --- | --- | --- | --- |
| A5 one-stage - ảnh TRAIN bị loader loại | 10.587/10.952 (96,7%) | 10.952 ảnh TRAIN | `output/demo_20261114/M7/a5_audit.json::splits.train.rejected_images / scanned_images` | khớp nguồn |

A5 **không hợp lệ**. Mã xuất nhãn chuẩn hóa theo kích thước box, làm tọa độ vượt biên và loader bỏ phần lớn ảnh. Nguồn [audit A5](MODEL_REVIEW.md#a5---detector-one-stage-88-lớp).

### Câu không được dùng trên slide

- "real-time". Báo thời gian render và thời lượng clip theo bảng.
- "bỏ giọng biển để giảm latency xử lý". Render không chạy TTS. TTS chạy khi phát theo Mục 9 của `D:/DSP391m/PLAN_DEMO_20261114.md`.
- Bất kỳ tỷ lệ phần trăm nào thiếu số ca hoặc mẫu số tương ứng.
