# Kết quả thử nghiệm Lane Preprocessing (Hướng B)

**Ngày:** 02/11/2026 · **Nhánh:** `experiment/lane-preprocessing-20261102` · **Nguồn:** `f6bdf8f51c0171d480d17bb786a01aa56df6c86f`
**Kết luận:** **FAIL ở giai đoạn 1 (lọc nhanh theo khung).** Không biến thể nào đạt tiêu chí vào giai đoạn 3. `preprocessing chưa cải thiện đủ trên bộ clip thử`. Giữ `lane_preprocess: none`.

Đây là đánh giá trực quan của AI, không phải ground truth người (`human_gt: false`). Số mẫu có lane không phải recall hay accuracy.

## Trạng thái từng bước

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 0 | Nhánh từ `f6bdf8f`, chép PLAN.md | PASS (commit `de46017`) |
| 1 | Code + config + test, mặc định `none` | PASS (commit `7624e7f`) |
| 2 | Khóa 30 mốc + 2 khung chẩn đoán, hash 2 video (`frames.json`) | PASS |
| 3 | Lọc nhanh 5 biến thể, chấm ĐÚNG/SAI/KHÔNG CHẮC | FAIL, không biến thể đạt |
| 4 | Render đầy đủ, hồi quy 900 mẫu, chấm temporal | NOT RUN (chỉ chạy cho biến thể đạt) |
| 5 | Kiểm tra crop 40% trên | NOT RUN (dừng ở cổng bước 3 theo plan) |
| 6 | Evidence và commit | PASS (commit này) |

## Bước 3: bảng lọc nhanh

Cách chạy: UFLD-v2 checkpoint hiện có, `display_row_ratio: 0.5`, không temporal, một khung mỗi lần. Polylines vẽ lên **ảnh gốc** rồi ghép 5 cột. Chấm lại A0 theo cùng cách (không dùng số 3/9 của bản render có temporal).

Đếm trên 15 khung ngày và 15 khung đêm (ĐÚNG / SAI / KHÔNG CHẮC):

| Biến thể | Ngày (60 s) | Đêm (90 s) | Tổng 30 khung |
|---|---|---|---|
| A0 none | 0 / 1 / 0 | 1 / 9 / 3 | **1 / 10 / 3** |
| A1a gamma 0.8 | 0 / 1 / 0 | 1 / 9 / 3 | 1 / 10 / 3 |
| A1b gamma 0.6 | 0 / 1 / 0 | 1 / 10 / 3 | 1 / 11 / 3 |
| A2a CLAHE clip 2.0, tile 8 | 0 / 0 / 0 | 1 / 7 / 3 | 1 / 7 / 3 |
| A2b CLAHE clip 3.0, tile 8 | 0 / 1 / 0 | 1 / 7 / 3 | 1 / 8 / 3 |

Khung chi tiết (số đường hiển thị sau bộ lọc candidate; bảng đầy đủ trong `evidence/stage1_counts.json`):

| Clip, giây | A0 | A1a | A1b | A2a | A2b | Ghi chú |
|---|---|---|---|---|---|---|
| Ngày 42 | 1 (S) | 1 (S) | 1 (S) | 0 | 1 (S) | Đường bên lề, không phải vạch |
| Đêm 15 | 4 | 4 | 4 | 4 | 4 | A2 đổi đường ngang phẳng thành zigzag |
| Đêm 21 | 3 | 3 | 3 | 3 | 3 | Zigzag qua người đi xe |
| Đêm 27 | 3 | 3 | 3 | 3 | 3 | Có 1 ĐÚNG ở mọi biến thể (vạch trắng phải) |
| Đêm 51 | 1 | 1 | 2 | 0 | 0 | A1b thêm 1 đoạn chéo qua đường |
| Đêm 69 | 2 | 2 | 2 | 1 | 1 | Đường đi qua xe đỏ và xe máy |
| Ngày 55 (chẩn đoán) | 0 | 0 | 0 | 0 | 0 | Không đạt ngưỡng hiển thị ở mọi biến thể |
| Ngày 59 (chẩn đoán) | 0 | 0 | 0 | 0 | 0 | Không đạt ngưỡng hiển thị ở mọi biến thể |

Ảnh ghép (mỗi hàng một khung, 5 cột A0 | A1a | A1b | A2a | A2b):
- `docs/experiments/lane_preprocessing_20261102/screen_day.jpg` (17 hàng: 15 khung + 2 chẩn đoán)
- `docs/experiments/lane_preprocessing_20261102/screen_night.jpg` (15 hàng)

**Cổng bước 3 (plan mục 3):** phải đồng thời (1) ĐÚNG tăng so với A0 và có ít nhất 1 ĐÚNG ở clip ngày, (2) SAI không tăng.
- ĐÚNG ở clip ngày: **0 ở mọi biến thể**, không đạt điều kiện (1).
- ĐÚNG tổng: không biến thể nào tăng so với A0 (1 → 1).
- SAI: A1b tăng (10 → 11), vi phạm điều kiện (2).

**Kết quả:** không biến thể nào đạt. Theo plan, dừng ở đây, không render (bước 4) và không kiểm tra crop (bước 5).

### Lưu ý về độ tin cậy của bảng

- Số A0 (1/10/3) khác số 3/9 của báo cáo LANE_RELAX. Lý do: bảng này không có temporal, chấm trên ô 448 px thay vì ảnh 1920×1080, và cách chấm vài đường đêm khác. Đây là sai lệch của người chấm (AI), không phải sai lệch của model. Vì vậy so sánh chỉ có ý nghĩa **giữa các biến thể trong cùng bảng**.
- Kết luận FAIL không phụ thuộc vào chi tiết chấm đêm. Điều kiện ĐÚNG ở clip ngày = 0 ở mọi biến thể là kết quả rõ ràng: các đường ngày 42 s đều nằm trên lề, không nằm trên vạch.
- Với 1 khung ngày được chấm, A1a và A1b cho đúng cùng một đường như A0 (lề trái); A2b thay bằng đường lề phải. Không biến thể nào đưa đường lên vạch sơn trắng/vàng ở khung ngày.

## Bước 4 và 5

NOT RUN. Không có biến thể đạt cổng bước 3, nên không render 2 clip, không chạy hồi quy 360 ngày + 540 đêm, không đo thời gian xử lý, không kiểm tra crop. Không có số liệu thời gian xử lý cho bước này.

## Hồi quy, decode và test

- Hồi quy sign / context / speech / camera_view: **NOT RUN** (không có biến thể nào được render).
- Test đã chạy trên commit `7624e7f` (code + config + test):
  - `pytest -q tests`: **623 passed, 11 skipped, 38 deselected**, 0 failed. Trong đó có 10 test mới ở `tests/unit/test_lane_preprocess_20261102.py` (none giữ nguyên ảnh, gamma/CLAHE giữ shape/dtype/range, gamma<1 làm sáng, lỗi đầu vào, lượt UFLD thứ hai chỉ dùng cho hiển thị và camera_view lấy từ ảnh gốc, `none` chỉ một lượt, precision không chạy preprocess).
  - `node --test frontend/src/*.test.js`: **14 passed**, 0 failed.
- Số test này là thật, chạy trong môi trường `D:/miniconda/envs/itas-mm` với `PYTHONNOUSERSITE=1`.

## Input / output (SHA-256)

| Mục | Giá trị |
|---|---|
| `output/lane_relax_20261102/frozen_samples.json` | `31d136e6b6c13e301471c0b70500b77a6d0ecb2507d8752374828332bdee7cf9` |
| `docs/demo/itas-mm_demo_video.mp4` (ngày, 60 s, 30 fps, 4200 khung) | `21b5f2889e5ac4bc6b6e13bb8c4ff01f1feeed084b05c90760b8c22e1926ff4a` |
| `output/final_20261102/demo_clips/night_hanoi_YQl9c5gCjhg_30s_90s.mp4` (đêm, 90 s, 60 fps, 5400 khung) | `c22caa280981dbba4b738e6296a5d70550bce227143e89c3c57a2ced96791b21` |
| `docs/experiments/lane_preprocessing_20261102/screen_day.jpg` | `ee3f6e5b750a97ca2d047f2662082f97395679d2af4825c2334656a31ac0e56f` |
| `docs/experiments/lane_preprocessing_20261102/screen_night.jpg` | `c95c939d0148f5fbfc4f312b4ccf10fa06863b5324fa7599d6023e1714428192` |

Bản gốc chưa nén (không commit, trong `output/`): `screen_day.jpg` `497e4e07…`, `screen_night.jpg` `64376dcd…`.

## Hash commit

| Commit | Nội dung |
|---|---|
| `f6bdf8f` | Nguồn (nhánh `fix/lane-simplified-demo-20261102`) |
| `de46017` | Add lane preprocessing experiment plan |
| `7624e7f` | code + config + test |
| (commit này) | experiment results |

## Tệp trong thư mục này

- `PLAN.md`: plan gốc, nội dung không đổi.
- `frames.json`: danh sách 30 mốc + 2 chẩn đoán, fps, số khung, SHA-256 của 2 video (ghi trước khi chạy biến thể).
- `evidence/stage1_counts.json`: số đường và số hàng tồn tại theo từng slot UFLD cho mỗi khung × biến thể (160 bản ghi).
- `evidence/stage1_grades.json`: chấm ĐÚNG/SAI/KHÔNG CHẮC theo từng đường và tổng hợp.
- `tools/make_frames.py`, `tools/screen_frames.py`, `tools/grade_stage1.py`: script tái lập.

## Quyết định

- Giữ `offline_video_lane.lane_preprocess: none` trong `configs/perception.yaml`. Không bật biến thể nào.
- Không đề xuất merge preprocessing. Code trong commit `7624e7f` là tùy chọn tắt mặc định; giữ lại làm nền cho lần thử sau, hoặc bỏ nếu người phụ trách quyết định không cần.
- Không tuyên bố cải thiện accuracy hay tốc độ. Không có bằng chứng nào trong bảng này cho thấy biến thể nào tốt hơn A0.
- Lần thử tiếp theo nếu có: chấm độc lập bởi người (có ground truth) và chấm trên ảnh độ phân giải gốc, không dùng ô 448 px.
