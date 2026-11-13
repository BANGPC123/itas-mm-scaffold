# M14 REPRO: tab Ảnh, photo_61s.jpg, khởi động bằng start.ps1

## Kết luận

- Tab Ảnh không nhận dạng được biển trên `photo_61s.jpg` **không phải do thiếu `ITAS_WARMUP=1`**. Cả 3 lần không warmup và 3 lần có warmup đều ra đúng 1 biển, mã `unknown`, `recognition_source = crop_too_small`.
- Nguyên nhân có bằng chứng: detector bắt được 1 hộp 15.6 x 30 px (`bbox_xyxy` [749.6, 461.0, 765.2, 491.0]). Cạnh ngắn 15.6 px nhỏ hơn `sign_recognizer.min_short_side: 32` (`configs/perception.yaml:86`), nên recognizer bỏ qua hộp này và gán `unknown`. Video dùng ngưỡng riêng `video_min_short_side: 12` (`configs/perception.yaml:87`), vì vậy cùng một biển nhỏ có thể được nhận trên video mà không nhận trên ảnh.
- Số "signRows 11" của M10 là **thước đo sai**, không phải kết quả khác. Xem mục 1.
- Warmup **không đổi kết quả** nhưng đổi thời gian lần gọi đầu: lần đầu không warmup mất 33.3 s, có warmup 0.1 s. Đây là rủi ro cho buổi demo, không phải lỗi nhận dạng.
- Không đổi model, ngưỡng, hay code. `start.ps1` không được sửa (điều kiện "lần đầu sai, lần sau đúng" của M14.3 không xảy ra).

## 1. Thước đo signRows của M10

- M10 đếm `document.querySelectorAll('.sign-list .sign-row')` (`output/demo_20261114/M10/e2e/e2e_cdp_20261114.mjs:107`). Selector này khớp cả `VideoTab.jsx:256` và `VideoTab.jsx:283` lẫn `ImageTab.jsx:145`.
- `frontend/src/App.jsx:79-80` giữ cả hai tab trong DOM (`hidden`), nên danh sách biển của tab Video vẫn có mặt khi M10 chuyển sang tab Ảnh. Số 11 gộp hàng của tab Video và tab Ảnh.
- Bằng chứng trực tiếp: `output/demo_20261114/M10/screens/08_image_result.png` (tab Ảnh, chụp trong cùng lượt M10) hiển thị **"0 nhận dạng · 1 phát hiện"** và một hàng "Chưa nhận dạng". M10 cũng ra 1 biển trên ảnh này.
- Thước đo đúng cho M14: số phần tử `perception.signs` trong response của `POST /pipeline/run`. Mỗi phần tử tương ứng một hàng `.sign-row` của tab Ảnh (`frontend/src/logic.js:62`, `imageRows`).

## 2. Tái hiện (start.ps1, không sửa, gọi `POST /pipeline/run` 3 lần liên tiếp)

Ảnh: `output/demo_20261114/M12/e2e_final/photo_61s.jpg`, SHA1 `ebd86eb38cfa7f2f207428ecfe036f0b5bf92975`, giống M10. Form gửi giống `frontend/src/ImageTab.jsx:37-42` (`latitude 10.7769`, `longitude 106.7009`, `synthesize_audio=false`, `annotate=true`, `auxiliary_sign_review=false`). Script: `output/demo_20261114/M14/repro_start_ps1.ps1`.

| Lượt | ITAS_WARMUP | HTTP | Thời gian curl | `timings_ms.total` | Số biển | Mã | recognition_source |
|---|---|---|---:|---:|---:|---|---|
| cold 1 | không đặt | 200 | 33.3 s | 1109 ms | 1 | unknown | crop_too_small |
| cold 2 | không đặt | 200 | < 0.05 s | 13 ms | 1 | unknown | crop_too_small |
| cold 3 | không đặt | 200 | < 0.05 s | 16 ms | 1 | unknown | crop_too_small |
| warm 1 | = 1 | 200 | 0.1 s | 71 ms | 1 | unknown | crop_too_small |
| warm 2 | = 1 | 200 | 0.1 s | 38 ms | 1 | unknown | crop_too_small |
| warm 3 | = 1 | 200 | 0.1 s | 38 ms | 1 | unknown | crop_too_small |

Nguồn: `M14/cold/runs.txt`, `M14/warm/runs.txt`, `M14/*/run{1,2,3}.json`. Trong cả 6 lượt, `confidence = 0.378`, `recognition_confidence = null`.

Ghi chú: "< 0.05 s" là do script làm tròn một chữ số, nên hiển thị 0.

## 3. Thời gian khởi động (start.ps1 đến khi /health trả 200)

| Chế độ | Khởi động | Ghi chú |
|---|---:|---|
| không warmup | 3.2 s | `M14/cold/startup.txt` |
| ITAS_WARMUP=1 | 10.7 s | `M14/warm/startup.txt`; `output/product_startup_status.json` ghi `startup_seconds 6.25` (chỉ phần warmup trong lifespan) |

Với warmup, phần khởi động tăng khoảng 7.5 s, và lần gọi đầu đã nằm trong khoảng đó.

## 4. Lần gọi đầu 33 s khi không warmup

- Pipeline đo được 1109 ms ở lần đầu (chủ yếu `detection` 1047 ms, khởi tạo CUDA), còn `curl` đo 33.3 s. Khoảng **32 s nằm ngoài các bước được pipeline đo**, trên đường xử lý request.
- Chưa xác định chính xác đoạn 32 s đó (nghi là import/khởi tạo lười ở lần gọi đầu). Chưa có bằng chứng về nguyên nhân cụ thể; chỉ biết ITAS_WARMUP=1 loại bỏ nó trong cả 3 lượt có warmup.
- Hệ quả cho demo: nếu không warmup, lần upload đầu tiên trên tab Ảnh có thể đứng khoảng 33 s.

## 5. E2E tab Ảnh

- Lần E2E này dùng cùng ảnh và cùng form mà UI gửi, qua API. Chưa chạy lại trình duyệt trong M14.
- E2E trước đó trên `start.ps1` (M12, `output/demo_20261114/M12/e2e_final/shots/08_image_result.png`) cho cùng kết quả: 0 nhận dạng, 1 phát hiện. Bước tra cứu luật trên tab Ảnh không có biển để bấm.
- Phương án trong DEMO_SCRIPT (tra cứu luật trên tab Video, mục P.102) vẫn đúng.

## 6. Thay đổi

- Không có thay đổi code, config, model, hay ngưỡng.
- `start.ps1` không đổi. Đề xuất (chưa làm, cần chủ dự án quyết): thêm `$env:ITAS_WARMUP = '1'` để lần upload đầu không bị chậm 33 s, đánh đổi là khởi động lâu hơn khoảng 7.5 s.
- Đề xuất riêng cho biển nhỏ trên ảnh (`min_short_side` 32 so với `video_min_short_side` 12): đây là ngưỡng, không đổi trong vòng đóng băng.

## Giới hạn

- Chỉ 1 ảnh, 1 lần khởi động cho mỗi chế độ (3 lượt liên tiếp mỗi chế độ). Không đo recall.
- Không xác nhận được loa.
- Đoạn 32 s ở lần gọi đầu chưa được định vị.
