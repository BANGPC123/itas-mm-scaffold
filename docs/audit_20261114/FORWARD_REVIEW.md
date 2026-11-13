# M11 - Rà soát cảnh báo vật cản

## Luật chấm cố định trước khi xem ảnh

- `hop_ly`: nhận đúng lớp và đúng box, vật nằm trên làn xe mình đi hoặc có dấu hiệu đang cắt vào làn, và đủ gần theo vị trí chân hoặc điểm tiếp xúc mặt đường ở phần dưới khung. Xét phối cảnh, vạch làn hoặc quỹ đạo xe mình và đối chiếu ảnh trước đó.
- `khong_hop_ly`: có bằng chứng rõ cho ít nhất một lỗi. Ví dụ xe đỗ lề, xe ngược chiều hoặc cùng chiều ở làn khác và không cắt vào, vật ở xa, sai lớp hoặc box sai vật.
- `khong_ro`: hai ảnh thiếu thông tin để quyết định, chẳng hạn ánh sáng kém, che khuất, không phân biệt được làn hay hướng chuyển động.
- Dùng cùng luật cho cả hai mức. Chấm từ ảnh sự kiện và ảnh trước 0,5 s theo Mục 8 của `D:/DSP391m/PLAN_DEMO_20261114.md`. Dùng TTC proxy để đối chiếu metadata. Tỷ lệ hợp lý lấy số `hop_ly` chia cho toàn bộ cảnh báo trong nhóm, kể cả `khong_ro`. Mỗi cảnh báo giữ một dòng. Cột `owner_verdict` dành cho chủ dự án.
- AI-reviewed, không phải nhãn người. Kết quả đo độ hợp lý trên các cảnh báo đã phát trong các clip này. Phép rà soát chỉ lấy mẫu các sự kiện đã phát nên chưa đo cảnh báo bị bỏ sót.

## v1 - Render M5

Nguồn: [phiếu clip đêm](../../output/demo_20261114/M11/v1/night_90s/review_sheet.csv), [phiếu video demo](../../output/demo_20261114/M11/v1/demo_video/review_sheet.csv), [bảng đếm](../../output/demo_20261114/M11/v1/counts.json). Mỗi ảnh sự kiện có khung xanh lam đánh dấu bbox từ metadata, đã quy đổi theo kích thước ảnh đầu ra. Ảnh trước đó giữ nguyên lớp phủ có trong video.

| Clip | Mức | Hợp lý / tổng | Tỷ lệ | Không hợp lý | Không rõ |
| --- | --- | ---: | ---: | ---: | ---: |
| night_90s | chu_y | 3/6 | 50,0% | 2 | 1 |
| night_90s | nguy_co | 0/2 | 0,0% | 2 | 0 |
| night_90s | Mọi mức | 3/8 | 37,5% | 4 | 1 |
| demo_video | chu_y | 4/16 | 25,0% | 10 | 2 |
| demo_video | nguy_co | 0/7 | 0,0% | 7 | 0 |
| demo_video | Mọi mức | 4/23 | 17,4% | 17 | 2 |
| Gộp hai clip | chu_y | 7/22 | 31,8% | 12 | 3 |
| Gộp hai clip | nguy_co | 0/9 | 0,0% | 9 | 0 |
| Gộp hai clip | Mọi mức | 7/31 | 22,6% | 21 | 3 |

### Ca không hợp lý tiêu biểu

- Clip đêm, 61,833 s, track 735, nguy_co. Box theo người lái xe máy, trong khi cảnh báo ghi người đi bộ. Nguồn dòng 002 của phiếu clip đêm.
- Video demo, 29,000 đến 29,833 s, track 166, 202 và 207, nguy_co. Cả ba box nằm trong dãy xe đỗ lề phải. Nguồn dòng 002 đến 004 của phiếu video demo.
- Video demo, 117,833 s, track 390, chu_y. Ô tô tiếp tục ở bên kia vạch làn trái. Nguồn dòng 020 của phiếu video demo.

### Lớp phủ trong render

Một số box cảnh báo gốc của clip đêm lệch xuống phải so với box trong metadata đã quy đổi. Ví dụ dòng 004, ảnh sự kiện cho thấy box cam gốc ở phần đường bên phải, còn khung xanh quy đổi bao xe bán tải ở giữa. Phiếu đánh giá vị trí vật theo bbox metadata. Nguồn ảnh 004_event.jpg và sources.json của clip đêm.

## v2 - Render M10

Nguồn: [phiếu clip đêm](../../output/demo_20261114/M11/v2/night_90s/review_sheet.csv), [phiếu video demo](../../output/demo_20261114/M11/v2/demo_video/review_sheet.csv), [phiếu clip ngắn](../../output/demo_20261114/M11/v2/demo_short_30s/review_sheet.csv), [bảng đếm](../../output/demo_20261114/M11/v2/counts.json). Dùng luật chấm đã ghi ở đầu báo cáo. Clip ngắn không có cảnh báo trong cả summary và predictions. Vì mẫu số bằng 0, tỷ lệ của clip ngắn không xác định.

| Clip | Mức | Hợp lý / tổng | Tỷ lệ | Không hợp lý | Không rõ |
| --- | --- | ---: | ---: | ---: | ---: |
| night_90s | chu_y | 2/4 | 50,0% | 1 | 1 |
| night_90s | nguy_co | 0/5 | 0,0% | 5 | 0 |
| night_90s | Mọi mức | 2/9 | 22,2% | 6 | 1 |
| demo_video | chu_y | 6/14 | 42,9% | 7 | 1 |
| demo_video | nguy_co | 0/1 | 0,0% | 1 | 0 |
| demo_video | Mọi mức | 6/15 | 40,0% | 8 | 1 |
| demo_short_30s | chu_y | 0/0 | không xác định | 0 | 0 |
| demo_short_30s | nguy_co | 0/0 | không xác định | 0 | 0 |
| demo_short_30s | Mọi mức | 0/0 | không xác định | 0 | 0 |
| Gộp ba clip | chu_y | 8/18 | 44,4% | 8 | 2 |
| Gộp ba clip | nguy_co | 0/6 | 0,0% | 6 | 0 |
| Gộp ba clip | Mọi mức | 8/24 | 33,3% | 14 | 2 |

### Ca không hợp lý tiêu biểu

- Clip đêm, 21,667 s, track 110, nguy_co. Box bám người ngồi lái xe máy, nhưng cảnh báo ghi người đi bộ. Nguồn dòng 001 của phiếu clip đêm.
- Clip đêm, 63,333 s, track 615, nguy_co. Bbox quy đổi chỉ bao phần dưới đuôi SUV đỏ ở luồng trái. Nguồn dòng 003 và ảnh 003_event.jpg của clip đêm.
- Video demo, 60,667 s, track 281, nguy_co. Ô tô bạc ở sát bó vỉa, phía bên kia vạch làn. Nguồn dòng 006 của phiếu video demo.

Box gốc của clip đêm v2 vẫn lệch xuống phải ở các ca như track 342 tại 71,000 s. Khung xanh quy đổi đánh dấu xe bán tải phía trước. Nguồn ảnh 005_event.jpg và sources.json của clip đêm v2.

Tỷ lệ gộp v2 cao hơn v1 trên cảnh báo đã phát. Số sự kiện và mức cảnh báo giữa hai render thay đổi. Phép rà soát này chưa đo recall và chưa có nhãn người. Nguồn hai bảng đếm.

### Đối chiếu dữ liệu render

Lần trích v2 đầu tiên dừng khi so toàn bộ trường JSON vì summary M10 thêm hai trường giọng nói sau khi ghi predictions. Script bỏ qua đúng hai trường đó khi so, rồi kiểm tra toàn bộ metadata cảnh báo còn lại và số dòng. Nguồn v2-extract.log, v2-extract-verified.log và sources.json từng clip.

## v3 - Render M13

**Kết luận trên holdout demo_video:** v3 có 5/10 (50,0%) cảnh báo hợp lý, tất cả ở mức `chu_y`. Tổng số cảnh báo giảm từ 15 ở v2 xuống 10 ở v3. Số ca hợp lý giảm từ 6 xuống 5 dù tỷ lệ tăng từ 40,0% lên 50,0%. V3 không phát `nguy_co` trên holdout, nên chưa có mẫu để đánh giá mức này. Nguồn [đếm v2](../../output/demo_20261114/M11/v2/counts.json) và [đếm v3](../../output/demo_20261114/M11/v3/counts.json).

**Clip đêm là dev**, đã dùng để chỉnh ngưỡng theo `output/demo_20261114/M13/THRESHOLDS.md` trong repo chính. Kết quả v3 của clip đêm là 2/5 (40,0%) hợp lý, trong đó ca `nguy_co` duy nhất tại 60,667 s, track 342, gán người lái xe máy thành người đi bộ. Nguồn [phiếu dev](../../output/demo_20261114/M11/v3/night_90s/review_sheet.csv).

### So sánh v1, v2, v3 trên hai clip

| Bản | Clip | Vai trò | chu_y hợp lý | nguy_co hợp lý | Mọi mức hợp lý |
| --- | --- | --- | --- | --- | --- |
| v1 | night_90s | dev | 3/6 (50,0%) | 0/2 (0,0%) | 3/8 (37,5%) |
| v1 | demo_video | holdout | 4/16 (25,0%) | 0/7 (0,0%) | 4/23 (17,4%) |
| v2 | night_90s | dev | 2/4 (50,0%) | 0/5 (0,0%) | 2/9 (22,2%) |
| v2 | demo_video | holdout | 6/14 (42,9%) | 0/1 (0,0%) | 6/15 (40,0%) |
| v3 | night_90s | dev | 2/4 (50,0%) | 0/1 (0,0%) | 2/5 (40,0%) |
| v3 | demo_video | holdout | 5/10 (50,0%) | 0/0 (không xác định) | 5/10 (50,0%) |

Nguồn [v1](../../output/demo_20261114/M11/v1/counts.json), [v2](../../output/demo_20261114/M11/v2/counts.json), [v3](../../output/demo_20261114/M11/v3/counts.json). AI-reviewed, chưa đo recall. Tỷ lệ dùng toàn bộ cảnh báo đã phát, gồm cả ca không rõ.

### Box vẽ trên video

- **night_90s, dev:** 5/5 box khớp bbox metadata sau khi quy đổi từ nguồn 1920x1080 sang canvas 1280x720. [Đối chiếu cạnh](../../output/demo_20261114/M11/v3/night_90s/overlay_evidence.json) · [Ví dụ frame gốc](../../output/demo_20261114/M11/v3/night_90s/002_native.jpg)
- **demo_video, holdout:** 10/10 box khớp. Nguồn và canvas cùng kích thước 1280x720. [Đối chiếu cạnh](../../output/demo_20261114/M11/v3/demo_video/overlay_evidence.json) · [Ví dụ frame gốc](../../output/demo_20261114/M11/v3/demo_video/010_native.jpg)

Kiểm tra trên frame giải mã trước khi vẽ khung xanh đánh dấu. Mỗi cạnh được đối chiếu theo màu viền trong dải ±4 px quanh bbox đã quy đổi và xem lại trực quan. Hai file `overlay_evidence.json` lưu kết quả theo sự kiện. Ở M10, phép kiểm tra đối chứng cho 0/9 box đêm khớp và 15/15 box video demo khớp. Nguồn `v3-overlay-control.log`.

### Ca không hợp lý tiêu biểu trên holdout

- 65,333 s, track 307, xe máy đã qua vạch sang đường và còn xa. Nguồn dòng 003.
- 102,667 s, track 500, box ôm người lái xe máy nhưng nhãn là người đi bộ. Nguồn dòng 005.
- 111,667 s và 117,667 s, track 330, ô tô giữ làn bên trái. Nguồn dòng 007 và 008.

Nguồn [phiếu holdout](../../output/demo_20261114/M11/v3/demo_video/review_sheet.csv).
