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

## M17 - holdout mới, 2 clip đông xe ban ngày

Hai clip mới dài 90 s mỗi clip. Dùng luật cố định ở đầu báo cáo và cùng script trích ảnh như v3, với ảnh tại sự kiện và trước 0,5 s. Số câu/phút lấy cue cảnh báo đã lên lịch trong render. **AI-reviewed, chưa đo recall.**

| Clip | Mức | n cảnh báo | hop_ly | khong_hop_ly | khong_ro | Tỷ lệ hop_ly | Cue / phút |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| day_hanoi_many_signs | chu_y | 2 | 1 | 0 | 1 | 50,0% |  |
| day_hanoi_many_signs | nguy_co | 3 | 1 | 2 | 0 | 33,3% |  |
| day_hanoi_many_signs | Mọi mức | 5 | 2 | 2 | 1 | 40,0% | 2,67 (4 cue / 90 s) |
| day_west_lake | chu_y | 11 | 6 | 2 | 3 | 54,5% |  |
| day_west_lake | nguy_co | 0 | 0 | 0 | 0 | Không xác định |  |
| day_west_lake | Mọi mức | 11 | 6 | 2 | 3 | 54,5% | 2,67 (4 cue / 90 s) |
| Gộp hai clip M17 | chu_y | 13 | 7 | 2 | 4 | 53,8% |  |
| Gộp hai clip M17 | nguy_co | 3 | 1 | 2 | 0 | 33,3% |  |
| Gộp hai clip M17 | Mọi mức | 16 | 8 | 4 | 4 | 50,0% | 2,67 (8 cue / 180 s) |
| demo_video v3 (holdout) | Mọi mức | 10 | 5 | 5 | 0 | 50,0% | 2,57 (6 cue / 140 s) |
| **Gộp 3 holdout** | Mọi mức | 26 | 13 | 9 | 4 | 50,0% | 2,625 (14 cue / 320 s) |

Gộp ba holdout gồm demo_video v3 và hai clip M17. Ở mức `nguy_co`, cả 3 mẫu trong nhóm gộp đều đến từ clip Hà Nội, với 1/3 ca hợp lý.

Nguồn [phiếu Hà Nội](../../output/demo_20261114/M11/M17/day_hanoi_many_signs/review_sheet.csv), [phiếu Hồ Tây](../../output/demo_20261114/M11/M17/day_west_lake/review_sheet.csv), [đếm M17](../../output/demo_20261114/M11/M17/counts.json), [đếm v3](../../output/demo_20261114/M11/v3/counts.json). Cue và thời lượng từ `M17/clips/<clip>/render/summary.json` và `M13/clips/demo_video/render/summary.json` dưới `D:/DSP391m/project/itas-mm-scaffold/output/demo_20261114/`.

### Box vẽ trên video

Hà Nội 5/5 và Hồ Tây 11/11 box khớp bbox metadata quy đổi từ nguồn 1920x1080 sang canvas 1280x720. Dùng kiểm tra v3 trên frame giải mã gốc, với dải +/-4 px và ít nhất 60% điểm khớp màu trên từng cạnh, rồi đối chiếu ảnh. [Hà Nội](../../output/demo_20261114/M11/M17/day_hanoi_many_signs/overlay_evidence.json), [Hồ Tây](../../output/demo_20261114/M11/M17/day_west_lake/overlay_evidence.json).

### Nguyên nhân các ca khong_hop_ly theo M13.4

| Nhóm | Hà Nội | Hồ Tây | Tổng / 4 ca | Ca trong phiếu |
| --- | ---: | ---: | ---: | --- |
| C2 - Ngoài làn, không thấy cắt vào | 0 | 2 | 2 | Hồ Tây 002, Hồ Tây 011 |
| C1 - Người lái xe máy bị gán là người đi bộ | 1 | 0 | 1 | Hà Nội 001 |
| C3 - Sát lề hoặc ở xa | 1 | 0 | 1 | Hà Nội 004 |
| C4 - Box chỉ bao một phần xe | 0 | 0 | 0 |  |
| Khác | 0 | 0 | 0 |  |

Nhóm lấy từ `D:/DSP391m/project/itas-mm-scaffold/output/demo_20261114/M13/ERROR_ANALYSIS.md`, dòng 15. Mỗi ca có một nguyên nhân chính.

### Quan sát giao thông đông

Ước lượng trên khung sự kiện. Thấp là 0-3 xe máy nhìn thấy, vừa là 4-7, cao là từ 8 trở lên. Xe chở hai người tính một xe. Xe bị che khuất làm ước lượng kém chắc chắn.

| Clip | Thấp | Vừa | Cao | n khung sự kiện |
| --- | --- | --- | --- | ---: |
| Hà Nội | 004 | 001 / 002 / 003 / 005 | 0 ca | 5 |
| Hồ Tây | 0 ca | 001 / 002 / 006 / 008 / 009 / 010 | 003 / 004 / 005 / 007 / 011 | 11 |

Các ô mật độ liệt kê ID trong phiếu. Rà soát thêm khung mỗi 3 s dọc clip và ảnh lân cận ở các đoạn dưới đây. Đối chiếu timestamp với log cảnh báo vật cản, cả bốn cửa sổ đều có 0 cảnh báo.

| Clip | Khoảng thời gian | Mật độ ước lượng | Tình huống quan sát |
| --- | --- | --- | --- |
| Hà Nội | 17,5-18,5 s | vừa | Xe máy áo đen xuất hiện sát bên trái và tiến lên gần hướng xe mình. |
| Hà Nội | 41,5-42,5 s | vừa | Xe máy tối màu xuất hiện sát mép phải cạnh SUV trắng rồi chạy lên gần hướng xe mình. |
| Hồ Tây | 62,5-63,5 s | cao | Xe máy mũ vàng dịch từ phải về phía trước xe mình khi một xe giao hàng chạy sát bên phải. |
| Hồ Tây | 68,5-69,5 s | cao | Nhiều xe máy cắt ngang phía trước từ hai phía tại nút giao. |

Đây là quan sát định tính từ các khung lấy mẫu. Chưa có danh sách đầy đủ các tình huống cần cảnh báo hoặc nhãn người để tính recall. [Ảnh Hà Nội](../../output/demo_20261114/M11/M17/day_hanoi_many_signs/traffic_followup.jpg), [ảnh Hồ Tây](../../output/demo_20261114/M11/M17/day_west_lake/traffic_followup.jpg), [phiếu quan sát](../../output/demo_20261114/M11/M17/traffic_observations.csv).
