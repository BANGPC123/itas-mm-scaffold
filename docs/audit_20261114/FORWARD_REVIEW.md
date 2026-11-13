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
