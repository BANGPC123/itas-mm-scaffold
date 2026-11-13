# M13.4 - Phân loại ca `khong_hop_ly` của M11 v2 (trước khi sửa logic lọc)

Viết trước M13.5. Không sửa ngưỡng hay code lọc trong bước này.

## Nguồn

- Phiếu M11 v2 (bản sao, không sửa): `output/demo_20261114/M13/source_m11_v2/night_90s_review_sheet.csv`, `demo_video_review_sheet.csv`, `counts.json`.
  Bản gốc: `work/audit-20261114` tại `output/demo_20261114/M11/v2/` (commit `deee1bf`).
- Render M10 được chấm: `output/demo_20261114/M10/clips/{night_90s,demo_video}/render/`.
- Luật chấm (AI-reviewed, không phải nhãn người): `docs/audit_20261114/FORWARD_REVIEW.md` ở nhánh audit.
- Số đếm M11 v2: 24 cảnh báo đã chấm (night 9, demo 15). `hop_ly` 8, `khong_hop_ly` 14, `khong_ro` 2.

## Bảng 14 ca `khong_hop_ly`

Phân loại: **C1** người lái xe máy bị gán nhãn người đi bộ; **C2** vật ở làn khác, bên kia vạch, không cắt vào; **C3** sát lề/bó vỉa, xa, vật nhỏ; **C4** box chỉ bao một phần xe.

| Clip | ID | t (s) | track | lớp | mức | Lý do của phiếu (tóm tắt) | Loại |
|---|---|---|---|---|---|---|---|
| night_90s | 001 | 21.667 | 110 | person | nguy_co | Box bám người ngồi trên xe máy xanh | C1 |
| night_90s | 002 | 60.667 | 342 | person | nguy_co | Người lái xe máy nhập từ trái; track 342 sau đó là truck | C1 (+A1) |
| night_90s | 003 | 63.333 | 615 | car | nguy_co | Box chỉ bao phần dưới đuôi SUV đỏ | C4 (+A2) |
| night_90s | 004 | 63.333 | 606 | person | nguy_co | Box trên người lái xe máy | C1 (+A2) |
| night_90s | 006 | 72.0 | 601 | person | nguy_co | Người ngồi xe máy giữa hai ô tô | C1 |
| night_90s | 009 | 86.333 | 762 | person | chu_y | Người lái xe máy chở hàng | C1 |
| demo_video | 002 | 40.667 | 205 | person | chu_y | Người ngồi xe máy | C1 |
| demo_video | 004 | 47.0 | 205 | person | chu_y | Người lái xe máy cùng làn | C1 |
| demo_video | 006 | 60.667 | 281 | car | nguy_co | Ô tô bạc sát bó vỉa, bên kia vạch, chưa nhập làn | C3 |
| demo_video | 007 | 65.333 | 307 | motorcycle | chu_y | Xe máy nhỏ bên kia vạch, còn xa | C2 |
| demo_video | 009 | 94.333 | 472 | motorcycle | chu_y | Xe máy nhỏ mép phải, gần vỉa hè, còn xa | C3 |
| demo_video | 010 | 102.667 | 500 | person | chu_y | Người điều khiển xe máy bên phải nút giao | C1 |
| demo_video | 012 | 111.667 | 330 | car | chu_y | Ô tô chạy bên trái vạch đứt, tách làn | C2 |
| demo_video | 013 | 117.667 | 330 | car | chu_y | Ô tô làn trái, chưa cắt vào | C2 |

Đếm: C1 = 8 (night 5, demo 3); C2 = 3 (demo); C3 = 2 (demo); C4 = 1 (night). Tổng 14.

## Ghi chú theo loại

- **A1 - đổi lớp trong cùng track (night 342):** cùng ID 342 là `person` lúc 60.667 s rồi `truck` lúc 71.0 s và 77.0 s. ID bị tái dùng hoặc gán lại khi box đổi. Cảnh báo `person` ở 60.667 s là sai. Hai cảnh báo `truck` sau đó được chấm `hop_ly`, nên không bỏ được cả track. Cần reset lịch sử khi đổi lớp (M13.5).
- **A2 - hai cảnh báo nguy_co cùng thời điểm (night 63.333 s):** track 615 (ô tô) và 606 (người lái xe máy) cùng lúc. Đây là cùng một lỗi của M13.3 (đã sửa phần phát giọng). Không phải lỗi của bộ lọc.
- **A3 - lỗi box ở mọi cảnh báo do quy đổi (M13.1):** box cam trong video vẽ ở tọa độ nguồn 1920x1080 lên canvas 1280x720. Phiếu M11 chấm trên box đã quy đổi nên không bị ảnh hưởng ở cột phán quyết. Người xem video thấy box lệch. Đã sửa ở M13.1.
- **A4 - không có ca `khong_hop_ly` nào có box chạm mép.** Ca `khong_ro` night 008 có y2 = 1077.7 trên khung 1080 (chạm mép dưới). Ca này không phải bằng chứng cho bộ lọc "không chạm mép", nên ghi nhận nhưng không dùng để chọn ngưỡng.
- **A5 - track 330 (demo):** cùng một track có 011 `hop_ly` (ô tô ngay trước), 012 và 013 `khong_hop_ly` (làn trái). Không đổi lớp. Phân biệt đúng-sai nằm ở vị trí ngang, không ở ID.

## Bộ lọc tương ứng (đề xuất, chưa viết code)

| Loại lỗi | Ca | Bộ lọc dự kiến (M13.5) | Ghi chú |
|---|---|---|---|
| C1 rider = person | 8 | Gộp: person có tâm nằm trong box motorcycle thì bỏ person, dùng box xe máy | Chỉ khi tìm được motorcycle cùng khung; kiểm bằng replay trên night |
| A1 đổi lớp cùng track | night 002 | Reset lịch sử track khi lớp đổi; yêu cầu tuổi track tối thiểu | Không xóa track; chỉ bỏ mẫu cũ |
| C3 sát lề, xa, nguy_co | demo 006 | Vùng giữa hẹp và không chạm mép cho nguy_co | Chỉ chỉnh trên night; demo là holdout |
| Trượt ngang | night 002 (vào từ trái) | Lọc tốc độ dịch ngang tâm box | Có thể không đủ; xem replay |
| C2, C3 chu_y, C4 | demo 007, 009, 012, 013; night 003 | Không có bộ lọc nào trong danh sách M13.5 | Để lại, ghi rõ trong báo cáo |

Hệ quả đã biết: các ca C2, C3 (chu_y), C4 không có bộ lọc nào trong danh sách. Chúng sẽ còn sau M13.5 nếu không có bộ lọc mới. Nếu cần xử lý, phải mở thêm bước có ghi lại quyết định.
