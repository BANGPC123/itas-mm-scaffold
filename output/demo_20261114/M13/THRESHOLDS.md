# M13.5 - Bộ lọc và ngưỡng (chỉnh trên clip đêm; video demo là holdout)

## Quy tắc

- Mỗi bộ lọc có một cờ trong `configs/perception.yaml: forward_warning`. Đặt cờ về `false` là quay về quy tắc M10 cho bộ lọc đó.
- Chỉ chỉnh ngưỡng trên `night_90s` (phiếu M11 v2, `source_m11_v2/night_90s_review_sheet.csv`). Video demo không dùng để chọn ngưỡng.
- Phát lại không cần GPU: `scripts/forward_replay_20261114.py` chạy trên detections đã cache (`dets_night_90s.jsonl`, tạo bởi `scripts/forward_detect_cache_20261114.py`) với cùng `ForwardObjectWarner`, `VoiceGate` như renderer.
- Kiểm tra parity: với mọi cờ tắt, phát lại cho đúng 9 cảnh báo của M10 (cùng thời điểm, track, lớp, mức). Số câu nói khác M10 (5 so với 6) vì M13.3.

## Kết quả phát lại trên clip đêm (đối chiếu phiếu M11 v2)

Cột: số ca `khong_hop_ly` / `hop_ly` / `khong_ro` còn lại; tổng cảnh báo; số câu nói.

| Cấu hình | khong_hop | hop | khong_ro | cảnh báo | câu nói |
|---|---:|---:|---:|---:|---:|
| M10 (tất cả cờ tắt) | 6 | 2 | 1 | 9 | 5 |
| chỉ `rider_merge` | 5 | 2 | 1 | 8 | 5 |
| chỉ `nguy_zone` | 4 | 2 | 1 | 7 | 4 |
| chỉ `lateral` (0,25) | 4 | 2 | 1 | 7 | 5 |
| `track_class_reset` | 1 | 1 | 1 | 3 | 2 |
| `track_min_age_samples=3` | 6 | 2 | 1 | 9 | 5 |
| **Cấu hình chọn** (rider + nguy_zone + lateral 0,25) | **2** | **2** | **1** | **5** | **3** |
| Cấu hình chọn, `lateral_max_per_h=0,15` | 1 | 2 | 1 | 4 | 2 |
| Cấu hình chọn, `lateral_max_per_h=0,35` | 3 | 2 | 1 | 6 | 4 |

Nguồn số: `replay_night_final_config.json` và các lần chạy `scripts/forward_replay_20261114.py --review`. Ghi chú: `lateral 0,15` làm mất cảnh báo `hop_ly` 005 (xe bán tải, track 342, 71,0 s). Ngưỡng 0,25 nằm giữa hai ngưỡng đó, nên độ nhạy cao. Đây là lý do cần holdout.

## Quyết định từng bộ lọc

| Cờ | Giá trị | Lý do |
|---|---|---|
| `rider_merge_enabled` | true | Bỏ người nằm trong box xe máy. Trên clip đêm nó tự có tác dụng (bỏ 006), nhưng khi bật cùng bộ lọc khác thì không còn tác dụng riêng. Giữ vì đúng với C1 và không làm mất `hop_ly` nào. |
| `nguy_zone_enabled` | true, 0,36 / 0,12, biên 2% | Một mình bỏ box 606 (người lái xe máy, 004). Cùng với lateral thì bỏ thêm 615 (ô tô, box cắt, 003). Không mất `hop_ly` nào. |
| `lateral_filter_enabled` | true, 0,25 | Bỏ 001 và 006 (người lái xe máy đang dịch ngang). Không mất `hop_ly` nào. |
| `track_class_reset_enabled` | false | Bỏ được 002 và 004 nhưng cũng mất `hop_ly` 005. Trong báo cáo không đánh đổi được, nên tắt. |
| `track_min_age_samples` | 0 | Không có tác dụng trên clip đêm. Tắt. |
| `nguy_co_enabled` | true | Công tắc cho M13.6. Chỉ đặt false nếu video demo còn nguy_co vô lý theo FORWARD_REVIEW.md. |

## Hạn chế

- Ngưỡng `lateral_max_per_h` nhạy cảm (xem bảng).
- Mẫu đêm chỉ có 9 cảnh báo đã chấm (phiếu M11 v2). Đây không phải tập kiểm định, và số liệu chỉ đúng cho clip này.
- Các ca C2 (xe ở làn khác, `chu_y`), C3 (sát lề, xa) và C4 (box chỉ bao một phần xe) không có bộ lọc riêng. Ca 003 (615) không còn trong cấu hình chọn, nhưng đó là do lateral và vùng hẹp, không phải do sửa box.
