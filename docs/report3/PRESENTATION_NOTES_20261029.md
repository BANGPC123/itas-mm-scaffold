# ITAS-MM - Ghi chú trình bày Report 3 (29/10/2026)

Nguồn số liệu: `docs/report3/REPORT3_MODEL_RESULTS.md` (nhánh `work/kiet-dataset-v2`, sau PR #6). Mọi nhãn là nhãn AI (Claude, GPT-6.1 sol) cộng spot-check 44 crop của owner; không có human ground truth.

## 1. Ý chính (3 câu)
1. Fine-tune detector cải thiện có ý nghĩa thống kê trên TEST; recognizer v2 (87 mã QCVN) ít gắn sai mã hơn hẳn bản cũ, đổi lại trả lời ít hơn.
2. Ngoài thực địa (video Hà Nội), model mới ở ngưỡng 0,87 có precision 95,7-100% theo cả ba cách tính nhãn; bản cũ chỉ khoảng 72-77%.
3. Lane và các khâu end-to-end chưa đo được bằng ground truth; hệ thống vẫn HOLD, chỉ là prototype xử lý video offline.

## 2. Số nên nói
| Nội dung | Số liệu | Mục |
|---|---|---|
| Detector, TEST, recall | 0,8509 -> 0,8754 (+0,0245; CI 0,0170-0,0317) | V.4 |
| Detector, TEST, mAP50 | 0,9194 -> 0,9357 (+0,0163; CI 0,0122-0,0198) | V.4 |
| Recognizer, TEST, 30 mã cũ, crop >= 32 px | accuracy 0,9371 -> 0,9697; macro-F1 0,9056 -> 0,9594 | V.5 |
| Recognizer, crop < 16 px | accuracy 0,5139 -> 0,7839 | V.5 |
| Điểm vận hành (>= 32 px): cũ @0,63 / mới @0,87 | phủ 0,975 / 0,904; đúng trong số trả lời 0,956 / 0,989; đúng trên tổng 0,932 / 0,893 | V.5 |
| Precision thực địa, Hà Nội + V0, mới @0,87 | 96,3% (nhãn Claude), 95,7% (nhãn sol), 100% (chỉ phần đồng thuận) | V.7 |
| Cùng chỉ tiêu, cũ @0,63 + gate cũ | 72,1% / 72,0% / 77,2% | V.7 |
| Tái lập train bằng script đã commit | VAL macro-F1 0,9511 -> 0,9500; TEST 0,9697 -> 0,9697; cùng epoch 18 | V.6 |
| Đồng thuận hai AI (Cohen kappa) | 0,62-0,81 theo nguồn; sol khớp owner 29/32 | VI.6 |
| Tốc độ | khoảng 22 khung/giây trên GPU 4 GB | VI.5 |

## 3. Giới hạn nên nói trước
- Nhãn do AI gán; TEST đã dùng ở các giai đoạn trước nên không phải holdout sạch.
- Precision tính trên nhãn "đã xác nhận" (theo track), không phải recall của biển ngoài đời. Proxy recall chỉ 46 biển: cũ 0,630; mới 0,739.
- Con số 100% là trên tập hai AI đồng thuận (tập dễ hơn); khoảng tin cậy lấy theo đoạn 60 giây nên lạc quan; thực địa chỉ 3 video Hà Nội + V0 (V0 thuộc TEST).
- Model mới trả lời ít hơn bản cũ (phủ 90,4% so với 97,5% ở crop >= 32 px) để đổi lấy ít sai hơn khoảng 4 lần.
- Lane: không có ground truth Việt Nam, không có F1. Đánh giá định tính 60 khung bằng UFLD-v2 gốc (chưa bật phương án dự phòng camera-aware, chưa làm mượt): 0 khung "good"; ban ngày 3 partial / 15 fail / 12 không thấy vạch; ban đêm 9 / 15 / 6. Tính riêng khung có vạch nhìn thấy: fail 83% (ngày), 62,5% (đêm), 71% (gộp).
- Demo end-to-end (camera đến giọng nói) chưa được kiểm tra lại trên video mới. Chưa làm: SAHI (đang tắt), contrastive learning, GPS thật; QCVN trích dẫn bản 2024 (đề cương ghi 2019).
- Chưa làm P4: khoảng 206/403 box detector trên TRAIN không khớp nhãn là biển thật bị thiếu nhãn.
- Khôi phục trên máy khác và quyền dữ liệu chưa kiểm chứng. Release: HOLD / NOT_APPROVED.

## 4. Câu dễ bị hỏi
- **Nhãn AI có tin được không?** Hai AI khác hãng khớp ở mức kappa 0,62-0,81; owner đồng ý với nhãn Claude 36/37 câu có trả lời. Vẫn không thay được human ground truth, nên báo cáo ghi rõ là "AI-reviewed".
- **Sao giữ ngưỡng 0,87, không hạ xuống 0,64?** Luật đăng ký trước: hạ ngưỡng chỉ khi precision không giảm quá 0,02. Ở 0,64 precision còn 88,7% / 84,7% / 91,1% so với 96,3% / 95,7% / 100%, cả ba cách đều không đạt.
- **Gate v2 có tác dụng không?** Không: cấu hình có gate cho kết quả y hệt cấu hình không gate trên Hà Nội, nên không bật.
- **Vì sao số thực địa thấp hơn số trên dataset?** Có khoảng cách miền dữ liệu (dataset sạch, video thực tế mờ, nhỏ, ban đêm); pilot V0: precision detector 0,641 -> 0,729 sau fine-tune.
- **Lane dùng được không?** Chưa. Nói thẳng là chưa đo bằng ground truth và đánh giá định tính cho thấy lane detector gốc thường không bắt đúng vạch; việc tiếp theo là bộ lane ground truth Việt Nam. Nếu bị hỏi về F1: đã thử vẽ GT lane bằng AI trên 30 khung, F1 0,032 theo giao thức đăng ký trước, nhưng GT lệch định nghĩa (nhiều đường bám lề thay vì vạch sơn) nên không dùng làm kết quả; chỉ ghi ở giới hạn (Report 3, V.8 và VII.3).

## 5. Nên đọc lại trước khi trình bày
- `docs/report3/REPORT3_MODEL_RESULTS.md`: mục V.5, V.7, V.8, VI.6, VII.3, VII.5.
- Hình: `docs/report3/figures/fig7_final_field_precision.png`, `fig8_final_operating_points.png`, `fig9_final_reviewer_agreement.png`.
