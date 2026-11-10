"""Report completed measurements separately from pending human/holdout work."""
from pathlib import Path
import json,hashlib
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/evaluation/improvements_20261009'
def read(path):return json.loads((OUT/path).read_text())

def main():
    lane=read('lane_val/results.json');recognition=read('recognizer_small/results.json');calibration=read('recognizer_small/calibration.json')
    video=read('video_verified/results.json');negative=read('negative_cases.json')
    selected=next(r for r in recognition['history'] if r['epoch']==recognition['selected_epoch'])
    cases=[]
    for name in ['software-tests.xml','video-tests.xml']:
        for case in ET.parse(OUT/name).getroot().iter('testcase'):
            cases.append({'name':case.get('name'),'pass':all(case.find(x) is None for x in ['failure','error','skipped'])})
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))+[ROOT/'models/sign_recognizer/robust_val.pt',ROOT/'models/lane/culane_res18.pth',ROOT/'models/sign_detector/best.pt']
    hashes={}
    for path in files:
        with path.open('rb') as file:hashes[str(path.relative_to(ROOT))]=hashlib.file_digest(file,'sha256').hexdigest()
    deployment={'date':'2026-10-09','source_hashes':hashes,'lane_filter_geometry':lane['selected_filter_geometry'],
        'recognizer':'models/sign_recognizer/robust_val.pt','recognizer_threshold':calibration['selected']['threshold'],'video_min_short_side':16,
        'human_labels_verified':0,'independent_holdout_verified':False,'status':'development_candidate_pending_human_review_and_holdout'}
    (OUT/'deployment.json').write_text(json.dumps(deployment,indent=2))
    baseline=recognition['baseline'];raw=lane['guard_without_geometry_filter'];old=lane['guard_with_legacy_filter']
    p=lambda n:f'{n*100:.2f}%'
    text=f'''# ITAS-MM: sửa lỗi và kiểm chứng ngày 09/10/2026

## Trạng thái

Đã hoàn thành phần tự động: đánh giá VAL làn, từ chối ảnh đồng màu, fine-tune classifier trên TRAIN, chọn checkpoint/ngưỡng trên VAL, ghép bằng chứng video theo phiên và đo độ trễ. Chưa hoàn tất xác nhận nhãn bằng người và holdout độc lập: người dùng đã nhận rà soát và sẽ cung cấp video mới, hiện chưa có đường dẫn/nhãn xác nhận.

## Làn đường

Đủ {lane['n']:,} ảnh VAL CULane chính thức; không dùng TEST để chọn ngưỡng. Archive `driver_23_30frame.tar.gz` tải qua nguồn UFLD-v2/CULane, giải nén đúng danh sách VAL; `data/raw/culane/val-verification.json` không thiếu file.

| Chính sách | Precision | Recall | F1 |
|---|---:|---:|---:|
| Guard ảnh + bộ lọc hình học cũ | {p(old['precision'])} | {p(old['recall'])} | {p(old['f1'])} |
| Guard ảnh, bỏ bộ lọc hình học cũ | {p(raw['precision'])} | {p(raw['recall'])} | {p(raw['f1'])} |

Cấu hình được chọn: `filter_geometry={lane['selected_filter_geometry']}`. Guard từ chối ảnh có khoảng mức xám toàn ảnh ≤2; không coi ảnh tối thật là ảnh lỗi chỉ vì độ sáng thấp. Guard từ chối {lane['input_guard_rejections']} ảnh VAL. Đây chỉ là kiểm tra thiếu thông tin rất bảo thủ, chưa phải bộ phân loại camera lệch, sương mù hoặc cảnh ngoài đường.

Metric Python chuyển từ evaluator CULane; chưa đối chiếu binary C++, không TTA, ảnh nguyên khung. VAL từng được dùng trong phát triển checkpoint gốc, không gọi nó là holdout độc lập. TEST đo ngày 08/10 được giữ nguyên trong thư mục cũ.

## Biển nhỏ/mờ

Fine-tune trên {recognition['train_n']} crop TRAIN; chọn epoch {recognition['selected_epoch']} bằng {recognition['val_n']} crop VAL. Không dùng TEST trong lượt chọn này, không lật ngang biển. Trọng số gốc `best.pt` giữ nguyên, runtime dùng `robust_val.pt`.

| Điều kiện VAL | Accuracy gốc | Accuracy mới |
|---|---:|---:|
| Crop sạch | {p(baseline['clean']['accuracy'])} | {p(selected['clean']['accuracy'])} |
| Gaussian 5×5, sigma 1,2 | {p(baseline['blur']['accuracy'])} | {p(selected['blur']['accuracy'])} |
| Giảm cạnh ngắn xuống 16px | {p(baseline['small']['accuracy'])} | {p(selected['small']['accuracy'])} |
| 16px + Gaussian 3×3, sigma 0,7 | {p(baseline['small_blur']['accuracy'])} | {p(selected['small_blur']['accuracy'])} |

VAL gốc toàn crop ≥32px; crop 16px/mờ là biến đổi tổng hợp từ VAL, không thay thế biển nhỏ ngoài thực địa. Nhãn mã biển vẫn AI-verified. Ngưỡng confidence chọn trên VAL: {calibration['selected']['threshold']:.2f}; calibration chỉ gồm lớp đã biết, chưa có OOD benchmark.

Ảnh đơn giữ cạnh ngắn tối thiểu 32px. Video cho phép 16px nhưng cần ít nhất hai quan sát cùng mã trong cửa sổ 0,75 giây. Giữ nhãn/confidence của crop có điểm độ rõ tốt nhất còn hạn; không ghép ảnh thành siêu phân giải. Khi mã mâu thuẫn, mất detection, ghép box không duy nhất, tua hoặc đổi phiên/ROI thì hủy bằng chứng. Giới hạn 64 track/phiên, 8 phiên và hết hạn phiên sau 15 giây.

## Video và độ trễ

Đo {video['frames']} frame, {video['sample_fps']:.1f} lượt/giây, từ giây {video['start_seconds']} trong {video['duration_requested']} giây của `test_video.mp4`. Pipeline có crop đường 65%; số đo cuối sau cấu hình làn đã chọn.

- Xử lý trung bình: {video['processing_ms']['mean']:.2f} ms; P95: {video['processing_ms']['p95']:.2f} ms; P99: {video['processing_ms']['p99']:.2f} ms.
- Tỷ lệ frame xử lý >100ms: {p(video['processing_ms']['fraction_over_100ms'])}.
- Có {video['confirmation_delay_ms']['events']} track được xác nhận; độ trễ xác nhận trung vị {video['confirmation_delay_ms']['median']} ms, P95 {video['confirmation_delay_ms']['p95']} ms.

Độ trễ tính từ detection đầu tiên của track đến xác nhận mã bằng hai frame, không phải lúc biển thật bắt đầu xuất hiện. Chưa gồm mạng, hiển thị hoặc TTS; video hiện không phát TTS. Các nhãn xác nhận chưa được đối chiếu GT, nên số track/nhãn không phải accuracy. Video vẫn phát native FPS, inference tối đa 10Hz và một request đang chạy; không tuyên bố hết mọi giật lag.

## Kiểm thử

{sum(c['pass'] for c in cases)}/{len(cases)} test phần mềm đạt. Với model làn thật: {', '.join(f"{r['case']}={r['lanes']} làn" for r in negative)}. Các ảnh đồng màu trả `rejection_reason=insufficient_image_information`. Random noise không bị guard nhận là ảnh đồng màu; model không phát hiện làn trong ca thử cố định này.

## Người duyệt và holdout

[Mở trang rà soát](http://127.0.0.1:8000/review/test_video_development/): 24 thời điểm phát triển, mỗi thời điểm có ảnh ±0,2 giây. Lấy ảnh có nhiều detection trong từng đoạn thời gian để dễ rà lỗi, do đó không dùng bộ này để ước lượng tần suất lỗi ngoài đường. Draft box/mã là gợi ý từ model; người dùng sửa cả nhãn sai và thiếu, vẽ hai biên làn, ghi loại xe, overhead, vai trò biển, parent_id và kết quả áp dụng mong đợi.

Chỉ `human_confirmed=true` có tên và thời điểm người duyệt mới được nhập. Chương trình kiểm tra SHA nguồn, id/thời gian/kích thước ảnh, tọa độ, quan hệ cha-con; không tự nâng nhãn AI lên human verified. Trường hợp không đọc được giữ unknown/uncertain. Script chấm ba logic từ metadata người duyệt; chưa phải accuracy toàn pipeline vì đầu vào là GT.

```powershell
python scripts/evaluate_reviewed_context.py "C:\\duong-dan\\reviewed_labels.json"
python scripts/prepare_video_review.py "D:\\video-chuyen-di-moi.mp4" --holdout --output output/review/independent_holdout
```

Lệnh holdout từ chối dùng lại video phát triển, lưu hash hệ thống trước khi chuẩn bị ảnh và lấy mẫu theo thời gian, không chọn mẫu theo model. Vẫn cần người dùng xác nhận chuyến đi độc lập: SHA khác không chứng minh hai video không trùng cảnh. Sau khi xem kết quả holdout, không tiếp tục chỉnh rồi báo lại cùng tập như kiểm chứng độc lập.

## Tái lập

`scripts/finetune_recognizer_robustness.py`, `scripts/calibrate_robust_recognizer.py`, `scripts/evaluate_lane_val.py`, `scripts/benchmark_temporal_video.py`, `scripts/build_improvement_report.py`. Các kết quả JSON, dự đoán từng frame, JUnit và hash triển khai nằm trong thư mục này. Lượt `recognizer/` là thí nghiệm tăng cường nhẹ trước đó; lựa chọn cuối là `recognizer_small/`.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps({'report':str(OUT/'REPORT.md'),'tests':len(cases),'human_verified':0,'holdout_verified':False},indent=2))

if __name__=='__main__':main()
