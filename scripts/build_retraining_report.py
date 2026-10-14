"""Summarize completed runs without converting development-video counts into accuracy."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/retraining_20261009'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def main():
    d=read('detector/results.json');r=read('recognizer/results.json');l=read('lane/results.json')
    dt=read('detector_test.json');rt=read('recognizer_test.json');v=read('video_full/summary.json');lat=read('video_latency/results.json');review=read('video_review_comparison.json')
    chosen=next(x for x in r['history'] if x['epoch']==r['selected_epoch'])['val']
    calibration=read('recognizer/calibration.json')
    acceptance='; '.join(f"{k}: coverage {v['coverage']*100:.2f}%, accuracy trên phần chấp nhận {v['accepted_accuracy']*100:.2f}%" for k,v in calibration['selected']['conditions'].items())
    testcases=[]
    for f in ['software-tests.xml','video-tests.xml']:
        testcases.extend(ET.parse(OUT/f).getroot().iter('testcase'))
    passed=sum(not any(t.find(x) is not None for x in ['failure','error','skipped']) for t in testcases)
    percent=lambda x:f'{100*x:.2f}%'
    lane_metrics=(f"F1 trên 9.675 ảnh VAL: {percent(l['baseline_val']['f1'])} → {percent(l['candidate_val']['f1'])}."
                  if 'candidate_val' in l else 'Ứng viên không vượt vòng sàng lọc VAL 512 ảnh; không thay checkpoint gốc.')
    text=f'''# ITAS-MM — Huấn luyện lại và kiểm thử video, 09/10/2026

## Model đã huấn luyện

| Model | Lượt train mới | Tiêu chí chọn |
|---|---|---|
| YOLO11n | 8 epoch, 10.952 ảnh TRAIN | mAP50–95 VAL; bản mới được chọn: {d['candidate_selected']} |
| MobileNetV3-Small | 5 epoch; {r['training_n']} crop gồm 1.800 crop nền TRAIN | Giữ accuracy của bốn điều kiện VAL trong 1 điểm %, giảm nhận nhầm nền; chọn epoch {r['selected_epoch']} |
| UFLD-v2 ResNet18 | 2 epoch, 4.096 ảnh CULane TRAIN driver_23, nhãn segmentation chính thức | 512 ảnh VAL để sàng lọc; chỉ đổi nếu vượt baseline trên VAL đầy đủ. Bản mới được chọn: {l['candidate_selected']} |

Đây là fine-tuning từ trọng số đã có, không phải huấn luyện từ đầu. Lượt UFLD dùng tập con và loader PyTorch trên Windows, không tái lập toàn bộ quy trình DALI 50 epoch của tác giả. Trọng số gốc vẫn được giữ. `deployment.json` ghi đường dẫn và SHA256 của bản được đưa vào ứng dụng.

{lane_metrics}

## Kết quả biển báo

YOLO VAL mAP50–95: {percent(d['baseline']['metrics/mAP50-95(B)'])} → {percent(d['candidate']['metrics/mAP50-95(B)'])}.
YOLO TEST mAP50: {percent(dt['metrics']['metrics/mAP50(B)'])}; mAP50–95: {percent(dt['metrics']['metrics/mAP50-95(B)'])}.

Classifier VAL crop sạch: {percent(r['baseline']['clean']['accuracy'])} → {percent(chosen['clean']['accuracy'])}.
Classifier VAL crop 16px + mờ tổng hợp: {percent(r['baseline']['small_blur']['accuracy'])} → {percent(chosen['small_blur']['accuracy'])}.
Tỷ lệ 400 crop nền VAL bị chấp nhận thành biển: {percent(r['baseline']['background']['accepted_fraction'])} → {percent(chosen['background']['accepted_fraction'])}.

Classifier TEST tất cả crop: accuracy {percent(rt['metrics']['all']['accuracy'])}, macro-F1 {rt['metrics']['all']['macro_f1']:.4f}.
TEST crop ≥32px: {percent(rt['metrics']['ge32']['accuracy'])}; crop <32px: {percent(rt['metrics']['lt32']['accuracy'])}.
Ngưỡng confidence chọn trên VAL: {calibration['selected']['threshold']:.2f}. {acceptance}. Lớp nền được từ chối trước khi ghép bằng chứng video. Bộ phân loại có 30 mã biển và một lớp nền `unknown`, chưa bao phủ mọi biển giao thông.

Crop nền lấy ngoài box đã gán nhãn và có khoảng đệm; giả định nhãn localization đủ. Đây không phải bộ OOD đầy đủ do người duyệt. Nhãn mã biển nguồn vẫn chưa được xác nhận độc lập bằng người. Không dùng video người dùng làm dữ liệu train.

## Kiểm thử trên video người dùng

Quét nguồn đầy đủ: {v['complete_source_scan']}; thời lượng {v['source_duration_seconds']:.1f} giây; {v['sampled_frames']} frame lấy mẫu {v['sampled_fps']:.1f} FPS. Video kết quả được lấy mẫu 3 FPS để rà soát, không dùng nó đo độ mượt phát video gốc. Giao diện phát video gốc bằng trình phát native và chạy inference riêng.

Đo inference 10Hz trong 120 giây: trung bình {lat['processing_ms']['mean']:.2f} ms; P95 {lat['processing_ms']['p95']:.2f} ms; P99 {lat['processing_ms']['p99']:.2f} ms; tỷ lệ >100ms {percent(lat['processing_ms']['fraction_over_100ms'])}.
Độ trễ xác nhận nhãn P95: {lat['confirmation_delay_ms']['p95']} ms. Tính từ detection đầu tiên đến hai lần nhận cùng mã; chưa gồm TTS, mạng, hiển thị hay lúc biển thật bắt đầu nhìn thấy.

Đối chiếu crop trong 24 ảnh do AI rà soát (không phải accuracy có nhãn người):

| Chỉ số chẩn đoán | Trước | Sau |
|---|---:|---:|
| Crop có mã trùng nhãn AI | {review['baseline']['known_crop_agreement']}/{review['baseline']['known_crop_n']} | {review['candidate']['known_crop_agreement']}/{review['candidate']['known_crop_n']} |
| Crop âm bị gán mã biển | {review['baseline']['negative_crop_false_accepts']}/{review['baseline']['negative_crop_n']} | {review['candidate']['negative_crop_false_accepts']}/{review['candidate']['negative_crop_n']} |

Video đã dùng phát hiện lỗi và chỉnh ROI nên là dữ liệu phát triển, không phải holdout độc lập. Không gọi số lượng dự đoán là accuracy. Chưa có GT đáng tin cậy cho làn đang đi, hiệu lực biển theo loại xe và toàn bộ biển phụ trong video; chưa kết luận ba logic đã chính xác ngoài thực địa.

## Kiểm tra phần mềm và sử dụng

{passed}/{len(testcases)} test phần mềm đạt. Frontend đã build. Mở http://127.0.0.1:8000/ để chạy ảnh/video; mục kết quả huấn luyện mới có đường dẫn video đã kiểm thử. TTS không bật trong lượt replay benchmark.

Các lệnh tái lập: `retrain_detector_candidate.py`, `prepare_lane_retraining.py`, `retrain_recognizer_candidate.py`, `retrain_lane_candidate.py`, `finalize_retraining.py`, `build_retraining_report.py` trong thư mục scripts. Các log, lịch sử epoch, manifest train, kết quả VAL/TEST, video và dự đoán từng frame nằm trong thư mục báo cáo này.
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8')
    print(OUT/'REPORT.md')

if __name__=='__main__':main()
