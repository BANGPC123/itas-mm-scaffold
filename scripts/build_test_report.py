"""Build the report from completed revalidation artifacts, without training."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/revalidation_20261008'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def pct(value):return 'N/A' if value is None else f'{value*100:.2f}%'


def main():
    detector=read('detector.json');recognizer=read('recognizer.json');conditions=read('detector_conditions.json')
    pipeline=read('product/sign_pipeline_results.json');lane=read('culane_full/results.json');negative=read('negative_cases.json')
    if not lane['complete_test']:raise RuntimeError('CULane TEST evaluation is incomplete')
    (OUT/'lane.json').write_text(json.dumps({
        'status':'MEASURED_COMPLETE_CULANE_TEST', 'accuracy':None,
        'f1':lane['metrics']['deployed_filter']['all']['f1'],
        'raw_adapter_f1':lane['metrics']['raw_adapter']['all']['f1'],
        'evaluated_images':lane['evaluated_images'],
        'details':'culane_full/results.json', 'negative_cases':negative,
        'note':'Lane F1 uses a Python port of the CULane metric; local driving video remains unlabeled.'
    },indent=2),encoding='utf-8')
    cases=[]
    for filename,group in [('software-tests.xml','Python'),('video-tests.xml','Video JavaScript')]:
        for case in ET.parse(OUT/filename).getroot().iter('testcase'):
            state='FAIL' if case.find('failure') is not None or case.find('error') is not None else ('SKIP' if case.find('skipped') is not None else 'PASS')
            cases.append({'group':group,'name':case.get('classname','')+'.'+case.get('name',''),'status':state,'seconds':case.get('time','')})
    with (OUT/'test_cases.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['group','name','status','seconds']);writer.writeheader();writer.writerows(cases)
    original=recognizer['conditions']['original'];m=detector['metrics']
    summary={'detector':m,'recognizer_original':{key:original[key] for key in ['n','accuracy','macro_f1','ge32','lt32','runtime_coverage','runtime_accepted_accuracy']},
             'pipeline':pipeline,'lane':lane,'software_passed':sum(c['status']=='PASS' for c in cases),'software_total':len(cases),
             'negative_passed':sum(c['pass_no_false_detections'] for c in negative),'negative_total':len(negative),
             'training_or_threshold_tuning_performed':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'recognizer_per_class.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file);writer.writerow(['code','precision','recall','f1','support'])
        for name,values in original['per_class'].items():
            if isinstance(values,dict) and name not in ['macro avg','weighted avg']:
                writer.writerow([name,values['precision'],values['recall'],values['f1-score'],int(values['support'])])
    names=list(recognizer['conditions']);labels=['Original','Dark 40%','Bright 180%','Blur 9px','JPEG Q20','Half resolution']
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(names))
    ax.bar(x-.18,[recognizer['conditions'][n]['accuracy']*100 for n in names],.36,label='All TEST crops (3058)')
    ax.bar(x+.18,[recognizer['conditions'][n]['ge32']['accuracy']*100 for n in names],.36,label='Original size >=32px (1179)')
    ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,100);ax.set_ylabel('Classification accuracy (%)');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.08),ncol=2);ax.set_title('Synthetic crop robustness: fixed checkpoint')
    fig.tight_layout();fig.savefig(OUT/'recognizer_conditions.png',dpi=150);plt.close(fig)
    groups=[n for n in lane['metrics']['raw_adapter'] if n not in ['all','test7_cross']]
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(groups))
    ax.bar(x-.18,[(lane['metrics']['raw_adapter'][n]['f1'] or 0)*100 for n in groups],.36,label='Raw adapter, no TTA')
    ax.bar(x+.18,[(lane['metrics']['deployed_filter'][n]['f1'] or 0)*100 for n in groups],.36,label='With geometry filter')
    ax.set_xticks(x,[n.split('_',1)[1] for n in groups],rotation=15);ax.set_ylim(0,100);ax.set_ylabel('Lane F1 (%)');ax.legend();ax.set_title('Complete CULane TEST: effect of geometry filtering')
    fig.tight_layout();fig.savefig(OUT/'lane_conditions.png',dpi=150);plt.close(fig)
    text=['# ITAS-MM: accuracy từng model và toàn bộ test case',
          '\nNgày đánh giá: 08/10/2026. Checkpoint và ngưỡng được giữ nguyên; không huấn luyện hoặc chọn ngưỡng bằng TEST. Mục tiêu là đo lại chất lượng thực tế, gồm các trường hợp thất bại.',
          '\n## 1. Kết quả chính',
          '\n| Thành phần | Tập đánh giá | Kết quả |\n|---|---|---|',
          f'| Model 1: UFLD-v2, adapter trước lọc | 34.680 ảnh CULane TEST | F1 **{pct(lane["metrics"]["raw_adapter"]["all"]["f1"])}** |',
          f'| Model 1: sau bộ lọc đang dùng | Cùng 34.680 ảnh | F1 **{pct(lane["metrics"]["deployed_filter"]["all"]["f1"])}** |',
          f'| Model 2a: YOLO11n phát hiện biển | 2.326 ảnh / 4.888 box TEST | mAP50 **{pct(m["metrics/mAP50(B)"])}**, mAP50–95 **{pct(m["metrics/mAP50-95(B)"])}** |',
          f'| Model 2b: MobileNetV3Small, 30 mã biển | 3.058 crop chuẩn TEST | Accuracy **{pct(original["accuracy"])}**, macro-F1 **{pct(original["macro_f1"])}** |',
          f'| Classifier: biển có cạnh ngắn chuẩn ≥32px | 1.179 crop | Accuracy **{pct(original["ge32"]["accuracy"])}** |',
          f'| Classifier: biển có cạnh ngắn chuẩn <32px | 1.879 crop | Accuracy **{pct(original["lt32"]["accuracy"])}** |',
          f'| Detector + classifier, quy tắc từ chối bật | Toàn bộ biển TEST có nhãn mã | Phục hồi đúng mã **{pct(pipeline["correct_code_recall_all_released"])}** |',
          f'| Detector + classifier, biển chuẩn ≥32px | 1.179 biển | Phục hồi đúng mã **{pct(pipeline["correct_code_recall_ge32"])}** |',
          '\nKhông gọi mAP/F1 là classification accuracy. Accuracy của classifier trên crop đã biết vị trí không đại diện cho toàn hệ thống. Nhãn mã biển hiện là AI-verified, chưa có human verification độc lập.',
          '\n## 2. Model làn đường: đủ 9 nhóm CULane',
          '\n| Nhóm | F1 trước lọc | F1 sau lọc | TP / FP / FN sau lọc |\n|---|---:|---:|---|']
    for name in sorted(lane['metrics']['raw_adapter']):
        if name=='all':continue
        a=lane['metrics']['raw_adapter'][name];b=lane['metrics']['deployed_filter'][name]
        text.append(f'| {name} | {pct(a["f1"])} | {pct(b["f1"])} | {b["tp"]} / {b["fp"]} / {b["fn"]} |')
    text += ['\nNhóm cross không có làn GT: cần đọc số false positive, không diễn giải F1=0 như accuracy=0%.',
             '\n![F1 theo điều kiện](lane_conditions.png)',
             '\nBộ lọc hình học làm mất nhiều làn thật. Không coi việc ít đường vẽ hơn là accuracy tốt hơn; chưa tinh chỉnh filter trên TEST trong lượt này. Khuyến nghị đánh giá lại việc bật mặc định và hiệu chỉnh bằng VAL có nhãn, rồi kiểm tra trên một holdout khác.',
             '\nPhương pháp: checkpoint CULane ResNet18 gốc, FP16 CUDA, ảnh đủ khung 1640×590, không crop ROI 65% của video nội thất, không TTA. Metric Python port từ mã CULane trong UFLD-v2: spline tự nhiên theo độ dài cung, 50 mẫu/đoạn, bề rộng 30px, ghép Hungarian và IoU >0,5. Có kiểm thử perfect-match, disjoint, empty và biên IoU. Chưa chạy binary C++ chính thức nên cần đối chiếu binary trước khi công bố benchmark chính thức.',
             '\nRepository tác giả công bố F1 75,0 cho CULane ResNet18; cấu hình test của tác giả bật TTA. Đó là kết quả tham khảo của tác giả, không phải số đo sản phẩm này. [UFLD-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2), [nguồn và giấy phép CULane](https://xingangpan.github.io/projects/CULane.html). Dataset tải qua liên kết trong INSTALL.md, đủ 34.680 cặp ảnh/nhãn, commit nguồn c903880678454dfd9b55a63022368db05c00bc6d.',
             '\n## 3. Detector: ảnh gốc và suy giảm chất lượng',
             '\nNgưỡng confidence 0,35; ghép box một-một ở IoU ≥0,5. Đây là P/R/F1 tại ngưỡng runtime, khác precision/recall từ sweep AP.',
             '\n| Điều kiện | Precision | Recall | F1 | TP / FP / FN |\n|---|---:|---:|---:|---|']
    for name,c in conditions['conditions'].items():text.append(f'| {name} | {pct(c["precision"])} | {pct(c["recall"])} | {pct(c["f1"])} | {c["tp"]} / {c["fp"]} / {c["fn"]} |')
    text += ['\n## 4. Classifier: ảnh gốc và suy giảm chất lượng',
             '\n| Điều kiện | Accuracy tất cả | Macro-F1 | Tỷ lệ chấp nhận runtime | Accuracy trong số được chấp nhận |\n|---|---:|---:|---:|---:|']
    for name,c in recognizer['conditions'].items():text.append(f'| {name} | {pct(c["accuracy"])} | {pct(c["macro_f1"])} | {pct(c["runtime_coverage"])} | {pct(c["runtime_accepted_accuracy"])} |')
    text += ['\n![Ảnh hưởng chất lượng crop](recognizer_conditions.png)',
             '\nẢnh tối nhân cường độ 0,4; ảnh sáng nhân 1,8 và clip 255; Gaussian 9×9 sigma=2; JPEG quality=20; giảm đôi chiều rộng/cao. Detector áp dụng lên ảnh đầy đủ, classifier áp dụng trên crop chuẩn. Đây là stress test tổng hợp, không phải accuracy ngoài thực địa ban đêm và không đánh giá phản quang.',
             '\nRuntime classifier dùng confidence ≥0,5 và cạnh ngắn crop thực ≥32px. Crop thực dùng floor/ceil nên số accepted có thể khác phân nhóm theo kích thước bbox metadata. Accuracy cao ở nhóm accepted phải đọc cùng coverage; từ chối không được tính là dự đoán đúng. Mã biển ngoài 30 lớp chưa có OOD benchmark.',
             '\nChi tiết đủ 30 lớp: [recognizer_per_class.csv](recognizer_per_class.csv). Nhầm lẫn từng lớp: [ma trận](recognizer_confusion_matrix.png).',
             '\n## 5. Test âm tính có kỳ vọng không phát hiện',
             '\n| Input | Số biển | Số làn | Không phát hiện giả |\n|---|---:|---:|---|']
    for c in negative:text.append(f'| {c["case"]} | {c["sign_detections"]} | {c["lane_boundaries"]} | {"PASS" if c["pass_no_false_detections"] else "FAIL"} |')
    text += ['\nẢnh trắng hoàn toàn còn sinh làn giả: đây là lỗi thực tế chưa khắc phục trong lượt đánh giá cố định này. Ưu tiên kiểm tra chất lượng đầu vào và cơ chế từ chối; không được dùng confidence để kết luận ảnh có đường thật.',
             '\n## 6. Toàn bộ test case phần mềm',
             f'\nĐã chạy {len(cases)} test; {sum(c["status"]=="PASS" for c in cases)} PASS. Test phần mềm đạt không đồng nghĩa accuracy mô hình đạt yêu cầu. Một cảnh báo deprecation Starlette/AnyIO không làm test thất bại.',
             '\n| Nhóm | Test case | Trạng thái |\n|---|---|---|']
    for c in cases:text.append(f'| {c["group"]} | `{c["name"]}` | {c["status"]} |')
    text += ['\n## 7. Giới hạn và bước tiếp theo',
             '\nĐã chạy toàn bộ test hiện có và các nhóm điều kiện mô tả ở trên; không tuyên bố đã bao phủ mọi tình huống giao thông. Chưa có nhãn video Việt Nam để đo lane/sign accuracy trên test_video.mp4, nhãn biển chính/phụ, overhead và loại xe tự động; logic ngữ cảnh mới được kiểm thử bằng ca có metadata xác định. Kiểm thử TTS hiện là mock/cache/retry, không xác nhận uptime của dịch vụ trực tuyến.',
             '\nƯu tiên: sửa cơ chế từ chối ảnh trắng; đánh giá lại filter làm giảm recall làn; bổ sung ảnh mờ/biển nhỏ và hard negatives; kiểm chứng nhãn mã biển bằng người; kiểm thử ban đêm thực tế. Mọi thay đổi sau khi xem TEST này phải được xác nhận trên dữ liệu holdout mới để tránh báo cáo lạc quan.',
             '\nVideo đã được kiểm tra toàn thời lượng ở lượt trước (3.580 frame mẫu), xem `output/video_tests/test_video_final/verification.json`; không ghi nhận đó là lượt chạy lại trong phiên accuracy này. Các số latency/FPS trước đây không phải accuracy.',
             '\nTệp JSON/CSV, đường cong AP, confusion matrix, checkpoint hash và JUnit được giữ trong thư mục này để tái lập.']
    (OUT/'TEST_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(OUT/'TEST_REPORT.md'),'software_tests':len(cases),'culane_images':lane['evaluated_images']},indent=2))


if __name__=='__main__':main()
