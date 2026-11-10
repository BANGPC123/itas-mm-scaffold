"""Apply the recorded visual review of the 24 development frames, preserving proposals."""
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'output/review/test_video_development'
OUT = ROOT / 'output/review/test_video_ai_review'

# Each decision refers to the original frame/sign identifiers, not a new model prediction.
DECISIONS = {
0: [('P.131a','Một vạch chéo trên nền xanh; cấm đỗ'),('P.130','Hai vạch chéo trên nền xanh; cấm dừng và đỗ'),(None,'Cabin xe tải, không phải biển')],
1: [('P.130','Dấu chéo đỏ trên nền xanh'),('unknown','Biển xanh chỉ hướng theo làn; không phải đường một chiều'),('P.117','Biển giới hạn chiều cao trên cầu; chưa chép trị số')],
2: [('P.124d','Ký hiệu cấm rẽ phải và quay đầu'),('P.130','Dấu chéo đỏ'),('P.131a','Một vạch chéo đỏ')],
3: [(None,'Bảng quảng cáo cửa hàng'),(None,'Bảng quảng cáo cửa hàng'),('unknown','Biển xanh chỉ hướng theo làn; không phải chỗ quay xe')],
4: [('unknown','Bảng chữ xanh hướng dẫn tại đèn tín hiệu; không phải tốc độ tối đa'),(None,'Đèn tín hiệu màu vàng'),('P.130','Dấu chéo đỏ'),('P.131a','Một vạch chéo đỏ')],
5: [('P.130','Dấu chéo đỏ'),('P.131a','Một vạch chéo đỏ'),(None,'Logo đỏ trên mặt tiền cửa hàng')],
6: [('P.130','Dấu chéo đỏ'),('P.131a','Một vạch chéo đỏ'),(None,'Logo tròn vàng cửa hàng'),('unknown','Cụm biển xanh phân làn theo loại xe; ngoài bộ lớp hiện tại'),(None,'Chữ quảng cáo cửa hàng')],
7: [(None,'Logo xanh cửa hàng, không phải vòng xuyến'),('P.131a','Một vạch chéo đỏ'),('unknown','Biển tròn xanh viền đỏ bị lá che; chưa phân biệt chắc vạch chéo'),(None,'Cột A và vùng cảnh rộng, không phải biển')],
8: [('P.131a','Một vạch chéo đỏ'),('P.130','Dấu chéo đỏ'),('unknown','Biển tròn cấm có biểu tượng xe; chưa đọc rõ loại xe')],
9: [(None,'Đèn tín hiệu đỏ'),(None,'Hình/bảng quảng cáo dưới tán cây')],
10: [('P.131a','Một vạch chéo đỏ'),('P.130','Dấu chéo đỏ'),('unknown','Biển cấm xe tải; không phải cấm vượt; ngoài 30 lớp'),('unknown','Cụm biển xanh phân làn theo loại xe; không phải cấm dừng đỗ')],
11: [('unknown','Biển xanh chỉ hướng; quá nhỏ để xác định mã'),(None,'Mái che màu đỏ của cửa hàng'),('unknown','Vùng biển tròn đỏ quá mờ để xác định nội dung')],
12: [(None,'Logo quảng cáo trên bảng đỏ'),(None,'Logo quảng cáo trên bảng đỏ')],
13: [('unknown','Biển cấm xe tải có trị số tải trọng; không phải tốc độ tối đa'),('P.131a','Một vạch chéo đỏ'),('P.131a','Một vạch chéo đỏ'),('unknown','Biển cấm xe tải; ngoài bộ lớp hiện tại'),('unknown','Cụm biển xanh phân làn theo loại xe'),('unknown','Cụm biển xanh phân làn theo loại xe'),(None,'Box trùng một phần cụm biển s5; không phải chỗ quay xe')],
14: [('unknown','Cấm rẽ trái với biểu tượng ô tô; không ép thành lớp cấm rẽ trái chung'),('unknown','Biển tròn xanh viền đỏ nhìn nghiêng; chưa đọc rõ'),('unknown','Biển tròn đỏ nhỏ nhìn nghiêng; chưa đọc rõ')],
15: [('P.131a','Một vạch chéo đỏ'),(None,'Bảng quảng cáo cửa hàng'),(None,'Đèn tín hiệu đỏ'),(None,'Logo quảng cáo cửa hàng')],
16: [('P.131a','Một vạch chéo đỏ'),('unknown','Bảng chữ xanh nhỏ; chưa xác định nội dung')],
17: [('P.130','Dấu chéo đỏ, biển bị che một phần'),(None,'Bảng chữ vàng của cửa hàng')],
18: [('unknown','Biển điểm dừng xe buýt; ngoài 30 lớp'),('unknown','Biển tam giác cảnh báo nhỏ; chưa đọc chắc biểu tượng'),(None,'Bảng quảng cáo đỏ của cửa hàng')],
19: [('unknown','Biển điểm dừng xe buýt; không phải tốc độ tối đa'),(None,'Cột A trong xe; không phải dốc lên nguy hiểm'),('unknown','Biển tam giác cảnh báo nhỏ; chưa đọc chắc biểu tượng')],
20: [(None,'Cột A trong xe'),(None,'Đồng hồ trên taplo; không phải biển tốc độ')],
21: [('P.130','Dấu chéo đỏ'),('unknown','Biển cấm ô tô; ngoài 30 lớp'),('W.225','Hình hai trẻ em trên biển tam giác')],
22: [('P.130','Dấu chéo đỏ'),('unknown','Biển cấm ô tô; không phải cấm rẽ trái; ngoài 30 lớp'),('W.225','Hình hai trẻ em trên biển tam giác')],
23: [('P.131a','Một vạch chéo đỏ'),('unknown','Cụm biển xanh phân làn theo loại xe; không phải người đi bộ')],
}

def main():
    original = json.loads((SOURCE / 'draft.json').read_text(encoding='utf-8'))
    packet = copy.deepcopy(original)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE / 'images', OUT / 'images', dirs_exist_ok=True)
    changes = []
    for i, sample in enumerate(packet['samples']):
        decisions = DECISIONS[i]
        assert len(decisions) == len(sample['signs'])
        kept = []
        for sign, (label, reason) in zip(sample['signs'], decisions):
            action = 'removed' if label is None else 'relabelled' if label != sign['label'] else 'retained'
            changes.append({'frame': sample['id'], 'sign': sign['id'], 'action': action,
                            'before': copy.deepcopy(sign), 'after_label': label, 'reason': reason})
            if label is None:
                continue
            sign['label'] = label
            sign['visual_review_note'] = reason
            sign['annotation_source'] = 'ai_visual_review'
            sign['panel_role'] = 'primary' if label != 'unknown' else 'unknown'
            sign['overhead'] = True if i == 1 and sign['id'] == 's3' else None
            kept.append(sign)
        sample['signs'] = kept
        sample.update(human_confirmed=False, reviewer='', reviewed_at=None,
                      ai_reviewed=True, ai_review_date='2026-10-09',
                      notes='AI visual review of existing boxes plus selected visible omissions. '
                            'Not exhaustive ground truth. Lanes and applicability remain unverified.')

    def add(frame, sid, box, note, parent=None, text=None):
        sign = dict(id=sid, label='unknown', bbox_xyxy=box,
                    panel_role='supplementary' if parent else 'primary', allowed_vehicles=[],
                    overhead=None, parent_id=parent, expected_applicability='uncertain',
                    visual_review_note=note, annotation_source='ai_visual_review')
        if text:
            sign['observed_text'] = text
        packet['samples'][frame]['signs'].append(sign)
        changes.append(dict(frame=f'frame_{frame:03d}', sign=sid, action='added', after=sign, reason=note))

    add(12, 'added_direction', [646, 9, 753, 107], 'Biển xanh chỉ hướng địa danh bị bỏ sót; không tách biểu tượng in trong bảng thành biển độc lập')
    add(13, 'added_time', [1090, 108, 1134, 127], 'Biển phụ thời gian dưới biển xe tải; chữ nhỏ cần kiểm tra lại', 's1')
    packet['samples'][13]['signs'][0]['panel_role'] = 'primary'
    for frame, boxes in [(21, [[47,350,115,391],[219,317,291,345]]),
                         (22, [[218,326,281,364],[368,299,439,325]])]:
        for sid in ['s1', 's2']:
            next(s for s in packet['samples'][frame]['signs'] if s['id'] == sid)['panel_role'] = 'primary'
        add(frame, 'added_time_stop', boxes[0], 'Biển phụ thời gian dưới biển cấm dừng đỗ', 's1', '6:00-8:00; 16:00-18:00')
        add(frame, 'added_time_car', boxes[1], 'Biển phụ thời gian dưới biển cấm ô tô', 's2', '6:00-22:00')

    packet['review_provenance'] = {
        'type': 'ai_visual_review', 'date': '2026-10-09', 'human_verified': False,
        'scope': 'All 24 central frames and all 76 original boxes visually inspected; '
                 'neighbor crops inspected for frames 0,2,4,6,7,14,18,23. Six selected omissions added. '
                 'Not exhaustive lane, sign, or applicability ground truth.',
        'training_use': 'Do not automatically promote these annotations to verified training or test labels.'}
    (OUT / 'ai_reviewed.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
    template = (ROOT / 'scripts/video_review_template.html').read_text(encoding='utf-8')
    template = template.replace('Rà soát nhãn video — Mã và tên biển', 'Nhãn đã được AI rà soát — Chưa có xác nhận của người')
    template = template.replace('<p id="purpose">', '<p>Đã kiểm tra trực quan 24 ảnh và 76 box gốc; các ghi chú giải thích quyết định được hiển thị bên dưới. Biển ngoài bộ lớp giữ mã unknown. Chưa xác nhận làn hoặc biển áp dụng.</p><p id="purpose">')
    template = template.replace('label.append(select);target.append(label)', "label.append(select);if(s.visual_review_note){let note=document.createElement('span');note.textContent=' — '+s.visual_review_note;label.append(note)}target.append(label)")
    (OUT / 'index.html').write_text(template.replace('__PACKET__', json.dumps(packet, ensure_ascii=False).replace('<', '\\u003c')), encoding='utf-8')
    counts = Counter(c['action'] for c in changes)
    summary = dict(frames=24, original_boxes=76, actions=dict(counts),
                   remaining_boxes=sum(len(s['signs']) for s in packet['samples']), human_verified=0)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
