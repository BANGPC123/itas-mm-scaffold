"""Resume a user's saved labeling session without granting human confirmation to unfinished frames."""
from pathlib import Path
from collections import Counter
import argparse,csv,json,re,hashlib

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
from src.evaluation.label_review_schema import validate_review

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--review',type=Path,default=PACK/'itasmm_human_review_uploaded_20261011.json')
    args=p.parse_args()
    source=json.loads((PACK/'review_manifest.json').read_text(encoding='utf-8'))
    submitted=json.loads(args.review.read_text(encoding='utf-8'))
    summary=validate_review(submitted,source)
    review_html=PACK/'index_ai_assisted.html'
    html=review_html.read_text(encoding='utf-8')
    encoded=json.dumps(submitted,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    replacement='const manifest='+encoded+';'
    html,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:replacement+'\nconst original=',html,count=1,flags=re.S)
    if n!=1:raise RuntimeError('Cannot embed saved work')
    assert 'const original=' in html
    html=html.replace('Nguồn VIDEO PHÁT TRIỂN — không phải independent TEST',
                      'Tiếp tục nhãn đã lưu: chỉ những khung được bấm xác nhận mới là human-confirmed.')
    html=html.replace("$('counter').textContent=(idx+1)+'/'+frames.length",
        "$('counter').textContent=(idx+1)+'/'+frames.length+' | '+frames.filter(x=>x.human_confirmed).length+'/'+frames.length+' đã xác nhận'")
    html=html.replace(
        "if(f.signs.some(s=>s.judgment==='unreviewed')){say('Vẫn còn box chưa được phân loại');return}",
        "if(f.signs.some(s=>s.judgment==='unreviewed')){say('Vẫn còn box chưa được phân loại');return}"
        "if(f.signs.some(s=>s.judgment==='sign' && !String(s.label||'').trim())){say('Có biển thật đang để trống nhãn; nhập unknown nếu chưa đọc rõ');return}")
    out=PACK/'index_tiep_tuc_nhan_da_duyet.html'
    if out.exists():raise FileExistsError('Resume UI already exists; refuse overwrite')
    out.write_text(html,encoding='utf-8')

    notes=[]
    reference={(f['id'],s['id']):s for f in source['frames'] for s in f['signs']}
    for f in submitted['frames']:
        if not f['human_confirmed']:
            notes.append({'frame':f['id'],'box_id':'','issue':'FRAME_UNCONFIRMED','details':'Cần bấm xác nhận sau khi duyệt'})
        if f['lane_status']=='unreviewed':
            notes.append({'frame':f['id'],'box_id':'','issue':'LANE_STATUS_UNREVIEWED','details':'Chọn tình trạng làn'})
        for s in f['signs']:
            if s['judgment']=='sign' and not str(s.get('label','')).strip():
                notes.append({'frame':f['id'],'box_id':s['id'],'issue':'EMPTY_SIGN_LABEL','details':'Sửa mã/nhập unknown nếu chưa đọc rõ'})
            if (s.get('ai_visual_assessment') and s['judgment'] != s['ai_visual_assessment']):
                notes.append({'frame':f['id'],'box_id':s['id'],'issue':'CONFLICT_WITH_AI_SUGGESTION',
                              'details':'Người duyệt cần kiểm tra lại; AI có thể sai, không tự thay lựa chọn đã lưu'})
            if s.get('ai_bbox_needs_correction') and s['judgment']=='sign':
                orig=reference.get((f['id'],s['id']))
                if orig is not None and orig['bbox_xyxy']==s['bbox_xyxy']:
                    notes.append({'frame':f['id'],'box_id':s['id'],'issue':'SIGN_BBOX_UNCHANGED_FLAGGED_FOR_REVIEW',
                                  'details':'Box từng được AI đánh dấu là lệch; hãy xem lại tọa độ'})
    csv_path=PACK/'validation_uploaded_20261011_issues.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=['frame','box_id','issue','details'])
        w.writeheader();w.writerows(notes)
    report={
        'source_file':str(args.review),
        'source_hash_sha256':hashlib.sha256(args.review.read_bytes()).hexdigest(),
        'source_group':'test_video_development',
        'independent_field_test':False,
        'resume_ui':str(out),
        'validation':summary,
        'label_counts':dict(Counter(s['judgment'] for f in submitted['frames'] for s in f['signs'])),
        'manual_drawn':sum(s.get('source')=='human_drawn' for f in submitted['frames'] for s in f['signs']),
        'issues':dict(Counter(x['issue'] for x in notes)),
        'issues_csv':str(csv_path),
        'no_model_training_or_checkpoint_change':True,
    }
    (PACK/'validation_uploaded_20261011_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
