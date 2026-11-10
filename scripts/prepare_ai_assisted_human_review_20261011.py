"""Generate AI-assisted human review UI, keeping all choices UNREVIEWED by default."""
from pathlib import Path
import re,json,csv
root=Path(r'D:\itas-mm\output\review\field_review_pack_20261011')
folder=root/'ai_visual_review'
assisted=json.loads((folder/'ai_assessment.json').read_text(encoding='utf-8'))
ui=(root/'index.html').read_text(encoding='utf-8')
embedded='const manifest='+json.dumps(assisted,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+';'
ui,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:embedded+'\nconst original=',ui,count=1,flags=re.S)
if n!=1:raise RuntimeError('Could not patch review manifest')
needle="""<small class="note"> Gợi ý AI: '+esc(s.ai_suggested_label||'mới')+'</small>"""
new="""<small class="note"> Mã ban đầu: '+esc(s.ai_original_label||s.ai_suggested_label||'mới')+' | <b>Nhận xét AI (CHƯA xác minh): '+esc(s.ai_visual_assessment||'chưa rõ')+'</b> | Mã gợi ý: '+esc(s.ai_label_suggestion||'unknown')+' | '+(s.ai_bbox_needs_correction?'Cần sửa tọa độ box | ':'')+esc(s.ai_visual_note||'')+'</small>"""
assert needle in ui
ui=ui.replace(needle,new,1)
ui=ui.replace('Nguồn VIDEO PHÁT TRIỂN — không phải independent TEST','Gợi ý thị giác từ AI, NGƯỜI DUYỆT cần xác minh từng box; KHÔNG phải ground truth')
(root/'index_ai_assisted.html').write_text(ui,encoding='utf-8')
with (folder/'ai_assessment.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['frame','time_s','box_id','bbox_xyxy','source','original_code','ai_visual_status','ai_code_suggestion','visually_supported_code','needs_rebox','ai_panel_role_suggestion','note'])
    writer.writeheader()
    for frame in assisted['frames']:
        for s in frame['signs']:
            writer.writerow({'frame':frame['id'],'time_s':frame['timestamp_seconds'],
            'box_id':s['id'],'bbox_xyxy':json.dumps(s['bbox_xyxy']),
            'source':s['source'],'original_code':s['ai_original_label'],
            'ai_visual_status':s['ai_visual_assessment'],
            'ai_code_suggestion':s['ai_label_suggestion'],
            'visually_supported_code':s['ai_code_visually_supported'],
            'needs_rebox':s['ai_bbox_needs_correction'],
            'ai_panel_role_suggestion':s['ai_panel_role_suggestion'],
            'note':s['ai_visual_note']})
print(json.dumps({'review_ui':str(root/'index_ai_assisted.html'),
                 'csv':str(folder/'ai_assessment.csv'),'human_confirmed':0},ensure_ascii=False))
