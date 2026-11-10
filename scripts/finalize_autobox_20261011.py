"""Human-safe curation of AI auto boxes with scene-level assistant visual review.

This is a source-specific second-stage curation for the 24 known frames.
It does NOT purport to be a reusable ad-vs-sign classifier or human GT.
"""
from pathlib import Path
from collections import Counter
import copy,json,re,cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
FOLDER=PACK/'autobox_20261011'
proposal=json.loads((FOLDER/'autobox_review_draft.json').read_text(encoding='utf-8'))
original=json.loads((PACK/'itasmm_human_review_uploaded_20261011.json').read_text(encoding='utf-8'))

# Frame-specific scene assessments from examining the annotated full frame.
# This is NOT independently human verified and MUST NEVER become GT automatically.
triage={
    'frame_000':['uncertain','Far left, 10-px narrow region, insufficient readable sign pixels'],
    'frame_005':['sign','Blue vehicle/lane-use panel to the left of the stacked prohibition signs'],
    'frame_006':['uncertain','Vertical structure or sign above the existing stacked signs, unreadable'],
    'frame_007':['not_sign','Blue storefront/commercial symbol, not a traffic restriction'],
    'frame_008':['not_sign','Vertical commercial storefront/brand light board'],
    'frame_013':['sign','Circular prohibition sign left of the street corner, missed/misaligned by earlier boxes'],
    'frame_018':['not_sign','Small roadside advertising or storefront object'],
    'frame_019':['not_sign','Commercial shopfront sign/advertisement'],
}
assert len(proposal['frames'])==len(original['frames'])==24
final=copy.deepcopy(proposal)
rejected=[];review_tasks=[];statuses=Counter()
for i,(frame,old) in enumerate(zip(final['frames'],original['frames'])):
    kept=[]
    if frame['id']!=old['id']:raise AssertionError('Source frame order changed')
    for s in frame['signs']:
        if s.get('source')!='ai_autobox_20261011':
            kept.append(s)
            if s.get('ai_auto_bbox_changed'):
                review_tasks.append({'frame':frame['id'],'box_id':s['id'],
                    'type':'CHECK_SNAPPED_BOX','old_box':s['ai_auto_original_bbox'],
                    'new_box':s['bbox_xyxy']})
            continue
        assessment,reason=triage[frame['id']]
        statuses[assessment]+=1
        s['ai_auto_visual_assessment']=assessment
        s['ai_auto_visual_note']=reason
        s['ai_auto_label_provenance']='assistant_visual_nonhuman'
        if assessment=='not_sign':
            rejected.append({'frame':frame['id'],**s,'decision':'excluded_from_main_review_not_training_GT'})
            continue
        if assessment=='sign':
            # Save visual AI suggestion but do not call it human review.
            s['ai_auto_visual_label_candidate']=True
        else:
            s['ai_auto_visual_label_candidate']=False
        kept.append(s)
        review_tasks.append({'frame':frame['id'],'box_id':s['id'],
                             'type':'VERIFY_NEW_AI_BOX','visual':assessment,'bbox':s['bbox_xyxy']})
    frame['signs']=kept
    if old['human_confirmed'] and (len(kept)!=len(old['signs']) or any(s.get('ai_auto_bbox_changed') for s in kept)):
        assert frame['human_confirmed'] is False
    if frame['id']=='frame_023':
        assert 'ai_auto_previous_human_confirmation' in frame
    # Even the 2 likely-real new sign boxes are recommendations only.
    assert all(s['judgment']=='unreviewed' for s in kept if s.get('source')=='ai_autobox_20261011')
    assert all(s['ai_auto_visual_assessment']!='not_sign' for s in kept if s.get('source')=='ai_autobox_20261011')

dest=FOLDER/'autobox_curated_review.json'
dest.write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
(FOLDER/'rejected_model_candidates.json').write_text(json.dumps({
    'provenance':'Assistant visually rejected, NOT human verified or approved for training','items':rejected},
    ensure_ascii=False,indent=2),encoding='utf-8')
(FOLDER/'review_tasks.json').write_text(json.dumps(review_tasks,ensure_ascii=False,indent=2),encoding='utf-8')

# New review UI, user choices preserved, AI additions marked unreviewed.
template=(PACK/'index_tiep_tuc_nhan_da_duyet.html').read_text(encoding='utf-8')
serialized=json.dumps(final,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
template,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:'const manifest='+serialized+';\nconst original=',template,count=1,flags=re.S)
assert n==1
template=template.replace('Tiếp tục nhãn đã lưu: chỉ những khung được bấm xác nhận mới là human-confirmed.',
                          'AI đã khoanh lại 32 box và gợi ý các box mới. MỌI BOX AI MỚI chưa được con người xác nhận.')
new="""<small class="note"> '+(s.ai_auto_bbox_changed?'✅ YOLO chỉnh box | ':'')+(s.source==='ai_autobox_20261011'?'🔎 BOX MỚI TỪ AI | ':'')+ 'Mã tham khảo: '+esc(s.ai_auto_suggested_code||s.ai_suggested_label||'unknown')+' | conf YOLO: '+esc(s.ai_auto_yolo_confidence||'n/a')+' | '+esc(s.ai_auto_visual_note||'')+'</small>"""
template,n=re.subn(r'<small class="note">.*?</small>',lambda m:new,template,count=1,flags=re.S)
if n!=1:raise RuntimeError('Template label field changed; cannot safely generate UI')
template=template.replace("""ctx.strokeStyle=s.judgment==='sign'?'#54eb92'""",
                          """ctx.strokeStyle=s.source==='ai_autobox_20261011'?'#49d6fb':s.ai_auto_bbox_changed?'#56eb91':s.judgment==='sign'?'#54eb92'""")
(PACK/'index_AI_tu_bat_box.html').write_text(template,encoding='utf-8')
# Updated overlays based on final decisions; prior preview kept separately.
preview=[]
for f in final['frames']:
    im=cv2.imread(str(PACK/f['image']))
    for s in f['signs']:
        if s.get('source')!='ai_autobox_20261011' and not s.get('ai_auto_bbox_changed'):continue
        x1,y1,x2,y2=[int(round(v)) for v in s['bbox_xyxy']]
        color=(240,216,45) if s.get('source')=='ai_autobox_20261011' else (50,223,70)
        cv2.rectangle(im,(x1,y1),(x2,y2),color,2)
        cv2.putText(im,('AI NEW '+s['ai_auto_visual_assessment']) if s.get('source')=='ai_autobox_20261011' else 'AI REBOX',
                    (max(0,x1),max(18,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1)
    preview.append(cv2.resize(im,(640,360)))
for j in range(0,len(preview),4):
    cv2.imwrite(str(FOLDER/f'curated_contact_{j//4+1:02d}.jpg'),
                np.vstack([np.hstack(preview[j:j+2]),np.hstack(preview[j+2:j+4])]))
report={'frame_count':len(final['frames']),
        'existing_user_boxes':sum(len(f['signs']) for f in original['frames']),
        'auto_refined_existing_boxes':sum(s.get('ai_auto_bbox_changed',False) for f in final['frames'] for s in f['signs']),
        'raw_candidate_new':sum(statuses.values()),
        'visually_rejected_new':len(rejected),
        'new_kept_for_review':sum(s.get('source')=='ai_autobox_20261011' for f in final['frames'] for s in f['signs']),
        'new_triage':dict(statuses),
        'total_boxes_in_review':sum(len(f['signs']) for f in final['frames']),
        'human_confirmed_frames':sum(f.get('human_confirmed') is True for f in final['frames']),
        'lost_prior_confirmation_due_to_ai_changes':1,
        'all_ai_new_boxes_are_unreviewed':True,
        'ground_truth':False,
        'deployed_model_changed':False,
        'source':'video development; same original 24 frames'}
(FOLDER/'curated_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
