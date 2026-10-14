"""Non-destructive AI audit of sign vs supplementary roles and unique parent matching.

Changes only a new review DRAFT, never original user review / selected checkpoints.
No OCR text or legal meaning inferred. Reviewer must validate all suggestions.
"""
from pathlib import Path
import copy,json,cv2,collections,re
from src.perception.supplementary_panel_proposals import bbox_iou

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
BASE=PACK/'autobox_20261011'
src=json.loads((BASE/'autobox_curated_review.json').read_text(encoding='utf-8'))
result=copy.deepcopy(src)
# Explicit source-frame geometry inspection; these are independent primary panels, not
# secondary plates: blue lane allocation boards or circular no-parking sign.
role_fixes={
    ('frame_001','s2'):('primary','Blue lane-use allocation sign, not a white supplementary panel'),
    ('frame_006','s4'):('primary','Blue vehicle/lane allocation sign, NOT white subplate'),
    ('frame_012','added_direction'):('primary','Overhead blue direction guide sign, not supplementary plate'),
    ('frame_013','s2'):('primary','Blue circular prohibition sign; distinct sign, not white hours subplate'),
    ('frame_023','s2'):('primary','Blue rectangular lane/vehicle allocation sign below a prohibition sign'),
}
audit=[];stats=collections.Counter()
def choose_parent(sub, mains):
    x1,y1,x2,y2=sub['bbox_xyxy'];sw=x2-x1;sh=y2-y1
    if sw<=0 or sh<=0:return None,'invalid bbox'
    scores=[]
    for main in mains:
        if main['id']==sub['id']:continue
        a,b,c,d=main['bbox_xyxy'];mw=c-a;mh=d-b
        if mw<=0 or mh<=0:continue
        overlap=max(0,min(x2,c)-max(x1,a))/max(1,min(sw,mw))
        gap=y1-d
        center=abs((x1+x2)-(a+c))/max(1,mw+sw)
        if overlap<.44 or not (-.18*mh<=gap<=1.1*mh):continue
        # Same-sign overlap shouldn't be considered a separate panel.
        if bbox_iou(sub['bbox_xyxy'],main['bbox_xyxy'])>.32:continue
        score=max(0,gap)/mh+.65*center+.15*abs(sw-mw)/max(sw,mw)
        scores.append((score,main['id'],round(overlap,3),round(gap,2)))
    scores.sort()
    if not scores:return None,'no geometrically aligned main sign'
    if len(scores)>1 and scores[1][0]-scores[0][0]<.14:
        return None,'multiple equally plausible parent signs'
    return scores[0][1],'unique geometry candidate; legal/semantic meaning unresolved'

for frame in result['frames']:
    if frame['human_confirmed']:
        frame['ai_role_previous_confirmation']={'reviewer':frame.get('reviewer'),
                                                 'reviewed_at':frame.get('reviewed_at')}
        frame['human_confirmed']=False;frame['reviewed_at']=None
    for sign in frame['signs']:
        item=(frame['id'],sign['id'])
        if sign['judgment']=='sign' and sign['label']=='':
            sign['ai_original_empty_code']=''
            sign['label']='unknown'
            sign['ai_role_needs_revalidation']=True
            stats['blank_code_normalized_unknown']+=1
        if item in role_fixes:
            suggested,note=role_fixes[item]
            assert sign['judgment']=='sign' and sign['panel_role']=='supplementary',item
            sign['ai_original_panel_role']=sign['panel_role']
            sign['panel_role']=suggested
            sign['ai_role_change_reason']=note
            sign['ai_role_needs_revalidation']=True
            stats['misclassified_primary_role_proposals']+=1
            audit.append({'frame':frame['id'],'box_id':sign['id'],'action':'ROLE_PROPOSED_PRIMARY',
                          'explanation':note,'old_role':'supplementary'})
    mains=[s for s in frame['signs'] if s['panel_role']=='primary' and s['judgment']=='sign']
    for sign in frame['signs']:
        if sign['panel_role']!='supplementary' or sign['judgment']!='sign':continue
        if sign['label'] in ('P.131a','P.130','P.127'):
            sign['ai_previous_invalid_panel_code']=sign['label']
            sign['label']='unknown'
            sign['ai_role_needs_revalidation']=True
            stats['primary_code_on_plate_suppressed']+=1
        parent,reason=choose_parent(sign,mains)
        sign['ai_panel_parent_candidate_id']=parent
        sign['ai_panel_parent_reason']=reason
        sign['ai_plate_text_verified']=False
        sign['ai_panel_semantics']='unknown'
        if parent:stats['plate_parent_unique']+=1
        else:stats['plate_parent_unresolved']+=1
        audit.append({'frame':frame['id'],'box_id':sign['id'],'action':'PLATE_PARENT_CANDIDATE',
                      'parent_id':parent,'explanation':reason})
    frame['all_visible_signs_annotated']=False if any(s.get('ai_role_needs_revalidation') for s in frame['signs']) else frame.get('all_visible_signs_annotated',False)
    # Hand-read text candidates from full-resolution supplied images; NOT OCR or legal interpretation.
    # This is an assistant annotation to VERIFY, never an approved temporal restriction.
    visual_time_hints={
        ('frame_013','added_time'):'6:00-22:00',
        ('frame_021','manual_1791687609125_1'):'6:00-22:00',
        ('frame_022','manual_1791687638009_1'):'6:00-22:00',
        ('frame_022','manual_1791687659050_3'):'6:00-8:00; 16:00-18:00',
    }
    for sign in frame['signs']:
        hint=visual_time_hints.get((frame['id'],sign['id']))
        if hint is not None:
            assert sign['panel_role']=='supplementary' and sign['judgment']=='sign'
            sign['ai_suggested_time_text']=hint
            sign['ai_time_text_verified']=False
            sign['ai_time_text_provenance']='assistant_visual_transcription_unreviewed'
            stats['time_text_candidate_unverified']+=1


out=BASE/'autobox_roles_review.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'role_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
stats['frames']=len(result['frames'])
stats['total_boxes']=sum(len(f['signs']) for f in result['frames'])
stats['human_confirmed']=sum(f.get('human_confirmed') for f in result['frames'])
stats['independent_GT']=0
(BASE/'role_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
html=(PACK/'index_AI_tu_bat_box.html').read_text(encoding='utf-8')
encoded=json.dumps(result,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
html,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:'const manifest='+encoded+';\nconst original=',html,count=1,flags=re.S)
assert n==1
small="""+' | AI role: '+esc(s.ai_role_change_reason||'')+' | AI parent: '+esc(s.ai_panel_parent_candidate_id||'unresolved')+' | AI time UNVERIFIED: '+esc(s.ai_suggested_time_text||'not decoded')"""
needle="""+' | '+esc(s.ai_auto_visual_note||'')"""
if needle not in html:raise RuntimeError('UI version changed')
html=html.replace(needle,needle+small,1)
(PACK/'index_AI_bien_chinh_bien_phu.html').write_text(html,encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False),flush=True)
