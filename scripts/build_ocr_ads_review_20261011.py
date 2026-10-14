"""Merge OCR and high-conf background review into a new, non-GT QA interface.

NO accepted user label/box is overwritten and OCR text is explicitly unverified.
"""
from pathlib import Path
from collections import Counter
import csv,json,re,hashlib,copy

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
BASE=PACK/'autobox_20261011'
OCR=PACK/'ocr_experiments_20261011'
ADS=PACK/'advertisement_audit_20261011'
source=json.loads((BASE/'autobox_roles_review.json').read_text(encoding='utf-8'))
new=copy.deepcopy(source)
ocr=json.loads((OCR/'results.json').read_text(encoding='utf-8'))
ads=json.loads((ADS/'summary.json').read_text(encoding='utf-8'))
candidates={(r['frame'],r['supplementary_id']):r for r in ocr['rows']}
background={(r['frame'],r['id']):r for r in ads['high_confidence_background_suspects']}
events=[]
for frame in new['frames']:
    for s in frame['signs']:
        key=(frame['id'],s['id'])
        if key in candidates:
            p=candidates[key]
            s['ai_ocr_raw_variants']=[
                {'view':v['view'],'text':v['text'],'scores':v['scores']}
                for v in p['variants']]
            s['ai_ocr_time_candidate']=p['ocr_agreed_time_candidate']
            s['ai_ocr_status']='stable_two_views' if p['agreement_2_of_3'] else 'unresolved'
            s['ai_ocr_verified']=False
            s['ai_ocr_human_corroboration_required']=True
            events.append({'frame':frame['id'],'box':s['id'],
                'priority':1 if p['agreement_2_of_3'] else 3,
                'kind':'OCR_REVIEW',
                'model_text':'; '.join(p['ocr_agreed_time_candidate']) or 'NOT_RELIABLE',
                'visual_hint_not_gt':p['assistant_visual_hint_NOT_GT']})
        if key in background:
            p=background[key]
            assert s['judgment']=='not_sign'
            s['ai_advertising_false_accept_alert']=True
            s['ai_background_confusing_code']=p['predicted_code']
            s['ai_background_confusing_confidence']=p['prediction_conf']
            events.append({'frame':frame['id'],'box':s['id'],
                'priority':0,'kind':'HIGH_CONF_BACKGROUND',
                'model_text':p['predicted_code'],
                'recognition_confidence':p['prediction_conf'],
                'visual_hint_not_gt':p['ai_visual_note']})
    assert frame['human_confirmed'] is False

file=PACK/'ai_ocr_advertising_review_20261011.json'
file.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
base_ui=(PACK/'index_AI_bien_chinh_bien_phu.html').read_text(encoding='utf-8')
embedded=json.dumps(new,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
base_ui,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:'const manifest='+embedded+';\nconst original=',base_ui,count=1,flags=re.S)
assert n==1
# Labels are notes only, not imported to sign class or applicability.
needle="""+' | AI time UNVERIFIED: '+esc(s.ai_suggested_time_text||'not decoded')"""
assert needle in base_ui
addition="""+' | OCR: '+esc((s.ai_ocr_time_candidate||[]).join(', ')||s.ai_ocr_status||'n/a')+
 ' | NOT_SIGN high-conf issue: '+esc(s.ai_advertising_false_accept_alert
   ? (s.ai_background_confusing_code+' @ '+s.ai_background_confusing_confidence):'none')"""
base_ui=base_ui.replace(needle,needle+addition,1)
base_ui=base_ui.replace('AI QA →','AI QA →',1)
base_ui=base_ui.replace('</header>','<span style="color:#ffce80;font-weight:bold">OCR + quảng cáo: chưa xác minh / chỉ gợi ý</span></header>',1)
htmlfile=PACK/'index_AI_OCR_va_loc_quang_cao.html'
if htmlfile.exists():raise FileExistsError('Review UI exists; refusing overwrite')
htmlfile.write_text(base_ui,encoding='utf-8')
events.sort(key=lambda r:(r['priority'],r['frame']))
with (PACK/'ai_ocr_advertising_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
    names=['frame','box','priority','kind','model_text','recognition_confidence','visual_hint_not_gt']
    w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(events)
report={'frames':len(new['frames']),
    'total_boxes':sum(len(f['signs']) for f in new['frames']),
    'ocr_panel_research_regions':len(ocr['rows']),
    'ocr_stable_candidate_panels':sum(bool(r['agreement_2_of_3']) for r in ocr['rows']),
    'background_high_conf_suspects':len(background),
    'human_verified_frames':0,
    'modified_existing_user_judgments':False,
    'model_deployed_changed':False,'independent_test':False,
    'OCR_model_source':'RapidOCR ONNX CPU, isolated environment .ocr-venv',
    'source_group':'test_video_development',
    'files':{'review_ui':str(htmlfile),'qa_queue':str(PACK/'ai_ocr_advertising_review_queue.csv')}}
(PACK/'ai_ocr_advertising_review_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
