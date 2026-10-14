"""Audit advertising/background confusion of selected recognizer (research only).

These are user-judged boxes from a development video. Most frames are not
human-confirmed: figures are *review proxies*, NOT independent FPR/recall.
Do not auto-tune/deploy thresholds using the same development video.
"""
from pathlib import Path
import json,cv2,collections,math,numpy as np,csv
from src.perception.sign_recognizer import SignRecognizer
from src.utils.types import SignDetection
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
OUT=PACK/'advertisement_audit_20261011'
OUT.mkdir(exist_ok=True)
data=json.loads((PACK/'autobox_20261011/autobox_curated_review.json').read_text(encoding='utf-8'))
net=SignRecognizer(load_config('perception'))
if net.model is None:raise RuntimeError('Selected recognizer unavailable')
rows=[]
for frame in data['frames']:
    image=cv2.imread(str(PACK/frame['image']))
    labels=[s for s in frame['signs'] if s['judgment'] in ('sign','not_sign')]
    predictions=[SignDetection(label='unknown',confidence=1.,bbox_xyxy=tuple(s['bbox_xyxy']))
                 for s in labels]
    predictions=net.recognize(image,predictions,min_short_side=16)
    for s,p in zip(labels,predictions):
        x1,y1,x2,y2=s['bbox_xyxy']
        rows.append({'frame':frame['id'],'id':s['id'],'review_judgment_UNVERIFIED':s['judgment'],
                     'frame_human_confirmed':frame.get('human_confirmed') is True,
                     'box_w':round(x2-x1,2),'box_h':round(y2-y1,2),
                     'original_code':s['label'],'predicted_code':p.label,
                     'prediction_conf':round(float(p.recognition_confidence or 0),5),
                     'recognition_source':p.recognition_source,
                     'source_kind':s.get('source','unknown'),
                     'ai_visual_note':s.get('ai_visual_note','')})
with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as fp:
    writer=csv.DictWriter(fp,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
neg=[r for r in rows if r['review_judgment_UNVERIFIED']=='not_sign']
pos=[r for r in rows if r['review_judgment_UNVERIFIED']=='sign']
# As code is already filtered at .63, a threshold >= .63 is meaningful;
# cannot recover rejected raw probability from recognizer output.
thresholds={}
for cutoff in (.63,.7,.8,.9):
    accepted=lambda r:r['predicted_code']!='unknown' and r['prediction_conf']>=cutoff
    thresholds[str(cutoff)]={
        'negative_boxes_accepted_proxy':sum(accepted(r) for r in neg),
        'positive_boxes_accepted_proxy':sum(accepted(r) for r in pos),
        'negative_total':len(neg),'positive_total':len(pos)}
# Save contact sheet for review: highest confidence suspect errors.
risky=sorted([r for r in neg if r['predicted_code']!='unknown'],
            key=lambda x:x['prediction_conf'],reverse=True)
for idx,r in enumerate(risky[:25]):
    f=next(x for x in data['frames'] if x['id']==r['frame'])
    sign=next(x for x in f['signs'] if x['id']==r['id'])
    im=cv2.imread(str(PACK/f['image']))
    x1,y1,x2,y2=sign['bbox_xyxy']
    w=x2-x1;h=y2-y1
    x1=max(0,math.floor(x1-.35*w));y1=max(0,math.floor(y1-.35*h))
    x2=min(im.shape[1],math.ceil(x2+.35*w));y2=min(im.shape[0],math.ceil(y2+.35*h))
    crop=im[y1:y2,x1:x2]
    if crop.size:cv2.imwrite(str(OUT/f'negative_review_{idx:02d}.jpg'),crop)
report={'development_review_only':True,
        'reviewed_source_groups':1,
        'independent_field_negative_test':False,
        'human_verified_box_count':sum(r['frame_human_confirmed'] for r in rows),
        'sign_box_count_development_proxy':len(pos),
        'not_sign_box_count_development_proxy':len(neg),
        'thresholds_on_same_development_source_NOT_FOR_DEPLOY':thresholds,
        'high_confidence_background_suspects':risky[:20],
        'status':'Research-only: rank likely false positives for human review, no model/threshold change'}
(OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='high_confidence_background_suspects'},ensure_ascii=False),flush=True)
