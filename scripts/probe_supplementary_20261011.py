"""Evaluate supplementary candidate boxes against unverified user-drawn plates.

Proxy only, not accuracy/GT; intentionally no checkpoint changes.
"""
import json,cv2,collections,numpy as np
from pathlib import Path
from src.perception.supplementary_panel_proposals import propose_plates,bbox_iou

root=Path(__file__).resolve().parents[1]
pack=root/'output/review/field_review_pack_20261011'
data=json.loads((pack/'autobox_20261011/autobox_curated_review.json').read_text(encoding='utf-8'))
out=pack/'supplementary_probe_20261011'
out.mkdir(exist_ok=True)
found=0;total=0;allproposals=0;extras=[];hits=[];panels=[]
for f in data['frames']:
    image=cv2.imread(str(pack/f['image']))
    proposals=[]
    primaries=[s for s in f['signs'] if s.get('panel_role')=='primary' and s.get('judgment')=='sign']
    truth=[s for s in f['signs'] if s.get('panel_role')=='supplementary' and s.get('judgment')=='sign']
    total+=len(truth)
    for main in primaries:
        for c in propose_plates(image,main['bbox_xyxy']):
            box=c['bbox_xyxy']
            if any(bbox_iou(box,p['bbox_xyxy'])>.5 for p in proposals):continue
            proposals.append({**c,'parent_id':main['id']})
    allproposals+=len(proposals)
    for gt in truth:
        best=max([bbox_iou(gt['bbox_xyxy'],p['bbox_xyxy']) for p in proposals] or [0])
        if best>.3:found+=1
        hits.append({'frame':f['id'],'plate':gt['id'],'best_iou':round(best,2)})
    for p in proposals:
        match=max([bbox_iou(p['bbox_xyxy'],t['bbox_xyxy']) for t in truth] or [0])
        if match<.2:extras.append({'frame':f['id'],**p})
    if truth or proposals:
        pic=image.copy()
        for gt in truth:
            a,b,c,d=map(int,gt['bbox_xyxy'])
            cv2.rectangle(pic,(a,b),(c,d),(30,240,40),3)
        for p in proposals:
            a,b,c,d=map(int,p['bbox_xyxy'])
            cv2.rectangle(pic,(a,b),(c,d),(255,130,0),2)
        cv2.imwrite(str(out/(f['id']+'.jpg')),pic)
    panels.append({'frame':f['id'],'proposals':proposals})
result={'manual_supplementary_boxes_unverified':total,'matched_iou_gt_0.3':found,
        'all_cv_proposals':allproposals,'nonoverlap_with_manual':len(extras),
        'hits':hits,'potential_new':extras,
        'human_GT':False,'precision_or_recall_verified':False}
(out/'evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
(out/'proposals.json').write_text(json.dumps(panels,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('hits','potential_new')},ensure_ascii=False),flush=True)
print('NEW',json.dumps(extras[:30],ensure_ascii=False),flush=True)
