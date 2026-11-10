"""Build a provenance-preserving DEV hard-negative dataset from AI-reviewed removed boxes.
Not independently human verified. No canonical splits are modified.
A group-id is the source video; every example remains a TRAIN_CANDIDATE.
"""
from pathlib import Path
import cv2,json,math,hashlib
from collections import Counter
root=Path(__file__).resolve().parents[1]
review=root/'output/review/test_video_ai_review'
out=root/'data/audit/ai_removed_boxes_hardneg_20261011'
out.mkdir(parents=True,exist_ok=True)
actions=json.loads((review/'changes.json').read_text(encoding='utf-8'))
index=0;kept=[];excluded=[]
for rec in actions:
    if rec['action']!='removed':continue
    number=int(rec['frame'].split('_')[-1])
    image=cv2.imread(str(review/'images'/f'{number:03}_0.jpg'))
    if image is None:
        excluded.append([rec['frame'],'missing image']);continue
    h,w=image.shape[:2]
    box=rec['before']['bbox_xyxy']
    x1,y1,x2,y2=(float(x) for x in box)
    if not all(math.isfinite(x) for x in (x1,y1,x2,y2)):
        excluded.append([rec['frame'],'nonfinite']);continue
    xi=max(0,math.floor(x1));yi=max(0,math.floor(y1))
    xa=min(w,math.ceil(x2));ya=min(h,math.ceil(y2))
    if xa-xi<16 or ya-yi<16:
        excluded.append([rec['frame'],'too small']);continue
    if (xa-xi)*(ya-yi)>0.11*w*h:
        excluded.append([rec['frame'],'too large for object-level negative']);continue
    crop=image[yi:ya,xi:xa]
    name=f'negative_{index:03d}.png';index+=1
    path=out/name
    if not cv2.imwrite(str(path),crop):raise RuntimeError(name)
    kept.append({'image':name,'frame':rec['frame'],'source_box':box,
     'previous_predicted_code':rec['before']['label'],
     'review_reason':rec['reason'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
     'source_group':'test_video_development','review_type':'AI_ONLY_NOT_HUMAN_CONFIRMED',
     'split':'TRAIN_CANDIDATE_ONLY'})
result={'images':kept,'excluded':excluded,'summary':{'kept':len(kept),'excluded':len(excluded),
        'unique_video_sources':1,'human_verified':0},'caveat':'Not an independent field dataset. Never use same source as a claimed held-out TEST.'}
(out/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result['summary']))
