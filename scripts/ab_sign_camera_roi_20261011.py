"""Offline camera-sign-ROI experiment, without changing production configuration.

Compare 65% vs 52% sign ROI on development snapshots. This alone cannot
establish recall; full image/manual annotations are needed before adoption.
"""
from pathlib import Path
from collections import Counter
import json,cv2
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.utils.config import load_config
root=Path(__file__).resolve().parents[1]
conf=load_config('perception')
d=SignDetector(conf);c=SignRecognizer(conf)
images=sorted((root/'output/review/test_video_development/images').glob('*_0.jpg'))
rows=[];counter={'0.65':Counter(),'0.52':Counter()}
for path in images:
    img=cv2.imread(str(path))
    if img is None:continue
    row={'image':path.name}
    for ratio in (.65,.52):
        crop=img[:int(img.shape[0]*ratio)]
        signs=c.recognize(crop,d.detect(crop),min_short_side=16)
        result=[{'label':s.label,'confidence':round(s.confidence,3),'xmid':round((s.bbox_xyxy[0]+s.bbox_xyxy[2])/2/img.shape[1],3),'ymid':round((s.bbox_xyxy[1]+s.bbox_xyxy[3])/2/img.shape[0],3)} for s in signs]
        row[str(ratio)]=result
        counter[str(ratio)].update(s.label for s in signs)
    rows.append(row)
out=root/'output/evaluation/sign_camera_roi_ab_20261011.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'scope':'development snapshots; A/B is counts not accuracy', 'samples':len(rows),'counts':{k:dict(v) for k,v in counter.items()},'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'samples':len(rows),'counts':{k:dict(v) for k,v in counter.items()},'report':str(out)},ensure_ascii=False))
