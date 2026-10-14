"""Compare classifier inputs on fixed AI-reviewed boxes; no threshold fitting."""
import json
from pathlib import Path
import sys
import cv2
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.sign_recognizer import SignRecognizer
from src.utils.config import load_config
from src.utils.types import SignDetection

torch.set_num_threads(2)
cfg=load_config('perception');cfg['sign_recognizer']['device']='cpu'
model=SignRecognizer(cfg)
folder=ROOT/'output/review/test_video_ai_review'
packet=json.loads((folder/'ai_reviewed.json').read_text(encoding='utf-8'))
changes=json.loads((folder/'changes.json').read_text(encoding='utf-8'))
cap=cv2.VideoCapture(packet['source_video']);rows=[]
for sample in packet['samples']:
    cap.set(cv2.CAP_PROP_POS_MSEC,sample['timestamp_seconds']*1000);ok,native=cap.read()
    if not ok:raise RuntimeError(sample['id'])
    small=cv2.imread(str(folder/sample['images'][1]))
    targets=[(s['bbox_xyxy'],s['label']) for s in sample['signs'] if s['label']!='unknown']
    targets += [(c['before']['bbox_xyxy'],'unknown') for c in changes if c['frame']==sample['id'] and c['action']=='removed' and 'trùng' not in c['reason']]
    for name,image in [('1280',small),('1920',native)]:
        sx,sy=image.shape[1]/sample['width'],image.shape[0]/sample['height']
        signs=[SignDetection('unknown',1.,tuple(v*(sx if i%2==0 else sy) for i,v in enumerate(box))) for box,_ in targets]
        found=model.recognize(image,signs,min_short_side=16)
        rows.extend({'frame':sample['id'],'input':name,'expected':target,'predicted':s.label,'confidence':s.recognition_confidence} for (_,target),s in zip(targets,found))
cap.release();metrics={}
for name in ['1280','1920']:
    known=[r for r in rows if r['input']==name and r['expected']!='unknown'];bg=[r for r in rows if r['input']==name and r['expected']=='unknown']
    metrics[name]={'known_agreement':sum(r['predicted']==r['expected'] for r in known),'known_n':len(known),'known_rejected':sum(r['predicted']=='unknown' for r in known),
                   'background_false_accepts':sum(r['predicted']!='unknown' for r in bg),'background_n':len(bg)}
out=ROOT/'output/product_20261010/resolution_comparison.json'
out.write_text(json.dumps({'metrics':metrics,'rows':rows,'scope':'AI-reviewed development boxes, not human accuracy; stored JPEG versus source video.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metrics),flush=True)
