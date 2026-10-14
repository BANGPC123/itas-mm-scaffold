"""Probe multi-scale YOLO localization on development review snapshots.

Produces *proposals*, not certified GT. Uses selected checkpoint unchanged.
"""
from pathlib import Path
import json,cv2,torch,time,collections
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
frames=json.loads((PACK/'itasmm_human_review_uploaded_20261011.json').read_text(encoding='utf-8'))['frames']
device='0' if torch.cuda.is_available() else 'cpu'
model=YOLO(str(ROOT/'models/sign_detector/retrained_20261009.pt'))
summary=[]
for frame in frames:
    im=cv2.imread(str(PACK/frame['image']))
    h,w=im.shape[:2]
    y_max=int(h*.75)
    out=[]
    for method,crop,xoff in [
        ('full',im[:y_max],0),
        ('tile_left',im[:y_max,:min(w,780)],0),
        ('tile_right',im[:y_max,max(0,w-780):],max(0,w-780)),
    ]:
        preds=model.predict(crop,conf=.20,iou=.5,imgsz=960 if method!='full' else 1280,device=device,verbose=False,max_det=100)[0]
        for box in preds.boxes:
            x1,y1,x2,y2=box.xyxy[0].tolist()
            out.append({'pass':method,'box':[round(x1+xoff,1),round(y1,1),round(x2+xoff,1),round(y2,1)],
                        'conf':round(float(box.conf[0]),3)})
    summary.append({'frame':frame['id'],'n':len(out),'detections':out})
    print(frame['id'],'pass count',len(out),flush=True)
dest=PACK/'ai_visual_review/multiscale_yolo_raw_20261011.json'
dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('FINISHED '+str(dest),flush=True)
