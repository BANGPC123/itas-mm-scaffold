"""Conservative detector finetune. Preserve original checkpoints, select only on VAL."""
from pathlib import Path
import json,time,hashlib
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/retraining_20261010/detector'
OUT.mkdir(exist_ok=True,parents=True)
source=ROOT/'models/sign_detector/retrained_20261009.pt'
data=str(ROOT/'configs/sign_dataset.yaml')
torch.set_num_threads(4)
started=time.time()
def evaluate(path,tag):
    r=YOLO(str(path)).val(data=data,split='val',device=0,batch=16,workers=0,
        project=str(OUT),name=tag,plots=False,verbose=False)
    return dict(r.results_dict)
baseline=evaluate(source,'baseline_val')
(OUT/'baseline.json').write_text(json.dumps({'checkpoint':str(source),'metrics':baseline},indent=2))
print('BASELINE',json.dumps(baseline),flush=True)
model=YOLO(str(source))
model.train(data=data,epochs=4,imgsz=640,batch=8,device=0,workers=0,
    project=str(OUT),name='train_4epochs',exist_ok=False,seed=391,
    deterministic=True,optimizer='AdamW',lr0=3e-5,lrf=.2,patience=4,
    amp=True,cache=False,hsv_h=.005,hsv_s=.15,hsv_v=.2,
    degrees=2,translate=.03,scale=.15,flipud=0,fliplr=0,mosaic=.2,
    mixup=0,copy_paste=0,plots=False,verbose=False)
candidate=Path(model.trainer.save_dir)/'weights/best.pt'
score=evaluate(candidate,'candidate_val')
selected=score['metrics/mAP50-95(B)']>baseline['metrics/mAP50-95(B)']
report={'baseline':baseline,'candidate':score,'selected':selected,
        'candidate_path':str(candidate),'baseline_path':str(source),
        'selection':'STRICT VAL mAP50-95 improvement, no TEST tuning',
        'epochs_budget':4,'data':'canonical TRAIN only','seconds':time.time()-started}
(OUT/'results.json').write_text(json.dumps(report,indent=2))
print('FINAL_RESULT',json.dumps(report),flush=True)
