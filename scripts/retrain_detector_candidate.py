"""Fine-tune the deployed detector; select on the existing VAL split only."""
from pathlib import Path
import json, shutil, time
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/retraining_20261009/detector'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    source=ROOT/'models/sign_detector/best.pt'
    shutil.copy2(source,OUT/'baseline.pt')
    data=str(ROOT/'configs/sign_dataset.yaml')
    started=time.time()
    baseline=YOLO(str(source)).val(data=data,split='val',device=0,batch=16,workers=0,
        project=str(OUT),name='baseline_val',plots=False,verbose=False)
    before=dict(baseline.results_dict)
    model=YOLO(str(source))
    model.train(data=data,epochs=8,imgsz=640,batch=16,device=0,workers=2,
        project=str(OUT),name='train',exist_ok=False,seed=391,deterministic=True,
        optimizer='AdamW',lr0=.0001,lrf=.1,patience=4,amp=True,cache=False,
        hsv_h=.005,hsv_s=.15,hsv_v=.25,degrees=2,translate=.03,scale=.2,
        flipud=0,fliplr=0,mosaic=0,mixup=0,copy_paste=0,plots=True)
    candidate=Path(model.trainer.save_dir)/'weights/best.pt'
    after=YOLO(str(candidate)).val(data=data,split='val',device=0,batch=16,workers=0,
        project=str(OUT),name='candidate_val',plots=False,verbose=False)
    improved=after.results_dict['metrics/mAP50-95(B)']>before['metrics/mAP50-95(B)']
    chosen=candidate if improved else OUT/'baseline.pt'
    shutil.copy2(chosen,OUT/'selected.pt')
    result=dict(baseline=before,candidate=dict(after.results_dict),candidate_selected=improved,
        selection='VAL mAP50-95; preserve baseline unless strictly better',
        training_data='Original TRAIN only; video annotations excluded',seconds=time.time()-started)
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
