"""Frozen-checkpoint TEST evaluation plus explicitly synthetic robustness cases."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.sign_recognizer import SignRecognizer, crop_tensor
from src.perception.sign_detector import SignDetector
from src.perception.lane_detector import LaneDetector
from src.utils.config import load_config

LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
SEM=ROOT/'data/audit/semantic/final_v1_1'


def transform(crop, condition):
    if condition=='dark_40pct': return np.clip(crop.astype(np.float32)*.4,0,255).astype(np.uint8)
    if condition=='bright_180pct': return np.clip(crop.astype(np.float32)*1.8,0,255).astype(np.uint8)
    if condition=='blur_9px': return cv2.GaussianBlur(crop,(9,9),2.)
    if condition=='jpeg_quality20':
        ok,data=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,20])
        if not ok: raise RuntimeError('JPEG encoding failed')
        return cv2.imdecode(data,cv2.IMREAD_COLOR)
    if condition=='half_resolution': return cv2.resize(crop,(max(1,crop.shape[1]//2),max(1,crop.shape[0]//2)),interpolation=cv2.INTER_AREA)
    return crop


def metrics(y,p,size,threshold,min_size):
    prediction=p.argmax(1)
    accepted=(p.max(1)>=threshold)&(size>=min_size)
    correct=prediction==y
    return {'n':len(y),'accuracy':float(correct.mean()),
            'macro_f1':float(f1_score(y,prediction,labels=list(range(p.shape[1])),average='macro',zero_division=0)),
            'accepted_n':int(accepted.sum()),'abstained_n':int((~accepted).sum()),
            'runtime_coverage':float(accepted.mean()),
            'runtime_accepted_accuracy':float(correct[accepted].mean()) if accepted.any() else None,
            'runtime_correct_fraction_all':float((correct&accepted).mean())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'output/evaluation/revalidation_20261008')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    device=0 if torch.cuda.is_available() else 'cpu'
    started=time.perf_counter()
    cfg=load_config('perception')
    manifest={}
    for path in ['models/sign_detector/best.pt','models/sign_recognizer/best.pt','models/lane/culane_res18.pth']:
        digest=hashlib.sha256()
        with (ROOT/path).open('rb') as file:
            for chunk in iter(lambda:file.read(8*1024*1024),b''):digest.update(chunk)
        manifest[path]=digest.hexdigest()
    (out/'checkpoints.json').write_text(json.dumps(manifest,indent=2))
    from ultralytics import YOLO
    model=YOLO(str(ROOT/cfg['sign_detector']['weights_path']))
    validation=model.val(data=str(ROOT/'configs/sign_dataset.yaml'),split='test',imgsz=640,batch=8,
        device=device,workers=0,plots=True,project=str(out),name='detector_test',exist_ok=True,
        half=False,conf=.001,iou=.7,verbose=False)
    detector_report={'split':'TEST','images':2326,'ground_truth_boxes':4888,
        'metrics':validation.results_dict,'speed_ms':validation.speed,
        'note':'Standard AP sweep; operating-threshold precision/recall measured separately in sign_pipeline_results.json.'}
    (out/'detector.json').write_text(json.dumps(detector_report,indent=2))
    del model,validation;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

    recognizer=SignRecognizer(cfg)
    frame=pd.read_csv(SEM/'recognition_manifest_v1.csv',keep_default_na=False)
    frame=frame[frame.split=='TEST'].reset_index(drop=True)
    classes=pd.read_csv(SEM/'recognition_classes_v1.csv').sort_values('class_index').canonical_sign_code.tolist()
    assert classes==recognizer.classes
    y=frame.class_index.to_numpy(dtype=int)
    results={}
    for condition in ['original','dark_40pct','bright_180pct','blur_9px','jpeg_quality20','half_resolution']:
        probabilities=[];sizes=[]
        for offset in range(0,len(frame),64):
            tensors=[]
            for row in frame.iloc[offset:offset+64].itertuples():
                image=cv2.imread(str(LOC/row.image_path))
                if image is None: raise RuntimeError(row.image_path)
                x1,y1,x2,y2=json.loads(row.bbox_xyxy);h,w=image.shape[:2]
                crop=image[max(0,math.floor(y1)):min(h,math.ceil(y2)),max(0,math.floor(x1)):min(w,math.ceil(x2))]
                if not crop.size: raise RuntimeError(row.canonical_box_id)
                crop=transform(crop,condition);sizes.append(min(crop.shape[:2]));tensors.append(crop_tensor(crop,recognizer.size))
            with torch.inference_mode():
                probabilities.extend(recognizer.model(torch.stack(tensors).to(recognizer.device)).softmax(1).cpu().numpy())
        p=np.asarray(probabilities);size=np.asarray(sizes)
        report=metrics(y,p,size,recognizer.threshold,recognizer.config.get('min_short_side',32))
        for name,mask in [('ge32',frame.short_side_px.to_numpy()>=32),('lt32',frame.short_side_px.to_numpy()<32)]:
            report[name]=metrics(y[mask],p[mask],size[mask],recognizer.threshold,recognizer.config.get('min_short_side',32))
        report['per_class']=classification_report(y,p.argmax(1),labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
        results[condition]=report
        pd.DataFrame({'box_id':frame.canonical_box_id,'truth':[classes[i] for i in y],
            'prediction':[classes[i] for i in p.argmax(1)],'confidence':p.max(1),'crop_short_side':size,
            'runtime_accepted':(p.max(1)>=recognizer.threshold)&(size>=32)}).to_csv(out/f'recognition_{condition}.csv',index=False)
        if condition=='original':
            matrix=confusion_matrix(y,p.argmax(1),labels=list(range(len(classes))))
            pd.DataFrame(matrix,index=classes,columns=classes).to_csv(out/'recognizer_confusion_matrix.csv')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(12,10))
            picture=ax.imshow(matrix/np.maximum(matrix.sum(1,keepdims=True),1),cmap='Blues',vmin=0,vmax=1)
            ax.set_xticks(range(len(classes)),classes,rotation=90);ax.set_yticks(range(len(classes)),classes)
            ax.set_xlabel('Predicted code');ax.set_ylabel('Ground truth');ax.set_title('Recognizer TEST: row-normalized confusion matrix')
            fig.colorbar(picture,ax=ax);fig.tight_layout();fig.savefig(out/'recognizer_confusion_matrix.png',dpi=140);plt.close(fig)
        print(json.dumps({'condition':condition,**{k:v for k,v in report.items() if k not in ['per_class','ge32','lt32']}}),flush=True)
        (out/'recognizer.json').write_text(json.dumps({'threshold':recognizer.threshold,'conditions':results,
            'notes':'Original TEST labels are AI-verified, not independently human-verified. Corruptions are synthetic crop tests, not real nighttime/video accuracy.'},indent=2))
    del recognizer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

    detector=SignDetector(cfg);lane=LaneDetector(cfg)
    negative_cases=[]
    rng=np.random.default_rng(391)
    for name,image in [('black',np.zeros((720,1280,3),np.uint8)),('white',np.full((720,1280,3),255,np.uint8)),
                       ('gray',np.full((720,1280,3),127,np.uint8)),('noise',rng.integers(0,256,(720,1280,3),dtype=np.uint8))]:
        signs=detector.detect(image);lines=lane.detect(image)
        negative_cases.append({'case':name,'sign_detections':len(signs),'lane_boundaries':len(lines.polylines),
                               'pass_no_false_detections':not signs and not lines.polylines})
    (out/'negative_cases.json').write_text(json.dumps(negative_cases,indent=2))
    (out/'lane.json').write_text(json.dumps({'accuracy':None,'f1':None,'status':'NOT_MEASURED_NO_LANE_GROUND_TRUTH',
        'negative_cases':negative_cases,'note':'Blank/noise cases and geometry tests cannot replace a labeled lane benchmark.'},indent=2))
    print(json.dumps({'negative_cases':negative_cases,'elapsed_seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
