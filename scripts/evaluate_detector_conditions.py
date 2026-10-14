"""Fixed-threshold localization P/R/F1 across all labeled TEST images."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.sign_detector import SignDetector
from src.utils.config import load_config


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    out=ROOT/'output/evaluation/revalidation_20261008'
    spec=importlib.util.spec_from_file_location('corruptions',ROOT/'scripts/evaluate_all_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=ROOT/'data/processed/canonical/itas_mm_v1/localization'
    samples=pd.read_csv(root/'samples.csv');samples=samples[samples.split=='TEST']
    labels=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv',keep_default_na=False)
    groups={k:np.array([json.loads(x) for x in v.bbox_xyxy],dtype=float) for k,v in labels[labels.split=='TEST'].groupby('sample_id')}
    detector=SignDetector(load_config('perception'));results={}
    for condition in ['original','dark_40pct','bright_180pct','blur_9px','jpeg_quality20','half_resolution']:
        tp=fp=fn=0;started=time.perf_counter()
        for i,row in enumerate(samples.itertuples(),1):
            image=cv2.imread(str(root/row.image_path));h,w=image.shape[:2]
            changed=module.transform(image,condition);ch,cw=changed.shape[:2]
            truth=groups[row.sample_id]*np.array([cw/w,ch/h,cw/w,ch/h])
            pred=np.array([sign.bbox_xyxy for sign in detector.detect(changed)],dtype=float).reshape(-1,4)
            hit=0
            if len(pred) and len(truth):
                top=np.maximum(pred[:,None,:2],truth[None,:,:2]);bottom=np.minimum(pred[:,None,2:],truth[None,:,2:])
                intersection=np.prod(np.maximum(0,bottom-top),axis=2)
                union=np.prod(pred[:,2:]-pred[:,:2],axis=1)[:,None]+np.prod(truth[:,2:]-truth[:,:2],axis=1)[None,:]-intersection
                iou=intersection/np.maximum(union,1e-9)
                a,b=linear_sum_assignment(-np.where(iou>=.5,1+iou/1000,0));hit=int((iou[a,b]>=.5).sum())
            tp+=hit;fp+=len(pred)-hit;fn+=len(truth)-hit
        results[condition]={'images':len(samples),'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
                            'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'elapsed_seconds':time.perf_counter()-started}
        print(json.dumps({'condition':condition,**results[condition]}),flush=True)
        (out/'detector_conditions.json').write_text(json.dumps({'conf':.35,'iou_match':.5,'conditions':results,
            'notes':'Synthetic full-image corruptions, fixed confidence. IoU Hungarian maximum-cardinality matching. Not nighttime field accuracy.'},indent=2))


if __name__=='__main__':main()
