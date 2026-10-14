"""CULane TEST evaluation of the deployed adapter, with/without its geometry filter.

Metric port follows UFLD-v2 evaluation/culane (MIT): natural arc-length spline,
50 samples per segment, width-30 masks, Hungarian IoU matching, IoU > 0.5.
This is not the authors' TTA evaluation binary. See third_party/UFLD-v2-LICENSE.
"""
import argparse
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.ufld_v2 import UFLDv2, plausible_lane


def lane_mask(points,width=1640,height=590):
    points=np.asarray(points,dtype=np.float32).astype(np.float64)
    mask=np.zeros((height,width),np.uint8)
    if len(points)<2:return mask
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    points=points[np.r_[True,lengths>1e-6]]
    if len(points)<2:return mask
    if len(points)>2:
        t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        dense=np.r_[np.concatenate([np.linspace(a,b,50,endpoint=False) for a,b in zip(t,t[1:])]),t[-1]]
        points=CubicSpline(t,points,bc_type='natural')(dense).astype(np.float32)
    cv2.polylines(mask,[np.rint(points).astype(np.int32)],False,1,30,lineType=cv2.LINE_8)
    return mask


def match_counts(matrix,prediction_count,truth_count):
    if not prediction_count or not truth_count:return 0,prediction_count,truth_count
    gi,pi=linear_sum_assignment(-matrix)
    tp=int((matrix[gi,pi]>.5).sum())
    return tp,prediction_count-tp,truth_count-tp


def score(predictions,truth):
    pm=[lane_mask(line) for line in predictions];gm=[lane_mask(line) for line in truth]
    matrix=np.zeros((len(gm),len(pm)))
    for i,a in enumerate(gm):
        for j,b in enumerate(pm):
            inter=cv2.countNonZero(cv2.bitwise_and(a,b))
            union=cv2.countNonZero(a)+cv2.countNonZero(b)-inter
            matrix[i,j]=inter/union if union else 0
    return matrix


def evaluate_prediction(name,categories,pred,truth):
    keep=[i for i,p in enumerate(pred) if plausible_lane(p,1640,590)]
    matrix=score(pred,truth)
    return {'image':name,'categories':categories,
            'raw_adapter':match_counts(matrix,len(pred),len(truth)),
            'deployed_filter':match_counts(matrix[:,keep],len(keep),len(truth))}


def aggregate(rows):
    result={}
    for mode in ['raw_adapter','deployed_filter']:
        groups=defaultdict(lambda:np.zeros(3,dtype=np.int64))
        for row in rows:
            groups['all']+=row[mode]
            for category in row['categories']:groups[category]+=row[mode]
        result[mode]={}
        for category,(tp,fp,fn) in groups.items():
            result[mode][category]={'tp':int(tp),'fp':int(fp),'fn':int(fn),
                'precision':float(tp/(tp+fp)) if tp+fp else None,
                'recall':float(tp/(tp+fn)) if tp+fn else None,
                'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/raw/culane/test')
    parser.add_argument('--output',type=Path,default=ROOT/'output/evaluation/revalidation_20261008/culane')
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    names=[line.split()[0].lstrip('/') for line in (args.data/'list/test.txt').read_text().splitlines() if line.strip()]
    assert len(names)==len(set(names))
    categories=defaultdict(list)
    for path in sorted((args.data/'list/test_split').glob('*.txt')):
        for line in path.read_text().splitlines():
            if line.strip():categories[line.split()[0].lstrip('/')].append(path.stem)
    missing=[name for name in names if not (args.data/name).is_file() or not (args.data/Path(name).with_suffix('.lines.txt')).is_file()]
    if missing and not args.allow_incomplete:raise RuntimeError(f'{len(missing)} TEST image/label pairs missing')
    missing_set=set(missing)
    available=[name for name in names if name not in missing_set]
    if args.limit:available=available[:args.limit]
    if not available:raise RuntimeError('No labeled TEST images available')
    identity={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in ['scripts/evaluate_culane.py','src/perception/ufld_v2.py']}
    with (ROOT/'models/lane/culane_res18.pth').open('rb') as checkpoint:
        identity['checkpoint']=hashlib.file_digest(checkpoint,'sha256').hexdigest()
    identity_path=out/'run_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise RuntimeError('Code/checkpoint changed; use a new output directory')
    identity_path.write_text(json.dumps(identity,indent=2))
    rows=[]
    records=out/'frames.jsonl'
    if records.exists():
        rows=[json.loads(line) for line in records.read_text().splitlines()]
        if len({r['image'] for r in rows})!=len(rows):raise RuntimeError('Duplicate resume records')
    completed={row['image'] for row in rows}
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=UFLDv2(ROOT/'models/lane/culane_res18.pth','cuda:0' if torch.cuda.is_available() else 'cpu',half=True)
    started=time.perf_counter()
    pending=deque()
    with records.open('a',encoding='utf-8') as file, ThreadPoolExecutor(max_workers=2) as workers:
        def save_row(row):
            rows.append(row);file.write(json.dumps(row)+'\n')
            if len(rows)%200==0:
                file.flush()
                progress={'evaluated':len(rows),'available':len(available),'expected':len(names),'elapsed_seconds':time.perf_counter()-started}
                (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
        for name in available:
            if name in completed:continue
            image=cv2.imread(str(args.data/name))
            if image is None or image.shape[:2]!=(590,1640):raise RuntimeError(f'Invalid CULane image {name}')
            truth=[]
            for line in (args.data/Path(name).with_suffix('.lines.txt')).read_text().splitlines():
                values=np.fromstring(line,sep=' ')
                if len(values)%2:raise RuntimeError(f'Invalid annotation {name}')
                if len(values)>=4:truth.append(values.reshape(-1,2).tolist())
            pred=model.detect(image,filter_geometry=False).polylines
            pending.append(workers.submit(evaluate_prediction,name,categories[name],pred,truth))
            if len(pending)>=4:save_row(pending.popleft().result())
        for future in pending:save_row(future.result())
    report={'expected_test_images':len(names),'evaluated_images':len(rows),'missing_pairs':len(missing),
            'complete_test':len(rows)==len(names),'metrics':aggregate(rows),
            'configuration':{'checkpoint':'models/lane/culane_res18.pth','precision':'FP16 CUDA / FP32 CPU',
                'tta':False,'road_roi':1.0,'width':30,'iou_strictly_greater_than':.5,
                'metric':'Python port of official natural-spline raster IoU metric; not official binary'},
            'elapsed_this_run_seconds':time.perf_counter()-started}
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
