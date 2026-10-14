"""Compare FP16/FP32 lane geometry on the same camera frames, not accuracy."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.perception.ufld_v2 import UFLDv2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source')
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for FP16 comparison')
    torch.set_num_threads(4)
    full=UFLDv2('models/lane/culane_res18.pth','cuda:0',half=False)
    half=UFLDv2('models/lane/culane_res18.pth','cuda:0',half=True)
    capture=cv2.VideoCapture(args.source)
    rows=[]
    for seconds in [0,10,24,120,240,360,600,840,960,1000]:
        capture.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
        ok,frame=capture.read()
        if not ok: raise RuntimeError(f'Cannot read {seconds}s')
        frame=cv2.resize(frame,(1280,720))[:468]
        a,b=full.detect(frame),half.detect(frame)
        differences=[]
        same_lengths=len(a.polylines)==len(b.polylines) and all(len(x)==len(y) for x,y in zip(a.polylines,b.polylines))
        if same_lengths:
            differences=[float(np.linalg.norm(np.array(x)-y,axis=1).max()) for x,y in zip(a.polylines,b.polylines)]
        rows.append({'seconds':seconds,'fp32_lanes':len(a.polylines),'fp16_lanes':len(b.polylines),
                     'identical_anchor_counts':same_lengths,'max_point_distance_px':max(differences,default=0) if same_lengths else None})
    capture.release()
    output={'frames':rows,'limitation':'Numerical agreement on 10 development frames; not lane accuracy.'}
    Path('output/performance/lane_precision.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
