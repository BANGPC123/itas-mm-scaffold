"""Frozen-checkpoint CULane VAL A/B for the lane sanity pass, not independent TEST."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import cv2
from src.perception.lane_detector import LaneDetector
from src.perception.lane_consistency import reject_obvious_artifacts
from src.utils.config import load_config
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import importlib.util
spec=importlib.util.spec_from_file_location('lane_metric',Path(__file__).with_name('evaluate_culane.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
match_counts,score=module.match_counts,module.score

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--limit',type=int,default=160)
    p.add_argument('--output',type=Path,default=ROOT/'output/evaluation/lane_sanity_val_20261010.json')
    a=p.parse_args()
    lines=(ROOT/'data/raw/culane/test/list/val.txt').read_text().splitlines()
    names=sorted(row.split()[0].lstrip('/') for row in lines if row.strip())[:a.limit]
    model=LaneDetector(load_config('perception'))
    totals={k:np.zeros(3,dtype=np.int64) for k in ('before','after')}
    changed=0
    started=time.perf_counter()
    cv2.setNumThreads(1)
    for index,name in enumerate(names):
        path=ROOT/'data/raw/culane/val'/name
        im=cv2.imread(str(path))
        if im is None:
            raise RuntimeError(str(path))
        truth=[]
        for line in path.with_suffix('.lines.txt').read_text().splitlines():
            z=np.fromstring(line,sep=' ')
            if len(z)>=4:truth.append(z.reshape(-1,2).tolist())
        lane=model.detect(im)
        before=lane.polylines[:]
        after=reject_obvious_artifacts(lane,im.shape).polylines
        changed+=len(before)!=len(after)
        for mode, pred in [('before',before),('after',after)]:
            matrix=score(pred,truth)
            totals[mode]+=match_counts(matrix,len(pred),len(truth))
        if (index+1)%40==0:print('processed',index+1,flush=True)
    def metric(c):
        tp,fp,fn=map(int,c)
        return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    result=dict(data='official CULane VAL subset, alphabetically selected',sampled=len(names),independent_holdout=False,
                same_checkpoint=True,changed_frames=changed,before=metric(totals['before']),after=metric(totals['after']),seconds=time.perf_counter()-started)
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
