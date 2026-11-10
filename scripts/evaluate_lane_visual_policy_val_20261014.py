"""Select offline *visual-only* UFLD confidence screens on CULane VAL.
Source-diverse validation labels; no TEST hyperparameter tuning. No ego-law use.
"""
from pathlib import Path
import json,hashlib,sys
from collections import defaultdict
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.ufld_v2 import UFLDv2
from scripts.evaluate_culane_holdout_20261014 import score,match_counts
from src.perception.lane_display_quality import display_quality

def accept(points,shape,policy):
    h,w=shape[:2]
    xy=np.asarray(points,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<3 or not np.isfinite(xy).all():return False
    if np.any(xy[:,0]<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]<0) or np.any(xy[:,1]>=h):return False
    if policy=='raw':return True
    if policy=='legacy':return display_quality(points,shape)
    span=np.ptp(xy[:,1]);run=np.ptp(xy[:,0])
    if policy.startswith('slope'):
        ratio=float(policy.split('_')[1])
        return span>=h*.015 and run<ratio*max(span,1)
    if policy.startswith('span'):
        fraction=float(policy.split('_')[1])
        return span>=h*fraction
    if policy=='rough_geometry':
        if span<h*.02 or run>5*span:return False
        ordered=xy[np.argsort(xy[:,1])]
        dx=np.diff(ordered[:,0])
        direction=np.sign(dx[np.abs(dx)>w*.012])
        return sum(direction[i]*direction[i-1]<0 for i in range(1,len(direction)))<3
    raise ValueError(policy)

def main():
    data=ROOT/'data/raw/culane/val'
    imgs=sorted(data.rglob('*.jpg'),key=lambda f:hashlib.sha256(str(f.relative_to(data)).encode()).digest())[:240]
    policies=['raw','legacy','slope_2.5','slope_4','slope_8','span_0.01','span_0.03','span_0.06','rough_geometry']
    totals=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    model=UFLDv2(ROOT/'models/lane/selected_20261009.pth',
        'cuda:0' if torch.cuda.is_available() else 'cpu',half=torch.cuda.is_available())
    cv2.setNumThreads(1);torch.set_num_threads(4)
    for index,path in enumerate(imgs,1):
        frame=cv2.imread(str(path))
        gt=[]
        for line in path.with_suffix('.lines.txt').read_text().splitlines():
            value=np.fromstring(line,sep=' ')
            if len(value)>=4 and len(value)%2==0:gt.append(value.reshape(-1,2).tolist())
        raw=model.detect(frame,filter_geometry=False).polylines
        mat=score(raw,gt)
        for method in policies:
            keep=[i for i,p in enumerate(raw) if accept(p,frame.shape,method)]
            totals[method]+=match_counts(mat[:,keep],len(keep),len(gt))
        if index%40==0:print('LANE_VAL',index,len(imgs),flush=True)
    result={}
    for name,(tp,fp,fn) in totals.items():
        result[name]={'tp':int(tp),'fp':int(fp),'fn':int(fn),
        'precision':float(tp/(tp+fp)) if tp+fp else 0,
        'recall':float(tp/(tp+fn)) if tp+fn else 0,
        'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0}
    out=ROOT/'output/evaluation/lane_visual_policy_val_20261014'
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'split':'VAL','images':len(imgs),
        'metrics':result,'no_field_accuracy_claim':True},indent=2),encoding='utf8')
    print('LANE_POLICIES_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
