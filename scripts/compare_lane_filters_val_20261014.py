"""Optimize *display-only* lane screens on CULane VAL, never TEST.

Compare original raw visual proposals, legacy strict geometry and candidate
direction/continuity screens with the same GT match metric. This does not
generate Vietnam ground truth or change ego-lane/legal reasoning.
"""
from pathlib import Path
import sys,hashlib,json
import numpy as np,cv2,torch
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.ufld_v2 import UFLDv2,plausible_lane
from src.perception.lane_display_quality import display_quality
from scripts.evaluate_culane_holdout_20261014 import score,match_counts

def candidate(line,shape,mode):
    xy=np.asarray(line,dtype=float)
    h,w=shape[:2]
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<5 or not np.isfinite(xy).all():return False
    xs=xy[:,0];ys=xy[:,1]
    if min(xs)<0 or max(xs)>=w or min(ys)<0 or max(ys)>=h:return False
    if np.ptp(ys)<(.08 if mode=='gentle' else .14)*h:return False
    if np.ptp(xs)>(2.5 if mode=='gentle' else 1.8)*np.ptp(ys):return False
    sortedx=xy[np.argsort(ys),0]
    if np.quantile(np.abs(np.diff(sortedx)),.9)>w*(.16 if mode=='gentle' else .08):return False
    return True


def main():
    root=ROOT/'data/raw/culane/val'
    imgs=sorted(root.rglob('*.jpg'),key=lambda p:hashlib.sha256(str(p.relative_to(root)).encode()).digest())[:240]
    model=UFLDv2(ROOT/'models/lane/selected_20261009.pth','cuda:0' if torch.cuda.is_available() else 'cpu',half=torch.cuda.is_available())
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    totals=defaultdict(lambda:np.zeros(3,dtype=np.int64))
    rows=[]
    for i,img in enumerate(imgs,1):
        frame=cv2.imread(str(img));assert frame is not None
        truth=[]
        for line in img.with_suffix('.lines.txt').read_text().splitlines():
            vals=np.fromstring(line,sep=' ')
            if len(vals)>=4:truth.append(vals.reshape(-1,2).tolist())
        raw=model.detect(frame,filter_geometry=False).polylines
        # Assumes CULane standard 1640x590.
        assert frame.shape[:2]==(590,1640)
        matrix=score(raw,truth)
        flags={
          'raw':[j for j in range(len(raw))],
          'plausible':[j for j,p in enumerate(raw) if plausible_lane(p,1640,590)],
          'existing_visual':[j for j,p in enumerate(raw) if display_quality(p,frame.shape)],
          'gentle':[j for j,p in enumerate(raw) if candidate(p,frame.shape,'gentle')],
          'balanced':[j for j,p in enumerate(raw) if candidate(p,frame.shape,'balanced')],
        }
        for name,keep in flags.items():
            totals[name]+=match_counts(matrix[:,keep],len(keep),len(truth))
        rows.append({'image':str(img.relative_to(root)),**{k:len(v) for k,v in flags.items()},'gt':len(truth)})
        if i%40==0:print('VAL_LANE_PROGRESS',i,'/',len(imgs),flush=True)
    metrics={}
    for k,(tp,fp,fn) in totals.items():
        metrics[k]={'tp':int(tp),'fp':int(fp),'fn':int(fn),
         'precision':tp/(tp+fp) if tp+fp else None,
         'recall':tp/(tp+fn) if tp+fn else None,
         'f1':2*tp/(2*tp+fp+fn) if tp+fp+fn else None}
    out=ROOT/'output/evaluation/lane_filter_val_20261014'
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'source':'CULane labeled VAL','sample_policy':'hash-ranked top 240 VAL images, predeclared', 'images':len(imgs),'metrics':metrics,'not_vietnam_accuracy':True},indent=2),encoding='utf8')
    print('FINAL_LANE_VAL',json.dumps(metrics),flush=True)
if __name__=='__main__':main()
