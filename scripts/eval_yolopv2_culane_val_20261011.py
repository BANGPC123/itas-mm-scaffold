"""Fair 70-frame official CULane VAL comparison using the actual evaluator.

Pretrained CULane UFLD vs BDD YOLOPv2 vs conservative no-lane fallback.
VAL is NOT an independent Vietnam TEST.
"""
from pathlib import Path
import sys,cv2,torch,json,time,numpy as np,importlib.util
from src.perception.ufld_v2 import UFLDv2
from src.perception.yolopv2_lane import YOLOPv2Lane
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('culane_metric',ROOT/'scripts/evaluate_culane.py')
metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
OUT=ROOT/'output/evaluation/yolopv2_culane_val_20261011'
if (OUT/'results.json').is_file():raise FileExistsError('Refuse overwrite')
OUT.mkdir(parents=True,exist_ok=True)
cv2.setNumThreads(1);torch.set_num_threads(4)
ufld=UFLDv2(ROOT/'models/lane/selected_20261009.pth','cuda:0',half=True)
yolo=YOLOPv2Lane(ROOT/'models/lane/candidates/yolopv2_official_20261011/yolopv2.pt',
                  device='cuda:0',half=True,roi=1.,threshold=.56)
names=[s.split()[0].lstrip('/') for s in (ROOT/'data/raw/culane/test/list/val.txt').read_text().splitlines() if s.strip()][:70]
data=ROOT/'data/raw/culane/val'
rows=[];start=time.perf_counter()
for i,name in enumerate(names):
    img=cv2.imread(str(data/name))
    if img is None:continue
    h,w=img.shape[:2]
    gt=[]
    for s in (data/Path(name).with_suffix('.lines.txt')).read_text().splitlines():
        nums=np.fromstring(s,sep=' ')
        if nums.size>=4:gt.append(nums.reshape(-1,2).tolist())
    a=ufld.detect(img,filter_geometry=False).polylines
    b=yolo.detect(img).polylines
    cand=a if a else b
    out={'name':name,'gt_lanes':len(gt),'ufld':metric.match_counts(metric.score(a,gt),len(a),len(gt)),
         'yolopv2':metric.match_counts(metric.score(b,gt),len(b),len(gt)),
         'fallback':metric.match_counts(metric.score(cand,gt),len(cand),len(gt))}
    rows.append(out)
    if i%15==0:print('CULANE_VAL',i,len(names),'GT',len(gt),'BASE',len(a),'YOLO',len(b),flush=True)
def metrics(k):
    tp,fp,fn=np.sum([r[k] for r in rows],axis=0).astype(int).tolist()
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(1,tp+fp),
            'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)}
report={'source':'CULane official VAL, 70 images','not_independent_Vietnam_gt':True,
  'results':{k:metrics(k) for k in ('ufld','yolopv2','fallback')},
  'checked_frames':len(rows),'image_rows':rows,
  'seconds':round(time.perf_counter()-start,1),
  'release_rule':'Fallback only if no F1 regression vs UFLD on this VAL and visually plausible on user video.'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('CULANE_VAL_RESULT',json.dumps({k:v for k,v in report.items() if k!='image_rows'},ensure_ascii=False),flush=True)
