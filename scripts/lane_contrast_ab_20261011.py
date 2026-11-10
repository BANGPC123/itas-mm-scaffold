"""Research-only lane contrast recovery A/B. Never promotes itself to runtime.

Uses official CULane VAL (not independent) and public Vietnam videos (unlabeled).
Counts and GT scores documented separately. Selected weight unchanged.
"""
from pathlib import Path
import json,time,cv2,torch,numpy as np
from src.perception.ufld_v2 import UFLDv2
from src.perception.image_quality import insufficient_information
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('culane_metric',ROOT/'scripts/evaluate_culane.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=ROOT/'output/evaluation/lane_contrast_ab_20261011'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);cv2.setNumThreads(1)
lane=UFLDv2(ROOT/'models/lane/selected_20261009.pth',
            'cuda:0' if torch.cuda.is_available() else 'cpu',half=True)
def clahe(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(6,6)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2BGR)
def gamma(img):
    lut=(np.power(np.arange(256)/255.0,.8)*255).clip(0,255).astype(np.uint8)
    return cv2.LUT(img,lut)
modes={'original':lambda x:x,'clahe':clahe,'gamma_08':gamma}
vals=[]
path=ROOT/'data/raw/culane/test/list/val.txt'
names=[r.split()[0].lstrip('/') for r in path.read_text().splitlines() if r.strip()][:70]
data=ROOT/'data/raw/culane/val'
started=time.time()
for idx,name in enumerate(names):
    image=cv2.imread(str(data/name))
    label_file=data/Path(name).with_suffix('.lines.txt')
    if image is None or not label_file.is_file():continue
    truth=[]
    for line in label_file.read_text().splitlines():
        points=np.fromstring(line,sep=' ')
        if len(points)>=4:truth.append(points.reshape(-1,2).tolist())
    per={}
    for mode,transform in modes.items():
        pred=lane.detect(transform(image),filter_geometry=False).polylines
        per[mode]=m.evaluate_prediction(name,[],pred,truth)['raw_adapter']
    vals.append({'image':name,'n_gt':len(truth),'counts':per})
    if idx%20==0:print('LANE_VAL',idx,len(names),round(time.time()-started,1),flush=True)
video=[]
for folder in ['commons_hanoi_20261011','commons_saigon_govap_20261011']:
    pack=ROOT/'output/review'/folder
    manifest=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
    step=max(1,len(manifest['frames'])//6)
    for f in manifest['frames'][::step][:6]:
        img=cv2.imread(str(pack/f['image']))
        counts={}
        for mode,transform in modes.items():
            pred=lane.detect(transform(img),filter_geometry=False)
            counts[mode]={'n_lanes':len(pred.polylines),'ego_boundaries':len(pred.ego_boundaries),
                          'reason':pred.rejection_reason}
        video.append({'source':folder,'frame':f['id'],'raw_coverage_only_not_gt':counts})
stat={}
for mode in modes:
    a=np.sum([x['counts'][mode] for x in vals],axis=0).astype(int)
    tp,fp,fn=a.tolist()
    stat[mode]={'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None}
report={'scope':'EXPERIMENTAL_AB_NOT_PRODUCTION',
        'val_source':'official CULane VAL; not independent Vietnam-road GT',
        'val_frames':len(vals),'culane_val_sample':stat,
        'new_vietnam_road_images':len(video),
        'vietnam_coverage':{mode:sum(r['raw_coverage_only_not_gt'][mode]['n_lanes']>0 for r in video)
                            for mode in modes},
        'vietnam_rows':video,
        'status':'CANDIDATE_EVIDENCE_ONLY; do not turn artificial lines into verified ego-lane scope',
        'selected_checkpoint_changed':False,
        'elapsed_seconds':round(time.time()-started,1)}
(OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('LANE_AB_RESULT',json.dumps({k:v for k,v in report.items() if k!='vietnam_rows'},ensure_ascii=False),flush=True)
