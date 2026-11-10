"""Check user-supplied YOLO11 sign checkpoint on our held-out canonical VAL localization set.
Do not tune on TEST or declare better code recognition from localization alone.
"""
from pathlib import Path
import sys,json,time,hashlib
import pandas as pd, numpy as np,cv2,torch
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ultralytics import YOLO
from src.perception.sign_detector import SignDetector
from src.utils.config import load_config
from scripts.evaluate_product import iou_matrix

WEIGHT=ROOT/'data/raw/user_archives_20261014/yolo11_optimized/weights/best.pt'
def match(boxes,gt):
    matrix=iou_matrix(boxes,gt)
    if not matrix.size:return 0
    a,b=linear_sum_assignment(-np.where(matrix>=.5,1+matrix/1000,0))
    return sum(float(matrix[i,j])>=.5 for i,j in zip(a,b))

def merged(primary,additional):
    out=list(primary)
    for box,score in additional:
        if score<.55:continue
        if any(float(iou_matrix([box],[p])[0,0])>=.35 for p in out):continue
        out.append(box)
    return out

def run():
    cv2.setNumThreads(2);torch.set_num_threads(4)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):
        original=SignDetector(load_config('perception'))
        additional=YOLO(str(WEIGHT))
        print('LOADED_YOLO11',additional.model.names,flush=True)
        sample=pd.read_csv(ROOT/'data/processed/canonical/itas_mm_v1/localization/samples.csv')
        sample=sample[sample.split=='VAL'].sort_values('image_sha256').head(240)
        g=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv',keep_default_na=False)
        grouped={k:v for k,v in g[g.split=='VAL'].groupby('sample_id')}
        counts={k:{'gt':0,'tp':0,'pred':0} for k in ('existing960','aux416','aux640','union416','union640')}
        latencies={k:[] for k in ('existing960','aux416','aux640')}
        row_data=[]
        for n,row in enumerate(sample.itertuples(),1):
            image=cv2.imread(str(ROOT/'data/processed/canonical/itas_mm_v1/localization'/row.image_path))
            truth=[json.loads(b) for b in grouped[row.sample_id].bbox_xyxy]
            t=time.perf_counter();original.image_size=960;p=[s.bbox_xyxy for s in original.detect(image)];latencies['existing960'].append((time.perf_counter()-t)*1000)
            dd={}
            for sz in (416,640):
                t=time.perf_counter()
                det=additional.predict(image,conf=.35,iou=.45,imgsz=sz,device=0,verbose=False)
                aux=[(tuple(box.xyxy[0].tolist()),float(box.conf[0])) for res in det for box in res.boxes]
                latencies['aux'+str(sz)].append((time.perf_counter()-t)*1000)
                dd['aux'+str(sz)]=[b for b,c in aux if c>=.55]
                dd['union'+str(sz)]=merged(p,aux)
            dd['existing960']=p
            for key,boxes in dd.items():
                rec=counts[key];rec['gt']+=len(truth);rec['pred']+=len(boxes);rec['tp']+=match(boxes,truth)
            row_data.append({'image':row.sample_id,'gt':len(truth),**{k:len(v) for k,v in dd.items()}})
            if n%40==0:print('PROGRESS',n,'/',len(sample),flush=True)
        for k,d in counts.items():
            d['recall']=d['tp']/max(d['gt'],1);d['precision']=d['tp']/max(d['pred'],1)
        report={'checkpoint_sha256':hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
          'source':'VAL canonical labeled bbox; no TEST tuning; not Vietnam route independent',
          'split':'VAL','images':len(sample),'match_iou':.5,'auxiliary_confidence':.55,
          'results':counts,'latency_ms':{k:round(float(np.mean(v)),2) for k,v in latencies.items()},
          'notes':['Code recall needs separate recognition benchmark',
                   'Training-mAP from zip is not comparable to this dataset',
                   'Auxiliary boxes are hypotheses and not verified traffic regulations']}
        out=ROOT/'output/import_yolo11_comment_20261014';out.mkdir(parents=True,exist_ok=True)
        (out/'val_localization.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print('FINAL',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
