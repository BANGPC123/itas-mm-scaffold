"""Compare 960+640 rescue to a cautious third 1280 pass on frozen VAL."""
from pathlib import Path
import sys,time,copy,json,hashlib,argparse
from collections import defaultdict
import cv2,numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.perception.sign_multiscale import recognize_views
from src.perception.offline_sign_quality import safe_refinement
from scripts.evaluate_sign_candidates import score

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--split',choices=['VAL','TEST'],default='VAL')
 ap.add_argument('--limit',type=int,default=None)
 args=ap.parse_args()
 cv2.setNumThreads(2);torch.set_num_threads(4)
 from threadpoolctl import threadpool_limits
 with threadpool_limits(limits=1,user_api='blas'):
  root=ROOT/'data/processed/canonical/itas_mm_v1/localization'
  samples=pd.read_csv(root/'samples.csv')
  samples=samples[samples.split==args.split].sort_values('image_sha256')
  samples=samples.head(args.limit if args.limit is not None else (400 if args.split=='VAL' else len(samples)))
  labels=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv',keep_default_na=False)
  groups={k:v.reset_index(drop=True) for k,v in labels[labels.split==args.split].groupby('sample_id')}
  d=SignDetector(load_config('perception'));rec=SignRecognizer(load_config('perception'))
  methods={'baseline':None}
  for dc in ((.65,.75,.85) if args.split=='VAL' else (.75,)):
   for rc in ((.88,.93,.97) if args.split=='VAL' else (.93,)):
    methods[f'highres_{dc:.2f}_{rc:.2f}']=(dc,rc)
  totals={k:defaultdict(int) for k in methods}
  examples=[]
  t0=time.perf_counter()
  for n,s in enumerate(samples.itertuples(),1):
   im=cv2.imread(str(root/s.image_path))
   d.image_size=960;a=d.detect(im)
   d.image_size=640;b=d.detect(im)
   base=recognize_views(rec,im,a,b,min_short_side=12)
   d.image_size=1280;high=rec.recognize(im,d.detect(im),min_short_side=12)
   truth=groups[s.sample_id]
   for method,limits in methods.items():
    vals=base if limits is None else safe_refinement(base,high,detector_confidence=limits[0],recognition_confidence=limits[1])
    score(vals,truth,totals[method],[],s.sample_id,method)
   if n%100==0:print('VAL_QUALITY_PROGRESS',n,'of',len(samples),'s',int(time.perf_counter()-t0),flush=True)
  for k,c in totals.items():
   c['code_recall']=c['correct']/max(1,c['released_gt'])
   c['known_code_precision']=c['correct']/max(1,c['evaluable_known'])
   c['code_f1']=2*c['correct']/max(1,c['released_gt']+c['evaluable_known'])
  report={'split':args.split,'images':len(samples),
    'source':'canonical released-code GT; IoU >=.5; no TEST used',
    'runtime':{'weight_hash':hashlib.sha256((ROOT/'models/sign_detector/retrained_20261009.pt').read_bytes()).hexdigest()},
    'methods':{k:dict(v) for k,v in totals.items()},
    'elapsed_sec':time.perf_counter()-t0,
    'field_video_accuracy_proven':False}
  out=ROOT/f'output/evaluation/offline_sign_quality_{args.split.lower()}_20261014';out.mkdir(parents=True,exist_ok=True)
  (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
  print('OFFLINE_SIGN_VAL_COMPLETE',json.dumps({k:(v['correct'],v['evaluable_known'],round(v['known_code_precision'],4),round(v['code_recall'],4),round(v['code_f1'],4)) for k,v in report['methods'].items()}),flush=True)

if __name__=='__main__':main()
