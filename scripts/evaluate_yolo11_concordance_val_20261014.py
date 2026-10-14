"""Frozen VAL audit for imported YOLO11 + independent MobileNet agreement.

Do not choose threshold using TEST. Count correct code with one-to-one IoU>0.5,
including unmatched confirmed predictions as false positives.
"""
import json,sys,hashlib,copy
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,pandas as pd,torch
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.perception.auxiliary_sign_review import AuxiliarySignReviewer
from src.perception.sign_concordance import label_agreement, SECONDARY_TO_CODE
from scripts.evaluate_product import iou_matrix
from src.utils.config import load_config

def main():
 torch.set_num_threads(4);cv2.setNumThreads(1)
 cfg=load_config('perception')
 main_detector=SignDetector(cfg);recognizer=SignRecognizer(cfg)
 extra=AuxiliarySignReviewer(cfg['auxiliary_sign_review'])
 assert extra.status=='ready'
 loc=ROOT/'data/processed/canonical/itas_mm_v1/localization'
 sample=pd.read_csv(loc/'samples.csv')
 sample=sample[sample.split=='VAL'].sort_values('image_sha256').head(240)
 gt=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv',keep_default_na=False)
 by_id={k:v.reset_index(drop=True) for k,v in gt[gt.split=='VAL'].groupby('sample_id')}
 totals=defaultdict(int)
 detail=[]
 for idx,row in enumerate(sample.itertuples(),1):
  image=cv2.imread(str(loc/row.image_path))
  main_detector.image_size=960
  primary=main_detector.detect(image)
  auxiliary=extra.detect(image,primary)
  agreed=label_agreement(recognizer,image,auxiliary)
  known=[x for x in agreed if x['agreement_status']=='cross_model_visual_agreement']
  truth=by_id[row.sample_id]
  released=truth[truth.semantic_final_disposition.str.startswith('RELEASED')]
  truth_boxes=[json.loads(v) for v in released.bbox_xyxy]
  truth_codes=list(released.canonical_sign_code)
  boxes=[x['bbox_xyxy'] for x in known]
  mat=iou_matrix(boxes,truth_boxes)
  if len(boxes) and len(truth_boxes):
   pi,gi=linear_sum_assignment(-np.where(mat>=.5,1+mat/1000,0))
   pairs=[(p,g) for p,g in zip(pi,gi) if mat[p,g]>=.5]
  else:pairs=[]
  matches=set()
  correct=0
  for p,g in pairs:
   if known[p]['recognizer_label']==truth_codes[g]:
    correct+=1
    matches.add(p)
  totals['images']+=1
  totals['baseline_localization_boxes']+=len(primary)
  totals['auxiliary_review_boxes']+=len(agreed)
  totals['strong_agreement_boxes']+=len(known)
  totals['strong_agreement_matched_correct_code']+=correct
  totals['strong_agreement_false_or_wrong']+=len(known)-correct
  totals['released_gt']+=len(released)
  if known:detail.append({'sample_id':row.sample_id,'candidates':known,'matched_correct_code':correct})
  if idx%40==0:print('CONCORDANCE_VAL',idx,'/',len(sample),dict(totals),flush=True)
 totals['agreement_precision']=totals['strong_agreement_matched_correct_code']/max(1,totals['strong_agreement_boxes'])
 report={'split':'VAL','images':len(sample),'iou':.5,
  'checkpoint_sha256':hashlib.sha256(extra.weights.read_bytes()).hexdigest(),
  'method':'YOLO11 >=.80, MobileNetV3 same mapped code >=.90, min 24px, distinct from primary >=.35 IoU',
  'results':dict(totals),'no_legal_effect':True,'not_independent_road_accuracy':True}
 out=ROOT/'output/evaluation/yolo11_concordance_val_20261014'
 out.mkdir(parents=True,exist_ok=True)
 (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 (out/'examples.json').write_text(json.dumps(detail[:30],ensure_ascii=False,indent=2),encoding='utf8')
 print('CONCORDANCE_FINAL',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
