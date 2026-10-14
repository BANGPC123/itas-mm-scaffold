"""Use existing Vietnamese labeled VAL to assess whether YOLO11 labels can rescue
unrecognized *already localized* original sign boxes.

No model tuning/test-set selection from TEST. This is off-line candidate study
and no candidate changes legal applicability automatically.
"""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
from collections import defaultdict
import cv2,pandas as pd,torch
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.perception.sign_multiscale import recognize_views
from src.perception.sign_concordance import SECONDARY_TO_CODE
from src.perception.auxiliary_sign_review import box_iou,AuxiliarySignReviewer
from src.utils.config import load_config
from scripts.evaluate_sign_candidates import score

def rescue(base,aux,*,threshold=.75,iou=.40):
    output=copy.deepcopy(base)
    for s in output:
        if s.label!='unknown' or s.recognition_source in (
            'outside_supported_reference','symbol_shape_unsupported','background_rejected'):
            continue
        matches=[p for p in aux if p['score']>=threshold
             and box_iou(p['bbox'],s.bbox_xyxy)>=iou and p['code'] is not None]
        matches=sorted(matches,key=lambda r:r['score'],reverse=True)
        if not matches:continue
        candidate=matches[0]
        if len(matches)>1 and matches[1]['code']!=candidate['code'] and matches[1]['score']>=candidate['score']-.12:
            continue
        s.label=candidate['code']
        s.recognition_confidence=candidate['score']
        s.recognition_source='aux_localized_label_candidate'
        s.label_verified=False
        s.context_verified=False
    return output

def run(limit=400):
    torch.set_num_threads(4);cv2.setNumThreads(2)
    cfg=load_config('perception')
    det=SignDetector(cfg);rec=SignRecognizer(cfg)
    aux=AuxiliarySignReviewer(cfg['auxiliary_sign_review'])
    assert aux.status=='ready'
    root=ROOT/'data/processed/canonical/itas_mm_v1/localization'
    samples=pd.read_csv(root/'samples.csv')
    samples=samples[samples.split=='VAL'].sort_values('image_sha256').head(limit)
    gt=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv',keep_default_na=False)
    groups={k:v for k,v in gt[gt.split=='VAL'].groupby('sample_id')}
    methods={'baseline':None}
    for iou in (.3,.45,.6):
        for threshold in (.50,.65,.8,.9):
            methods[f'rescue_{iou}_{threshold}']=(iou,threshold)
    totals={name:defaultdict(int) for name in methods}
    raw_count=0;eligible=0
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):
        for count,s in enumerate(samples.itertuples(),1):
            image=cv2.imread(str(root/s.image_path))
            det.image_size=960;aa=det.detect(image)
            det.image_size=640;bb=det.detect(image)
            primary=recognize_views(rec,image,aa,bb,min_short_side=12)
            # Direct raw model inference deliberately includes overlaps with
            # primary boxes; the normal preview suppresses such boxes.
            preds=aux.model.predict(image,conf=.25,iou=.45,imgsz=640,
               device=0 if torch.cuda.is_available() else 'cpu',verbose=False,max_det=80)
            overlays=[]
            for row in preds:
                for b in row.boxes:
                    ident=int(b.cls[0]);code=SECONDARY_TO_CODE.get(ident)
                    overlays.append({'bbox':b.xyxy[0].tolist(),'score':float(b.conf[0]),
                        'code':code})
            raw_count+=len(overlays)
            eligible+=sum(x['code'] is not None for x in overlays)
            for name,settings in methods.items():
                predicted=primary if settings is None else rescue(primary,overlays,iou=settings[0],threshold=settings[1])
                score(predicted,groups[s.sample_id],totals[name],[],s.sample_id,name)
            if count%80==0:print('VAL_RESCUE_PROGRESS',count,len(samples),'raw',raw_count,flush=True)
    summary={}
    for name,m in totals.items():
        tp=m['correct'];p=m['evaluable_known'];g=m['released_gt']
        summary[name]={'correct':tp,'pred_known':p,'released_gt':g,
           'precision':tp/max(1,p),'recall':tp/max(1,g),'f1':2*tp/max(1,p+g),
           'wrong_known':m['wrong_known'],'unmatched_known':m['unmatched_known']}
    result={'split':'VAL','images':len(samples),'results':summary,
           'aux_predictions':raw_count,'mapped_aux_predictions':eligible,
           'candidate_selection_on_test':False,
           'warnings':['If precision/F1 not improved, never promote auxiliary guesses into primary signs.']}
    out=ROOT/'output/evaluation/yolo11_primary_rescue_val_20261014'
    out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('RESCUE_RESULT',json.dumps(summary),flush=True)
if __name__=='__main__':run()
