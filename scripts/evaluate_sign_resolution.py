"""Compare localization/code recovery with existing canonical VAL/TEST GT."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,time
import cv2,pandas as pd,numpy as np,torch
from scipy.optimize import linear_sum_assignment
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.utils.config import load_config
from scripts.evaluate_product import iou_matrix

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['VAL','TEST'],required=True)
    ap.add_argument('--min-short-side',type=int,default=32);ap.add_argument('--limit',type=int)
    ap.add_argument('--reference-bank',type=Path);ap.add_argument('--reference-margin',type=float,default=.015)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    loc=ROOT/'data/processed/canonical/itas_mm_v1/localization'
    samples=pd.read_csv(loc/'samples.csv');samples=samples[samples.split==args.split].sort_values('image_sha256')
    if args.limit:samples=samples.head(args.limit)
    gt=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv',keep_default_na=False)
    groups={key:g for key,g in gt[gt.split==args.split].groupby('sample_id')}
    cfg=load_config('perception')
    if args.reference_bank:
        cfg['sign_recognizer']['reference_gate_path']=str(args.reference_bank)
        cfg['sign_recognizer']['reference_gate_margin']=args.reference_margin
    det=SignDetector(cfg);rec=SignRecognizer(cfg);rows=[];totals={}
    for size in [640,960]:
        totals[size]={'images':0,'released_gt':0,'released_ge32':0,'correct':0,'correct_ge32':0,
            'detections':0,'localization_tp':0,'all_gt':0,'evaluable_known':0,'wrong_known':0,'unmatched_known':0}
    for number,s in enumerate(samples.itertuples(),1):
        image=cv2.imread(str(loc/s.image_path));g=groups[s.sample_id].reset_index(drop=True)
        boxes=[json.loads(x) for x in g.bbox_xyxy];released=g.semantic_final_disposition.str.startswith('RELEASED')
        for size,c in totals.items():
            det.image_size=size;signs=rec.recognize(image,det.detect(image),min_short_side=args.min_short_side)
            signs=[q for q in signs if q.recognition_source!='background_rejected']
            c['images']+=1;c['detections']+=len(signs);c['released_gt']+=int(released.sum());c['all_gt']+=len(g)
            c['released_ge32']+=int((released&(g.short_side_px>=32)).sum())
            mat=iou_matrix([q.bbox_xyxy for q in signs],boxes)
            pi,gi=linear_sum_assignment(-np.where(mat>=.5,1+mat/1000,0))
            matched=set()
            for p,j in zip(pi,gi):
                if mat[p,j]<.5:continue
                matched.add(p);c['localization_tp']+=1
                if not released.iloc[j]:continue
                q=signs[p];truth=g.iloc[j];correct=q.label==truth.canonical_sign_code
                c['correct']+=int(correct)
                c['correct_ge32']+=int(correct and truth.short_side_px>=32)
                c['evaluable_known']+=int(q.label!='unknown');c['wrong_known']+=int(q.label!='unknown' and not correct)
                rows.append({'sample_id':s.sample_id,'size':size,'truth':truth.canonical_sign_code,
                    'prediction':q.label,'short_side_px':float(truth.short_side_px),'iou':float(mat[p,j])})
            unmatched=sum(q.label!='unknown' for p,q in enumerate(signs) if p not in matched)
            c['evaluable_known']+=unmatched;c['unmatched_known']+=unmatched
        if number%100==0:print('SIGN_EVAL',args.split,number,'/',len(samples),flush=True)
    for c in totals.values():
        c['code_recall']=c['correct']/max(1,c['released_gt'])
        c['code_recall_ge32']=c['correct_ge32']/max(1,c['released_ge32'])
        c['known_code_precision']=c['correct']/max(1,c['evaluable_known'])
        c['localization_precision']=c['localization_tp']/max(1,c['detections'])
        c['localization_recall']=c['localization_tp']/max(1,c['all_gt'])
    args.output.mkdir(parents=True)
    report={'split':args.split,'images':len(samples),'source':'canonical released semantic GT; IoU .5 one-to-one',
        'selection':'SHA256 order','min_crop_short_side':args.min_short_side,'recognizer_threshold':rec.threshold,'results':totals,
        'reference_bank':str(args.reference_bank) if args.reference_bank else cfg['sign_recognizer'].get('reference_gate_path'),
        'reference_margin':getattr(rec.reference_gate,'margin',None),
        'masked_semantics_excluded_from_known_code_precision':True,'video_field_accuracy':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(args.output/'matched_predictions.csv',index=False)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
