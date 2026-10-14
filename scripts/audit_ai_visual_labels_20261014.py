"""Compare frozen, provenance-tagged AI visual annotations with live pipeline.

NOT human-verified GT. Metrics are diagnostic annotation alignment only.
No training, checkpoint selection, or field release gate may use these labels.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from collections import Counter
from PIL import ImageFont
import cv2
import numpy as np
import requests
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.field_metrics import lane_pixel_counts

OUT=ROOT/'output/ai_visual_gt_20261014'
STATUS={'annotated','no_visible_lane','uncertain','not_assessable'}

def read_data():
    labels=json.loads((OUT/'ai_visual_labels.json').read_text(encoding='utf8'))
    sources=json.loads((OUT/'selection.json').read_text(encoding='utf8'))
    assert labels.get('human_verified') is False
    assert set(labels['annotations'])==set(sources)
    return labels,sources

def image_for(source):
    path=Path(source['image'])
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==source['original_image_sha256'],path
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    assert image is not None
    w,h=source['label_image_size']
    if image.shape[:2]!=(h,w):
        image=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
    return image

def run_api(image):
    ok,data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok:raise RuntimeError('JPEG encoding failed')
    result=requests.post('http://127.0.0.1:8000/pipeline/run',
        files={'image':('review.jpg',data.tobytes(),'image/jpeg')},
        data={'latitude':'10.7769','longitude':'106.7009',
              'synthesize_audio':'false','annotate':'false',
              'road_bottom':'1','lane_visual_roi':'true'},timeout=120)
    result.raise_for_status()
    return result.json()

def polyline(draw,segments,color,thickness=3):
    for coords in segments:
        if len(coords)>=2:
            cv2.polylines(draw,[np.array(coords,np.int32)],False,color,thickness)

def overlay_frame(image,gt,pred,name):
    dst=image.copy()
    lane=pred['perception']['lane']
    polyline(dst,lane.get('display_polylines') or [],(0,190,255),2)
    polyline(dst,gt.get('road_edge_markings',[]),(30,55,220),4)
    polyline(dst,gt.get('lanes',[]),(55,220,55),4)
    for sign in gt.get('signs',[]):
        a,b,c,d=map(int,sign['bbox'])
        cv2.rectangle(dst,(a,b),(c,d),(255,0,200),2)
    for nonsign in gt.get('non_sign_objects',[]):
        a,b,c,d=map(int,nonsign['bbox'])
        cv2.rectangle(dst,(a,b),(c,d),(210,100,0),2)
    for sign in pred['perception']['signs']:
        if sign['label']!='unknown':
            a,b,c,d=map(int,sign['bbox_xyxy'])
            cv2.rectangle(dst,(a,b),(c,d),(0,240,245),1)
    cv2.rectangle(dst,(0,0,dst.shape[1],52),(20,30,40),-1)
    cv2.putText(dst,name+' | '+gt['lane_status'],(10,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(dst,'GT visible paint=GREEN | edge=RED | AI lane=YELLOW',(10,42),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    return dst

def main():
    labels,sources=read_data()
    predpath=OUT/'live_model_predictions.jsonl'
    rows={}
    if predpath.exists():
        for line in predpath.read_text(encoding='utf8').splitlines():
            if line.strip():
                item=json.loads(line);rows[item['id']]=item
    review=OUT/'comparisons'
    review.mkdir(exist_ok=True)
    failures={}
    with predpath.open('a',encoding='utf8') as sink:
        for i,(name,gt) in enumerate(labels['annotations'].items(),1):
            assert gt['lane_status'] in STATUS
            im=image_for(sources[name])
            if name not in rows:
                try:
                    result=run_api(im)
                    row={'id':name,'pipeline':result,'input_size':[im.shape[1],im.shape[0]],
                         'source_sha256':sources[name]['original_image_sha256']}
                    rows[name]=row
                    sink.write(json.dumps(row,ensure_ascii=False)+'\n');sink.flush()
                except Exception as exc:
                    failures[name]=type(exc).__name__+': '+str(exc)[:190]
                    print('REVIEW_FAIL',name,failures[name],flush=True)
                    continue
            result=rows[name]['pipeline']
            check=overlay_frame(im,gt,result,name)
            cv2.imwrite(str(review/(name+'.jpg')),check,[cv2.IMWRITE_JPEG_QUALITY,89])
            print('REVIEW_PROGRESS',i,'/',len(labels['annotations']),name,gt['lane_status'],
                  'pred_lines',len(result['perception']['lane'].get('display_polylines') or []),flush=True)

    valid=[];frame_details=[]
    totals=Counter()
    counts_sum=Counter()
    sign_checks=[]
    for name,gt in labels['annotations'].items():
        if name not in rows:continue
        pred=rows[name]['pipeline']
        h,w=rows[name]['input_size'][1],rows[name]['input_size'][0]
        points=pred['perception']['lane'].get('display_polylines') or []
        status=gt['lane_status']
        row={'id':name,'source':sources[name]['source'],'lane_status':status,
            'manual_visible_markings':len(gt.get('lanes',[])),
            'predicted_line_count':len(points)}
        totals['processed']+=1
        totals['reviewed_'+status]+=1
        if status in ('annotated','no_visible_lane'):
            totals['scorable']+=1
            if status=='annotated':
                totals['positive_scorable']+=1
                totals['positive_with_prediction']+=int(bool(points))
            else:
                totals['no_visible_scorable']+=1
                totals['no_visible_false_positive']+=int(bool(points))
            lc=lane_pixel_counts((h,w,3),gt.get('lanes',[]),points,tolerance=6)
            if lc:
                valid.append(name)
                for k,v in lc.items():counts_sum[k]+=v
                row['lane_pixel_counts']=lc
        if gt.get('signs') or gt.get('non_sign_objects'):
            known=[s for s in pred['perception']['signs'] if s['label']!='unknown']
            sign_checks.append({'frame':name,
                'visually_reviewed_signs':gt.get('signs',[]),
                'visually_reviewed_non_sign_objects':gt.get('non_sign_objects',[]),
                'model_known_signs':[{'label':s['label'],'bbox_xyxy':s['bbox_xyxy'],
                   'car_scope':s['car_context'].get('status')} for s in known],
                'not_exhaustively_reviewed':True})
        frame_details.append(row)
    a,b,c,d=(counts_sum[k] for k in
        ['matched_pred_pixels','pred_pixels','matched_gt_pixels','gt_pixels'])
    precision=a/b if b else None
    recall=c/d if d else None
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0.
    report={'schema_version':'itasmm-ai-visual-agreement-v1',
        'annotation_source':'ChatGPT visual/manual AI labeling, NOT HUMAN GROUND TRUTH',
        'human_verified':False,'field_accuracy_validated':False,
        'visual_review_frames':len(labels['annotations']),
        'scored_frames':totals['scorable'],
        'positive_visible_lane_frames':totals['positive_scorable'],
        'negative_no_visible_lane_frames':totals['no_visible_scorable'],
        'forward_positives_with_model_line':totals['positive_with_prediction'],
        'no_visible_lane_frames_with_model_false_line':totals['no_visible_false_positive'],
        'excluded_uncertain_or_nonforward':len(labels['annotations'])-totals['scorable'],
        'alignment_metric':{
            'precision_at_6px':precision,'recall_at_6px':recall,'f1_at_6px':f1,
            'matched_prediction_pixels':a,'prediction_pixels':b,
            'matched_label_pixels':c,'label_pixels':d,
            'score_is_not_independent_accuracy':True,
            'labels_do_not_interpolate_occlusions':True},
        'sign_spotcheck':sign_checks,
        'frames':frame_details,'errors':failures,
        'camera_GPS_is_placeholder':True,
        'scope':'Research error analysis only; prohibits model selection on this pseudo-GT',
        'checklist':['User must manually approve or correct label marks',
                     'Collect source-disjoint human-reviewed positive lane examples',
                     'Blind evaluator and separate independent test split',
                     'Obtain exhaustive sign labels before precision/recall']}
    (OUT/'visual_agreement_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('SCORE',json.dumps({k:v for k,v in report.items() if k not in
       ('frames','sign_spotcheck')},ensure_ascii=False),flush=True)
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
