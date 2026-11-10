"""Spot-check existing UFLD-v2 on new, independently sourced Vietnam videos.

Stores prediction overlays, never turns predicted polylines into reviewer GT.
"""
from pathlib import Path
import cv2,json,numpy as np
from src.perception.lane_detector import LaneDetector
from src.utils.config import load_config
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/commons_lane_spotcheck_20261011'
OUT.mkdir(parents=True,exist_ok=True)
lane=LaneDetector(load_config('perception'))
sources=['commons_hanoi_20261011','commons_saigon_govap_20261011']
rows=[]
for src in sources:
    pack=ROOT/'output/review'/src
    doc=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
    frames=doc['frames']
    step=max(1,len(frames)//6)
    for f in frames[::step][:6]:
        im=cv2.imread(str(pack/f['image']))
        if im is None:raise RuntimeError('Image corrupt')
        result=lane.detect(im)
        polylines=result.polylines
        overlay=im.copy()
        for pts in polylines:
            if len(pts)<2:continue
            path=np.asarray(pts,dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(overlay,[path],False,(0,255,255),3)
        name=f'{src}_{f["id"]}.jpg'
        cv2.imwrite(str(OUT/name),overlay,[cv2.IMWRITE_JPEG_QUALITY,90])
        row={'source_group':f['source_group'],'frame_id':f['id'],'backend':result.backend,
             'polylines_predicted':len(polylines),
             'ego_boundaries_predicted':len(result.ego_boundaries),
             'rejection_reason':result.rejection_reason,
             'human_ground_truth':False,'preview':name}
        rows.append(row)
        print('LANE_QA '+json.dumps(row),flush=True)
report={'frames_evaluated':len(rows),'sources':sources,'model_checkpoint':'models/lane/selected_20261009.pth',
       'status':'VISUAL_PREDICTIONS_ONLY_NO_METRICS',
       'human_verified_lane_lines':0,
       'notes':'New footage has no human lane polylines yet. Cannot report independent F1.',
       'results':rows}
(OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE '+json.dumps({'frames':len(rows),'with_predicted_lanes':sum(x['polylines_predicted']>0 for x in rows)},ensure_ascii=False),flush=True)
