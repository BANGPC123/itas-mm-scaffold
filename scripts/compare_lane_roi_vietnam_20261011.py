"""Compare old shared 65% and new full-frame lane input on 24 Vietnam video snapshots.

No human lane GT. Counts and 3 visual overlays are diagnostic only.
"""
from pathlib import Path
import cv2,json,numpy as np
from src.perception.lane_detector import LaneDetector
from src.utils.config import load_config
root=Path(__file__).resolve().parents[1]
folder=root/'output/review/test_video_development/images'
files=sorted(folder.glob('*_0.jpg'))
detector=LaneDetector(load_config('perception'))
rows=[];panels=[]
for j,p in enumerate(files):
    frame=cv2.imread(str(p))
    if frame is None:continue
    stats={'image':p.name}
    overlay=frame.copy()
    for name,img in [('old65',frame[:int(.65*frame.shape[0])]),('newfull',frame)]:
        prediction=detector.detect(img)
        stats[name]={'lanes':len(prediction.polylines),'ego_raw':len(prediction.ego_boundaries)}
        if name=='newfull':
            for polyline in prediction.polylines:
                cv2.polylines(overlay,[np.array(polyline,np.int32)],False,(40,220,30),2)
    rows.append(stats)
    if j in (0,5,12,18):
        cv2.putText(overlay,f'{p.name} : full UFLD {stats["newfull"]["lanes"]} lines',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,0,0),3)
        panels.append(cv2.resize(overlay,(640,360)))
result={'development_only':True,'human_ground_truth':False,'samples':len(rows),
    'frames_with_lane_old':sum(r['old65']['lanes']>0 for r in rows),
    'frames_with_lane_new':sum(r['newfull']['lanes']>0 for r in rows),
    'total_lane_predictions_old':sum(r['old65']['lanes'] for r in rows),
    'total_lane_predictions_new':sum(r['newfull']['lanes'] for r in rows),
    'rows':rows}
out=root/'output/evaluation/lane_roi_vietnam_snapshots_20261011.json'
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
if panels:
    cv2.imwrite(str(root/'output/evaluation/lane_roi_vietnam_visual_20261011.jpg'),
        np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,90])
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
