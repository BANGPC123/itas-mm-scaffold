"""ROI A/B on bundled UFLD highway video; detection counts, NOT F1."""
from pathlib import Path
import cv2,json,time
from src.utils.config import load_config
from src.perception.lane_detector import LaneDetector
from src.perception.lane_consistency import credible_ego
root=Path(__file__).resolve().parents[1]
cap=cv2.VideoCapture(str(root/'data/demo/ufld_example.mp4'))
model=LaneDetector(load_config('perception'))
rows=[]
for frameid in range(0,1000,40):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frameid)
    ok,frame=cap.read()
    if not ok:break
    entry={'frame':frameid}
    for label,im in [('shared_65',frame[:int(frame.shape[0]*.65)]),('full_lane',frame)]:
        begin=time.perf_counter()
        lane=model.detect(im)
        entry[label]={'polylines':len(lane.polylines),'ego_raw':len(lane.ego_boundaries),
          'ego_credible':credible_ego(lane.ego_boundaries,im.shape),
          'milliseconds':round((time.perf_counter()-begin)*1000,2)}
    rows.append(entry)
cap.release()
summary={name:{'frames_with_lanes':sum(bool(r[name]['polylines']) for r in rows),
               'mean_polylines':round(sum(r[name]['polylines'] for r in rows)/len(rows),2),
               'credible_ego_frames':sum(r[name]['ego_credible'] for r in rows),
               'mean_ms':round(sum(r[name]['milliseconds'] for r in rows)/len(rows),2)}
 for name in ('shared_65','full_lane')}
report={'video':'data/demo/ufld_example.mp4','frames':len(rows),'ground_truth':False,'summary':summary,'rows':rows}
dest=root/'output/evaluation/lane_roi_ab_20261011.json'
dest.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'n':len(rows),'summary':summary,'output':str(dest)}),flush=True)
