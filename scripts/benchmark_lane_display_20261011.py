"""Lane display jitter diagnostic on the bundled UFLD video.

No lane GT, so report ONLY temporal curve displacement, never precision/F1.
"""
from pathlib import Path
import json,cv2,numpy as np,time
from src.perception.lane_detector import LaneDetector
from src.perception.lane_temporal import LaneStabilizer,_curve_match
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/demo/ufld_example.mp4'
cap=cv2.VideoCapture(str(source))
fps=cap.get(cv2.CAP_PROP_FPS)
frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if fps<=0:raise RuntimeError('Video FPS unavailable')
detector=LaneDetector(load_config('perception'))
track=LaneStabilizer()
previous_raw=[];previous_display=[]
stats={'frames_checked':0,'frames_with_lane':0,'matched_curves':0,'raw_motion_x':[],'display_motion_x':[]}
start=time.perf_counter()
for frame_id in range(0,min(frames,750),10):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
    ok,img=cap.read()
    if not ok:continue
    roi=img
    lane=detector.detect(roi)
    display=track.update(lane.polylines,frame_id/fps,roi.shape)
    assert len(display)==len(lane.polylines)
    stats['frames_checked']+=1
    stats['frames_with_lane']+=bool(lane.polylines)
    for a,b in zip(lane.polylines,display):
        old=[(dist,k) for k,p in enumerate(previous_raw)
             if (dist:=_curve_match(a,p,roi.shape[1])) is not None]
        if not old:continue
        _,idx=min(old)
        rawmotion=_curve_match(a,previous_raw[idx],roi.shape[1])
        displaymotion=_curve_match(b,previous_display[idx],roi.shape[1]) if idx<len(previous_display) else None
        if rawmotion is None or displaymotion is None:continue
        stats['matched_curves']+=1
        stats['raw_motion_x'].append(rawmotion)
        stats['display_motion_x'].append(displaymotion)
    previous_raw=lane.polylines
    previous_display=display
cap.release()
result={'source':str(source),'frames':stats['frames_checked'],'frames_with_lane':stats['frames_with_lane'],
 'matched_curves':stats['matched_curves'],
 'mean_interframe_displacement_px_raw':float(np.mean(stats['raw_motion_x'])) if stats['raw_motion_x'] else None,
 'mean_interframe_displacement_px_display':float(np.mean(stats['display_motion_x'])) if stats['display_motion_x'] else None,
 'seconds':time.perf_counter()-start,
 'accuracy_verified':False,'scope':'bundled demo, full-frame lane input, 10-frame sampling, no human GT. Display-only smoothing; no extrapolation of missing lanes.'}
out=ROOT/'output/evaluation/lane_temporal_demo_20261011.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
