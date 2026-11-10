"""Non-destructive end-to-end smoke test on review snapshots (not independent TEST)."""
from pathlib import Path
import json, time, cv2
from src.pipeline.orchestrator import Pipeline

root=Path(__file__).resolve().parents[1]
images=root/'output/review/test_video_development/images'
samples=[images/f'{i:03}_0.jpg' for i in (0,3,8,12,18,23)]
missing=[str(p) for p in samples if not p.is_file()]
if missing:raise RuntimeError('Missing video review snapshots: '+str(missing))
pipeline=Pipeline()
rows=[]
for path in samples:
    frame=cv2.imread(str(path))
    if frame is None:raise RuntimeError('Cannot decode '+str(path))
    t=time.perf_counter()
    result=pipeline.run(frame,10.7769,106.7009,False,road_bottom=.65,annotate=False)
    bad=[s.label for s in result.perception.signs if s.applicability=='applies' and not(s.label_verified and s.context_verified)]
    if bad or result.reasoning.speech_text:raise AssertionError('Unverified video frame generated a safety instruction '+str(bad))
    rows.append({'file':path.name,'size':frame.shape[:2],
                 'predicted_signs':[{'code':s.label,'scope':s.applicability} for s in result.perception.signs],
                 'lane_polylines':len(result.perception.lane.polylines),
                 'ego_boundaries':len(result.perception.lane.ego_boundaries),
                 'spoke':bool(result.reasoning.speech_text),
                 'elapsed_ms':round((time.perf_counter()-t)*1000,2)})
out=root/'output/evaluation/strict_pipeline_smoke_20261011.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'scope':'6 development video snapshots, no ground-truth accuracy',
   'pass':True,'screenshots':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pass':True,'snapshots':rows,'report':str(out)},ensure_ascii=False),flush=True)
