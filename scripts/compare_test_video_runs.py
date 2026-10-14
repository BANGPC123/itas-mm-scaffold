"""Compare two development-video replays without calling prediction counts accuracy."""
from pathlib import Path
import json
import csv
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'output/retraining_20261009/video_full'
new=ROOT/'output/video_tests/test_video_20261010_after_hardening'
a=json.loads((old/'summary.json').read_text(encoding='utf-8'))
b=json.loads((new/'summary.json').read_text(encoding='utf-8'))
keys=['sampled_frames','frames_with_detections','frames_with_known_predictions','frames_with_lane_boundaries','frames_with_ego_geometry','detection_observations','known_code_observations','unknown_observations','pipeline_mean_ms_excluding_first','pipeline_p95_ms_excluding_first','elapsed_wall_seconds']
differences={k:{'previous':a.get(k),'current':b.get(k),'delta':round(b.get(k,0)-a.get(k,0),3)} for k in keys}
with (old/'predictions.jsonl').open(encoding='utf-8') as f:
    pa={row['frame_id']:row for line in f if (row:=json.loads(line))}
with (new/'predictions.jsonl').open(encoding='utf-8') as f:
    pb={row['frame_id']:row for line in f if (row:=json.loads(line))}
common=sorted(pa.keys()&pb.keys())
changed=[]
for i in common:
    x,y=pa[i],pb[i]
    labels_old=Counter(z['label'] for z in x.get('signs',[]))
    labels_new=Counter(z['label'] for z in y.get('signs',[]))
    if x['lane_boundaries']!=y['lane_boundaries'] or x['ego_geometry_available']!=y['ego_geometry_available'] or labels_old!=labels_new:
        changed.append({'frame_id':i,'second':y['timestamp_seconds'],
          'lane_before':x['lane_boundaries'],'lane_after':y['lane_boundaries'],
          'ego_before':x['ego_geometry_available'],'ego_after':y['ego_geometry_available'],
          'signs_before':dict(labels_old),'signs_after':dict(labels_new)})
report={'scope':'Development video, no human ground truth. Repeated predictions are not distinct signs or accuracy.', 'same_frame_count':len(common),
        'frames_only_previous':len(pa)-len(common),'frames_only_current':len(pb)-len(common),
        'comparisons':differences,'changes_count':len(changed),'changed_examples':changed[:30]}
(new/'comparison_vs_20261009.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
