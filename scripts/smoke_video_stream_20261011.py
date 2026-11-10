"""Bundled demo stream integration smoke for both vision models and all 3 rules.

Real checkpoint inference, 5 FPS sampled, NO legal advice or TTS on unknown
scope. No human ground truth; this is runtime functionality not accuracy.
"""
from pathlib import Path
import json,cv2,time,statistics
from src.pipeline.orchestrator import Pipeline
root=Path(__file__).resolve().parents[1]
video=root/'data/demo/ufld_example.mp4'
cap=cv2.VideoCapture(str(video))
pipe=Pipeline()
fps=cap.get(cv2.CAP_PROP_FPS)
rows=[];start=time.perf_counter()
for idx in range(0,100,5):
    cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
    ok,frame=cap.read()
    if not ok:break
    result=pipe.run(frame,10.7769,106.7009,False,annotate=False,road_bottom=.65,
        stream_id='smoke-demo',frame_timestamp=idx/fps,exclude_rects=[[0,.9,.12,.99]])
    assert all(s.applicability!='applies' for s in result.perception.signs), 'Unverified sign got legal scope'
    assert result.reasoning.speech_text==''
    assert result.perception.lane.display_polylines is not None
    assert len(result.perception.lane.display_polylines)==len(result.perception.lane.polylines)
    rows.append({'frame':idx,'signs':[s.label for s in result.perception.signs],
        'supplement_proposals':sum(s.is_compound for s in result.perception.signs),
        'lane_count':len(result.perception.lane.polylines),
        'credible_ego':bool(result.perception.lane.ego_boundaries),
        'spoken':bool(result.reasoning.speech_text),'total_ms':round(result.timings_ms['total'],2)})
cap.release()
summary={'video':str(video),'sampled_frames':len(rows),'frames_with_lane':sum(x['lane_count']>0 for x in rows),
 'frames_with_ego':sum(x['credible_ego'] for x in rows),
 'known_sign_observations':sum(sum(s!='unknown' for s in x['signs']) for x in rows),
 'unauthorized_speech_events':sum(x['spoken'] for x in rows),
 'mean_pipeline_ms':round(statistics.mean(x['total_ms'] for x in rows[1:]),2),
 'elapsed_seconds':round(time.perf_counter()-start,2),
 'verdict':'runtime_smoke_pass_not_accuracy',
 'independent_human_ground_truth':False,'frames':rows}
dest=root/'output/evaluation/video_stream_integration_20261011.json'
dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='frames'},ensure_ascii=False),flush=True)
