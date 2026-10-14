"""Replay video at 10 sampled frames/s; record confirmation delay, not field accuracy."""
from pathlib import Path
import argparse,json,time,sys
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.pipeline.orchestrator import Pipeline

def main():
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--start',type=float,default=0);p.add_argument('--output',type=Path,default=Path('output/evaluation/improvements_20261009/video'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    pipe=Pipeline();cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS)
    first=round(a.start*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,first);step=max(1,round(fps/10))
    rows=[];delays={};counts={'detections':0,'recognized':0,'pending':0};started=time.perf_counter();frame_id=first
    with (a.output/'frames.jsonl').open('w',encoding='utf-8') as log:
        while cap.grab():
            ts=frame_id/fps
            if ts>=a.start+a.seconds:break
            if (frame_id-first)%step==0:
                ok,image=cap.retrieve()
                if not ok:raise RuntimeError('Video decode failed')
                image=cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
                result=pipe.run(image,10.7769,106.7009,False,road_bottom=.65,annotate=False,stream_id='benchmark',frame_timestamp=ts)
                signs=result.perception.signs
                for s in signs:
                    if s.confirmation_delay_ms is not None:delays.setdefault(str(s.track_id),s.confirmation_delay_ms)
                row={'frame_id':frame_id,'time_seconds':ts,'total_ms':result.timings_ms['total'],'timings_ms':result.timings_ms,'signs':[asdict(s) for s in signs]}
                rows.append(row);log.write(json.dumps(row)+'\n')
                counts['detections']+=len(signs);counts['recognized']+=sum(s.label!='unknown' for s in signs)
                counts['pending']+=sum(s.recognition_source=='temporal_pending' for s in signs)
                if len(rows)%200==0:print(json.dumps({'frames':len(rows),'timestamp':ts}),flush=True)
            frame_id+=1
    cap.release();warm=[r['total_ms'] for r in rows[10:]];delay=list(delays.values())
    report={'sample_fps':fps/step,'frames':len(rows),'start_seconds':a.start,'duration_requested':a.seconds,'counts':counts,
        'processing_ms':{'mean':float(np.mean(warm)),'p95':float(np.percentile(warm,95)),'p99':float(np.percentile(warm,99)),'fraction_over_100ms':float(np.mean(np.array(warm)>100))},
        'confirmation_delay_ms':{'events':len(delay),'median':float(np.median(delay)) if delay else None,'p95':float(np.percentile(delay,95)) if delay else None},
        'accuracy':None,'latency_definition':'Video timestamp from first tracked detection to two-frame label confirmation; excludes network/display/TTS; not ground-truth first visibility',
        'wall_seconds':time.perf_counter()-started}
    (a.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
