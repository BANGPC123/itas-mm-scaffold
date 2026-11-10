"""Make a ready-to-play *offline MP4* capstone demo, no real camera/GPS.

Renders detected traffic signs, raw/display lane polylines, simulated GPS zone
and a citation index. Narration is explicitly DEMO status text, never a
ground-truth legal or road safety command.
"""
from pathlib import Path
import argparse,cv2,json,os,time,hashlib,sys,subprocess
import numpy as np
from src.pipeline.orchestrator import Pipeline
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'data/demo/ufld_example.mp4')
    p.add_argument('--out',type=Path,default=ROOT/'output/capstone_demo_20261011')
    p.add_argument('--seconds',type=float,default=12)
    p.add_argument('--fps',type=int,default=15)
    p.add_argument('--analysis-interval',type=float,default=.2,
                   help='minimum video time between inference calls; overlays expire after 0.6s')
    p.add_argument('--lat',type=float,default=10.7769)
    p.add_argument('--lon',type=float,default=106.7009)
    p.add_argument('--with-narration',action='store_true')
    args=p.parse_args()
    if not(1<=args.seconds<=120 and 1<=args.fps<=30 and .04<=args.analysis_interval<=2):
        raise ValueError('Invalid output length/FPS/analysis interval')
    if args.out.exists() and (args.out/'summary.json').exists():raise FileExistsError('Do not overwrite previous capstone video')
    args.out.mkdir(parents=True,exist_ok=True)
    v=cv2.VideoCapture(str(args.source))
    if not v.isOpened():raise ValueError('Cannot open provided video')
    orig_fps=v.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(orig_fps) or orig_fps<=1:raise ValueError('Input video FPS unknown')
    total=v.get(cv2.CAP_PROP_FRAME_COUNT)
    if total<=0:raise ValueError('Input video length unknown')
    fps=min(args.fps,max(1,int(orig_fps)))
    length=min(args.seconds,total/orig_fps)
    target_frames=max(1,int(length*fps))
    writer=cv2.VideoWriter(str(args.out/'video_annotated.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
    if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
    model=Pipeline()
    stream_id='capstone_offline_'+hashlib.sha256(str(args.source.resolve()).encode()).hexdigest()[:12]
    report=[];start=time.perf_counter()
    last_analysis=-100.;last_result=None;scored=0
    try:
        for k in range(target_frames):
            timestamp=k/fps
            v.set(cv2.CAP_PROP_POS_MSEC,timestamp*1000)
            ok,frame=v.read()
            if not ok:break
            if frame.shape[:2]!=(720,1280):
                frame=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
            must=last_result is None or timestamp-last_analysis>=args.analysis_interval
            if must:
                result=model.run(frame,args.lat,args.lon,False,road_bottom=.65,
                                 annotate=False,stream_id=stream_id,frame_timestamp=timestamp)
                last_result=result;last_analysis=timestamp;scored+=1
                report.append({'video_s':round(timestamp,3),
                  'lane_polylines':len(result.perception.lane.polylines),
                  'ego_boundaries':len(result.perception.lane.ego_boundaries),
                  'simulated_zone':result.context.zone_label,
                  'verified_zone_from_signs':result.verified_zone_status,
                  'signs':[{'code':s.label,'bbox_xyxy':list(s.bbox_xyxy),
                       'confidence':round(float(s.confidence),3),
                       'scope':s.applicability,'verified':bool(s.context_verified and s.label_verified)}
                           for s in result.perception.signs],
                  'legal_reference_source_count':len(result.reasoning.references),
                  'legal_references':[{'code':r.get('code'), 'page':r.get('page'),
                          'document':r.get('edition') or r.get('document'),'source_url':r.get('source_url')}
                           for r in result.reasoning.references],
                  'legal_penalty_verified':False,
                  'pipeline_timings_ms':result.timings_ms})
            overlay=frame.copy()
            if last_result is not None and timestamp-last_analysis<=.6:
                ln=last_result.perception.lane
                lines=ln.display_polylines if ln.display_polylines is not None else ln.polylines
                for pts in lines:
                    if len(pts)>=2:
                        cv2.polylines(overlay,[np.asarray(pts,dtype=np.int32)],False,(65,230,90),3,cv2.LINE_AA)
                for sign in last_result.perception.signs:
                    box=tuple(int(v) for v in sign.bbox_xyxy)
                    color=(0,215,255) if sign.applicability=='uncertain' else (45,215,90)
                    cv2.rectangle(overlay,box[:2],box[2:],color,2)
                    cv2.putText(overlay,sign.label,(max(0,box[0]),max(28,box[1]-8)),
                                cv2.FONT_HERSHEY_SIMPLEX,.72,color,2,cv2.LINE_AA)
            cv2.rectangle(overlay,(0,0),(1280,80),(15,27,40),-1)
            cv2.putText(overlay,'ITAS-MM  |  OFFLINE MP4 DEMO   /   YOLO11n + UFLD-v2 + GPS(sim) + Legal PDF RAG',
                (18,30),cv2.FONT_HERSHEY_SIMPLEX,.72,(245,245,245),2,cv2.LINE_AA)
            name=last_result.context.zone_label if last_result else 'unknown'
            cv2.putText(overlay,f'Zone(sim): {name} | t={timestamp:5.1f}s | citations: {len(last_result.reasoning.references) if last_result else 0}',
                (18,63),cv2.FONT_HERSHEY_SIMPLEX,.66,(155,220,245),2,cv2.LINE_AA)
            cv2.rectangle(overlay,(0,671),(1280,719),(12,20,31),-1)
            cv2.putText(overlay,'RESEARCH DEMO: AI predictions unverified; no legal fine or mandatory driving action inferred',
                (18,701),cv2.FONT_HERSHEY_SIMPLEX,.57,(185,205,225),2,cv2.LINE_AA)
            writer.write(overlay)
            if (k+1)%60==0:print('RENDERING_FRAME',k+1,'/',target_frames,flush=True)
    finally:
        writer.release();v.release()
    video=args.out/'video_annotated.mp4'
    if not video.is_file() or video.stat().st_size<100_000:
        raise RuntimeError('Rendering did not produce a valid MP4')
    narration=None;combined=None
    if args.with_narration:
        try:
            from src.interaction.tts_engine import TtsEngine
            narration=Path(TtsEngine(load_config('interaction')).synthesize(
                'Đây là bản demo ITAS MM. GPS đang được mô phỏng.'))
        except Exception as exc:
            print('TTS_UNAVAILABLE_VIDEO_STILL_VALID',type(exc).__name__,flush=True)
        if narration and narration.is_file():
            try:
                import imageio_ffmpeg
                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
                combined=args.out/'video_with_narration.mp4'
                command=[ffmpeg,'-y','-loglevel','error','-i',str(video),'-i',str(narration),
                    '-filter_complex','[1:a]apad[a]','-map','0:v:0','-map','[a]',
                    '-c:v','copy','-c:a','aac','-t',str(target_frames/fps),
                    '-movflags','+faststart',str(combined)]
                process=subprocess.run(command,text=True,capture_output=True,timeout=90)
                if process.returncode!=0:
                    combined=None
                    print('NONFATAL_AUDIO_MUX_FAILED',process.stderr[:1000],flush=True)
            except Exception as exc:
                combined=None
                print('NONFATAL_AUDIO_MUX_FAILED',type(exc).__name__,flush=True)
    with (args.out/'events.jsonl').open('w',encoding='utf-8') as fp:
        for row in report:fp.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={
        'source':str(args.source.resolve()),'source_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'video':str(video),'video_with_narration':str(combined) if combined and combined.is_file() else None,
        'video_bytes':video.stat().st_size,
        'video_frames_requested':target_frames,'fps_output':fps,
        'inference_events':len(report),
        'sign_observations':sum(len(x['signs']) for x in report),
        'lane_positive_inference_frames':sum(x['lane_polylines']>0 for x in report),
        'offloaded_legal_source_reference_hits':sum(x['legal_reference_source_count'] for x in report),
        'simulated_gps':[args.lat,args.lon],
        'legal_penalties_verified':False,'camera_hardware_required':False,
        'narration_is_neutral_research_demo':True,'model_weights_modified':False,
        'elapsed_seconds':round(time.perf_counter()-start,2),
        'ground_truth_accuracy_evaluated':False}
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CAPSTONE_MP4_READY',json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
