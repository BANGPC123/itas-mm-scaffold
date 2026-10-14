"""Complete native-FPS development-video replay, before/after evidence and TTS.

The before implementation is read from an explicit local Git commit. This is
development validation, never an independent accuracy benchmark without GT.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,copy,json,time,subprocess,hashlib,types,base64
from dataclasses import asdict
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from src.pipeline.orchestrator import Pipeline
from src.perception.sign_catalog import display_name
from scripts.render_product_video import mix_speech
from src.perception.overlay_motion import OverlayMotion

RUNTIME_FILES=['configs/perception.yaml','src/pipeline/orchestrator.py','src/utils/types.py',
    'src/perception/sign_detector.py','src/perception/sign_multiscale.py',
    'src/perception/crosswalk_veto.py',
    'src/context/car_context.py','src/context/time_scope.py','src/context/sign_scope_principles.py',
    'src/perception/camera_lane.py','src/perception/sign_recognizer.py','src/perception/sign_open_set.py',
    'src/perception/supplementary_reader.py','src/perception/supplementary_semantics.py',
    'src/perception/supplementary_panel_proposals.py','src/perception/lane_temporal.py',
    'src/perception/traffic_symbols.py','src/perception/ocr_line_order.py','src/perception/overlay_motion.py',
    'src/perception/road_surface_veto.py','src/perception/temporal_signs.py','scripts/supplementary_ocr_worker.py',
    'scripts/complete_test_video.py','data/reference/qcvn2024_symbols.json',
    'models/sign_recognizer/reference_gate_20261011_final.npz']


def draw_frame(frame,payload,lines,boxes,timestamp,font,fps,analysis_hz):
    h,w=frame.shape[:2]
    for xy in lines:
        if len(xy)>=2:cv2.polylines(frame,[np.asarray(xy,np.int32)],False,(40,220,255),2)
    labels=[]
    for sign,box in zip(payload['signs'],boxes):
        if box is None or min(box)<0:continue
        x1,y1,x2,y2=map(round,box)
        color=(120,120,120) if sign['label']=='unknown' else (0,220,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        labels.append((x1,max(68,y1-22),display_name(sign['label']),color[::-1]))
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pil)
    draw.rectangle((0,0,w,57),fill=(15,23,34))
    draw.text((10,3),f'ITAS-MM | {timestamp/60:.2f} phút | Video {fps:g} FPS | AI {analysis_hz:g} Hz | Ô tô con',font=font,fill='white')
    view=payload['lane'].get('camera_view','unknown')
    message='Camera quay ngang/vào cabin: không gán biển trong gương cho hướng đi.' if view=='side_or_cabin' else 'Vạch sơn quan sát được; không nối nét qua xe hoặc vùng thiếu bằng chứng.'
    draw.text((10,29),message,font=font,fill='white')
    for x,y,text,color in labels:
        x=max(4,min(x,w-int(draw.textlength(text,font=font))-8))
        box=draw.textbbox((x,y),text,font=font)
        draw.rectangle((box[0]-2,box[1]-2,box[2]+2,box[3]+2),fill=(15,23,34));draw.text((x,y),text,font=font,fill=color)
    contexts=[]
    for sign in payload['signs']:
        if sign['label']=='unknown':continue
        context=sign.get('car_context',{})
        if context:contexts.append(sign['label']+': '+context['text'])
    if contexts:
        wrapped=[]
        for text in dict.fromkeys(contexts):
            line=''
            for word in text.split():
                if line and draw.textlength(line+' '+word,font=font)>w-24:wrapped.append(line);line=word
                else:line=(line+' '+word).strip()
            if line:wrapped.append(line)
        wrapped=wrapped[:3];top=h-24*len(wrapped)-8
        draw.rectangle((0,top,w,h),fill=(15,23,34))
        for i,text in enumerate(wrapped):draw.text((10,top+3+24*i),text,font=font,fill=(222,237,255))
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--analysis-hz',type=float,default=3)
    ap.add_argument('--baseline',default='167647521558ed6d5bdef2bb77156515ff7a7552')
    ap.add_argument('--baseline-predictions',type=Path)
    ap.add_argument('--baseline-side',choices=['before','after'],default='before')
    args=ap.parse_args();out=args.output
    runtime_snapshot={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in RUNTIME_FILES}
    if out.exists():raise FileExistsError('Use a new output directory; prior evidence is immutable')
    out.mkdir(parents=True);(out/'snapshots').mkdir()
    cap=cv2.VideoCapture(str(args.source));fps=cap.get(cv2.CAP_PROP_FPS)
    expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));sw,sh=int(cap.get(3)),int(cap.get(4))
    if not cap.isOpened() or fps<=0:raise ValueError('Unreadable source')
    step=max(1,round(fps/args.analysis_hz));w,h=1280,round(sh*1280/sw)//2*2
    from threadpoolctl import threadpool_limits
    blas_limit=threadpool_limits(limits=1,user_api='blas')
    started=time.perf_counter();pipeline=Pipeline();baseline=copy.copy(pipeline)
    if not args.baseline_predictions:
        old=subprocess.check_output(['git','show',args.baseline+':src/pipeline/orchestrator.py'],cwd=ROOT)
        namespace={'__name__':'itas_baseline_replay'};exec(compile(old,'baseline_orchestrator.py','exec'),namespace)
        baseline.run=types.MethodType(namespace['Pipeline'].run,baseline)
    baseline.auto_supplementary_enabled=False
    baseline_log=None
    if args.baseline_predictions:
        prior=json.loads((args.baseline_predictions.parent/'summary.json').read_text(encoding='utf-8'))
        if not prior['complete_source_scan'] or prior['source_frames']!=expected or abs(prior['analysis_hz']-fps/step)>.001:
            raise ValueError('Baseline scan does not match source or analysis cadence')
        digest=hashlib.sha256()
        with args.source.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=prior['source_sha256']:raise ValueError('Different source bytes')
        baseline_log=args.baseline_predictions.open(encoding='utf-8')
    # Shared immutable weights, separate stream state. YOLO size switches only
    # between sequential calls, never while another inference thread runs.
    after_size=pipeline.sign_detector.image_size
    cv2.setNumThreads(2)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    encoder=subprocess.Popen([ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast',
        '-crf','22','-threads','2','-pix_fmt','yuv420p',str(out/'silent.mp4')],stdin=subprocess.PIPE)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    counts={k:{'samples':0,'known_detections':0,'unknown_detections':0,'lane_positive_samples':0,
        'lane_polylines':0,'unique_tracks':set(),'observation_samples':0,'legal_speech_samples':0,
        'ocr_candidates':0,'ocr_errors':0,'latencies':[]} for k in ['before','after']}
    motion=OverlayMotion();a=None;lines=[];boxes=[]
    events=[];last_spoken={};snapshots=[];index=0;failures=[]
    def record(result,ts,key):
        c=counts[key];signs=result.perception.signs;lane=result.perception.lane
        lines=lane.display_polylines if lane.display_polylines is not None else lane.polylines
        c['samples']+=1;c['known_detections']+=sum(s.label!='unknown' for s in signs)
        c['unknown_detections']+=sum(s.label=='unknown' for s in signs)
        c['lane_positive_samples']+=bool(lines);c['lane_polylines']+=len(lines)
        c['unique_tracks'].update(s.track_id for s in signs if s.track_id is not None)
        c['observation_samples']+=bool(result.reasoning.observation_text)
        c['legal_speech_samples']+=bool(result.reasoning.speech_text)
        c['ocr_candidates']+=len(result.perception.supplementary_candidates)
        c['ocr_errors']+=sum(bool(x.get('error')) for x in result.perception.supplementary_candidates)
        c['latencies'].append(result.timings_ms['total'])
        return {'timestamp':ts,'signs':[asdict(s) for s in signs],'lane':asdict(lane),
            'supplementary_candidates':result.perception.supplementary_candidates,
            'observation':result.reasoning.observation_text,'speech':result.reasoning.speech_text,
            'timings_ms':result.timings_ms,'car_summary':getattr(result.reasoning,'car_summary','')}
    try:
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as log:
            while True:
                ok,frame=cap.read()
                if not ok:break
                ts=index/fps
                if index%step==0:
                    try:
                        if baseline_log:
                            payload=json.loads(next(baseline_log))[args.baseline_side]
                            if abs(payload['timestamp']-ts)>.001:raise ValueError('Misaligned baseline timestamp')
                            from src.utils.types import SignDetection,LaneInfo,PerceptionResult,ReasoningResult
                            before=types.SimpleNamespace(
                                perception=PerceptionResult([SignDetection(**x) for x in payload['signs']],LaneInfo(**payload['lane']),payload['supplementary_candidates']),
                                reasoning=ReasoningResult('',[],'Urban',speech_text=payload['speech'],observation_text=payload['observation']),
                                timings_ms=payload['timings_ms'])
                        else:
                            pipeline.sign_detector.image_size=640
                            before=baseline.run(frame,10.7769,106.7009,False,road_bottom=.65,
                                annotate=False,stream_id='completion-before',frame_timestamp=ts,lane_visual_roi=True)
                        pipeline.sign_detector.image_size=after_size
                        after=pipeline.run(frame,10.7769,106.7009,False,road_bottom=.65,
                            annotate=False,stream_id='completion-after',frame_timestamp=ts,lane_visual_roi=True)
                    except Exception as e:
                        failures.append({'timestamp':ts,'error':repr(e)});raise
                    b=record(before,ts,'before');a=record(after,ts,'after')
                    log.write(json.dumps({'before':b,'after':a},ensure_ascii=False)+'\n')
                    text=after.reasoning.speech_text or after.reasoning.observation_speech_text
                    if text and ts-last_spoken.get(text,-100)>20:
                        events.append({'timestamp':ts,'text':text});last_spoken[text]=ts
                    fresh=cv2.resize(frame,(w,h))
                    lines=[[(round(x*w/sw),round(y*h/sh)) for x,y in xy] for xy in a['lane']['display_polylines'] or []]
                    boxes=[[round(v*(w/sw if i%2==0 else h/sh)) for i,v in enumerate(s.bbox_xyxy)] for s in after.perception.signs]
                    motion.reset(fresh,lines,boxes)
                    if index%(round(fps)*30)==0 or expected-index<step:
                        raw=cv2.resize(frame,(w,h));old_view=raw.copy()
                        for xy in b['lane']['display_polylines'] or []:
                            cv2.polylines(old_view,[np.asarray([(round(x*w/sw),round(y*h/sh)) for x,y in xy],np.int32)],False,(0,255,0),2)
                        new_view=draw_frame(raw.copy(),a,lines,boxes,ts,font,fps,fps/step)
                        name=f'{int(ts):04d}';cv2.imwrite(str(out/f'snapshots/before_{name}.jpg'),old_view)
                        cv2.imwrite(str(out/f'snapshots/after_{name}.jpg'),new_view);snapshots.append(name)
                rendered=cv2.resize(frame,(w,h))
                if index%step:lines,boxes=motion.advance(rendered)
                rendered=draw_frame(rendered,a,lines,boxes,ts,font,fps,fps/step)
                encoder.stdin.write(rendered.tobytes());index+=1
                if index%900==0:
                    progress={'source_frames':index,'expected':expected,'video_seconds':index/fps,
                        'elapsed_seconds':round(time.perf_counter()-started,1)}
                    (out/'progress.json').write_text(json.dumps(progress));print('PROGRESS',json.dumps(progress),flush=True)
    finally:
        cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
        if pipeline.supplementary_reader:pipeline.supplementary_reader.close()
        if baseline_log:baseline_log.close()
    if code:raise RuntimeError('Encoder failed')
    report={'source':str(args.source),'source_frames':expected,'output_frames':index,'output_fps':fps,
        'duration_seconds':index/fps,'analysis_hz':fps/step,'complete_source_scan':index==expected,
        'baseline_commit':args.baseline,'baseline_side':args.baseline_side,'after_video_ocr_async':True,
        'overlay_geometry_tracked_each_source_frame':True,'car_context_automatic':True,
        'recording_datetime_supplied':False,
        'baseline_predictions_reused_from_verified_full_scan':bool(args.baseline_predictions),
        'failures':failures,'snapshots':snapshots,'independent_accuracy_available':False,
        'video_is_development_after_fixes':True,'source_analysis_size':[sw,sh],'output_size':[w,h]}
    for key,c in counts.items():
        values=np.asarray(c.pop('latencies'))[1:]
        c['unique_tracks']=len(c['unique_tracks'])
        c['latency_ms']={'mean':float(values.mean()),'p50':float(np.quantile(values,.5)),
            'p95':float(np.quantile(values,.95))}
        c['prediction_coverage_is_not_accuracy']=True
    report['before_after']=counts
    report['runtime_sha256']=runtime_snapshot
    report['runtime_source_changed_during_export']=any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=sha for f,sha in runtime_snapshot.items())
    if report['runtime_source_changed_during_export']:raise RuntimeError('Runtime changed during export; generate a fresh immutable run')
    (out/'inference_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'narration_requests.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    print('INFERENCE_COMPLETE',json.dumps(report,ensure_ascii=True),flush=True)
    pipeline.tts_engine.timeout_seconds=20
    spoken,tts_failures=mix_speech(events,pipeline.tts_engine,out,index/fps,max_failures=None)
    report.update(speech_events=spoken,tts_failures=tts_failures,wall_seconds=time.perf_counter()-started)
    digest=hashlib.sha256()
    with args.source.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    report['source_sha256']=digest.hexdigest()
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FULL_VIDEO_COMPLETE',flush=True)

if __name__=='__main__':main()
