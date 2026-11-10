"""Re-run vehicle/plate semantics over verified, unchanged lane/sign inference.

Decodes every original frame and exports fresh moving geometry and car text.
Reuse of the frozen ML predictions is explicit and checked against their hashes;
this export's speed is never advertised as model-inference performance.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,copy,hashlib,json,subprocess,time,cv2,numpy as np,imageio_ffmpeg
from dataclasses import asdict
from PIL import ImageFont
from scripts.complete_test_video import draw_frame,RUNTIME_FILES
from scripts.render_product_video import mix_speech
from src.context.car_context import assess_car,car_summary
from src.perception.supplementary_reader import SupplementaryReader
from src.perception.overlay_motion import OverlayMotion
from src.utils.types import SignDetection
from src.utils.config import load_config
from src.interaction.tts_engine import TtsEngine
from src.perception.sign_recognizer import SignRecognizer
from src.perception.temporal_signs import TemporalSigns
from src.perception.road_surface_veto import RoadSurfaceVeto
from src.perception.camera_lane import camera_view_from_ground
from src.perception.sign_catalog import display_name
from src.context.sign_applicability import SignApplicability
from src.interaction.ai_observation import ai_observation
from src.interaction.sign_speech import sign_speech


def main():
    ap=argparse.ArgumentParser();ap.add_argument('core_directory',type=Path)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    core=args.core_directory;out=args.output
    if out.exists():raise FileExistsError('Use a new immutable output directory')
    previous=json.loads((core/'summary.json').read_text(encoding='utf-8'))
    verified=json.loads((core/'verification.json').read_text(encoding='utf-8'))
    if not verified['passed'] or not previous['complete_source_scan'] or previous.get('runtime_source_changed_during_export'):
        raise ValueError('The core scan must be verified and use frozen sources')
    semantic_files={'src/perception/camera_lane.py','src/context/car_context.py','src/perception/supplementary_reader.py','scripts/supplementary_ocr_worker.py',
                    'src/perception/supplementary_panel_proposals.py'}
    for name,sha in previous['runtime_sha256'].items():
        if name not in semantic_files and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Core source changed: '+name)
    source=Path(previous['source'])
    with source.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=previous['source_sha256']:raise ValueError('Source bytes changed')
    files=[*RUNTIME_FILES,'src/perception/supplementary_motion.py','src/perception/supplementary_tracking.py',
           'scripts/refine_video_car_context.py']
    runtime={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    # The only camera-module changes allowed relative to the verified scan
    # are the direction cue and its call. Paint geometry must stay identical.
    import ast,inspect
    old_camera=ROOT/'tmp/camera_lane_before_view_fix.py'
    if hashlib.sha256(old_camera.read_bytes()).hexdigest()!=previous['runtime_sha256']['src/perception/camera_lane.py']:
        raise ValueError('Missing exact frozen camera source')
    def geometry_source(value):
        tree=ast.parse(value)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name=='camera_view_from_ground':
                node.body=[ast.Pass()];node.args.args=node.args.args[:1];node.args.defaults=[]
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='camera_view_from_ground':
                    call.args=call.args[:1]
        return ast.dump(tree,include_attributes=False)
    if geometry_source(old_camera.read_text())!=geometry_source((ROOT/'src/perception/camera_lane.py').read_text()):
        raise ValueError('Lane geometry changed; a new full ML scan is required')
    out.mkdir(parents=True);(out/'snapshots').mkdir();started=time.perf_counter()
    reader=SupplementaryReader();reader._ocr(np.full((60,180,3),240,np.uint8))
    import torch
    torch.set_num_threads(4)
    cfg=load_config('perception');recognizer=SignRecognizer(cfg);temporal=TemporalSigns()
    surface=RoadSurfaceVeto(cfg['lane_visual_fallback']['surface_veto_path'],'cuda:0' if torch.cuda.is_available() else 'cpu')
    applicability=SignApplicability(cfg.get('applicability'))
    cap=cv2.VideoCapture(str(source));fps=cap.get(cv2.CAP_PROP_FPS);expected=int(cap.get(7))
    sw,sh=int(cap.get(3)),int(cap.get(4));w,h=previous['output_size']
    step=round(fps/previous['analysis_hz'])
    if expected!=previous['source_frames'] or [sw,sh]!=previous['source_analysis_size']:raise ValueError('Source metadata changed')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();cv2.setNumThreads(2)
    encoder=subprocess.Popen([ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2',
        '-pix_fmt','yuv420p',str(out/'silent.mp4')],stdin=subprocess.PIPE)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);motion=OverlayMotion()
    events=[];last_spoken={};index=0;samples=0;statuses={};conditions={};ocr_count=0;ocr_errors=0;snapshots=[]
    a=None;lines=[];boxes=[];failures=[]
    detection_counts={'known_detections':0,'unknown_detections':0,'observation_samples':0,'legal_speech_samples':0}
    track_ids=set()
    qa={round(t*fps/step)*step for t in [8,110,148.67,207.33,660,689.33,900,960,1193]}
    try:
        with (core/'predictions.jsonl').open(encoding='utf-8') as old,(out/'predictions.jsonl').open('w',encoding='utf-8') as log:
            while True:
                ok,frame=cap.read()
                if not ok:break
                ts=index/fps
                if index%step==0:
                    payload=json.loads(next(old));a=payload['after']
                    if abs(a['timestamp']-ts)>.00001:raise ValueError('Core prediction timestamp mismatch')
                    signs=[SignDetection(**x) for x in a['signs']];tick=time.perf_counter()
                    for sign in signs:
                        sign.label='unknown';sign.recognition_confidence=None;sign.recognition_source='single_frame'
                        sign.car_context={};sign.track_id=None
                    roi=frame[:round(frame.shape[0]*.65)]
                    signs=recognizer.recognize(roi,signs,min_short_side=cfg['sign_recognizer']['video_min_short_side'])
                    signs=temporal.update(roi,signs,ts)
                    camera_frame=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
                    ground,_=surface.scene_probabilities(camera_frame)
                    view=camera_view_from_ground(ground,camera_frame)
                    a['lane']['camera_view']=view
                    if view=='side_or_cabin':
                        temporal.tracks=[]
                        for sign in signs:sign.label='unknown';sign.recognition_source='side_or_reflected_view'
                    from src.utils.types import LaneInfo
                    signs=applicability.apply(signs,LaneInfo(**a['lane']),frame.shape,'car')
                    for sign in signs:sign.name_vi=display_name(sign.label).split(' — ',1)[-1]
                    a['observation']=ai_observation(signs,video=True)
                    a['speech']=sign_speech(signs)
                    detection_counts['known_detections']+=sum(sign.label!='unknown' for sign in signs)
                    detection_counts['unknown_detections']+=sum(sign.label=='unknown' for sign in signs)
                    detection_counts['observation_samples']+=bool(a['observation'])
                    detection_counts['legal_speech_samples']+=bool(a['speech'])
                    track_ids.update(sign.track_id for sign in signs if sign.track_id is not None)
                    plates=reader.read_video(frame,signs,stream_id='car-final-replay',timestamp=ts)
                    for sign in signs:
                        sign.car_context=assess_car(sign,frame,plates,video=True,camera_view=view)
                        status=sign.car_context['status'];statuses[status]=statuses.get(status,0)+1
                        for cond in sign.car_context['conditions']:conditions[cond['type']]=conditions.get(cond['type'],0)+1
                    a['signs']=[asdict(s) for s in signs];a['supplementary_candidates']=plates
                    a['car_summary']=car_summary(signs)
                    a['semantic_replay_ms']=(time.perf_counter()-tick)*1000
                    a['lane_and_detector_boxes_reused_from_verified_scan']=True
                    a['recognition_and_camera_direction_recomputed']=True
                    log.write(json.dumps(payload,ensure_ascii=False)+'\n');samples+=1
                    ocr_count+=len(plates);ocr_errors+=sum(bool(p.get('error')) for p in plates)
                    text=a['speech'] or a['observation']
                    if text and ts-last_spoken.get(text,-100)>20:
                        events.append({'timestamp':ts,'text':text});last_spoken[text]=ts
                    lines=[[(round(x*w/sw),round(y*h/sh)) for x,y in p] for p in a['lane']['display_polylines'] or []]
                    boxes=[[round(v*(w/sw if i%2==0 else h/sh)) for i,v in enumerate(s.bbox_xyxy)] for s in signs]
                    motion.reset(cv2.resize(frame,(w,h)),lines,boxes)
                rendered=cv2.resize(frame,(w,h))
                if index%step:lines,boxes=motion.advance(rendered)
                rendered=draw_frame(rendered,a,lines,boxes,ts,font,fps,fps/step)
                encoder.stdin.write(rendered.tobytes())
                if index in qa or index%(round(fps)*30)==0:
                    name=f'after_{ts:08.3f}.jpg';cv2.imwrite(str(out/'snapshots'/name),rendered);snapshots.append(name)
                index+=1
                if index%900==0:
                    progress={'source_frames':index,'expected':expected,'video_seconds':index/fps,'elapsed_seconds':round(time.perf_counter()-started,1)}
                    (out/'progress.json').write_text(json.dumps(progress));print('PROGRESS',json.dumps(progress),flush=True)
            if old.readline():raise ValueError('Core predictions contain unexpected additional frames')
    except Exception as e:
        failures.append({'frame':index,'error':repr(e)});raise
    finally:
        cap.release();encoder.stdin.close();code=encoder.wait(timeout=60);reader.close()
    if code or index!=expected:raise RuntimeError('Video export failed/incomplete')
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha for name,sha in runtime.items()):
        raise RuntimeError('Runtime source changed during export')
    report=copy.deepcopy(previous)
    report.update(source_frames=expected,output_frames=index,complete_source_scan=index==expected,
        snapshots=snapshots,failures=failures,runtime_sha256=runtime,runtime_source_changed_during_export=False,
        core_predictions_reused_from_verified_scan=str(core),core_prediction_sha256=hashlib.sha256((core/'predictions.jsonl').read_bytes()).hexdigest(),
        core_runtime_sha256=previous['runtime_sha256'],car_context_recomputed_every_analysis_frame=True,
        recognition_recomputed_every_analysis_frame=True,camera_direction_recomputed_every_analysis_frame=True,
        lane_geometry_source_ast_matches_verified_scan=True,
        car_status_counts=statuses,car_condition_counts=conditions,car_context_is_independent_gt=False,
        export_speed_is_not_model_inference_performance=True)
    report['before_after']['after'].update(detection_counts,unique_tracks=len(track_ids),ocr_candidates=ocr_count,ocr_errors=ocr_errors)
    (out/'inference_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'narration_requests.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    engine=TtsEngine(load_config('interaction'));engine.timeout_seconds=20
    try:spoken,tts_failures=mix_speech(events,engine,out,index/fps,max_failures=None)
    finally:engine.close()
    report.update(speech_events=spoken,tts_failures=tts_failures,wall_seconds=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CAR_VIDEO_COMPLETE',json.dumps({'frames':index,'samples':samples,'statuses':statuses,'conditions':conditions,'ocr_errors':ocr_errors}),flush=True)


if __name__=='__main__':main()
