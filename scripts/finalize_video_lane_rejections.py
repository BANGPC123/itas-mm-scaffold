"""Replay verified ML geometry with fresh camera/paint rejection and car context.

Byte and AST guards prove that the retained detector, recognizer, lane seeds,
paint refinement and vehicle/time rules have not changed. This reuses ML core
outputs, not their latency as a new throughput measurement.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,ast,copy,hashlib,json,shutil,subprocess,time
from dataclasses import asdict
import cv2,imageio_ffmpeg,numpy as np,torch
from PIL import ImageFont
from src.perception.camera_lane import camera_view_from_ground
from src.perception.crosswalk_veto import reject_crosswalk_segments
from src.perception.road_surface_veto import RoadSurfaceVeto
from src.perception.overlay_motion import OverlayMotion
from src.perception.video_text_overlay import VideoTextOverlay
from src.context.car_context import assess_car,car_summary
from src.utils.types import SignDetection


def normalized_camera(source,new=False):
    tree=ast.parse(source)
    if new:
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name=='camera_view_from_ground':
                node.body=[s for s in node.body if not (isinstance(s,ast.If) and 'np.quantile(yy, 0.1)' in ast.unparse(s.test))]
            if isinstance(node,ast.FunctionDef) and node.name=='detect':
                node.body=[s for s in node.body if not (
                    isinstance(s,ast.If) and ast.unparse(s.test)=="camera_view != 'forward'" or
                    isinstance(s,ast.ImportFrom) and s.module=='src.perception.crosswalk_veto' or
                    isinstance(s,ast.Assign) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Name) and s.value.func.id=='reject_crosswalk_segments')]
    return ast.dump(tree,include_attributes=False)


def normalized_car(source,new=False):
    tree=ast.parse(source)
    if new:
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.IfExp) and ast.unparse(node.value.test)=="camera_view == 'side_or_cabin'":
                node.value=node.value.body
    return ast.dump(tree,include_attributes=False)


def normalized_export(source,new=False):
    tree=ast.parse(source)
    if new:
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RUNTIME_FILES' for t in node.targets):
                node.value.elts=[e for e in node.value.elts if not (isinstance(e,ast.Constant) and e.value=='src/perception/crosswalk_veto.py')]
    return ast.dump(tree,include_attributes=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('core',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    core=args.core;old=json.loads((core/'summary.json').read_text(encoding='utf-8'));verified=json.loads((core/'verification.json').read_text(encoding='utf-8'))
    if not verified['passed'] or not old['complete_source_scan']:raise RuntimeError('Unverified core scan')
    if args.output.exists():raise FileExistsError(args.output)
    changed={'src/perception/camera_lane.py','src/context/car_context.py','scripts/complete_test_video.py'}
    for name,digest in old['runtime_sha256'].items():
        if name not in changed and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Core changed: '+name)
    guards={}
    for name,archive,normalizer in [('src/perception/camera_lane.py','camera_before_final_filter.py',normalized_camera),
            ('src/context/car_context.py','car_before_final_filter.py',normalized_car),
            ('scripts/complete_test_video.py','export_before_final_filter.py',normalized_export)]:
        prior=(core.parent/archive).read_bytes()
        if hashlib.sha256(prior).hexdigest()!=old['runtime_sha256'][name]:raise RuntimeError('Wrong archived source')
        guards[name]=normalizer(prior.decode('utf-8'))==normalizer((ROOT/name).read_text(encoding='utf-8'),True)
    if not all(guards.values()):raise RuntimeError('Geometry/vehicle rules changed outside allowed rejection/text changes')
    source=Path(old['source'])
    with source.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=old['source_sha256']:raise RuntimeError('Source bytes changed')
    args.output.mkdir(parents=True);out=args.output;(out/'snapshots').mkdir()
    from scripts.complete_test_video import RUNTIME_FILES
    files=RUNTIME_FILES+['src/perception/video_text_overlay.py','scripts/finalize_video_lane_rejections.py']
    snapshot={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    torch.set_num_threads(4);cv2.setNumThreads(2)
    from threadpoolctl import threadpool_limits
    limit=threadpool_limits(limits=1,user_api='blas')
    surface=RoadSurfaceVeto(ROOT/'models/lane/candidates/segformer_cityscapes_20261011','cuda:0' if torch.cuda.is_available() else 'cpu')
    cap=cv2.VideoCapture(str(source));fps=cap.get(5);step=round(fps/old['analysis_hz']);sw,sh=old['source_analysis_size'];w,h=old['output_size']
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p',str(out/'silent.mp4')],stdin=subprocess.PIPE)
    renderer=VideoTextOverlay(ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18),fps,fps/step)
    motion=OverlayMotion();index=0;payload=None;lines=[];boxes=[];removed=0;view_changes=0;analysis=0;started=time.perf_counter()
    try:
        with (core/'predictions.jsonl').open(encoding='utf-8') as prior,(out/'predictions.jsonl').open('w',encoding='utf-8') as log:
            while True:
                ok,frame=cap.read()
                if not ok:break
                ts=index/fps;rendered=cv2.resize(frame,(w,h))
                if index%step==0:
                    row=json.loads(next(prior));payload=copy.deepcopy(row['after']);analysis+=1
                    if abs(payload['timestamp']-ts)>.001:raise RuntimeError('Misaligned cached ML timestamp')
                    scale=min(1,1280/sw);small=cv2.resize(frame,(1280,round(sh*scale)),interpolation=cv2.INTER_AREA) if sw>1280 else frame
                    tick=time.perf_counter();ground,_=surface.scene_probabilities(small);view=camera_view_from_ground(ground,small)
                    previous=payload['lane']['camera_view']
                    if view!=previous and view!='unknown':raise RuntimeError('Unexpected camera change beyond uncertainty rejection')
                    view_changes+=view!=previous
                    n,ids,stats,_=cv2.connectedComponentsWithStats((ground>.55).astype(np.uint8),8)
                    if n>1:ground=np.where(ids==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])),ground,0)
                    paint=payload['lane']['display_polylines'] or [];old_points=sum(map(len,paint))
                    kept=reject_crosswalk_segments(small,paint,ground_probability=ground,coordinate_scale=scale) if view=='forward' else []
                    removed+=old_points-sum(map(len,kept))
                    original={tuple(p) for line in paint for p in line}
                    if any(tuple(p) not in original for line in kept for p in line):raise RuntimeError('Rejection filter invented geometry')
                    lane=payload['lane'];lane.update(camera_view=view,polylines=kept,display_polylines=kept,
                        lane_lines=[[*a,*b] for p in kept for a,b in zip(p,p[1:])],
                        rejection_reason='camera_direction_uncertain' if view!='forward' else 'visual_paint_not_verified_lane_scope' if kept else 'no_supported_road_paint')
                    signs=[SignDetection(**s) for s in payload['signs']]
                    for s in signs:s.car_context=assess_car(s,frame,payload['supplementary_candidates'],video=True,camera_view=view)
                    payload['signs']=[asdict(s) for s in signs];payload['car_summary']=car_summary(signs)
                    payload['lane_rejection_stage_ms']=(time.perf_counter()-tick)*1000
                    payload['core_ml_timings_reused']=True
                    log.write(json.dumps({'before':row['before'],'after':payload},ensure_ascii=False)+'\n')
                    lines=[[(round(x*w/sw),round(y*h/sh)) for x,y in p] for p in kept]
                    boxes=[[round(v*(w/sw if j%2==0 else h/sh)) for j,v in enumerate(s.bbox_xyxy)] for s in signs]
                    motion.reset(rendered,lines,boxes)
                    if index%900==0 or old['source_frames']-index<step:
                        cv2.imwrite(str(out/f'snapshots/after_{int(ts):04d}.jpg'),renderer.draw(rendered.copy(),payload,lines,boxes,ts))
                else:lines,boxes=motion.advance(rendered)
                encoder.stdin.write(renderer.draw(rendered,payload,lines,boxes,ts).tobytes());index+=1
                if index%900==0:
                    p={'source_frames':index,'expected':old['source_frames'],'elapsed_seconds':round(time.perf_counter()-started,1)}
                    (out/'progress.json').write_text(json.dumps(p),encoding='utf-8');print('FILTER_PROGRESS',json.dumps(p),flush=True)
    finally:cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
    if code or index!=old['source_frames']:raise RuntimeError('Incomplete native-frame replay')
    if any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=digest for f,digest in snapshot.items()):raise RuntimeError('Source changed during final replay')
    shutil.copyfile(core/'speech.wav',out/'speech.wav');shutil.copyfile(core/'narration_requests.json',out/'narration_requests.json')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(out/'silent.mp4'),'-i',str(out/'speech.wav'),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','128k','-shortest',str(out/'test_video_annotated.mp4')],check=True)
    report={**old,'runtime_sha256':snapshot,'runtime_source_changed_during_export':False,
        'final_rejection_filter_replay':True,'core_scan_directory':str(core),'upstream_ast_guards':guards,
        'detector_recognizer_lane_geometry_reused_from_verified_fresh_ml':True,
        'camera_surface_model_and_car_context_recomputed':True,'lane_rejection_only_no_new_points':True,
        'removed_lane_points':removed,'camera_uncertainty_changes':view_changes,'output_frames':index,
        'speech_reused_names_unchanged':True,'wall_seconds':time.perf_counter()-started,
        'core_latency_fields_reused_not_a_new_ml_fps_measurement':True}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FILTER_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['runtime_sha256','speech_events','before_after','snapshots']},ensure_ascii=True),flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_complete_video.py'),str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
