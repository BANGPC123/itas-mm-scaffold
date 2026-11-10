"""Graphics-only replay of verified final predictions, with exact provenance."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,copy,hashlib,json,shutil,subprocess,time
import cv2,imageio_ffmpeg,numpy as np
from PIL import ImageFont
from src.perception.overlay_motion import OverlayMotion
from src.perception.video_text_overlay_safe import SafeVideoTextOverlay


def main():
    ap=argparse.ArgumentParser();ap.add_argument('core',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--wait-for-verification',action='store_true');args=ap.parse_args();core=args.core
    if args.wait_for_verification:
        deadline=time.monotonic()+1800
        while not (core/'verification.json').is_file():
            if time.monotonic()>deadline:raise RuntimeError('Verification deadline')
            time.sleep(1)
    old=json.loads((core/'summary.json').read_text(encoding='utf-8'));verified=json.loads((core/'verification.json').read_text(encoding='utf-8'))
    if not verified['passed']:raise RuntimeError('Unverified final predictions')
    if args.output.exists():raise FileExistsError(args.output)
    for name,digest in old['runtime_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Prediction source changed: '+name)
    source=Path(old['source'])
    with source.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=old['source_sha256']:raise RuntimeError('Source changed')
    args.output.mkdir(parents=True);out=args.output;(out/'snapshots').mkdir()
    files=list(old['runtime_sha256'])+['src/perception/video_text_overlay_safe.py','scripts/render_verified_video.py']
    snapshot={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    cv2.setNumThreads(2);cap=cv2.VideoCapture(str(source));fps=cap.get(5);step=round(fps/old['analysis_hz'])
    sw,sh=old['source_analysis_size'];w,h=old['output_size']
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p',str(out/'silent.mp4')],stdin=subprocess.PIPE)
    renderer=SafeVideoTextOverlay(ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18),fps,fps/step)
    motion=OverlayMotion();index=0;payload=None;lines=[];boxes=[];started=time.perf_counter()
    coverage={k:0 for k in ['samples','known_detections','unknown_detections','lane_positive_samples','lane_polylines','observation_samples','legal_speech_samples','ocr_candidates','ocr_errors']}
    tracks=set();statuses={};time_conditions=0
    try:
        with (core/'predictions.jsonl').open(encoding='utf-8') as prior,(out/'predictions.jsonl').open('w',encoding='utf-8') as log:
            while True:
                ok,frame=cap.read()
                if not ok:break
                ts=index/fps;rendered=cv2.resize(frame,(w,h))
                if index%step==0:
                    row=json.loads(next(prior));payload=row['after']
                    if abs(payload['timestamp']-ts)>.001:raise RuntimeError('Misaligned prediction')
                    signs=payload['signs'];paint=payload['lane']['display_polylines'] or []
                    coverage['samples']+=1;coverage['known_detections']+=sum(s['label']!='unknown' for s in signs)
                    coverage['unknown_detections']+=sum(s['label']=='unknown' for s in signs)
                    coverage['lane_positive_samples']+=bool(paint);coverage['lane_polylines']+=len(paint)
                    coverage['observation_samples']+=bool(payload['observation']);coverage['legal_speech_samples']+=bool(payload['speech'])
                    coverage['ocr_candidates']+=len(payload['supplementary_candidates']);coverage['ocr_errors']+=sum(bool(c.get('error')) for c in payload['supplementary_candidates'])
                    tracks.update(s['track_id'] for s in signs if s.get('track_id') is not None)
                    for s in signs:
                        context=s.get('car_context',{});key=context.get('status','unknown');statuses[key]=statuses.get(key,0)+1
                        time_conditions+=any(c['type']=='time' for c in context.get('conditions',[]))
                    log.write(json.dumps(row,ensure_ascii=False)+'\n')
                    lines=[[(round(x*w/sw),round(y*h/sh)) for x,y in p] for p in paint]
                    boxes=[[round(v*(w/sw if j%2==0 else h/sh)) for j,v in enumerate(s['bbox_xyxy'])] for s in signs]
                    motion.reset(rendered,lines,boxes)
                    if index%900==0 or old['source_frames']-index<step:cv2.imwrite(str(out/f'snapshots/after_{int(ts):04d}.jpg'),renderer.draw(rendered.copy(),payload,lines,boxes,ts))
                else:lines,boxes=motion.advance(rendered)
                encoder.stdin.write(renderer.draw(rendered,payload,lines,boxes,ts).tobytes());index+=1
                if index%900==0:
                    p={'source_frames':index,'expected':old['source_frames'],'elapsed_seconds':round(time.perf_counter()-started,1)}
                    (out/'progress.json').write_text(json.dumps(p),encoding='utf-8');print('RENDER_PROGRESS',json.dumps(p),flush=True)
    finally:cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
    if code or index!=old['source_frames']:raise RuntimeError('Incomplete graphics replay')
    if any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=digest for f,digest in snapshot.items()):raise RuntimeError('Runtime changed during rendering')
    shutil.copyfile(core/'speech.wav',out/'speech.wav');shutil.copyfile(core/'narration_requests.json',out/'narration_requests.json')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-i',str(out/'silent.mp4'),'-i',str(out/'speech.wav'),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','128k','-shortest',str(out/'test_video_annotated.mp4')],check=True)
    coverage.update(unique_tracks=len(tracks),prediction_coverage_is_not_accuracy=True,
                    latency_ms={'reused_core_only':True,'see':'core_scan_before_after'},car_statuses=statuses,time_condition_observations=time_conditions)
    report={**old,'runtime_sha256':snapshot,'core_scan_before_after':old['before_after'],
        'before_after':{'before':old['before_after']['before'],'after':coverage},'graphics_only_replay':True,
        'predictions_reused_from_verified_final_filter':str(core),'explanatory_text_avoids_sign_boxes':True,
        'wall_seconds':time.perf_counter()-started,'output_frames':index}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RENDER_COMPLETE',json.dumps(coverage),flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_complete_video.py'),str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
