"""Whole-source operational replay and explicitly AI-reviewed negative frames.

This source was selected after freezing the detector and was not used to train
or tune it. Eleven reviewed stills are negative controls for visible road paint,
not a human-annotated lane/tracking accuracy benchmark.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,math,subprocess,time
from dataclasses import asdict
import cv2,imageio_ffmpeg,numpy as np,torch
from PIL import ImageFont
from src.pipeline.orchestrator import Pipeline
from src.perception.overlay_motion import OverlayMotion
from scripts.complete_test_video import draw_frame,RUNTIME_FILES

REVIEWED_NEGATIVES=[0,12,24,36,48,60,72,84,96,108,118]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--wait-for-verification',type=Path);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.wait_for_verification:
        deadline=time.monotonic()+3600
        while not args.wait_for_verification.is_file():
            if time.monotonic()>deadline:raise RuntimeError('Main video verification deadline')
            time.sleep(1)
        verified=json.loads(args.wait_for_verification.read_text(encoding='utf-8'))
        if not verified['passed']:raise RuntimeError('Main video verification failed')
    args.output.mkdir(parents=True);out=args.output
    torch.set_num_threads(4);cv2.setNumThreads(2)
    from threadpoolctl import threadpool_limits
    limit=threadpool_limits(limits=1,user_api='blas')
    pipeline=Pipeline();cap=cv2.VideoCapture(str(args.source))
    expected=int(cap.get(7));fps=cap.get(5);sw,sh=int(cap.get(3)),int(cap.get(4))
    if fps<=0 or not cap.isOpened():raise RuntimeError('Source unavailable')
    step=max(1,round(fps/3));w,h=960,round(sh*960/sw)//2*2
    snapshot={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in RUNTIME_FILES}
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','23',
        '-threads','2','-pix_fmt','yuv420p',str(out/'encoding.mp4')],stdin=subprocess.PIPE)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    motion=OverlayMotion();rows=[];index=0;payload=None;lines=[];boxes=[];started=time.perf_counter()
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            ts=index/fps;view=cv2.resize(frame,(w,h))
            if index%step==0:
                tick=time.perf_counter()
                result=pipeline.run(frame,20.808892,105.466138,False,road_bottom=.65,annotate=False,
                                    stream_id='independent-hoa-binh',frame_timestamp=ts,lane_visual_roi=True)
                lane=result.perception.lane
                payload={'signs':[asdict(s) for s in result.perception.signs],'lane':asdict(lane)}
                lines=[[(round(x*w/sw),round(y*h/sh)) for x,y in p] for p in lane.display_polylines or []]
                boxes=[[round(v*(w/sw if j%2==0 else h/sh)) for j,v in enumerate(s.bbox_xyxy)] for s in result.perception.signs]
                motion.reset(view,lines,boxes)
                row={'timestamp':ts,'lane_segments':len(lines),'lane_points':sum(map(len,lines)),
                     'known_signs':[s.label for s in result.perception.signs if s.label!='unknown'],
                     'camera_view':lane.camera_view,'car_contexts':[s.car_context for s in result.perception.signs],
                     'legal_speech':result.reasoning.speech_text,'pipeline_ms':(time.perf_counter()-tick)*1000}
                rows.append(row)
                if any(abs(ts-v)<.001 for v in REVIEWED_NEGATIVES):
                    cv2.imwrite(str(out/f'review_{int(ts):03d}.jpg'),draw_frame(view.copy(),payload,lines,boxes,ts,font,fps,fps/step))
            else:lines,boxes=motion.advance(view)
            rendered=draw_frame(view,payload,lines,boxes,ts,font,fps,fps/step)
            cv2.putText(rendered,'Source: Obakeneko, CC BY 3.0 | ITAS-MM overlay',(8,h-8),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
            encoder.stdin.write(rendered.tobytes());index+=1
            if index%900==0:print('EXTERNAL_PROGRESS',index,'/',expected,flush=True)
    finally:
        cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
        if pipeline.supplementary_reader:pipeline.supplementary_reader.close()
        pipeline.tts_engine.close()
    if code:raise RuntimeError('Encoder failed')
    final=out/'independent_car_annotated.mp4';(out/'encoding.mp4').replace(final)
    decode=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(final),'-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True)
    decoded=[int(s.split('=',1)[1]) for s in decode.stdout.splitlines() if s.startswith('frame=')]
    controls=[r for r in rows if any(abs(r['timestamp']-t)<.001 for t in REVIEWED_NEGATIVES)]
    checks={'all_source_frames':index==expected==decoded[-1],
            'decode_without_errors':decode.returncode==0 and not decode.stderr.strip(),
            'all_analysis_samples':len(rows)==math.ceil(expected/step),
            'no_unverified_legal_speech':not any(r['legal_speech'] for r in rows),
            'all_ai_negative_controls_present':len(controls)==len(REVIEWED_NEGATIVES),
            'runtime_unchanged':all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==s for f,s in snapshot.items())}
    with args.source.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    report={'passed':all(checks.values()),'checks':checks,'source':str(args.source),'source_sha256':digest,
            'source_frames':expected,'output_frames':index,'fps':fps,'analysis_hz':fps/step,'analysis_samples':len(rows),
            'wall_seconds':time.perf_counter()-started,'video_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
            'video_size':final.stat().st_size,'video_mtime_ns':final.stat().st_mtime_ns,
            'negative_control_count':len(controls),'negative_controls_with_lane_overlay':sum(bool(r['lane_segments']) for r in controls),
            'known_sign_observations':sum(len(r['known_signs']) for r in rows),'runtime_sha256':snapshot,
            'source_disjoint_from_user_video':True,'used_for_sign_training_or_selection':False,'used_for_lane_postprocessing_selection':True,
            'reference_kind':'AI visual inspection of 11 original stills: no discernible longitudinal painted lane mark',
            'human_verified':False,'independent_human_lane_gt':False,'lane_f1_available':False,
            'historical_video_current_legal_applicability_verified':False,'silent_auxiliary_video':True,
            'attribution':'Obakeneko, Driving in Kim Boi and Luong Son Districts, CC BY 3.0; ITAS-MM overlay added',
            'source_page':'https://commons.wikimedia.org/wiki/File:Driving_in_Kim_Boi_and_Luong_Son_Districts.ogv',
            'license_url':'https://creativecommons.org/licenses/by/3.0','controls':controls,'rows':rows}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXTERNAL_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['rows','runtime_sha256']},ensure_ascii=True),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
