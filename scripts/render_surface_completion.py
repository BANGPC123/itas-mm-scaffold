"""Native-FPS final export using frozen core predictions plus surface veto.

The core perception pass already processed every sampled timestamp. This pass
decodes every source frame again, checks the source hash, filters display
curves, recomputes the stricter narration policy and verifies the final export.
It does not represent a new independent accuracy benchmark.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,copy,hashlib,json,time,subprocess
import cv2,numpy as np,torch,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from src.perception.road_surface_veto import RoadSurfaceVeto
from src.interaction.ai_observation import ai_observation
from src.interaction.tts_engine import TtsEngine
from src.utils.config import load_config
from src.utils.types import SignDetection
from src.perception.sign_catalog import display_name
from scripts.render_product_video import mix_speech

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output
    if out.exists():raise FileExistsError('Prior evidence is immutable')
    base=json.loads((args.base/'summary.json').read_text(encoding='utf-8'))
    checked=json.loads((args.base/'verification.json').read_text(encoding='utf-8'))
    if not base['complete_source_scan'] or not checked['passed']:raise ValueError('Core scan is not verified')
    source=Path(base['source']);digest=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=base['source_sha256']:raise ValueError('Source has changed')
    out.mkdir(parents=True);(out/'snapshots').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(2)
    cfg=load_config('perception')['lane_visual_fallback']
    veto=RoadSurfaceVeto(cfg['surface_veto_path'],'cuda:0' if torch.cuda.is_available() else 'cpu')
    cap=cv2.VideoCapture(str(source));fps=cap.get(5);expected=int(cap.get(7));sw,sh=int(cap.get(3)),int(cap.get(4))
    if expected!=base['source_frames'] or abs(fps-base['output_fps'])>.001:raise ValueError('Metadata mismatch')
    step=round(fps/base['analysis_hz']);w,h=base['output_size'];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    encoder=subprocess.Popen([ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2',
        '-pix_fmt','yuv420p',str(out/'silent.mp4')],stdin=subprocess.PIPE)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);started=time.perf_counter()
    overlay=np.zeros((h,w,3),np.uint8);mask=np.zeros((h,w),np.uint8);index=0
    surface_times=[];surface_removed=0;lane_positive=0;lane_lines=0;observations=0
    events=[];last_spoken={};snapshots=[]
    def intersects(a,b):return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]
    try:
        with (args.base/'predictions.jsonl').open(encoding='utf-8') as frozen, (out/'predictions.jsonl').open('w',encoding='utf-8') as log:
            while True:
                ok,frame=cap.read()
                if not ok:break
                ts=index/fps
                if index%step==0:
                    row=json.loads(next(frozen));a=row['after'];b=row['before']
                    if abs(a['timestamp']-ts)>.001:raise ValueError('Misaligned core predictions')
                    candidates=a['lane']['polylines'] if a['lane']['backend']=='camera_road_paint_visual' else (a['lane']['display_polylines'] or [])
                    t=time.perf_counter();lines,removed=veto.filter(frame,candidates)
                    surface_ms=(time.perf_counter()-t)*1000;surface_times.append(surface_ms);surface_removed+=removed
                    a['lane']['display_polylines']=lines
                    a['surface_veto']={'removed':removed,'latency_ms':surface_ms}
                    signs=[SignDetection(**s) for s in a['signs']]
                    observation=ai_observation(signs,video=True);a['observation']=observation
                    observations+=bool(observation);lane_positive+=bool(lines);lane_lines+=len(lines)
                    log.write(json.dumps(row,ensure_ascii=False)+'\n')
                    text=a['speech'] or observation
                    if text and ts-last_spoken.get(text,-100)>20:
                        events.append({'timestamp':ts,'text':text});last_spoken[text]=ts
                    overlay[:]=0;labels=[]
                    for xy in lines:
                        points=np.asarray([(round(x*w/sw),round(y*h/sh)) for x,y in xy],np.int32)
                        cv2.polylines(overlay,[points],False,(40,220,255),2)
                    for sign in signs:
                        x1,y1,x2,y2=[round(v*(w/sw if i%2==0 else h/sh)) for i,v in enumerate(sign.bbox_xyxy)]
                        color=(120,120,120) if sign.label=='unknown' else (0,220,255)
                        cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
                        label='?' if sign.label=='unknown' else display_name(sign.label)
                        labels.append((x1,y1,label,color[::-1]))
                    pil=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pil)
                    draw.rectangle((0,0,w,57),fill=(15,23,34))
                    draw.text((10,3),f'ITAS-MM | {ts/60:.2f} phút | Video {fps:g} FPS | AI {fps/step:g} Hz',font=font,fill='white')
                    draw.text((10,29),'Vạch và biển AI dự đoán; điều kiện chưa xác minh được giữ trạng thái chưa xác định.',font=font,fill='white')
                    occupied=[]
                    for x,y,label,color in sorted(labels,key=lambda item:item[1]):
                        tw=int(draw.textlength(label,font=font));x=max(4,min(x,w-tw-8));y=max(62,y-24)
                        box=(x-2,y-2,x+tw+4,y+22)
                        while any(intersects(box,previous) for previous in occupied) and y<h-26:
                            y+=24;box=(x-2,y-2,x+tw+4,y+22)
                        draw.rectangle(box,fill=(15,23,34));draw.text((x,y),label,font=font,fill=color);occupied.append(box)
                    overlay=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR);mask=np.any(overlay!=0,axis=2).astype(np.uint8)*255
                    if index%(round(fps)*30)==0 or expected-index<step:
                        raw=cv2.resize(frame,(w,h));old_view=raw.copy()
                        for xy in b['lane']['display_polylines'] or []:
                            cv2.polylines(old_view,[np.asarray([(round(x*w/sw),round(y*h/sh)) for x,y in xy],np.int32)],False,(0,255,0),2)
                        new_view=raw.copy();cv2.copyTo(overlay,mask,new_view);name=f'{int(ts):04d}'
                        cv2.imwrite(str(out/f'snapshots/before_{name}.jpg'),old_view)
                        cv2.imwrite(str(out/f'snapshots/after_{name}.jpg'),new_view);snapshots.append(name)
                rendered=cv2.resize(frame,(w,h));cv2.copyTo(overlay,mask,rendered);encoder.stdin.write(rendered.tobytes());index+=1
                if index%900==0:
                    progress={'source_frames':index,'expected':expected,'video_seconds':index/fps,'elapsed_seconds':round(time.perf_counter()-started,1)}
                    (out/'progress.json').write_text(json.dumps(progress));print('SURFACE_PROGRESS',json.dumps(progress),flush=True)
    finally:cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
    if code or index!=expected:raise RuntimeError('Incomplete export')
    report=copy.deepcopy(base);report['core_runtime_sha256']=report.pop('runtime_sha256')
    report.update(output_frames=index,complete_source_scan=index==expected,snapshots=snapshots,
        core_predictions_reused_from_verified_scan=str(args.base),source_sha256=digest.hexdigest(),
        surface_veto_removed_polylines=surface_removed,narration_min_video_short_side=32)
    c=report['before_after']['after'];c.update(lane_positive_samples=lane_positive,lane_polylines=lane_lines,observation_samples=observations)
    values=np.asarray(surface_times[1:])
    report['surface_filter_replay_latency_ms']={'mean':float(values.mean()),'p95':float(np.quantile(values,.95))}
    report['before_after_core_latency_is_cached_not_combined_live_latency']=True
    files=['configs/perception.yaml','src/pipeline/orchestrator.py','src/perception/road_surface_veto.py','src/interaction/ai_observation.py']
    report['runtime_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    (out/'inference_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'narration_requests.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
    cfg_tts=load_config('interaction');cfg_tts['tts']['timeout_seconds']=20
    spoken,errors=mix_speech(events,TtsEngine(cfg_tts),out,index/fps,max_failures=None)
    report.update(speech_events=spoken,tts_failures=errors,wall_seconds=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SURFACE_COMPLETE',json.dumps({'frames':index,'samples':len(surface_times),'removed':surface_removed,'speech_events':len(spoken),'tts_errors':len(errors)}),flush=True)

if __name__=='__main__':main()
