"""Real HTTP latency and source-disjoint operational checks after export."""
from pathlib import Path
import argparse,json,time,cv2,numpy as np,requests,hashlib
from src.utils.config import load_config
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--speech-suffix',default='')
args=parser.parse_args();out=args.output
if out.exists():raise FileExistsError('Immutable live evaluation')
out.mkdir();url='http://127.0.0.1:8000';rows=[];failures=[]
def run_frame(image,stream,ts):
 ok,blob=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
 if not ok:raise RuntimeError('JPEG encoding')
 t=time.perf_counter();r=requests.post(url+'/pipeline/run',files={'image':('frame.jpg',blob.tobytes(),'image/jpeg')},
 data={'latitude':10.7769,'longitude':106.7009,'synthesize_audio':'false','annotate':'false',
       'road_bottom':.65,'lane_visual_roi':'true','stream_id':stream,'frame_timestamp':ts},timeout=90)
 r.raise_for_status();result=r.json();return result,(time.perf_counter()-t)*1000
source=root/'output/video_quality_20261013/full_v4/summary.json'
summary=json.loads(source.read_text(encoding='utf-8'))
cap=cv2.VideoCapture(summary['source'])
for start in [0,270,540,660,900,1180]:
 for k in range(12):
  ts=start+k/3;cap.set(cv2.CAP_PROP_POS_MSEC,ts*1000);ok,image=cap.read()
  if not ok:raise RuntimeError('Decode failed')
  try:
   res,ms=run_frame(image,'http-user-'+str(start),ts)
   row={'source':'user','timestamp':ts,'http_ms':ms,'timings_ms':res['timings_ms'],
        'lanes':len(res['perception']['lane']['display_polylines'] or []),
        'ocr_candidates':len(res['perception'].get('supplementary_candidates',[])),
        'legal_speech':res['reasoning']['speech_text']}
   rows.append(row)
  except Exception as e:failures.append({'source':'user','timestamp':ts,'error':repr(e)})
cap.release()
external=root/'data/raw/external/vietnam_commons_street_20261011'
manifest=json.loads((external/'SOURCES.json').read_text(encoding='utf-8'))
for src in manifest['sources']:
 cap=cv2.VideoCapture(str(external/src['filename']))
 duration=cap.get(7)/cap.get(5)
 for ts in np.linspace(0,min(duration-.1,20),30):
  cap.set(cv2.CAP_PROP_POS_MSEC,float(ts)*1000);ok,image=cap.read()
  if not ok:failures.append({'source':src['id'],'timestamp':float(ts),'error':'decode'});continue
  image=cv2.resize(image,(1280,720))
  try:
   res,ms=run_frame(image,'http-external-'+hashlib.sha256(src['id'].encode()).hexdigest()[:12],float(ts))
   rows.append({'source':src['id'],'timestamp':float(ts),'http_ms':ms,'timings_ms':res['timings_ms'],
        'lanes':len(res['perception']['lane']['display_polylines'] or []),
        'legal_speech':res['reasoning']['speech_text']})
  except Exception as e:failures.append({'source':src['id'],'timestamp':float(ts),'error':repr(e)})
 cap.release()
def stats(values):
 return {'count':len(values),'mean_ms':float(np.mean(values)),'p50_ms':float(np.median(values)),
   'p95_ms':float(np.quantile(values,.95))} if values else {'count':0}
user=[r for r in rows if r['source']=='user'];other=[r for r in rows if r['source']!='user']
tts=[];tts_cfg=load_config('interaction')['tts']
for text in ['Kiểm thử giọng đọc tiếng Việt của bản video hoàn thiện.','Đo riêng độ trễ giọng nói; nhận diện video vẫn tiếp tục.']:
 if args.speech_suffix:text+=' '+args.speech_suffix
 key=hashlib.sha256((tts_cfg['voice_id']+'|'+tts_cfg['rate']+'|'+text).encode()).hexdigest()
 for attempt in range(2):
  cached=root/tts_cfg['output_dir']/(key+'.mp3')
  cache_hit_before=cached.is_file() and cached.stat().st_size>0
  t=time.perf_counter()
  try:
   r=requests.post(url+'/pipeline/speech',json={'text':text},timeout=45);r.raise_for_status()
   audio=requests.get(url+r.json()['audio_url'],timeout=10);audio.raise_for_status()
   tts.append({'text':text,'attempt':attempt,'latency_ms':(time.perf_counter()-t)*1000,
      'audio_bytes':len(audio.content),'http_status':r.status_code,'ok':True,'cache_hit_before':cache_hit_before})
  except Exception as e:tts.append({'text':text,'attempt':attempt,'ok':False,'error':repr(e)})
ranges=requests.get(url+'/pipeline/capstone-video/user-complete',headers={'Range':'bytes=0-1023'},timeout=10)
ready=requests.get(url+'/pipeline/video-completion-status',timeout=10);ready.raise_for_status()
report={'user_frames':len(user),'new_source_frames':len(other),'failures':failures,
 'cold_first_http_ms':user[0]['http_ms'] if user else None,
 'user_http_latency':stats([r['http_ms'] for r in user[1:]]),
 'user_pipeline_latency':stats([r['timings_ms']['total'] for r in user[1:]]),
 'external_http_latency':stats([r['http_ms'] for r in other]),
 'ocr_call_latency':stats([r['timings_ms'].get('supplementary_ocr',0) for r in user[1:]]),
 'unverified_legal_speech_frames':sum(bool(r['legal_speech']) for r in rows),
 'video_range_status':ranges.status_code,'video_range_bytes':len(ranges.content),
 'video_complete_status':ready.json()['complete'],'tts':tts,'tts_provider':tts_cfg['engine'],'rows':rows,
 'independent_video_gt':False,'source_disjoint_operational_test_is_not_accuracy':True,
 'latency_measured_without_concurrent_export_or_model_benchmark':True}
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('LIVE_EVALUATION',json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=True),flush=True)
if (failures or any(not row['ok'] for row in tts) or len(user)!=72 or len(other)!=60
    or ranges.status_code!=206 or len(ranges.content)!=1024 or not report['video_complete_status']
    or report['unverified_legal_speech_frames']):raise SystemExit(1)
