"""Operational benchmark: new-source temporal pipeline, OCR and TTS.

No labels inferred from the predictions are scored as ground truth.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,hashlib,cv2,numpy as np,torch
from src.pipeline.orchestrator import Pipeline
from src.perception.supplementary_reader import SupplementaryReader
from src.utils.types import SignDetection
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/auto_system_20261011'

def stats(values):
 return {'count':len(values),'mean_ms':float(np.mean(values)),'p50_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95))} if values else {'count':0}

def main():
 import argparse
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args();OUT=args.output
 torch.set_num_threads(4);cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'report.json').exists():raise FileExistsError('Immutable benchmark')
 t=time.perf_counter();p=Pipeline();init_ms=(time.perf_counter()-t)*1000
 raw=ROOT/'data/raw/external/vietnam_commons_street_20261011';sources=json.loads((raw/'SOURCES.json').read_text(encoding='utf-8'))['sources']
 rows=[];crashes=[];summary=[]
 for source in sources:
  cap=cv2.VideoCapture(str(raw/source['filename']));fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/max(1,fps)
  horizon=min(duration,30);timestamps=np.arange(0,horizon,.25)
  for ts in timestamps:
   cap.set(cv2.CAP_PROP_POS_MSEC,float(ts)*1000);ok,im=cap.read()
   if not ok:crashes.append({'source':source['id'],'timestamp':float(ts),'error':'decode'});continue
   im=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
   t=time.perf_counter()
   try:
    res=p.run(im,10.7769,106.7009,False,road_bottom=.65,annotate=False,stream_id=source['id'],frame_timestamp=float(ts))
    rows.append({'source':source['id'],'timestamp':float(ts),'wall_ms':(time.perf_counter()-t)*1000,'timings_ms':res.timings_ms,
      'track_ids':[s.track_id for s in res.perception.signs if s.track_id is not None],
      'signs':[{'label':s.label,'applicability':s.applicability} for s in res.perception.signs],
      'lane_count':len(res.perception.lane.polylines),'ocr_candidates':len(res.perception.supplementary_candidates),
      'legal_speech':bool(res.reasoning.speech_text),'tentative_observation':bool(res.reasoning.observation_text)})
   except Exception as e:crashes.append({'source':source['id'],'timestamp':float(ts),'error':type(e).__name__})
  cap.release()
  print('SYSTEM_SOURCE_DONE',source['id'],len(rows),flush=True)
  sr=[r for r in rows if r['source']==source['id']]
  summary.append({'source':source['id'],'source_sha256':source['sha256'],'frames':len(sr),'sampled_duration_seconds':float(horizon),
    'wall_latency':stats([r['wall_ms'] for r in sr][1:]),'unique_emitted_tracks':len({x for r in sr for x in r['track_ids']}),
    'legal_speech_frames':sum(r['legal_speech'] for r in sr),'track_accuracy':None,'false_alerts_per_hour':None})
 # Real OCR on previously used development regions. It tests execution only.
 reader=SupplementaryReader();pack=ROOT/'output/review/field_review_pack_20261011';data=json.loads((pack/'autobox_20261011/autobox_roles_review.json').read_text(encoding='utf-8'))
 pairs=json.loads((pack/'autobox_20261011/compound_pairs/pairs.json').read_text(encoding='utf-8'))
 ocr=[]
 for pair in pairs:
  f=next(x for x in data['frames'] if x['id']==pair['frame']);s=next(x for x in f['signs'] if x['id']==pair['supplementary_id'])
  im=cv2.imread(str(pack/f['image']));x1,y1,x2,y2=map(int,s['bbox_xyxy']);crop=im[y1:y2,x1:x2]
  t=time.perf_counter()
  try:
   result=reader._ocr(cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC))
   from src.perception.supplementary_semantics import parse_supplementary
   parsed=parse_supplementary(result.get('texts',[]),result.get('scores',[]))
   ocr.append({'frame':f['id'],'latency_ms':(time.perf_counter()-t)*1000,**parsed,'error':result.get('error')})
  except Exception as e:ocr.append({'frame':f['id'],'latency_ms':(time.perf_counter()-t)*1000,'error':str(e)})
 reader.close()
 if p.supplementary_reader:p.supplementary_reader.close()
 # Separate cold network/cache measurements; do not mix into inference FPS.
 tts=[]
 for text in ['Phát hiện biển: Cấm đỗ xe.','Phát hiện biển: Cấm rẽ trái.','Có thể có biển phụ, chưa đọc rõ điều kiện áp dụng.']:
  key=hashlib.sha256(f'{p.tts_engine.voice}|{p.tts_engine.rate}|{text}'.encode()).hexdigest();target=p.tts_engine.output_dir/(key+'.mp3');existed=target.is_file()
  for attempt in range(2):
   t=time.perf_counter()
   try:
    path=Path(p.tts_engine.synthesize(text));tts.append({'text':text,'cache_hit_before_call':existed or attempt>0,'latency_ms':(time.perf_counter()-t)*1000,'bytes':path.stat().st_size,'succeeded':True})
   except Exception as e:tts.append({'text':text,'cache_hit_before_call':existed,'latency_ms':(time.perf_counter()-t)*1000,'succeeded':False,'error':type(e).__name__})
  print('TTS_BENCH_DONE',len(tts),flush=True)
 ms=[r['wall_ms'] for r in rows][1:]
 report={'mode':'AUTOMATED_OPERATIONAL_RESEARCH_EVALUATION','human_ground_truth':False,'source_disjoint_from_training':True,'resolution':'1280x720',
   'sample_rate_hz':4,'streaming_capture_4Hz_is_NOT_full_video_FPS':True,'frames':len(rows),'initialization_ms':init_ms,'latency':stats(ms),
   'serial_inference_throughput_fps':1000/np.mean(ms) if ms else None,'gps_simulated':True,'failures':crashes,'sources':summary,
   'ocr':ocr,'ocr_accuracy':None,'lane_field_f1':None,'sign_false_positives':None,'legal_accuracy':None,'tts':tts,'rows':rows}
 (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('SYSTEM_BENCH_COMPLETE',len(rows),'failures',len(crashes),flush=True)
if __name__=='__main__':main()
