"""Automatic, abstaining supplementary-panel OCR for research display."""
from pathlib import Path
from collections import OrderedDict,Counter
from copy import deepcopy
import base64,json,queue,subprocess,threading,time,cv2
from src.perception.supplementary_panel_proposals import propose_plates
from src.perception.supplementary_semantics import parse_supplementary
ROOT=Path(__file__).resolve().parents[2]

class SupplementaryReader:
 def __init__(self):
  self.process=None;self.responses=queue.Queue();self.lock=threading.Lock();self.cache={};self.ready=False
  self.condition=threading.Condition();self.jobs=OrderedDict();self.completed=OrderedDict()
  self.streams=OrderedDict();self.closed=False;self.worker=None;self.epoch=0
 def _start(self):
  exe=ROOT/'.ocr-venv'/('Scripts/python.exe' if __import__('os').name=='nt' else 'bin/python')
  if not exe.is_file():raise RuntimeError('isolated_rapidocr_unavailable')
  with self.condition:
   if self.closed:raise RuntimeError('ocr_reader_closed')
   self.process=subprocess.Popen([str(exe),'-u',str(ROOT/'scripts/supplementary_ocr_worker.py')],
    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',bufsize=1)
  self.responses=queue.Queue()
  def collect(proc,q):
   for line in proc.stdout:q.put(line)
  threading.Thread(target=collect,args=(self.process,self.responses),daemon=True).start()
 def _ocr(self,crop):
  ok,data=cv2.imencode('.jpg',crop)
  if not ok:raise ValueError('encode_failed')
  if self.process is None or self.process.poll() is not None:
   self._start();self.ready=False
  self.process.stdin.write(json.dumps({'jpeg':base64.b64encode(data).decode()})+'\n');self.process.stdin.flush()
  try:
   value=json.loads(self.responses.get(timeout=30 if not self.ready else 8));self.ready=True;return value
  except queue.Empty:
   self._stop_process();raise RuntimeError('ocr_timeout')
 def read(self,image,signs,*,stream_id=None,timestamp=None):
  # No OCR of unrelated billboards. Only nearby geometric candidates below
  # a recognized primary sign; association remains an explicit hypothesis.
  primaries=[s for s in signs if not s.label.startswith('S') and s.label not in ('unknown','background')]
  if not primaries:return []
  key=(stream_id or '__still__',tuple((s.label,s.track_id,tuple(round(v/32) for v in s.bbox_xyxy)) for s in primaries[:2]))
  with self.lock:
   prev=self.cache.get(key)
   if stream_id is not None and prev and timestamp is not None and 0<=timestamp-prev[0]<3:
    return [{**r,'cached':True,'observation_timestamp':prev[0]} for r in prev[1]]
   rows=[]
   for s in primaries[:2]:
    for panel in propose_plates(image,s.bbox_xyxy,max_candidates=2):
     x1,y1,x2,y2=map(int,panel['bbox_xyxy']);crop=image[y1:y2,x1:x2]
     if crop.size==0:continue
     from src.perception.traffic_symbols import read_vehicle_icon
     icon=read_vehicle_icon(crop)
     extra=cv2.copyMakeBorder(cv2.resize(crop,None,fx=5,fy=5,interpolation=cv2.INTER_CUBIC),12,12,12,12,cv2.BORDER_CONSTANT,value=(245,245,245))
     crop=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
     gray=lambda view:cv2.cvtColor(cv2.createCLAHE(2,(5,5)).apply(cv2.cvtColor(view,cv2.COLOR_BGR2GRAY)),cv2.COLOR_GRAY2BGR)
     variants=[crop,gray(crop),extra,gray(extra)]
     parsed=[];error=None
     try:
      for view in variants:
       o=self._ocr(view)
       if 'error' in o:raise RuntimeError(o['error'])
       parsed.append(parse_supplementary(o['texts'],o['scores']))
     except Exception as e:error=str(e)[:80]
     time_votes=Counter(json.dumps(p['time_windows']) for p in parsed if p['time_windows'])
     agree=len(time_votes)==1 and max(time_votes.values(),default=0)>=2
     windows=json.loads(next(iter(time_votes))) if agree else []
     vehicle_votes=Counter(tuple(sorted(p['vehicle_candidates'])) for p in parsed if p['vehicle_candidates'])
     vehicles=list(next(iter(vehicle_votes))) if len(vehicle_votes)==1 and max(vehicle_votes.values(),default=0)>=2 else []
     polarities=[p['condition_polarity'] for p in parsed]
     polarity=polarities[0] if len(polarities)>=2 and len(set(polarities))==1 else 'unresolved'
     vehicle_agree=bool(vehicles) and polarity!='unresolved'
     if polarity=='unresolved':vehicles=[]
     if icon and not vehicles and polarity!='unresolved':
      vehicles=[icon['vehicle']];vehicle_agree=True
     rows.append({'primary_label':s.label,'primary_track_id':s.track_id,'bbox_xyxy':panel['bbox_xyxy'],
       'raw_texts':[p['raw_text'] for p in parsed],'time_windows':windows,
       'vehicle_candidates':vehicles,'condition_polarity':polarity,'vehicle_variant_agreement':vehicle_agree,'variant_agreement':agree,'status':'machine_candidate' if agree or vehicles else 'unreadable_or_ambiguous',
       'vehicle_icon':icon,'association_verified':False,'human_verified':False,'legal_effect_established':False,'error':error,'cached':False})
   if len(self.cache)>=32:self.cache.pop(next(iter(self.cache)))
   if stream_id is not None:self.cache[key]=(timestamp,rows)
   return rows

 def read_video(self,image,signs,*,stream_id,timestamp):
  from src.perception.supplementary_tracking import read_video_with_motion
  return read_video_with_motion(self,image,signs,stream_id=stream_id,timestamp=timestamp)

 def _video_worker(self):
  while True:
   with self.condition:
    self.condition.wait_for(lambda:self.closed or bool(self.jobs))
    if self.closed:return
    key,(image,signs,timestamp,version)=self.jobs.popitem(last=False)
   try:rows=self.read(image,signs,stream_id=None,timestamp=timestamp)
   except Exception as e:
    rows=[{'primary_label':s.label,'primary_track_id':s.track_id,'bbox_xyxy':list(s.bbox_xyxy),
       'raw_texts':[],'time_windows':[],'vehicle_candidates':[],'status':'ocr_unavailable',
       'association_verified':False,'human_verified':False,'legal_effect_established':False,
       'error':str(e)[:80],'cached':False} for s in signs]
   with self.condition:
    state=self.streams.get(key)
    if state and state['version']==version and not self.closed:
     self.completed[key]={'version':version,'timestamp':timestamp,'rows':rows,
        'boxes':{(s.label,s.track_id):s.bbox_xyxy for s in signs}}

 def close(self):
  with self.condition:
   self.closed=True;self.jobs.clear();self.completed.clear();self.condition.notify_all()
  self._stop_process()

 def _stop_process(self):
  process=self.process;self.process=None
  if process is None or process.poll() is not None:return
  # Windows venv launchers have a Python child; stop our owned tree too.
  try:
   import psutil
   children=psutil.Process(process.pid).children(recursive=True)
   for child in children:child.terminate()
  except (ImportError,ProcessLookupError):pass
  except Exception:pass
  try:process.terminate();process.wait(timeout=2)
  except (OSError,subprocess.TimeoutExpired):
   try:process.kill()
   except OSError:pass
