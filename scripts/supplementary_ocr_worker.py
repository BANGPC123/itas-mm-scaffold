"""Persistent isolated RapidOCR worker: stdout is a JSON-lines protocol."""
import contextlib,sys,json,base64
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.perception.ocr_line_order import printed_lines
import cv2,numpy as np
cv2.setNumThreads(2)
with contextlib.redirect_stdout(sys.stderr):
 from rapidocr import RapidOCR
 engine=RapidOCR(params={'EngineConfig.onnxruntime.intra_op_num_threads':2,
                         'EngineConfig.onnxruntime.inter_op_num_threads':1,
                         'Det.limit_type':'max','Det.limit_side_len':640})
def recognize_line(crop):
 with contextlib.redirect_stdout(sys.stderr):result=engine(crop,use_det=False,use_cls=False,use_rec=True)
 texts=list(getattr(result,'txts',[]) or []);scores=list(getattr(result,'scores',[]) or [])
 return (texts[0],float(scores[0])) if len(texts)==len(scores)==1 else None
for line in sys.stdin:
 try:
  data=json.loads(line);a=np.frombuffer(base64.b64decode(data['jpeg']),np.uint8)
  image=cv2.imdecode(a,cv2.IMREAD_COLOR)
  with contextlib.redirect_stdout(sys.stderr):result=engine(image,use_det=True,use_cls=True,use_rec=True)
  txts=list(getattr(result,'txts',[]) or [])
  scores=[float(x) for x in (getattr(result,'scores',[]) or [])]
  txts,scores=printed_lines(txts,scores,getattr(result,'boxes',None),image,recognize_line)
  print(json.dumps({'texts':txts,'scores':scores},ensure_ascii=True),flush=True)
 except Exception as e:print(json.dumps({'error':type(e).__name__}),flush=True)
