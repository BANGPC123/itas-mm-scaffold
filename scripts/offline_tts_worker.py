"""Isolated CPU Vietnamese TTS worker. Only JSON goes to stdout."""
from pathlib import Path
import sys,json,contextlib,time,argparse
ROOT=Path(__file__).resolve().parents[1]
for stream in (sys.stdin,sys.stdout,sys.stderr):
    stream.reconfigure(encoding='utf-8',errors='strict')
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
allowed=parser.parse_args().output_dir.resolve()
with contextlib.redirect_stdout(sys.stderr):
    from vieneu import Vieneu
    model=Vieneu(backend='onnx',precision='int8')
print(json.dumps({'ready':True,'provider':'vieneu-3.8.3-onnx-int8','voice':'Hải Đăng'}),flush=True)
for line in sys.stdin:
    try:
        request=json.loads(line);target=Path(request['wav_path']).resolve()
        if target.parent!=allowed or target.suffix!='.wav':raise ValueError('Unexpected output path')
        started=time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            audio=model.infer(request['text'],voice='Hải Đăng')
            model.save(audio,str(target))
        print(json.dumps({'wav_path':str(target),'synthesis_ms':(time.perf_counter()-started)*1000}),flush=True)
    except Exception as error:
        print(json.dumps({'error':type(error).__name__+': '+str(error)}),flush=True)
