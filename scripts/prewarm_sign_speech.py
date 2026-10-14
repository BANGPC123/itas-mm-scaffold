"""Cache the runtime's 30 standard observation phrases for Edge TTS."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.interaction.sign_speech import sign_speech
from src.interaction.tts_engine import TtsEngine
from src.perception.sign_catalog import NAMES_VI
from src.utils.config import load_config
from src.utils.types import SignDetection

engine=TtsEngine(load_config('interaction'));rows=[]
for code in NAMES_VI:
    text=sign_speech([SignDetection(code,1.,(0,0,1,1))]);start=time.perf_counter()
    try:
        path=engine.synthesize(text);rows.append({'code':code,'ok':True,'audio':path,'seconds':time.perf_counter()-start})
    except Exception as exc:
        rows.append({'code':code,'ok':False,'error':str(exc)})
    print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
out=ROOT/'output/product_20261010/tts_cache.json';out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
