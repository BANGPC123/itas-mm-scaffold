from pathlib import Path
import json,time,hashlib
from src.interaction.tts_engine import TtsEngine
from src.utils.config import load_config

engine=TtsEngine(load_config('interaction'))
records=[]
for text in ['Đã nhận dạng biển báo. Hãy quan sát điều kiện áp dụng thực tế.', 'Đây là bản thử nghiệm nhận diện biển báo giao thông.']:
    start=time.perf_counter()
    key=hashlib.sha256(f'{engine.voice}|{engine.rate}|{text}'.encode()).hexdigest()
    cached=(engine.output_dir/f'{key}.mp3').is_file()
    try:
        path=Path(engine.synthesize(text))
        records.append({'cache_before':cached,'ok':True,'bytes':path.stat().st_size,'file':str(path),'elapsed_seconds':time.perf_counter()-start})
    except Exception as error:
        records.append({'cache_before':cached,'ok':False,'error':type(error).__name__,'elapsed_seconds':time.perf_counter()-start})
out=Path('output/evaluation/tts_smoke.json')
out.write_text(json.dumps({'voice':engine.voice,'attempts':records,'note':'Earlier uncached synthesis succeeded once and another request returned NoAudioReceived. This smoke test does not evaluate pronunciation or service availability.'},indent=2))
print(out.read_text())
