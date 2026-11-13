"""M12: synthesize every forward-warning cue sentence with the configured TTS and check each file has audio."""
import json, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio_ffmpeg
from src.interaction.tts_engine import TtsEngine
from src.perception.forward_objects import prewarm_texts
from src.utils.config import load_config


def duration_s(path):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([ff, '-i', str(path)], capture_output=True, text=True)
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', proc.stderr)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def main():
    out = ROOT / 'output/demo_20261114/M12/prewarm'
    out.mkdir(parents=True, exist_ok=True)
    engine = TtsEngine(load_config('interaction'))
    rows = []
    t0 = time.perf_counter()
    engine.warmup()
    for text in prewarm_texts():
        started = time.perf_counter()
        path = Path(engine.synthesize(text)).resolve()
        rows.append({'text': text, 'file': path.relative_to(ROOT).as_posix(),
                     'bytes': path.stat().st_size, 'duration_s': round(duration_s(path), 2),
                     'seconds': round(time.perf_counter() - started, 2)})
    engine.close()
    result = {'provider': engine.provider, 'voice': engine.voice, 'total_seconds': round(time.perf_counter() - t0, 1),
              'sentences': rows, 'all_nonempty': all(r['bytes'] > 0 and r['duration_s'] > 0 for r in rows)}
    (out / 'prewarm.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PREWARM', json.dumps({'n': len(rows), 'all_nonempty': result['all_nonempty']}), flush=True)


if __name__ == '__main__':
    main()
