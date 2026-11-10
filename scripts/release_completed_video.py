"""Reuse verified native-frame overlays and generate a fresh offline soundtrack."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_product_video import mix_speech
from src.interaction.tts_engine import TtsEngine
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Release evidence is immutable')
    summary = json.loads((args.base / 'summary.json').read_text(encoding='utf-8'))
    verified = json.loads((args.base / 'verification.json').read_text(encoding='utf-8'))
    video = args.base / 'test_video_annotated.mp4'
    if (not verified['passed'] or not summary['complete_source_scan'] or summary['failures']
            or video.stat().st_size != verified['video_size']
            or video.stat().st_mtime_ns != verified['video_mtime_ns']):
        raise ValueError('The overlay source is not verified')
    args.output.mkdir(parents=True)
    for name in ('silent.mp4', 'predictions.jsonl', 'narration_requests.json'):
        try:
            os.link(args.base / name, args.output / name)
        except OSError:
            shutil.copy2(args.base / name, args.output / name)
    shutil.copytree(args.base / 'snapshots', args.output / 'snapshots', copy_function=os.link)
    events = json.loads((args.base / 'narration_requests.json').read_text(encoding='utf-8'))
    engine = TtsEngine(load_config('interaction'))
    started = time.perf_counter()
    try:
        spoken, errors = mix_speech(events, engine, args.output, summary['duration_seconds'], max_failures=None)
    finally:
        engine.close()
    report = copy.deepcopy(summary)
    report.update(speech_events=spoken, tts_failures=errors, speech_provider=engine.provider,
                  speech_voice=engine.voice, overlay_reused_from_verified=str(args.base),
                  offline_soundtrack_wall_seconds=time.perf_counter() - started,
                  offline_soundtrack_latency_is_not_live_latency=True)
    report['speech_runtime_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('configs/interaction.yaml', 'src/interaction/offline_speech.py',
                     'src/interaction/tts_engine.py', 'scripts/offline_tts_worker.py')
    }
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OFFLINE_RELEASE', json.dumps({'speech': len(spoken), 'errors': errors, 'frames': report['output_frames']}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
