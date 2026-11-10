"""Check final links and simultaneous HTTP video/speech after product startup."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import re
import threading
import time
import cv2
import numpy as np
import requests
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8000'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--speech-text',default='Kiểm tra xử lý đồng thời video và giọng đọc với luồng suy luận cố định.')
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError('Delivery evidence is immutable')
    args.output.mkdir(parents=True)
    summary = json.loads((ROOT / 'output/video_quality_20261013/full_v4/summary.json').read_text(encoding='utf-8'))
    capture = cv2.VideoCapture(summary['source'])
    frames = []
    try:
        for k in range(13):
            capture.set(cv2.CAP_PROP_POS_FRAMES, k * 10)
            ok, frame = capture.read()
            if not ok:raise RuntimeError('Source decode')
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:raise RuntimeError('JPEG encoding')
            frames.append(encoded.tobytes())
    finally:capture.release()

    def infer(k):
        start = time.perf_counter()
        response = requests.post(URL + '/pipeline/run', timeout=45,
            files={'image': ('frame.jpg', frames[k], 'image/jpeg')},
            data={'latitude': 10.7769, 'longitude': 106.7009, 'synthesize_audio': 'false',
                  'annotate': 'false', 'road_bottom': .65, 'lane_visual_roi': 'true',
                  'stream_id': 'delivery-probe', 'frame_timestamp': k / 3})
        response.raise_for_status()
        return {'frame': k, 'http_ms': (time.perf_counter() - start) * 1000,
                'legal_speech': response.json()['reasoning']['speech_text']}

    first = infer(0)
    tts_cfg=load_config('interaction')['tts']
    cache=ROOT/tts_cfg['output_dir']/(hashlib.sha256((tts_cfg['voice_id']+'|'+tts_cfg['rate']+'|'+args.speech_text).encode()).hexdigest()+'.mp3')
    cache_hit_before=cache.is_file() and cache.stat().st_size>0
    barrier = threading.Barrier(2)
    def video():
        barrier.wait(timeout=10)
        return [infer(k) for k in range(1, 13)]
    def speech():
        barrier.wait(timeout=10)
        start = time.perf_counter()
        response = requests.post(URL + '/pipeline/speech', timeout=45,
            json={'text': args.speech_text})
        response.raise_for_status()
        sound = requests.get(URL + response.json()['audio_url'], timeout=10)
        sound.raise_for_status()
        return {'http_ms': (time.perf_counter() - start) * 1000, 'audio_bytes': len(sound.content),
                'http_status': response.status_code,'cache_hit_before':cache_hit_before,'text':args.speech_text}
    with ThreadPoolExecutor(max_workers=2) as executor:
        vf, sf = executor.submit(video), executor.submit(speech)
        rows, spoken = vf.result(), sf.result()

    page = requests.get(URL, timeout=10)
    report = requests.get(URL + '/pipeline/video-completion-report', timeout=10)
    status = requests.get(URL + '/pipeline/video-completion-status', timeout=10)
    video_range = requests.get(URL + '/pipeline/capstone-video/user-complete',
                               headers={'Range': 'bytes=0-1023'}, timeout=10)
    assets = []
    for asset in re.findall(r'<script[^>]+src="([^"]+)"', page.text):
        response = requests.get(URL + asset, timeout=10)
        local = ROOT / 'frontend/dist' / asset.lstrip('/')
        assets.append({'path': asset, 'http_status': response.status_code,
                       'matches_current_build': local.is_file()
                       and hashlib.sha256(response.content).hexdigest() == hashlib.sha256(local.read_bytes()).hexdigest()})
    checks = {'frontend_200': page.status_code == 200,
              'current_assets_served': bool(assets) and all(a['http_status'] == 200 and a['matches_current_build'] for a in assets),
              'report_200': report.status_code == 200 and report.text.startswith('# ITAS-MM'),
              'verified_video_ready': status.status_code == 200 and status.json()['complete'],
              'video_range_206': video_range.status_code == 206 and len(video_range.content) == 1024,
              'simultaneous_video_speech': len(rows) == 12 and spoken['audio_bytes'] > 1000,
              'no_unverified_legal_speech': not any(row['legal_speech'] for row in [first, *rows])}
    result = {'passed': all(checks.values()), 'checks': checks, 'first_http_after_jpeg_warmup_ms': first['http_ms'],
              'concurrent_video_http_ms': {'mean': float(np.mean([r['http_ms'] for r in rows])),
                                           'p95': float(np.quantile([r['http_ms'] for r in rows], .95))},
              'speech': spoken, 'rows': rows, 'assets': assets, 'independent_accuracy': False}
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PRODUCT_DELIVERY', json.dumps(result), flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__ == '__main__':main()
