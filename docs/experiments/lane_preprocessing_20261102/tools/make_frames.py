"""Write the frozen frame manifest (frames.json) before any preprocessing variant is run."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'docs/experiments/lane_preprocessing_20261102/frames.json'
FROZEN = json.loads((ROOT / 'output/lane_relax_20261102/frozen_samples.json').read_text(encoding='utf-8'))
DIAGNOSTIC_TIMES = {'day': [55, 59]}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


clips = {}
for name, spec in FROZEN['clips'].items():
    source = ROOT / spec['source']
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    def entry(t, role):
        return {'time_s': t, 'source_frame': round(t * fps), 'role': role}
    clips[name] = {
        'source': spec['source'],
        'sha256': sha256(source),
        'bytes': source.stat().st_size,
        'fps': fps,
        'frame_count': frames,
        'max_seconds': spec['max_seconds'],
        'frames': [entry(t, 'acceptance') for t in spec['times_seconds']]
                  + [entry(t, 'diagnostic_not_scored') for t in DIAGNOSTIC_TIMES.get(name, [])],
    }

manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    'source_commit': 'f6bdf8f51c0171d480d17bb786a01aa56df6c86f',
    'frozen_from': 'output/lane_relax_20261102/frozen_samples.json',
    'frozen_samples_sha256': sha256(ROOT / 'output/lane_relax_20261102/frozen_samples.json'),
    'acceptance_frames': 30,
    'diagnostic_frames': 2,
    'human_gt': False,
    'clips': clips,
}
OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: {'sha256': v['sha256'], 'fps': v['fps'], 'frame_count': v['frame_count'], 'n': len(v['frames'])} for k, v in clips.items()}, indent=2))
