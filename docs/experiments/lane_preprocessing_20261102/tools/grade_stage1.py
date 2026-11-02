"""Record stage-1 AI visual grades (DUNG / SAI / KHONG_CHAC) and compress the composite sheets."""
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / 'docs/experiments/lane_preprocessing_20261102'
OUT = ROOT / 'output/lane_preprocess_20261102'
VARIANTS = ['A0', 'A1a', 'A1b', 'A2a', 'A2b']
ACCEPT_DAY = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]
ACCEPT_NIGHT = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87]

# Grades per line in the order drawn: D = DUNG, S = SAI, U = KHONG_CHAC. Frames not listed show 0 displayed lines.
grades = {
    ('day', 42): {'A0': ['S'], 'A1a': ['S'], 'A1b': ['S'], 'A2a': [], 'A2b': ['S']},
    ('night', 15): {'A0': ['S', 'S', 'U', 'S'], 'A1a': ['S', 'S', 'U', 'S'], 'A1b': ['S', 'S', 'U', 'S'],
                    'A2a': ['S', 'S', 'U', 'S'], 'A2b': ['S', 'S', 'U', 'S']},
    ('night', 21): {'A0': ['U', 'S', 'S'], 'A1a': ['U', 'S', 'S'], 'A1b': ['U', 'S', 'S'],
                    'A2a': ['S', 'S', 'U'], 'A2b': ['U', 'S', 'S']},
    ('night', 27): {'A0': ['U', 'S', 'D'], 'A1a': ['U', 'S', 'D'], 'A1b': ['U', 'S', 'D'],
                    'A2a': ['U', 'S', 'D'], 'A2b': ['U', 'S', 'D']},
    ('night', 51): {'A0': ['S'], 'A1a': ['S'], 'A1b': ['S', 'S'], 'A2a': [], 'A2b': []},
    ('night', 69): {'A0': ['S', 'S'], 'A1a': ['S', 'S'], 'A1b': ['S', 'S'], 'A2a': ['S'], 'A2b': ['S']},
}

records = json.loads((EXP / 'evidence/stage1_counts.json').read_text(encoding='utf-8'))
shown = {(r['clip'], r['time_s'], r['variant']): r['displayed_lines'] for r in records}
for (clip, t), per_variant in grades.items():
    for v in VARIANTS:
        lst = per_variant.get(v, [])
        assert len(lst) == shown[(clip, t, v)], (clip, t, v, lst, shown[(clip, t, v)])

totals = {}
for v in VARIANTS:
    count = {'D': 0, 'S': 0, 'U': 0}
    for (clip, t), per_variant in grades.items():
        if (clip == 'day' and t not in ACCEPT_DAY) or (clip == 'night' and t not in ACCEPT_NIGHT):
            continue
        for g in per_variant.get(v, []):
            count[g] += 1
    totals[v] = count

diagnostic = {v: shown[('day', 55, v)] + shown[('day', 59, v)] for v in VARIANTS}
payload = {
    'grader': 'AI visual review, 448 px cells, no temporal, not human ground truth',
    'grades': {f'{c}_{t}s': d for (c, t), d in grades.items()},
    'totals_acceptance_frames': totals,
    'diagnostic_frames_displayed_lines': diagnostic,
    'note': 'Frames and variants not listed have 0 displayed lines (see stage1_counts.json).',
}
(EXP / 'evidence/stage1_grades.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(totals))
print('diagnostic', diagnostic)

for name in ['screen_day', 'screen_night']:
    img = cv2.imread(str(OUT / f'{name}.jpg'))
    for quality in [80, 70, 60, 50]:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if len(buf) < 2_000_000:
            break
    (EXP / f'{name}.jpg').write_bytes(buf.tobytes())
    print(name, 'quality', quality, 'bytes', len(buf))
