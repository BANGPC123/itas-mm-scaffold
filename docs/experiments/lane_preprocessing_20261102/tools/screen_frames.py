"""Stage 1 screen: UFLD (ratio 0.5, no temporal) on each frozen frame under each preprocessing variant.

Writes per-frame row-existence counts to evidence/stage1_counts.json and the composite sheets
(5 columns: A0 | A1a | A1b | A2a | A2b; polylines drawn on the ORIGINAL frame) to output/.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.perception.image_quality import insufficient_information  # noqa: E402
from src.perception.lane_display_quality import visual_polylines  # noqa: E402
from src.perception.lane_preprocess import preprocess_for_lane  # noqa: E402
from src.perception.ufld_v2 import UFLDv2  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.types import LaneInfo  # noqa: E402

EXP = ROOT / 'docs/experiments/lane_preprocessing_20261102'
OUT = ROOT / 'output/lane_preprocess_20261102'
VARIANTS = [
    ('A0', 'none', {}),
    ('A1a', 'gamma', {'gamma': .8}),
    ('A1b', 'gamma', {'gamma': .6}),
    ('A2a', 'clahe', {'clahe_clip': 2., 'clahe_tile': 8}),
    ('A2b', 'clahe', {'clahe_clip': 3., 'clahe_tile': 8}),
]
CELL_W, CELL_H = 448, 252
COLOURS = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def build_detector():
    cfg = load_config('perception')['lane_detector']
    device = cfg.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return UFLDv2(cfg['weights_path'], device, half=cfg.get('half', False))


def head_row_counts(det, image):
    """Row-existence counts per UFLD lane slot, using the same heads and slot mapping as UFLDv2.detect."""
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    rgb = np.array(rgb.resize((1600, int(320 / .6)), Image.Resampling.BILINEAR))[-320:]
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).to(device=det.device, dtype=det.dtype) / 255
    x = (x - det.mean) / det.std
    with torch.inference_mode():
        pred = det.model(x.unsqueeze(0))
        existence = [v.argmax(1)[0].cpu().numpy() for v in pred[2:]]
    row_valid, col_valid = existence
    # detect() uses the row head for lane slots 1,2 and the column head for slots 0,3.
    return {
        'slot0_col_head': int(col_valid[:, 0].sum()),
        'slot1_row_head': int(row_valid[:, 1].sum()),
        'slot2_row_head': int(row_valid[:, 2].sum()),
        'slot3_col_head': int(col_valid[:, 3].sum()),
    }


def draw_cell(frame, polylines, label):
    canvas = frame.copy()
    for index, points in enumerate(polylines):
        colour = COLOURS[index % len(COLOURS)]
        cv2.polylines(canvas, [np.asarray(points, np.int32)], False, colour, 6)
    cell = cv2.resize(canvas, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(cell, (0, 0), (CELL_W, 22), (0, 0, 0), -1)
    cv2.putText(cell, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'cannot read source frame {index}')
    return frame


def main():
    manifest = json.loads((EXP / 'frames.json').read_text(encoding='utf-8'))
    det = build_detector()
    records = []
    OUT.mkdir(parents=True, exist_ok=True)
    (EXP / 'evidence').mkdir(parents=True, exist_ok=True)
    for clip, spec in manifest['clips'].items():
        cap = cv2.VideoCapture(str(ROOT / spec['source']))
        rows = []
        for entry in spec['frames']:
            frame = read_frame(cap, entry['source_frame'])
            cells = []
            for name, mode, kwargs in VARIANTS:
                image = preprocess_for_lane(frame, mode, **kwargs)
                counts = None if insufficient_information(image) else head_row_counts(det, image)
                lane = det.detect(image, filter_geometry=False, display_row_ratio=.5)
                displayed = visual_polylines(LaneInfo([], polylines=lane.relaxed_polylines), frame.shape, mode='candidate')
                record = {'clip': clip, 'time_s': entry['time_s'], 'role': entry['role'],
                          'source_frame': entry['source_frame'], 'variant': name, 'mode': mode, 'params': kwargs,
                          'row_counts': counts, 'canonical_lines': len(lane.polylines),
                          'relaxed_lines': len(lane.relaxed_polylines), 'displayed_lines': len(displayed),
                          'rejection_reason': getattr(lane, 'rejection_reason', None)}
                records.append(record)
                cells.append(draw_cell(frame, displayed, f"{name} t={entry['time_s']}s n={len(displayed)}"))
            rows.append(np.hstack(cells))
            print(clip, entry['time_s'], [r['displayed_lines'] for r in records[-len(VARIANTS):]], flush=True)
        cap.release()
        sheet = np.vstack(rows)
        cv2.imwrite(str(OUT / f'screen_{clip}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    (EXP / 'evidence/stage1_counts.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('done', len(records), 'records')


if __name__ == '__main__':
    main()
