"""Select small-sign localization candidates on VAL before a frozen TEST run."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse, copy, hashlib, json, time
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scripts.evaluate_product import iou_matrix
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.utils.config import load_config


def focused_detections(detector, image, full):
    h, w = image.shape[:2]
    if w < h * 1.2:
        return copy.deepcopy(full)
    width = round(w * .58)
    proposals = copy.deepcopy(full)
    for left in [0, w - width]:
        for sign in detector.detect(image[:, left:left + width]):
            a, b, c, d = sign.bbox_xyxy
            sign.bbox_xyxy = (a + left, b, c + left, d)
            # A clipped panel at a slice edge is not a whole sign crop.
            if (left > 0 and a < 3) or (left + width < w and c > width - 3):
                continue
            proposals.append(sign)
    boxes = [[a, b, c-a, d-b] for s in proposals for a, b, c, d in [s.bbox_xyxy]]
    selected = cv2.dnn.NMSBoxes(boxes, [s.confidence for s in proposals],
                              detector.confidence_threshold, detector.iou_threshold)
    return [proposals[int(i)] for i in np.asarray(selected).reshape(-1)]


def merge_views(first, second):
    merged = copy.deepcopy(first)
    used = set()
    matrix = iou_matrix([s.bbox_xyxy for s in first], [s.bbox_xyxy for s in second])
    pi, gi = linear_sum_assignment(-np.where(matrix >= .55, 1 + matrix / 1000, 0))
    for p, q in zip(pi, gi):
        if matrix[p, q] < .55:
            continue
        a, b = merged[p], second[q]
        weight = a.confidence + b.confidence
        a.bbox_xyxy = tuple((x*a.confidence+y*b.confidence)/weight for x,y in zip(a.bbox_xyxy,b.bbox_xyxy))
        a.confidence = max(a.confidence,b.confidence)
        a.localization_views = getattr(a,'localization_views',1)+1
        used.add(int(q))
    merged.extend(copy.deepcopy(s) for q,s in enumerate(second) if q not in used)
    boxes = [[a,b,c-a,d-b] for s in merged for a,b,c,d in [s.bbox_xyxy]]
    keep = cv2.dnn.NMSBoxes(boxes,[s.confidence for s in merged],.35,.45)
    return [merged[int(i)] for i in np.asarray(keep).reshape(-1)]


def strict_tiny(signs, image):
    import math
    h,w=image.shape[:2]
    for s in signs:
        a,b,c,d=s.bbox_xyxy
        side=min(min(w,math.ceil(c))-max(0,math.floor(a)),min(h,math.ceil(d))-max(0,math.floor(b)))
        if side<16 and s.label!='unknown' and ((s.recognition_confidence or 0)<.95 or s.confidence<.5 or getattr(s,'localization_views',1)<2):
            s.label='unknown';s.recognition_source='tiny_detail_unsupported'
    return signs


def score(signs, truth, totals, details, sample_id, method):
    signs = [s for s in signs if s.recognition_source != 'background_rejected']
    boxes = [json.loads(x) for x in truth.bbox_xyxy]
    released = truth.semantic_final_disposition.str.startswith('RELEASED').to_numpy()
    side = truth.short_side_px.to_numpy(float)
    mat = iou_matrix([s.bbox_xyxy for s in signs], boxes)
    pi, gi = linear_sum_assignment(-np.where(mat >= .5, 1 + mat / 1000, 0))
    matched = {int(g): int(p) for p, g in zip(pi, gi) if mat[p, g] >= .5}
    used = set(matched.values())
    totals['images'] += 1
    totals['localization_tp'] += len(matched)
    totals['all_gt'] += len(truth)
    totals['detections'] += len(signs)
    for j, row in enumerate(truth.itertuples()):
        p = matched.get(j)
        sign = signs[p] if p is not None else None
        if not released[j]:
            continue
        label = sign.label if sign is not None else 'missed'
        correct = label == row.canonical_sign_code
        bucket = 'lt16' if side[j] < 16 else '16to32' if side[j] < 32 else 'ge32'
        totals['released_gt'] += 1
        totals['correct'] += int(correct)
        totals[bucket + '_gt'] += 1
        totals[bucket + '_correct'] += int(correct)
        totals['evaluable_known'] += int(label not in ['unknown', 'missed'])
        totals['wrong_known'] += int(label not in ['unknown', 'missed'] and not correct)
        details.append({'sample_id': sample_id, 'method': method, 'truth': row.canonical_sign_code,
                        'prediction': label, 'short_side_px': side[j],
                        'recognition_source': sign.recognition_source if sign else 'not_detected',
                        'box': json.loads(row.bbox_xyxy)})
    unmatched = sum(s.label != 'unknown' for p, s in enumerate(signs) if p not in used)
    totals['evaluable_known'] += unmatched
    totals['unmatched_known'] += unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['VAL', 'TEST'], default='VAL')
    ap.add_argument('--limit', type=int)
    ap.add_argument('--methods', nargs='+', default=['base960_16','rescue_16','rescue_12'])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api='blas'):
        run(args)


def run(args):
    loc = ROOT / 'data/processed/canonical/itas_mm_v1/localization'
    samples = pd.read_csv(loc / 'samples.csv')
    samples = samples[samples.split == args.split].sort_values('image_sha256')
    if args.limit:
        samples = samples.head(args.limit)
    gt = pd.read_csv(ROOT / 'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv', keep_default_na=False)
    groups = {key: value.reset_index(drop=True) for key, value in gt[gt.split == args.split].groupby('sample_id')}
    cfg = load_config('perception')
    detector, recognizer = SignDetector(cfg), SignRecognizer(cfg)
    totals = {m: defaultdict(int) for m in args.methods}
    times = {m: [] for m in args.methods}
    details = []
    started = time.perf_counter()
    for number, sample in enumerate(samples.itertuples(), 1):
        image = cv2.imread(str(loc / sample.image_path))
        cache = {}
        detector.image_size = 960
        tick = time.perf_counter()
        cache['base960'] = detector.detect(image)
        cache['base960_ms'] = (time.perf_counter() - tick) * 1000
        if any(m.startswith(('base640','dual','tiles','rescue')) for m in args.methods):
            detector.image_size=640;tick=time.perf_counter()
            cache['base640']=detector.detect(image)
            cache['base640_ms']=(time.perf_counter()-tick)*1000
            cache['dual']=merge_views(cache['base960'],cache['base640'])
            cache['dual_ms']=cache['base960_ms']+cache['base640_ms']
        if any(m.startswith('tiles') for m in args.methods):
            h,w=image.shape[:2];tw,th=round(w*.65),round(h*.65)
            detector.image_size=640;tick=time.perf_counter();tile_signs=[]
            for left,top in [(0,0),(w-tw,0),(0,h-th),(w-tw,h-th)]:
                for s in detector.detect(image[top:top+th,left:left+tw]):
                    a,b,c,d=s.bbox_xyxy
                    if (left and a<3) or (top and b<3) or (left+tw<w and c>tw-3) or (top+th<h and d>th-3):continue
                    s.bbox_xyxy=(a+left,b+top,c+left,d+top);tile_signs.append(s)
            cache['tiles']=merge_views(cache['dual'],tile_signs)
            cache['tiles_ms']=cache['dual_ms']+(time.perf_counter()-tick)*1000
        if any(m.startswith('full1280') for m in args.methods):
            detector.image_size = 1280
            tick = time.perf_counter()
            cache['full1280'] = detector.detect(image)
            cache['full1280_ms'] = (time.perf_counter() - tick) * 1000
        if any(m.startswith('focus960') for m in args.methods):
            detector.image_size = 960
            tick = time.perf_counter()
            cache['focus960'] = focused_detections(detector, image, cache['base960'])
            cache['focus960_ms'] = (time.perf_counter() - tick) * 1000 + cache['base960_ms']
        for method in args.methods:
            name, minimum = method.rsplit('_', 1)
            tick = time.perf_counter()
            if name=='rescue':
                from src.perception.sign_multiscale import recognize_views
                signs=recognize_views(recognizer,image,cache['base960'],cache['base640'],min_short_side=int(minimum))
                cache['rescue_ms']=cache['base960_ms']+cache['base640_ms']
            else:signs = recognizer.recognize(image, copy.deepcopy(cache[name]), min_short_side=int(minimum.removesuffix('strict')))
            if minimum.endswith('strict'):signs=strict_tiny(signs,image)
            times[method].append(cache[name + '_ms'] + (time.perf_counter() - tick) * 1000)
            score(signs, groups[sample.sample_id], totals[method], details, sample.sample_id, method)
        if number % 100 == 0:
            print('PROGRESS', number, '/', len(samples), 'seconds', round(time.perf_counter() - started), flush=True)
    for name, c in totals.items():
        c['code_recall'] = c['correct'] / max(1, c['released_gt'])
        c['known_code_precision'] = c['correct'] / max(1, c['evaluable_known'])
        c['localization_recall'] = c['localization_tp'] / max(1, c['all_gt'])
        c['localization_precision'] = c['localization_tp'] / max(1, c['detections'])
        for bucket in ['lt16', '16to32', 'ge32']:
            c[bucket + '_recall'] = c[bucket + '_correct'] / max(1, c[bucket + '_gt'])
        c['mean_ms'] = float(np.mean(times[name]))
        c['p95_ms'] = float(np.percentile(times[name], 95))
    report = {'split': args.split, 'images': len(samples), 'selection': 'SHA256 order',
              'results': totals, 'elapsed_seconds': time.perf_counter() - started,
              'config_sha256': hashlib.sha256((ROOT / 'configs/perception.yaml').read_bytes()).hexdigest(),
              'source': 'released canonical GT, one-to-one IoU .5', 'field_video_gt': False,
              'candidate_selection_on_test': False}
    report['runtime_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
        'scripts/evaluate_sign_candidates.py','src/perception/sign_multiscale.py','src/perception/sign_detector.py',
        'src/perception/sign_recognizer.py','src/perception/sign_open_set.py','src/perception/traffic_symbols.py',
        'models/sign_detector/retrained_20261009.pt','models/sign_recognizer/retrained_20261009.pt',
        'models/sign_recognizer/reference_gate_20261011_final.npz']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(details).to_csv(args.output / 'details.csv', index=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
