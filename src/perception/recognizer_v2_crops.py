"""In-memory sign crops (with a margin for box jitter) and TRAIN-time augmentation for recognizer v2.

Evaluation uses `exact_crop`, which reproduces the floor/ceil box crop the runtime recognizer takes.
Augmentation never mirrors an image: left/right and turn directions carry meaning.
"""
from __future__ import annotations
import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

MARGIN = .12
MAX_SIDE = 256
SMALL_SHORT_SIDES = (10, 14, 18, 24, 32, 44)


def make_entry(image, bbox, margin=MARGIN, max_side=MAX_SIDE):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    ex1, ey1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    ex2, ey2 = min(w, math.ceil(x2)), min(h, math.ceil(y2))
    a, b = max(0, math.floor(x1 - margin * bw)), max(0, math.floor(y1 - margin * bh))
    c, d = min(w, math.ceil(x2 + margin * bw)), min(h, math.ceil(y2 + margin * bh))
    crop = image[b:d, a:c]
    inner = [ex1 - a, ey1 - b, ex2 - a, ey2 - b]
    if crop.size == 0 or ex2 <= ex1 or ey2 <= ey1:
        return None
    scale = min(1., max_side / max(crop.shape[:2]))
    if scale < 1.:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        inner = [round(v * scale) for v in inner]
        inner[2] = max(inner[2], inner[0] + 1)
        inner[3] = max(inner[3], inner[1] + 1)
    return dict(img=np.ascontiguousarray(crop), inner=inner)


def build_cache(rows, loc, workers=8):
    """rows: DataFrame with image_path and bbox_xyxy; returns entries in row order (None where unusable)."""
    groups = {}
    for position, row in enumerate(rows.itertuples()):
        groups.setdefault(row.image_path, []).append((position, json.loads(row.bbox_xyxy)))
    entries = [None] * len(rows)

    def work(item):
        path, boxes = item
        image = cv2.imread(str(Path(loc) / path))
        if image is None:
            return []
        return [(position, make_entry(image, bbox)) for position, bbox in boxes]

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(work, groups.items()):
            for position, entry in result:
                entries[position] = entry
    return entries


def save_cache(entries, path):
    with open(path, 'wb') as stream:
        pickle.dump(entries, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path):
    with open(path, 'rb') as stream:
        return pickle.load(stream)


def exact_crop(entry):
    x1, y1, x2, y2 = entry['inner']
    return entry['img'][y1:y2, x1:x2]


def jittered_crop(entry, rng):
    """Move each box edge by up to -8%..+10% of the box size inside the stored margin."""
    img = entry['img']
    h, w = img.shape[:2]
    x1, y1, x2, y2 = entry['inner']
    bw, bh = x2 - x1, y2 - y1
    nx1 = int(round(x1 + rng.uniform(-.10, .08) * bw))
    ny1 = int(round(y1 + rng.uniform(-.10, .08) * bh))
    nx2 = int(round(x2 + rng.uniform(-.08, .10) * bw))
    ny2 = int(round(y2 + rng.uniform(-.08, .10) * bh))
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(w, nx2), min(h, ny2)
    if nx2 - nx1 < 4 or ny2 - ny1 < 4:
        return exact_crop(entry)
    return img[ny1:ny2, nx1:nx2]


def augment(entry, rng):
    """Box jitter, brightness/contrast, light blur, +-5 degree rotation with scale, down-up resolution, JPEG."""
    crop = jittered_crop(entry, rng)
    if rng.random() < .55:
        short = rng.choice(SMALL_SHORT_SIDES)
        scale = min(1., short / min(crop.shape[:2]))
        if scale < 1.:
            small = cv2.resize(crop, (max(2, round(crop.shape[1] * scale)), max(2, round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            crop = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if rng.random() < .5:
        angle = rng.uniform(-5, 5)
        zoom = rng.uniform(.9, 1.1)
        h, w = crop.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, zoom)
        crop = cv2.warpAffine(crop, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .3:
        kernel = rng.choice([3, 5])
        crop = cv2.GaussianBlur(crop, (kernel, kernel), rng.uniform(.5, 1.5))
    if rng.random() < .8:
        crop = np.clip(crop.astype(np.float32) * rng.uniform(.6, 1.4) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    if rng.random() < .2:
        ok, encoded = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 80)])
        if ok:
            crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return crop
