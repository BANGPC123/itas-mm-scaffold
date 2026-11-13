"""A7: blind contact sheets of the 239-frame field pack for two independent labeling passes.

Only the box geometry is used. The pack's label fields (label, ai_suggested_label, panel_role, judgment,
AI-confidence field) are NOT read into the sheets. Each pass gets its own random permutation of box IDs and
its own sheet grid; the ID-to-box key of each pass is written to a sealed folder and must not be opened
until that pass's label file is complete.

Also renders the QCVN 41:2024 template reference sheet (from data/reference/qcvn2024_symbols.json) with
the code printed under each template, for pass comparison against the official figures.
"""

from __future__ import annotations

import ast
import base64
import json
import random
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = Path("D:/DSP391m/project/review_pack_20261019_239_proposals")
OUT = ROOT / "output/bench_20261112/A7"
MIN_SIDE = 16
CELL = 128
PASSES = {"pass1": {"seed": 20261112, "cols": 6, "rows": 6, "margin": 0.10},
          "pass2": {"seed": 20261113, "cols": 5, "rows": 7, "margin": 0.20}}


def load_boxes():
    manifest = json.loads((PACK / "review_manifest.json").read_text(encoding="utf-8"))
    items = []
    for frame in manifest["frames"]:
        signs = frame["signs"]
        signs = ast.literal_eval(signs) if isinstance(signs, str) else signs
        for box in signs:
            x1, y1, x2, y2 = [float(v) for v in box["bbox_xyxy"]]
            items.append({"frame_id": frame["id"], "image": frame["image"], "box_id": box["id"],
                          "bbox_xyxy": [x1, y1, x2, y2], "short_side_px": min(x2 - x1, y2 - y1)})
    return items


def crop(image, box, margin):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = side * (0.5 + margin)
    xa, ya = int(max(0, cx - half)), int(max(0, cy - half))
    xb, yb = int(min(w, cx + half)), int(min(h, cy + half))
    patch = image[ya:yb, xa:xb]
    if patch.size == 0:
        patch = np.zeros((CELL, CELL, 3), np.uint8)
    scale = CELL / max(patch.shape[:2])
    resized = cv2.resize(patch, (max(1, int(patch.shape[1] * scale)), max(1, int(patch.shape[0] * scale))))
    canvas = np.zeros((CELL, CELL, 3), np.uint8)
    yo = (CELL - resized.shape[0]) // 2
    xo = (CELL - resized.shape[1]) // 2
    canvas[yo:yo + resized.shape[0], xo:xo + resized.shape[1]] = resized
    return canvas


def sheet(crops, labels, cols, rows):
    grid = np.full((rows * (CELL + 22), cols * CELL, 3), 255, np.uint8)
    for k, (img, label) in enumerate(zip(crops, labels)):
        r, c = divmod(k, cols)
        y, x = r * (CELL + 22), c * CELL
        grid[y + 22:y + 22 + CELL, x:x + CELL] = img
        cv2.putText(grid, label, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return grid


def main() -> None:
    items = [it for it in load_boxes() if it["short_side_px"] >= MIN_SIDE]
    frames_cache = {}
    for name, cfg in PASSES.items():
        rng = random.Random(cfg["seed"])
        order = list(range(len(items)))
        rng.shuffle(order)
        key = {}
        sheet_dir = OUT / name / "sheets"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        per_sheet = cfg["cols"] * cfg["rows"]
        crops, labels = [], []
        n_sheets = 0
        for position, index in enumerate(order):
            item = items[index]
            if item["image"] not in frames_cache:
                frames_cache[item["image"]] = cv2.imread(str(PACK / item["image"]))
            image = frames_cache[item["image"]]
            label_id = f"{name[-1]}{position:03d}"
            key[label_id] = {"frame_id": item["frame_id"], "box_id": item["box_id"], "bbox_xyxy": item["bbox_xyxy"]}
            crops.append(crop(image, item["bbox_xyxy"], cfg["margin"]))
            labels.append(label_id)
            if len(crops) == per_sheet:
                n_sheets += 1
                cv2.imwrite(str(sheet_dir / f"sheet_{n_sheets:02d}.jpg"), sheet(crops, labels, cfg["cols"], cfg["rows"]))
                crops, labels = [], []
        if crops:
            n_sheets += 1
            cv2.imwrite(str(sheet_dir / f"sheet_{n_sheets:02d}.jpg"), sheet(crops, labels, cfg["cols"], cfg["rows"]))
        sealed = OUT / "sealed"
        sealed.mkdir(parents=True, exist_ok=True)
        (sealed / f"{name}_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
        print(json.dumps({"pass": name, "boxes": len(items), "sheets": n_sheets}), flush=True)
    # QCVN template reference (no blind content; codes are the official ones)
    symbols = json.loads((ROOT / "data/reference/qcvn2024_symbols.json").read_text(encoding="utf-8"))["templates"]
    ref_crops, ref_labels = [], []
    for code in sorted(symbols):
        raw = base64.b64decode(symbols[code]["mask_png_base64"])
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3 and mask.shape[2] == 4:
            alpha = mask[:, :, 3]
            gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
            gray = np.where(alpha > 0, gray, 255).astype(np.uint8)
        else:
            gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        scale = CELL / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, (max(1, int(bgr.shape[1] * scale)), max(1, int(bgr.shape[0] * scale))))
        canvas = np.full((CELL, CELL, 3), 255, np.uint8)
        yo, xo = (CELL - bgr.shape[0]) // 2, (CELL - bgr.shape[1]) // 2
        canvas[yo:yo + bgr.shape[0], xo:xo + bgr.shape[1]] = bgr
        ref_crops.append(canvas)
        ref_labels.append(code)
    ref_dir = OUT / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    per = 8
    for s in range(0, len(ref_crops), per * 4):
        chunk = ref_crops[s:s + per * 4]
        labs = ref_labels[s:s + per * 4]
        cv2.imwrite(str(ref_dir / f"qcvn_templates_{s // (per * 4) + 1:02d}.jpg"),
                    sheet(chunk, labs, per, (len(chunk) + per - 1) // per))
    print(json.dumps({"reference_templates": len(ref_crops)}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
