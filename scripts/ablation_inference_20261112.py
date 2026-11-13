"""A3: inference-mode ablation for the current sign detector (no training).

Modes:
  plain640        ultralytics predict at 640 (the reporting protocol; must reproduce A1 on TEST)
  plain960        ultralytics predict at 960 (video_image_size in configs/perception.yaml)
  multi960_640    detector-level analogue of the video path: 960 boxes kept as they are, 640 boxes
                  added when their IoU with every 960 box is below 0.3 (same score, no recognizer)
  sahi512_o0.2    sahi get_sliced_prediction with the production settings (512 slices, overlap 0.2,
                  NMS, match threshold 0.5)

Splits: TEST (canonical localization set, protocol of A1) and FIELD (120 owner-reviewed pilot frames,
boxes from bulk approval of detector proposals; circular, see docs). Scoring reuses the frozen runner's
matching at confidence 0.35, NMS 0.45, IoU 0.5. Predictions are stored at confidence 0.001 for AP.

Timing is NOT measured here: the GPU was shared with training. Re-time after the queue finishes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUNNER = ROOT / "output/kiet_wp_20261026/A/run_wp3_benchmarks.py"
FIELD_FRAMES = ROOT / "output/kiet_wp_20261026/C/human_corpus/frames.jsonl"
WEIGHTS = ROOT / "models/sign_detector/finetuned_20261026.pt"
OUT = ROOT / "output/bench_20261112/A3"
MODES = ["plain640", "plain960", "multi960_640", "sahi512_o0.2"]
AP_FLOOR = 0.001
NMS_IOU = 0.45


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("wp3_frozen_runner", FROZEN_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.maximum(0, br - tl), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


class Predictor:
    def __init__(self, weights: Path):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self._sahi = None

    def _plain(self, image, size):
        result = self.model.predict(image, imgsz=size, conf=AP_FLOOR, iou=NMS_IOU, max_det=300,
                                    device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.empty((0, 4)), np.empty((0,))
        return result.boxes.xyxy.detach().cpu().numpy(), result.boxes.conf.detach().cpu().numpy()

    def _sahi_predict(self, image):
        if self._sahi is None:
            from sahi import AutoDetectionModel

            self._sahi = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=str(WEIGHTS),
                confidence_threshold=AP_FLOOR, device=self.device)
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(image, self._sahi, slice_height=512, slice_width=512,
                                       overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                                       postprocess_type="NMS", postprocess_match_threshold=0.5, verbose=0)
        boxes, scores = [], []
        for pred in result.object_prediction_list:
            b = pred.bbox
            boxes.append([b.minx, b.miny, b.maxx, b.maxy])
            scores.append(pred.score.value)
        return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(scores, dtype=float)

    def predict(self, image, mode: str):
        if mode == "plain640":
            boxes, scores = self._plain(image, 640)
        elif mode == "plain960":
            boxes, scores = self._plain(image, 960)
        elif mode == "multi960_640":
            b960, s960 = self._plain(image, 960)
            b640, s640 = self._plain(image, 640)
            if len(b640):
                keep = xyxy_iou(b640.astype(float), b960.astype(float)).max(axis=1) < 0.3 if len(b960) else np.ones(len(b640), bool)
                b640, s640 = b640[keep], s640[keep]
            boxes = np.concatenate([b960, b640]) if len(b640) or len(b960) else np.empty((0, 4))
            scores = np.concatenate([s960, s640]) if len(s640) or len(s960) else np.empty((0,))
        elif mode == "sahi512_o0.2":
            boxes, scores = self._sahi_predict(image)
        else:
            raise ValueError(mode)
        return [{"box": [float(v) for v in b], "confidence": float(s)} for b, s in zip(boxes, scores)]


def test_records(module, predictor, mode):
    samples = pd.read_csv(module.LOC / "samples.csv", keep_default_na=False)
    labels = pd.read_csv(module.AUDIT / "semantic_labels_v1_1.csv", keep_default_na=False)
    samples = samples[samples["split"] == "TEST"].reset_index(drop=True)
    labels = labels[labels["split"] == "TEST"]
    truths = {sid: [json.loads(v) for v in g["bbox_xyxy"].tolist()] for sid, g in labels.groupby("sample_id", sort=False)}
    records = []
    for row in samples.itertuples():
        path = (module.LOC / row.image_path).resolve()
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(str(path))
        records.append({"sample_id": row.sample_id, "gts": truths.get(row.sample_id, []),
                        "predictions": predictor.predict(image, mode)})
    return records


def field_records(predictor, mode):
    rows = [json.loads(line) for line in FIELD_FRAMES.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Same eligibility as the 2026-10-26 pilot scorer: TEST frames with every visible sign annotated.
    rows = [r for r in rows if r.get("split") == "TEST" and r.get("all_visible_signs_annotated") is True]
    records = []
    for row in rows:
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        gts = [s["bbox_xyxy"] for s in row.get("signs", []) if s.get("judgment") == "sign"]
        records.append({"sample_id": row["frame_id"], "gts": gts, "predictions": predictor.predict(image, mode)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--splits", nargs="+", default=["TEST", "FIELD"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_frozen_runner()
    predictor = Predictor(WEIGHTS)
    summary = {}
    for split in args.splits:
        for mode in args.modes:
            started = time.perf_counter()
            records = test_records(module, predictor, mode) if split == "TEST" else field_records(predictor, mode)
            module.build_ap_records(records)
            metrics = module.metrics_from_images(records)
            metrics.pop("map50_95", None)
            result = {"split": split, "mode": mode, "weights_sha256": module.sha(WEIGHTS),
                      "images": len(records), "seconds_wall_shared_gpu": time.perf_counter() - started,
                      "metrics": metrics}
            (OUT / f"ablation_{mode}_{split.lower()}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            summary[f"{split}/{mode}"] = {k: metrics[k] for k in ("precision", "recall", "map50", "tp", "fp", "fn")}
            print(json.dumps({f"{split}/{mode}": summary[f"{split}/{mode}"]}), flush=True)
    (OUT / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
