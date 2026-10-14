"""YOLOv11 panel localization with optional SAHI slicing.

Missing checkpoints return no detections. Context grouping runs after recognition.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger
from src.utils.types import SignDetection

logger = get_logger(__name__)


class SignDetector:
    def __init__(self, perception_config: dict[str, Any]):
        sd_cfg = perception_config["sign_detector"]
        self.sahi_cfg = perception_config.get("sahi", {"enabled": False})

        self.confidence_threshold = sd_cfg["confidence_threshold"]
        self.iou_threshold = sd_cfg["iou_threshold"]
        self.image_size = sd_cfg.get('image_size',640)
        self.video_multiscale = sd_cfg.get('video_multiscale', False)
        self.secondary_image_size = sd_cfg.get('secondary_image_size', 640)
        self.device = sd_cfg["device"]
        if self.device == "auto":
            import torch
            self.device = "0" if torch.cuda.is_available() else "cpu"
        self.compound_group_distance_px = sd_cfg["compound_group_distance_px"]
        self.sahi_enabled = self.sahi_cfg.get("enabled", False)

        self._model = None
        self._sahi_model = None
        weights_path = sd_cfg.get("weights_path")

        if not weights_path or not Path(weights_path).exists():
            logger.warning(
                "No sign-detector weights found at %r. detect() will return "
                "an empty list until a trained model is provided. See "
                "README.md 'Training the sign detector'.",
                weights_path,
            )
            return

        self._load_model(weights_path)

    def _load_model(self, weights_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for sign detection. "
                "Install it with `pip install ultralytics`."
            ) from exc

        self._model = YOLO(weights_path)

        if self.sahi_enabled:
            try:
                from sahi import AutoDetectionModel

                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=weights_path,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device,
                )
            except ImportError:
                logger.warning(
                    "sahi is not installed; falling back to plain YOLO "
                    "inference without slicing. Small/distant signs may be "
                    "missed. Install with `pip install sahi`."
                )
                self.sahi_enabled = False

    def detect(self, image: np.ndarray) -> list[SignDetection]:
        """Run detection on a single BGR image (as read by OpenCV).

        Returns an empty list if no model is loaded — callers must not
        assume a non-empty result.
        """
        if self._model is None:
            return []

        raw_boxes = (
            self._detect_with_sahi(image)
            if self.sahi_enabled and self._sahi_model is not None
            else self._detect_plain(image)
        )
        return raw_boxes

    def detect_views(self, image: np.ndarray):
        """Keep native coordinates for two inference scales; restore state."""
        primary = self.detect(image)
        original_size = self.image_size
        try:
            self.image_size = self.secondary_image_size
            secondary = self.detect(image)
        finally:
            self.image_size = original_size
        return primary, secondary

    def _detect_plain(self, image: np.ndarray) -> list[SignDetection]:
        results = self._model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.image_size,
            verbose=False,
        )
        detections: list[SignDetection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                xyxy = tuple(box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append(
                    SignDetection(label=names[cls_id], confidence=conf, bbox_xyxy=xyxy)
                )
        return detections

    def _detect_with_sahi(self, image: np.ndarray) -> list[SignDetection]:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image,
            self._sahi_model,
            slice_height=self.sahi_cfg.get("slice_height", 512),
            slice_width=self.sahi_cfg.get("slice_width", 512),
            overlap_height_ratio=self.sahi_cfg.get("overlap_height_ratio", 0.2),
            overlap_width_ratio=self.sahi_cfg.get("overlap_width_ratio", 0.2),
            postprocess_type=self.sahi_cfg.get("postprocess_type", "NMS"),
            postprocess_match_threshold=self.sahi_cfg.get(
                "postprocess_match_threshold", 0.5
            ),
            verbose=0,
        )
        detections: list[SignDetection] = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            detections.append(
                SignDetection(
                    label=pred.category.name,
                    confidence=float(pred.score.value),
                    bbox_xyxy=(bbox.minx, bbox.miny, bbox.maxx, bbox.maxy),
                )
            )
        return detections

    def _group_compound_signs(self, detections):
        from src.context.sign_applicability import SignApplicability
        return SignApplicability().group(detections)
