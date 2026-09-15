"""Lane detection: classical computer-vision baseline.

This is intentionally NOT a learned model. It exists so the Perception
stage produces real, runnable output without requiring a training dataset
first. Replace with a learned row-wise / structure-aware architecture as
described in the project proposal — see README.md "Limitations".
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.utils.types import LaneInfo

logger = get_logger(__name__)


class LaneDetector:
    def __init__(self, perception_config: dict[str, Any]):
        cfg = perception_config["lane_detector"]
        self.canny_low = cfg["canny_low_threshold"]
        self.canny_high = cfg["canny_high_threshold"]
        self.hough_rho = cfg["hough_rho"]
        self.hough_theta = np.deg2rad(cfg["hough_theta_deg"])
        self.hough_threshold = cfg["hough_threshold"]
        self.hough_min_line_length = cfg["hough_min_line_length"]
        self.hough_max_line_gap = cfg["hough_max_line_gap"]
        self.roi_top_ratio = cfg["roi_top_ratio"]

    def detect(self, image: np.ndarray) -> LaneInfo:
        """Run lane detection on a single BGR image.

        Returns LaneInfo with an empty line list if no lines are found
        (e.g. blank/black input) rather than raising.
        """
        if image is None or image.size == 0:
            logger.warning("Received empty image; returning no lane lines.")
            return LaneInfo(lane_lines=[])

        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        mask = self._region_of_interest_mask(width, height)
        masked_edges = cv2.bitwise_and(edges, mask)

        raw_lines = cv2.HoughLinesP(
            masked_edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        if raw_lines is None:
            return LaneInfo(lane_lines=[])

        lane_lines = [tuple(int(v) for v in line[0]) for line in raw_lines]
        offset = self._estimate_center_offset(lane_lines, width)

        return LaneInfo(lane_lines=lane_lines, lane_center_offset_px=offset)

    def _region_of_interest_mask(self, width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        top_y = int(height * self.roi_top_ratio)
        polygon = np.array(
            [[(0, height), (width, height), (width, top_y), (0, top_y)]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, polygon, 255)
        return mask

    @staticmethod
    def _estimate_center_offset(
        lines: list[tuple[int, int, int, int]], frame_width: int
    ) -> float | None:
        """Rough lane-center offset: average x of all detected line
        endpoints minus the frame's horizontal center. This is a coarse
        heuristic, not a calibrated measurement — flagged as a known
        limitation of the classical-CV baseline.
        """
        if not lines:
            return None
        xs = [x for line in lines for x in (line[0], line[2])]
        avg_x = sum(xs) / len(xs)
        return avg_x - frame_width / 2
