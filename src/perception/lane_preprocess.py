"""Lane-only image preprocessing for the offline UFLD display path.

The output keeps shape, dtype and pixel grid, so UFLD coordinates map back to the original frame unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np

LANE_PREPROCESS_MODES = ('none', 'gamma', 'clahe')


def preprocess_for_lane(image_bgr: np.ndarray, mode: str = 'none', *,
                        gamma: float = .8, clahe_clip: float = 2., clahe_tile: int = 8) -> np.ndarray:
    if mode == 'none':
        return image_bgr
    if mode == 'gamma':
        if not gamma > 0:
            raise ValueError('lane_gamma must be greater than 0')
        table = np.clip(np.round(255. * (np.arange(256) / 255.) ** gamma), 0, 255).astype(np.uint8)
        return cv2.LUT(image_bgr, table)
    if mode == 'clahe':
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_tile), int(clahe_tile)))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    raise ValueError(f'lane_preprocess must be one of {LANE_PREPROCESS_MODES}')
