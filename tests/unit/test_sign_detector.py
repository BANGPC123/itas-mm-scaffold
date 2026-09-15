import numpy as np

from src.perception.sign_detector import SignDetector
from src.utils.types import SignDetection


def _perception_config_no_weights():
    return {
        "sign_detector": {
            "weights_path": None,
            "confidence_threshold": 0.35,
            "iou_threshold": 0.45,
            "device": "cpu",
            "compound_group_distance_px": 40,
        },
        "sahi": {"enabled": False},
    }


def test_detect_returns_empty_list_when_no_weights_configured():
    detector = SignDetector(_perception_config_no_weights())
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(dummy_image) == []


def test_group_compound_signs_merges_nearby_panels():
    detector = SignDetector(_perception_config_no_weights())
    detections = [
        SignDetection(label="panel_a", confidence=0.9, bbox_xyxy=(10, 10, 30, 30)),
        SignDetection(label="panel_b", confidence=0.8, bbox_xyxy=(35, 10, 55, 30)),
    ]
    grouped = detector._group_compound_signs(detections)
    assert len(grouped) == 1
    assert grouped[0].is_compound is True
    assert set(grouped[0].panel_labels) == {"panel_a", "panel_b"}


def test_group_compound_signs_keeps_distant_panels_separate():
    detector = SignDetector(_perception_config_no_weights())
    detections = [
        SignDetection(label="panel_a", confidence=0.9, bbox_xyxy=(0, 0, 10, 10)),
        SignDetection(label="panel_b", confidence=0.8, bbox_xyxy=(500, 500, 520, 520)),
    ]
    grouped = detector._group_compound_signs(detections)
    assert len(grouped) == 2
    assert all(not d.is_compound for d in grouped)


def test_group_compound_signs_empty_input_returns_empty_list():
    detector = SignDetector(_perception_config_no_weights())
    assert detector._group_compound_signs([]) == []
