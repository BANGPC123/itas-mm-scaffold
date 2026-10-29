import numpy as np
import subprocess
import sys

from scripts.report_recognizer_operating_points_20261029 import metrics, select_threshold


def test_metrics_count_unknown_low_confidence_and_wrong_labels():
    result = metrics(["A", "A", "B", "B"], ["A", "B", "unknown", "B"],
                     [0.9, 0.9, 0.99, 0.4], threshold=0.5)
    assert result == {"n": 4, "accepted": 2, "correct": 1, "coverage": 0.5,
                      "accepted_accuracy": 0.5, "end_to_end": 0.25, "wrong_label_rate": 0.25}


def test_threshold_uses_only_known_validation_rows():
    logits = np.log(np.array([[0.90, 0.05, 0.05], [0.85, 0.10, 0.05],
                              [0.80, 0.10, 0.10], [0.97, 0.02, 0.01]]))
    result = select_threshold(logits, [0, 0, 0, 2], unknown_index=2, min_coverage=2 / 3)
    assert np.isclose(result["threshold"], 0.85)
    assert result["coverage"] == 2 / 3
    assert result["background_accepted"] == 1.0


def test_threshold_fallback_does_not_claim_infeasible_target():
    logits = np.log(np.array([[0.10, 0.90], [0.10, 0.90], [0.90, 0.10]]))
    result = select_threshold(logits, [0, 0, 0], unknown_index=1, min_coverage=0.95)
    assert result["criterion"] == "fallback_accuracy"
    assert result["coverage"] == 1 / 3


def test_command_line_entrypoint_is_available():
    result = subprocess.run([sys.executable, "scripts/report_recognizer_operating_points_20261029.py", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--val-logits" in result.stdout and "--output" in result.stdout
