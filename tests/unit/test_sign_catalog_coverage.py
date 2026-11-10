import json
from pathlib import Path

from src.perception.sign_catalog import NAMES_VI, display_name

CLASSES_PATH = Path(__file__).resolve().parents[2] / "data/reference/recognizer_v2_classes.json"


def recognizer_codes():
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))["classes"]
    return [code for code in classes if code != "unknown"]


def test_every_recognizer_code_has_a_vietnamese_name():
    missing = [code for code in recognizer_codes() if not NAMES_VI.get(code, "").strip()]
    assert missing == []


def test_catalog_has_no_names_outside_recognizer_classes():
    extra = sorted(set(NAMES_VI) - set(recognizer_codes()))
    assert extra == []


def test_display_name_uses_code_and_name_for_every_recognizer_code():
    for code in recognizer_codes():
        assert display_name(code) == f"{code} — {NAMES_VI[code]}"
