import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "build_dataset_v2_manifest", ROOT / "scripts/build_dataset_v2_manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(key, sha, split, group):
    return {"sample_key": key, "image_sha256": sha, "split": split, "split_group_id": group}


def test_clean_rows_have_no_leakage():
    rows = [row("a", "1", "TRAIN", "g1"), row("b", "2", "TRAIN", "g1"), row("c", "3", "TEST", "g2")]
    assert module.leak_report(rows) == {}


def test_group_across_splits_is_leakage():
    rows = [row("a", "1", "TRAIN", "g1"), row("b", "2", "TEST", "g1")]
    assert "split_group_id" in module.leak_report(rows)


def test_same_image_in_two_splits_is_leakage():
    rows = [row("a", "1", "TRAIN", "g1"), row("b", "1", "VAL", "g2")]
    findings = module.leak_report(rows)
    assert "image_sha256" in findings and "duplicate_image_sha256" in findings


def test_duplicate_image_within_split_is_reported():
    rows = [row("a", "1", "TEST", "g1"), row("b", "1", "TEST", "g2")]
    assert "duplicate_image_sha256" in module.leak_report(rows)


def test_split_digest_is_order_independent_and_content_sensitive():
    rows = [row("a", "1", "TEST", "g"), row("b", "2", "TEST", "g")]
    assert module.split_digest(rows) == module.split_digest(list(reversed(rows)))
    assert module.split_digest(rows) != module.split_digest(rows[:1])
