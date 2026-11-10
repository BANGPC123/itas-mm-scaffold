"""Rebuild checks must never treat missing or tampered evidence as success."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import check_rebuild_readiness as check


def write_json(root: Path, path: str, value):
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value), encoding="utf-8")


def write_bytes(root: Path, path: str, value: bytes):
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(value)


def test_empty_checkout_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "ROOT", tmp_path)
    write_json(tmp_path, "docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json",
               {"file_count": 1, "files": [
                   {"path": "models/model.pt", "sha256": "0" * 64, "size_bytes": 99}
               ]})
    result = check.audit()
    assert result["lfs"]["verified_count"] == 0
    assert not result["clean_clone_core_assets_ready"]
    assert not result["historical_full_acceptance_ready"]


def test_verified_public_snapshot_cannot_promote_lost_human_data(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "ROOT", tmp_path)
    fake_weight = b"model-bytes"
    write_bytes(tmp_path, "models/model.pt", fake_weight)
    write_json(tmp_path, "docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json",
               {"file_count": 1, "files": [
                   {"path": "models/model.pt",
                    "sha256": hashlib.sha256(fake_weight).hexdigest(),
                    "size_bytes": len(fake_weight)}
               ]})
    rows = []
    for n in range(4):
        blob = ("%PDF research source " + str(n)).encode()
        filename = "law-" + str(n) + ".pdf"
        write_bytes(tmp_path,
                    "data/regulations/official_20261011/" + filename, blob)
        rows.append({
            "file": filename, "source_url": "https://example.test/law/" + filename,
            "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
    write_json(tmp_path, "data/regulations/official_20261011/sources.json",
               {"official_documents": rows})
    write_json(tmp_path, "docs/OFFICIAL_PDF_SNAPSHOT_20261022.json",
               {"sources": rows})
    write_json(tmp_path,
               "data/processed/legal_reference/official_corpus_20261011/manifest.json",
               {"chunks": 1, "source_count": 4})
    write_bytes(tmp_path,
                "data/processed/legal_reference/official_corpus_20261011/chunks.jsonl",
                b'{"id":"law-0:p1:c0"}\n')
    write_bytes(tmp_path,
                "third_party/Ultra-Fast-Lane-Detection-v2/utils/loss.py", b"")
    monkeypatch.setattr(
        check.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=check.UFLD_SHA))
    result = check.audit()
    assert result["clean_clone_core_assets_ready"]
    assert not result["historical_full_acceptance_ready"]
    assert result["legal_snapshot"]["pinned_snapshot_verified"]
    rows[0]["source_url"] = "https://example.test/changed"
    write_json(tmp_path, "docs/OFFICIAL_PDF_SNAPSHOT_20261022.json",
               {"sources": rows})
    assert not check.audit()["clean_clone_core_assets_ready"]
