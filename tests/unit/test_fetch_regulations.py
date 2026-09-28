import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.fetch_regulations import fetch_regulations


class FakeResponse:
    def __init__(self, payload: dict | None = None, content: bytes | None = None):
        self.content = content if content is not None else json.dumps(payload).encode()
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload if self._payload is not None else json.loads(self.content)


def write_config(path: Path, documents: list[dict]) -> Path:
    path.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    return path


def vbpl_document(document_id: str = "law") -> dict:
    return {
        "document_id": document_id,
        "source_kind": "vbpl",
        "item_id": "170620",
        "expected_document_number": "36/2024/QH15",
        "source_url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=170620",
    }


def test_vbpl_number_mismatch_fails_closed(tmp_path: Path):
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"documents": ["old"]}', encoding="utf-8")

    with patch(
        "scripts.fetch_regulations.requests.get",
        return_value=FakeResponse({"docNum": "wrong"}),
    ):
        with pytest.raises(ValueError, match="document number mismatch"):
            fetch_regulations(str(config), str(tmp_path / "raw"), str(manifest))

    assert manifest.read_text(encoding="utf-8") == '{"documents": ["old"]}'
    assert not (tmp_path / "raw").exists()


def test_same_raw_bytes_produce_same_sha256_and_paths(tmp_path: Path):
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    payload = b'{"docNum":"36/2024/QH15","id":170620}'

    with patch(
        "scripts.fetch_regulations.requests.get",
        return_value=FakeResponse(content=payload),
    ):
        first = fetch_regulations(
            str(config), str(tmp_path / "raw"), str(tmp_path / "manifest.json")
        )
        second = fetch_regulations(
            str(config), str(tmp_path / "raw"), str(tmp_path / "manifest.json")
        )

    assert first["documents"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert second["documents"][0]["sha256"] == first["documents"][0]["sha256"]
    assert second["documents"][0]["raw_file"] == first["documents"][0]["raw_file"]


def test_existing_raw_file_with_different_bytes_is_not_overwritten(tmp_path: Path):
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    raw_file = tmp_path / "raw" / "law" / "source.json"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"old bytes")

    with patch(
        "scripts.fetch_regulations.requests.get",
        return_value=FakeResponse({"docNum": "36/2024/QH15"}),
    ):
        with pytest.raises(ValueError, match="immutable raw artifact differs"):
            fetch_regulations(str(config), str(tmp_path / "raw"), str(tmp_path / "manifest.json"))

    assert raw_file.read_bytes() == b"old bytes"


def test_failed_fetch_does_not_replace_existing_manifest(tmp_path: Path):
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"version": "previous"}', encoding="utf-8")

    with patch(
        "scripts.fetch_regulations.requests.get", side_effect=RuntimeError("offline")
    ):
        with pytest.raises(RuntimeError, match="offline"):
            fetch_regulations(str(config), str(tmp_path / "raw"), str(manifest))

    assert manifest.read_text(encoding="utf-8") == '{"version": "previous"}'


def test_qcvn_download_uses_curated_official_attachment(tmp_path: Path):
    config_path = Path("config/legal-corpus.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    responses = {
        "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/170620": FakeResponse(
            {"docNum": "36/2024/QH15"}
        ),
        "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/173920": FakeResponse(
            {"docNum": "168/2024/NĐ-CP"}
        ),
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf": FakeResponse(
            content=b"official qcvn pdf"
        ),
    }

    with patch(
        "scripts.fetch_regulations.requests.get",
        side_effect=lambda url, **_: responses[url],
    ) as get:
        fetch_regulations(str(config_path), str(tmp_path / "raw"), str(tmp_path / "manifest.json"))

    qcvn = next(item for item in config["documents"] if item["source_kind"] == "attachment")
    get.assert_any_call(qcvn["attachment_url"], timeout=30)
