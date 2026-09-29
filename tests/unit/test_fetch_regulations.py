import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from scripts.fetch_regulations import fetch_regulations, fetch_with_retries, main


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def write_config(path: Path, documents: list[dict]) -> Path:
    path.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    return path


def vbpl_document(document_id: str = "law-36-2024-qh15") -> dict:
    return {
        "document_id": document_id,
        "document_number": "36/2024/QH15",
        "title": "Law on Road Traffic Order and Safety",
        "source_kind": "vbpl_json",
        "item_id": "170620",
        "expected_document_number": "36/2024/QH15",
        "source_url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=170620",
    }


def vbpl_response(number: str = "36/2024/QH15") -> FakeResponse:
    return FakeResponse(
        json.dumps(
            {
                "data": {
                    "docNum": number,
                    "title": "Law on Road Traffic Order and Safety",
                    "documentContent": {
                        "content": f"<p>{number} applies to road traffic safety and enforcement.</p>"
                    },
                }
            }
        ).encode()
    )


def test_retry_policy_retries_408_429_and_5xx_then_succeeds() -> None:
    responses = [
        FakeResponse(b"", 408),
        FakeResponse(b"", 429),
        FakeResponse(b"", 503),
        FakeResponse(b"ok"),
    ]

    with patch("scripts.fetch_regulations.requests.get", side_effect=responses) as get, patch(
        "scripts.fetch_regulations.time.sleep"
    ) as sleep:
        response = fetch_with_retries("https://example.test")

    assert response.content == b"ok"
    assert get.call_count == 4
    assert sleep.call_args_list == [((0.75,),), ((1.5,),), ((3.0,),)]


def test_source_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    corpus = tmp_path / "corpus.json"
    corpus.write_text('{"previous": true}', encoding="utf-8")

    with patch(
        "scripts.fetch_regulations.requests.get", return_value=vbpl_response("wrong")
    ):
        with pytest.raises(ValueError, match="document number mismatch"):
            fetch_regulations(config, tmp_path / "raw", corpus)

    assert corpus.read_text(encoding="utf-8") == '{"previous": true}'
    assert not (tmp_path / "raw").exists()


def test_same_raw_bytes_are_a_noop(tmp_path: Path) -> None:
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    raw_file = tmp_path / "raw" / "law-36-2024-qh15" / "source.json"

    with patch("scripts.fetch_regulations.requests.get", return_value=vbpl_response()):
        fetch_regulations(config, tmp_path / "raw", tmp_path / "corpus.json")
        first_bytes = raw_file.read_bytes()
        fetch_regulations(config, tmp_path / "raw", tmp_path / "corpus.json")

    assert raw_file.read_bytes() == first_bytes


def test_different_raw_bytes_are_not_overwritten(tmp_path: Path) -> None:
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    raw_file = tmp_path / "raw" / "law-36-2024-qh15" / "source.json"
    raw_file.parent.mkdir(parents=True)
    raw_file.write_bytes(b"existing official bytes")
    corpus = tmp_path / "corpus.json"
    corpus.write_text('{"previous": true}', encoding="utf-8")

    with patch("scripts.fetch_regulations.requests.get", return_value=vbpl_response()):
        with pytest.raises(ValueError, match="immutable raw artifact differs"):
            fetch_regulations(config, tmp_path / "raw", corpus)

    assert raw_file.read_bytes() == b"existing official bytes"
    assert corpus.read_text(encoding="utf-8") == '{"previous": true}'


def test_failed_source_does_not_replace_existing_corpus(tmp_path: Path) -> None:
    config = write_config(tmp_path / "config.json", [vbpl_document()])
    corpus = tmp_path / "corpus.json"
    corpus.write_text('{"previous": true}', encoding="utf-8")

    with patch(
        "scripts.fetch_regulations.requests.get", side_effect=requests.ConnectionError("offline")
    ), patch("scripts.fetch_regulations.time.sleep"):
        with pytest.raises(requests.ConnectionError, match="offline"):
            fetch_regulations(config, tmp_path / "raw", corpus)

    assert corpus.read_text(encoding="utf-8") == '{"previous": true}'


def test_duplicate_document_ids_are_rejected(tmp_path: Path) -> None:
    config = write_config(
        tmp_path / "config.json", [vbpl_document(), vbpl_document("law-36-2024-qh15")]
    )

    with patch("scripts.fetch_regulations.requests.get") as get:
        with pytest.raises(ValueError, match="duplicate document_id"):
            fetch_regulations(config, tmp_path / "raw", tmp_path / "corpus.json")

    get.assert_not_called()


def test_corpus_publication_contains_unique_source_and_chunk_ids(tmp_path: Path) -> None:
    decree_number = "168/2024/N\u0110-CP"
    documents = [
        vbpl_document(),
        {
            "document_id": "decree-168-2024-nd-cp",
            "document_number": decree_number,
            "title": "Decree on road traffic penalties",
            "source_kind": "official_html",
            "expected_document_number": decree_number,
            "source_url": "https://example.test/decree",
        },
        {
            "document_id": "qcvn-41-2024-bgtvt",
            "document_number": "QCVN 41:2024/BGTVT",
            "title": "National technical regulation on road signs",
            "source_kind": "official_pdf",
            "expected_document_number": "QCVN 41:2024/BGTVT",
            "source_url": "https://example.test/qcvn-page",
            "attachment_url": "https://example.test/qcvn.pdf",
        },
    ]
    config = write_config(tmp_path / "config.json", documents)
    decree_html = (
        f"<html><body><article itemprop='articleBody'><p>{decree_number} "
        "sets administrative penalties for road traffic violations.</p></article></body></html>"
    ).encode()
    responses = {
        "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/170620": vbpl_response(),
        "https://example.test/decree": FakeResponse(decree_html),
        "https://example.test/qcvn.pdf": FakeResponse(b"official qcvn pdf"),
    }

    with patch(
        "scripts.fetch_regulations.requests.get", side_effect=lambda url, **_: responses[url]
    ), patch(
        "scripts.fetch_regulations.extract_pdf_text",
        return_value="QCVN 41:2024/BGTVT sets road-sign requirements.",
    ):
        corpus = fetch_regulations(config, tmp_path / "raw", tmp_path / "corpus.json")

    assert set(corpus) == {"manifest", "sources", "chunks"}
    assert corpus["manifest"]["schema_version"] == "1"
    assert corpus["manifest"]["upstream"]["repository"] == "lqb464/LuatRAG"
    assert len(corpus["sources"]) == 3
    assert len({source["document_id"] for source in corpus["sources"]}) == 3
    assert len(corpus["chunks"]) == len({chunk["id"] for chunk in corpus["chunks"]})
    assert all(source["chunk_count"] > 0 for source in corpus["sources"])
    assert all(
        {
            "document_id",
            "document_number",
            "title",
            "source_kind",
            "source_url",
            "raw_file",
            "sha256",
            "text_sha256",
            "retrieved_at",
            "chunk_count",
        } <= source.keys()
        for source in corpus["sources"]
    )
    assert all("raw_sha256" not in source for source in corpus["sources"])


def test_cli_help_exits_without_network() -> None:
    with patch("scripts.fetch_regulations.requests.get") as get:
        with pytest.raises(SystemExit) as exit_code:
            main(["--help"])

    assert exit_code.value.code == 0
    get.assert_not_called()
