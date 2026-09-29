from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.normalize_regulations import normalize_regulations
from src.reasoning.legal_models import LegalSource
from src.reasoning.legal_normalizer import (
    extract_pdf_text,
    extract_vbpl_text,
    normalize_document,
)


def source() -> LegalSource:
    return LegalSource(
        source_url="https://example.gov.vn/source",
        raw_file="raw/law/source.json",
        sha256="a" * 64,
        retrieved_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )


def test_vbpl_html_normalizes_unicode_and_structure():
    text = extract_vbpl_text(
        {
            "content": "<style>ignored</style><p>\u0110ie\u0302\u0300u 1. <b>Quy đi\u0323nh</b></p>"
            "<script>ignored()</script><p>1. No\u0302\u0323i dung</p>",
        }
    )

    assert text == "Điều 1. Quy định\n1. Nội dung"


def test_vbpl_nested_document_content_normalizes_unicode_and_structure():
    text = extract_vbpl_text(
        {
            "data": {
                "documentContent": {
                    "content": "<p>Điều 1. <b>Quy định</b></p>"
                }
            }
        }
    )

    assert text == "Điều 1. Quy định"


def test_configured_document_missing_from_manifest_fails_before_normalized_writes(tmp_path: Path):
    raw = b'{"content":"<p>\\u0110i\\u1ec1u 1. Quy t\\u1eafc</p>"}'
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "source.json").write_bytes(raw)
    config = {
        "documents": [
            {
                "document_id": "law-1",
                "source_kind": "vbpl",
                "expected_document_number": "Law 1",
            },
            {
                "document_id": "law-2",
                "source_kind": "vbpl",
                "expected_document_number": "Law 2",
            },
        ]
    }
    manifest = {
        "documents": [
            {
                "document_id": "law-1",
                "raw_file": "raw/source.json",
                "normalized_file": "normalized/law-1.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_url": "https://example.gov.vn/law-1",
                "retrieved_at": "2026-09-29T00:00:00Z",
            }
        ]
    }
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="configured document missing from manifest: law-2"):
        normalize_regulations(config_path, manifest_path)

    assert not (tmp_path / "normalized").exists()


def test_duplicate_configured_document_id_fails_before_normalized_writes(tmp_path: Path):
    config = {
        "documents": [
            {
                "document_id": "law-1",
                "source_kind": "vbpl",
                "expected_document_number": "Law 1",
            },
            {
                "document_id": "law-1",
                "source_kind": "vbpl",
                "expected_document_number": "Law 1 duplicate",
            },
        ]
    }
    manifest = {"documents": []}
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate configured document_id: law-1"):
        normalize_regulations(config_path, manifest_path)

    assert not (tmp_path / "normalized").exists()


def test_duplicate_manifest_document_id_fails_before_normalized_writes(tmp_path: Path):
    config = {
        "documents": [
            {
                "document_id": "law-1",
                "source_kind": "vbpl",
                "expected_document_number": "Law 1",
            }
        ]
    }
    manifest = {
        "documents": [
            {"document_id": "law-1"},
            {"document_id": "law-1"},
        ]
    }
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate manifest document_id: law-1"):
        normalize_regulations(config_path, manifest_path)

    assert not (tmp_path / "normalized").exists()


def test_law_parser_preserves_article_clause_point():
    document, unresolved = normalize_document(
        "Điều 1. Quy tắc\n1. Người lái xe phải tuân thủ.\na) Dừng khi đèn đỏ.",
        document_id="law-1",
        title="Luật mẫu",
        document_type="law",
        source=source(),
    )

    point = document.articles[0].clauses[0].points[0]
    assert (document.articles[0].article_id, document.articles[0].clauses[0].clause_id) == (
        "1",
        "1",
    )
    assert point.point_id == "a"
    assert point.text == "Dừng khi đèn đỏ."
    assert unresolved == []


def test_law_structural_heading_is_unresolved_not_merged_into_point():
    document, unresolved = normalize_document(
        "Điều 1. Quy tắc\n1. Người lái xe phải tuân thủ.\na) Dừng khi đèn đỏ.\n"
        "CHƯƠNG II\nQuy định riêng\nĐiều 2. Quy tắc khác",
        document_id="law-1",
        title="Luật mẫu",
        document_type="law",
        source=source(),
    )

    assert document.articles[0].clauses[0].points[0].text == "Dừng khi đèn đỏ."
    assert unresolved == ["CHƯƠNG II\nQuy định riêng"]


def test_qcvn_parser_builds_nested_sections():
    document, unresolved = normalize_document(
        "1. QUY ĐỊNH CHUNG\n1.1. Phạm vi điều chỉnh\n1.2. Đối tượng áp dụng",
        document_id="qcvn-1",
        title="QCVN mẫu",
        document_type="qcvn",
        source=source(),
    )

    assert document.sections[0].section_id == "1"
    assert [article.article_id for article in document.sections[0].articles] == [
        "1.1",
        "1.2",
    ]
    assert unresolved == []


def test_qcvn_structural_heading_is_unresolved_not_merged_into_article():
    document, unresolved = normalize_document(
        "1. QUY ĐỊNH CHUNG\n1.1. Phạm vi điều chỉnh\nMỤC 2\nQuy định riêng\n"
        "2. QUY ĐỊNH KHÁC",
        document_id="qcvn-1",
        title="QCVN mẫu",
        document_type="qcvn",
        source=source(),
    )

    assert document.sections[0].articles[0].text == "Phạm vi điều chỉnh"
    assert unresolved == ["MỤC 2\nQuy định riêng"]


def test_unresolved_block_is_reported_without_assist():
    _, unresolved = normalize_document(
        "LỜI NÓI ĐẦU\nĐiều 1. Quy tắc",
        document_id="law-1",
        title="Luật mẫu",
        document_type="law",
        source=source(),
    )

    assert unresolved == ["LỜI NÓI ĐẦU"]


def test_assist_is_called_only_for_unresolved_block():
    calls: list[str] = []

    def assist(block: str) -> dict:
        calls.append(block)
        return {"article_id": "0", "text": "Lời nói đầu"}

    document, unresolved = normalize_document(
        "LỜI NÓI ĐẦU\nĐiều 1. Quy tắc",
        document_id="law-1",
        title="Luật mẫu",
        document_type="law",
        source=source(),
        assist=assist,
    )

    assert calls == ["LỜI NÓI ĐẦU"]
    assert [article.article_id for article in document.articles] == ["1", "0"]
    assert unresolved == []


def test_invalid_assist_payload_is_rejected():
    with pytest.raises(ValueError, match="invalid assist payload"):
        normalize_document(
            "LỜI NÓI ĐẦU",
            document_id="law-1",
            title="Luật mẫu",
            document_type="law",
            source=source(),
            assist=lambda _: {"article_id": "0", "text": "   "},
        )


def test_pdf_without_usable_text_fails_explicitly(monkeypatch, tmp_path):
    class BlankPage:
        def extract_text(self) -> str:
            return ""

    class BlankReader:
        pages = [BlankPage()]

    monkeypatch.setattr("src.reasoning.legal_normalizer.PdfReader", lambda _: BlankReader())

    with pytest.raises(ValueError, match="usable text"):
        extract_pdf_text(tmp_path / "scanned.pdf")
