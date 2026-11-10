from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.reasoning.document_loader import (
    DocumentChunk,
    build_corpus_chunks,
    chunk_regulation_text,
    load_regulation_corpus,
    split_long_paragraph,
    stable_checksum,
)


def _corpus() -> dict:
    return {
        "manifest": {
            "schema_version": "1",
            "upstream": {"repository": "lqb464/LuatRAG", "commit": "pinned"},
            "generated_at": "2026-09-29T00:00:00+00:00",
            "source_count": 1,
            "chunk_count": 2,
        },
        "sources": [
            {
                "document_id": "law-36-2024-qh15",
                "document_number": "36/2024/QH15",
                "title": "Law 36",
                "source_kind": "vbpl_json",
                "source_url": "https://example.test/law-36",
                "raw_file": "raw/law-36/source.json",
                "sha256": "a" * 64,
                "text_sha256": "b" * 64,
                "retrieved_at": "2026-09-29T00:00:00+00:00",
                "chunk_count": 2,
            }
        ],
        "chunks": [
            {
                "id": "law-36:7:first",
                "ordinal": 7,
                "document_id": "law-36-2024-qh15",
                "locator_type": "article",
                "locator": "Điều 7",
                "heading": "Điều 7",
                "text": "First legal chunk.",
                "checksum": "first",
            },
            {
                "id": "law-36:1:second",
                "ordinal": 1,
                "document_id": "law-36-2024-qh15",
                "locator_type": "article",
                "locator": "Điều 1",
                "heading": "Điều 1",
                "text": "Second legal chunk.",
                "checksum": "second",
            },
        ],
    }


def _write_corpus(path: Path, corpus: dict) -> Path:
    path.write_text(json.dumps(corpus), encoding="utf-8")
    return path


@pytest.mark.parametrize("field", ["sources", "chunks"])
def test_load_corpus_rejects_empty_sources_or_chunks(tmp_path: Path, field: str):
    corpus = _corpus()
    corpus[field] = []

    with pytest.raises(ValueError, match=field):
        load_regulation_corpus(_write_corpus(tmp_path / "corpus.json", corpus))


def test_load_corpus_rejects_duplicate_source_ids(tmp_path: Path):
    corpus = _corpus()
    corpus["sources"].append(copy.deepcopy(corpus["sources"][0]))
    corpus["manifest"]["source_count"] = 2

    with pytest.raises(ValueError, match="duplicate source id"):
        load_regulation_corpus(_write_corpus(tmp_path / "corpus.json", corpus))


def test_load_corpus_rejects_duplicate_chunk_ids(tmp_path: Path):
    corpus = _corpus()
    corpus["chunks"][1]["id"] = corpus["chunks"][0]["id"]

    with pytest.raises(ValueError, match="duplicate chunk id"):
        load_regulation_corpus(_write_corpus(tmp_path / "corpus.json", corpus))


def test_load_corpus_rejects_chunk_for_unknown_source(tmp_path: Path):
    corpus = _corpus()
    corpus["chunks"][0]["document_id"] = "unknown-source"

    with pytest.raises(ValueError, match="unknown source"):
        load_regulation_corpus(_write_corpus(tmp_path / "corpus.json", corpus))


def test_corpus_chunks_round_trip_provenance(tmp_path: Path):
    chunks = build_corpus_chunks(_write_corpus(tmp_path / "corpus.json", _corpus()))

    assert chunks == [
        DocumentChunk(
            text="First legal chunk.",
            source_file="raw/law-36/source.json",
            chunk_index=7,
            document_id="law-36-2024-qh15",
            locator_type="article",
            locator="Điều 7",
        ),
        DocumentChunk(
            text="Second legal chunk.",
            source_file="raw/law-36/source.json",
            chunk_index=1,
            document_id="law-36-2024-qh15",
            locator_type="article",
            locator="Điều 1",
        ),
    ]


def test_corpus_chunk_order_is_deterministic(tmp_path: Path):
    corpus_path = _write_corpus(tmp_path / "corpus.json", _corpus())

    assert [chunk.text for chunk in build_corpus_chunks(corpus_path)] == [
        "First legal chunk.",
        "Second legal chunk.",
    ]
    assert build_corpus_chunks(corpus_path) == build_corpus_chunks(corpus_path)


def test_stable_checksum_matches_pinned_luatrag_behavior():
    assert stable_checksum("abc") == "7aigaz"


def test_chunking_is_stable_for_same_text():
    text = (
        "Điều 1. Phạm vi điều chỉnh\n"
        "Người điều khiển phương tiện phải tuân thủ quy định này. "
        "Việc tuân thủ bảo đảm an toàn giao thông.\n\n"
        "Nội dung không có tiêu đề vẫn phải được định vị ổn định."
    )

    first = chunk_regulation_text("law-1", text, target_size=70, max_size=100)

    assert chunk_regulation_text("law-1", text, target_size=70, max_size=100) == first
    assert first[0]["id"] == f"law-1:0:{first[0]['checksum']}"
    assert first[0]["locator_type"] == "heading"
    assert first[0]["locator"] == "Điều 1. Phạm vi điều chỉnh"
    assert first[0]["heading"] == "Điều 1. Phạm vi điều chỉnh"
    assert all(chunk["checksum"] for chunk in first)

    unheaded = chunk_regulation_text(
        "law-2", "Nội dung không có tiêu đề vẫn phải được định vị ổn định."
    )

    assert unheaded[0]["locator_type"] == "chunk"
    assert unheaded[0]["locator"] == "Đoạn 1"


def test_split_long_paragraph_prefers_sentence_boundaries():
    assert split_long_paragraph("One. Two sentence.", 10) == ["One.", "Two senten", "ce."]
