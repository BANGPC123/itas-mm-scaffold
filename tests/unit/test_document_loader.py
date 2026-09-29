from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.reasoning.document_loader import build_corpus_chunks, build_legal_chunks
from src.reasoning.legal_models import (
    LegalArticle,
    LegalClause,
    LegalDocument,
    LegalPoint,
    LegalSection,
    LegalSource,
)


def _source() -> LegalSource:
    return LegalSource(
        source_url="https://example.gov.vn/law",
        raw_file="raw/law/source.json",
        sha256="a" * 64,
        retrieved_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )


def _document(**kwargs: object) -> LegalDocument:
    values = {
        "document_id": "law-36-2024-qh15",
        "title": "Law",
        "document_type": "law",
        "source": _source(),
        "articles": [],
        "sections": [],
    }
    values.update(kwargs)
    return LegalDocument(**values)


def test_point_becomes_chunk_with_exact_locator():
    document = _document(
        articles=[
            LegalArticle(
                article_id="6",
                clauses=[
                    LegalClause(
                        clause_id="1",
                        points=[LegalPoint(point_id="a", text="Stop at red lights.")],
                    )
                ],
            )
        ]
    )

    chunk = build_legal_chunks(document)[0]

    assert (
        chunk.text,
        chunk.source_file,
        chunk.document_id,
        chunk.locator_type,
        chunk.locator,
        chunk.chunk_index,
    ) == (
        "Stop at red lights.",
        "raw/law/source.json",
        "law-36-2024-qh15",
        "point",
        "Điều 6 Khoản 1 Điểm a",
        0,
    )


def test_clause_without_points_becomes_chunk():
    document = _document(
        articles=[
            LegalArticle(article_id="6", clauses=[LegalClause(clause_id="1", text="Clause text.")])
        ]
    )

    chunk = build_legal_chunks(document)[0]

    assert (chunk.text, chunk.locator_type, chunk.locator, chunk.chunk_index) == (
        "Clause text.",
        "clause",
        "Điều 6 Khoản 1",
        0,
    )


def test_article_without_clauses_becomes_chunk():
    document = _document(articles=[LegalArticle(article_id="6", text="Article text.")])

    chunk = build_legal_chunks(document)[0]

    assert (chunk.text, chunk.locator_type, chunk.locator, chunk.chunk_index) == (
        "Article text.",
        "article",
        "Điều 6",
        0,
    )


def test_qcvn_leaf_section_becomes_chunk():
    document = _document(
        document_id="qcvn-41-2024-bgtvt",
        document_type="qcvn",
        sections=[LegalSection(section_id="1.1", text="Scope.")],
    )

    chunk = build_legal_chunks(document)[0]

    assert (chunk.text, chunk.locator_type, chunk.locator, chunk.chunk_index) == (
        "Scope.",
        "section",
        "1.1",
        0,
    )


def test_projection_order_is_deterministic():
    document = _document(
        articles=[
            LegalArticle(article_id="2", text="Second article."),
            LegalArticle(
                article_id="3",
                clauses=[
                    LegalClause(
                        clause_id="1",
                        points=[
                            LegalPoint(point_id="a", text="First point."),
                            LegalPoint(point_id="b", text="Second point."),
                        ],
                    )
                ],
            ),
        ]
    )

    first = build_legal_chunks(document)

    assert build_legal_chunks(document) == first
    assert [(chunk.locator, chunk.chunk_index) for chunk in first] == [
        ("Điều 2", 0),
        ("Điều 3 Khoản 1 Điểm a", 0),
        ("Điều 3 Khoản 1 Điểm b", 0),
    ]


def test_long_leaf_split_preserves_locator_when_limit_is_explicit():
    document = _document(articles=[LegalArticle(article_id="6", text="abcdefghij")])

    chunks = build_legal_chunks(document, max_chunk_chars=4)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert {(chunk.document_id, chunk.locator_type, chunk.locator) for chunk in chunks} == {
        ("law-36-2024-qh15", "article", "Điều 6")
    }


def test_invalid_canonical_json_fails_before_projection(tmp_path, monkeypatch):
    (tmp_path / "invalid.json").write_text("{", encoding="utf-8")
    projected = False

    def record_projection(_: LegalDocument):
        nonlocal projected
        projected = True
        return []

    monkeypatch.setattr("src.reasoning.document_loader.build_legal_chunks", record_projection)

    with pytest.raises(json.JSONDecodeError):
        build_corpus_chunks(str(tmp_path))

    assert not projected
