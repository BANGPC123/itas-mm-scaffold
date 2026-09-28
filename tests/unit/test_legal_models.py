from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.reasoning.legal_models import (
    LegalArticle,
    LegalClause,
    LegalDocument,
    LegalPoint,
    LegalSection,
    LegalSource,
    SCHEMA_VERSION,
)


def _source() -> LegalSource:
    return LegalSource(
        source_url="https://example.gov.vn/law/2025",
        raw_file="law-2025.html",
        sha256="a" * 64,
        retrieved_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
    )


def _point(point_id: str = "a") -> LegalPoint:
    return LegalPoint(point_id=point_id, text="Vehicles must stop at a red light.")


def _clause(clause_id: str = "1") -> LegalClause:
    return LegalClause(clause_id=clause_id, points=[_point()])


def _article(article_id: str = "1") -> LegalArticle:
    return LegalArticle(article_id=article_id, clauses=[_clause()])


def test_valid_law_hierarchy():
    document = LegalDocument(
        document_id="law-2025",
        title="Road Traffic Law",
        document_type="law",
        source=_source(),
        articles=[_article()],
        sections=[],
    )

    assert document.schema_version == SCHEMA_VERSION == "1"
    assert document.articles[0].clauses[0].points[0].text.startswith("Vehicles")


def test_valid_qcvn_section_hierarchy():
    document = LegalDocument(
        document_id="qcvn-41",
        title="National Technical Regulation",
        document_type="qcvn",
        source=_source(),
        articles=[],
        sections=[LegalSection(section_id="I", articles=[_article()])],
    )

    assert document.sections[0].articles[0].article_id == "1"


def test_duplicate_articles_rejected():
    with pytest.raises(ValidationError, match="duplicate article_id: 1"):
        LegalDocument(
            document_id="law-2025",
            title="Road Traffic Law",
            document_type="law",
            source=_source(),
            articles=[_article(), _article()],
            sections=[],
        )


def test_duplicate_clauses_and_points_rejected():
    with pytest.raises(ValidationError, match="duplicate clause_id: 1"):
        LegalArticle(article_id="1", clauses=[_clause(), _clause()])

    with pytest.raises(ValidationError, match="duplicate point_id: a"):
        LegalClause(clause_id="1", points=[_point(), _point()])


def test_duplicate_sections_rejected():
    with pytest.raises(ValidationError, match="duplicate section_id: I"):
        LegalDocument(
            document_id="qcvn-41",
            title="National Technical Regulation",
            document_type="qcvn",
            source=_source(),
            articles=[],
            sections=[
                LegalSection(section_id="I", articles=[_article()]),
                LegalSection(section_id="I", articles=[_article("2")]),
            ],
        )


def test_empty_leaf_rejected():
    cases = [
        lambda: LegalPoint(point_id="a", text=" \t "),
        lambda: LegalClause(clause_id="1"),
        lambda: LegalArticle(article_id="1"),
        lambda: LegalSection(section_id="I"),
    ]

    for build in cases:
        with pytest.raises(ValidationError, match="text must contain non-whitespace characters"):
            build()


def test_missing_source_provenance_rejected():
    with pytest.raises(ValidationError, match="source\\n  Field required"):
        LegalDocument(
            document_id="law-2025",
            title="Road Traffic Law",
            document_type="law",
            articles=[_article()],
            sections=[],
        )
