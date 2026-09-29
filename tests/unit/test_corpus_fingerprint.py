from __future__ import annotations

from datetime import datetime, timezone

from src.reasoning.document_loader import compute_corpus_fingerprint
from src.reasoning.legal_models import LegalArticle, LegalDocument, LegalSource


def _document(*, raw_file: str = "raw/law/source.json", retrieved_at: datetime | None = None) -> LegalDocument:
    return LegalDocument(
        document_id="law-36-2024-qh15",
        title="Law 36",
        document_type="law",
        source=LegalSource(
            source_url="https://example.gov.vn/law-36",
            raw_file=raw_file,
            sha256="a" * 64,
            retrieved_at=retrieved_at or datetime(2026, 9, 29, tzinfo=timezone.utc),
        ),
        articles=[LegalArticle(article_id="1", text="Semantic legal text.")],
    )


def test_fingerprint_stable_for_same_semantic_documents():
    document = _document()

    first = compute_corpus_fingerprint([document], "1", "nomic-embed-text")
    second = compute_corpus_fingerprint([document], "1", "nomic-embed-text")

    assert first == second
    assert first == compute_corpus_fingerprint(
        [document.model_copy(deep=True)], "1", "nomic-embed-text"
    )


def test_fingerprint_changes_with_schema_or_embedding_model():
    document = _document()
    fingerprint = compute_corpus_fingerprint([document], "1", "nomic-embed-text")

    assert fingerprint != compute_corpus_fingerprint([document], "2", "nomic-embed-text")
    assert fingerprint != compute_corpus_fingerprint([document], "1", "other-embedding")


def test_fingerprint_ignores_retrieved_at_and_local_raw_path():
    original = _document()
    relocated = _document(
        raw_file="C:/different-machine/raw/law/source.json",
        retrieved_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
    )

    assert compute_corpus_fingerprint(
        [original], "1", "nomic-embed-text"
    ) == compute_corpus_fingerprint([relocated], "1", "nomic-embed-text")
