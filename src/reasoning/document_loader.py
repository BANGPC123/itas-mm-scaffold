"""Load validated canonical legal documents and project retrieval evidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.reasoning.legal_models import LegalArticle, LegalDocument


@dataclass
class DocumentChunk:
    text: str
    source_file: str
    chunk_index: int
    document_id: str = ""
    locator_type: str = ""
    locator: str = ""


def load_canonical_documents(normalized_dir: str) -> list[LegalDocument]:
    """Load normalized JSON documents in deterministic filename order."""
    directory = Path(normalized_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Normalized directory not found: {directory}")

    return [
        LegalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]


def _leaf_chunks(
    document: LegalDocument,
    text: str,
    locator_type: str,
    locator: str,
    max_chunk_chars: int | None,
) -> list[DocumentChunk]:
    if max_chunk_chars is None or len(text) <= max_chunk_chars:
        texts = [text]
    else:
        texts = [
            text[start : start + max_chunk_chars]
            for start in range(0, len(text), max_chunk_chars)
        ]

    return [
        DocumentChunk(
            text=chunk_text,
            source_file=document.source.raw_file,
            chunk_index=index,
            document_id=document.document_id,
            locator_type=locator_type,
            locator=locator,
        )
        for index, chunk_text in enumerate(texts)
    ]


def _article_chunks(
    document: LegalDocument,
    article: LegalArticle,
    max_chunk_chars: int | None,
    *,
    qcvn: bool = False,
) -> list[DocumentChunk]:
    if qcvn:
        article_locator = article.article_id
        article_type = "section"
    else:
        article_locator = f"Điều {article.article_id}"
        article_type = "article"

    if not article.clauses:
        return _leaf_chunks(
            document, article.text or "", article_type, article_locator, max_chunk_chars
        )

    chunks: list[DocumentChunk] = []
    for clause in article.clauses:
        clause_locator = (
            f"{article_locator}.{clause.clause_id}"
            if qcvn
            else f"{article_locator} Khoản {clause.clause_id}"
        )
        if not clause.points:
            chunks.extend(
                _leaf_chunks(
                    document,
                    clause.text or "",
                    "section" if qcvn else "clause",
                    clause_locator,
                    max_chunk_chars,
                )
            )
            continue

        for point in clause.points:
            point_locator = (
                f"{clause_locator}.{point.point_id}"
                if qcvn
                else f"{clause_locator} Điểm {point.point_id}"
            )
            chunks.extend(
                _leaf_chunks(
                    document,
                    point.text or "",
                    "section" if qcvn else "point",
                    point_locator,
                    max_chunk_chars,
                )
            )
    return chunks


def build_legal_chunks(
    document: LegalDocument, max_chunk_chars: int | None = None
) -> list[DocumentChunk]:
    """Project a canonical document's structural leaves into retrieval chunks."""
    if max_chunk_chars is not None and max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive")

    chunks: list[DocumentChunk] = []
    for article in document.articles:
        chunks.extend(_article_chunks(document, article, max_chunk_chars))

    for section in document.sections:
        if not section.articles:
            chunks.extend(
                _leaf_chunks(
                    document,
                    section.text or "",
                    "section",
                    section.section_id,
                    max_chunk_chars,
                )
            )
            continue
        for article in section.articles:
            chunks.extend(_article_chunks(document, article, max_chunk_chars, qcvn=True))
    return chunks


def build_corpus_chunks(
    normalized_dir: str, max_chunk_chars: int | None = None
) -> list[DocumentChunk]:
    """Load validated canonical JSON and project its legal leaves."""
    return [
        chunk
        for document in load_canonical_documents(normalized_dir)
        for chunk in build_legal_chunks(document, max_chunk_chars)
    ]
