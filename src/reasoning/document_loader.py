"""Load validated canonical legal documents and project retrieval evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

from src.reasoning.legal_models import LegalArticle, LegalDocument


@dataclass
class DocumentChunk:
    text: str
    source_file: str
    chunk_index: int
    document_id: str = ""
    locator_type: str = ""
    locator: str = ""


_HEADING = re.compile(
    r"^(chương\s+[ivxlcdm\d]+|mục\s+\d+|điều\s+\d+[a-z]?|khoản\s+\d+|phần\s+(?:thứ\s+)?[^\W_]+)\s*[.:]?",
    re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).replace("\0", "")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[\t\f\v]+", " ", value)
    value = re.sub(r" {2,}", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def stable_checksum(value: str) -> str:
    """Return LuatRAG's pinned FNV-1a checksum over UTF-16 code units."""
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    hashed = 2166136261
    for index in range(0, len(encoded), 2):
        hashed = (
            (hashed ^ int.from_bytes(encoded[index : index + 2], "little")) * 16777619
        ) & 0xFFFFFFFF
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while hashed:
        hashed, remainder = divmod(hashed, 36)
        result = alphabet[remainder] + result
    return result or "0"


def split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Split at sentence boundaries before falling back to a hard boundary."""
    if len(paragraph) <= max_size:
        return [paragraph]
    pieces: list[str] = []
    buffer = ""
    for sentence in re.split(r"(?<=[.!?;])\s+", paragraph):
        if len(sentence) > max_size:
            if buffer:
                pieces.append(buffer.strip())
                buffer = ""
            remaining = sentence.strip()
            while len(remaining) > max_size:
                boundary = remaining.rfind(" ", 0, max_size + 1)
                if boundary < max_size * 0.65:
                    boundary = max_size
                pieces.append(remaining[:boundary].strip())
                remaining = remaining[boundary:].lstrip()
            if remaining:
                pieces.append(remaining)
        else:
            if buffer and len(buffer) + len(sentence) + 1 > max_size:
                pieces.append(buffer.strip())
                buffer = ""
            buffer += (" " if buffer else "") + sentence
    if buffer.strip():
        pieces.append(buffer.strip())
    return pieces


def chunk_regulation_text(
    document_id: str, text: str, *, target_size: int = 950, max_size: int = 1400
) -> list[dict[str, object]]:
    """Build deterministic heading-aware retrieval records from regulation text."""
    text = re.sub(
        r"(?<=[.;!?])\s+(?=(?:chương\s+[ivxlcdm\d]+|mục\s+\d+|điều\s+\d+[a-z]?)\s*[.:]?)",
        "\n\n",
        _normalize_text(text),
        flags=re.IGNORECASE,
    )
    paragraphs = [
        part.strip()
        for part in re.split(
            r"\n{2,}|(?=^(?:Chương|Mục|Điều|Khoản)\s+)", text, flags=re.IGNORECASE | re.MULTILINE
        )
        if part.strip()
    ]
    chunks: list[dict[str, object]] = []
    buffer = ""
    heading = ""

    def flush() -> None:
        nonlocal buffer
        body = _normalize_text(buffer)
        buffer = ""
        if len(body) <= 20:
            return
        ordinal = len(chunks)
        locator = heading or f"Đoạn {ordinal + 1}"
        checksum = stable_checksum(f"{document_id}:{locator}:{body}")
        chunks.append(
            {
                "id": f"{document_id}:{ordinal}:{checksum}",
                "ordinal": ordinal,
                "document_id": document_id,
                "locator_type": "heading" if heading else "chunk",
                "locator": locator[:240],
                "heading": heading or None,
                "text": body,
                "checksum": checksum,
            }
        )

    for paragraph in paragraphs:
        marker = _HEADING.match(paragraph)
        if marker:
            if buffer:
                flush()
            title = re.split(r"(?<=[.!?])\s+|\n", paragraph[marker.end() :].strip(), maxsplit=1)[0][:150].strip()
            heading = (marker.group().strip() + (f" {title}" if title else ""))[:180]
        for piece in split_long_paragraph(paragraph, max_size):
            if buffer and len(buffer) + len(piece) + 2 > max_size:
                flush()
            buffer += ("\n\n" if buffer else "") + piece
            if len(buffer) >= target_size:
                flush()
    flush()
    return chunks


def compute_corpus_fingerprint(
    documents: list[LegalDocument], schema_version: str, embedding_model: str
) -> str:
    """Return a stable hash for the semantic canonical corpus and index inputs."""
    semantic_documents = []
    for document in documents:
        semantic_document = document.model_dump(mode="json")
        semantic_document["source"].pop("raw_file")
        semantic_document["source"].pop("retrieved_at")
        semantic_documents.append(semantic_document)

    canonical_documents = sorted(
        semantic_documents,
        key=lambda document: json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )
    canonical_data = json.dumps(
        {
            "schema_version": schema_version,
            "embedding_model": embedding_model,
            "documents": canonical_documents,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()


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
