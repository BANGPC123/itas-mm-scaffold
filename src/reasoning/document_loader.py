"""Load validated canonical legal documents and project retrieval evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import unicodedata


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


_SOURCE_FIELDS = (
    "document_id",
    "document_number",
    "title",
    "source_kind",
    "source_url",
    "raw_file",
    "sha256",
    "text_sha256",
    "retrieved_at",
)
_CHUNK_FIELDS = ("id", "document_id", "locator_type", "locator", "text", "checksum")


def _required_string(record: dict[str, Any], field: str, record_type: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_type} requires non-empty {field}")
    return value


def _required_count(record: dict[str, Any], field: str, record_type: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{record_type} requires non-negative {field}")
    return value


def _validate_corpus(corpus: dict[str, Any]) -> None:
    manifest = corpus.get("manifest")
    sources = corpus.get("sources")
    chunks = corpus.get("chunks")
    if not isinstance(manifest, dict):
        raise ValueError("corpus requires a manifest object")
    if not isinstance(sources, list) or not sources:
        raise ValueError("corpus requires non-empty sources")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("corpus requires non-empty chunks")

    _required_string(manifest, "schema_version", "manifest")
    _required_string(manifest, "generated_at", "manifest")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict):
        raise ValueError("manifest requires an upstream object")
    _required_string(upstream, "repository", "manifest upstream")
    _required_string(upstream, "commit", "manifest upstream")
    if _required_count(manifest, "source_count", "manifest") != len(sources):
        raise ValueError("manifest source_count does not match sources")
    if _required_count(manifest, "chunk_count", "manifest") != len(chunks):
        raise ValueError("manifest chunk_count does not match chunks")

    source_ids: set[str] = set()
    source_chunk_counts: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        for field in _SOURCE_FIELDS:
            _required_string(source, field, "source")
        document_id = source["document_id"]
        if document_id in source_ids:
            raise ValueError(f"duplicate source id: {document_id}")
        source_ids.add(document_id)
        source_chunk_counts[document_id] = _required_count(source, "chunk_count", "source")

    chunk_ids: set[str] = set()
    actual_chunk_counts = dict.fromkeys(source_ids, 0)
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("each chunk must be an object")
        for field in _CHUNK_FIELDS:
            _required_string(chunk, field, "chunk")
        _required_count(chunk, "ordinal", "chunk")
        heading = chunk.get("heading")
        if heading is not None and (not isinstance(heading, str) or not heading.strip()):
            raise ValueError("chunk heading must be null or non-empty text")
        chunk_id = chunk["id"]
        if chunk_id in chunk_ids:
            raise ValueError(f"duplicate chunk id: {chunk_id}")
        chunk_ids.add(chunk_id)
        document_id = chunk["document_id"]
        if document_id not in source_ids:
            raise ValueError(f"chunk references unknown source: {document_id}")
        actual_chunk_counts[document_id] += 1

    if actual_chunk_counts != source_chunk_counts:
        raise ValueError("source chunk_count does not match chunks")


def load_regulation_corpus(corpus_path: str | Path) -> dict:
    """Load the validated unified regulation corpus artifact."""
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if not isinstance(corpus, dict):
        raise ValueError("corpus root must be an object")
    _validate_corpus(corpus)
    return corpus


def compute_corpus_fingerprint(
    corpus: dict, schema_version: str, embedding_model: str
) -> str:
    """Return a stable hash for corpus semantics and index inputs."""
    _validate_corpus(corpus)
    semantic_corpus = json.loads(json.dumps(corpus))
    semantic_corpus["manifest"].pop("generated_at")
    for source in semantic_corpus["sources"]:
        source.pop("retrieved_at")
        source.pop("raw_file")
    canonical_data = json.dumps(
        {
            "schema_version": schema_version,
            "embedding_model": embedding_model,
            "corpus": semantic_corpus,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_data.encode("utf-8")).hexdigest()


def build_corpus_chunks(corpus_path: str | Path) -> list[DocumentChunk]:
    """Project corpus chunks into retrieval records in artifact order."""
    corpus = load_regulation_corpus(corpus_path)
    source_files = {
        source["document_id"]: source["raw_file"] for source in corpus["sources"]
    }
    return [
        DocumentChunk(
            text=chunk["text"],
            source_file=source_files[chunk["document_id"]],
            chunk_index=chunk["ordinal"],
            document_id=chunk["document_id"],
            locator_type=chunk["locator_type"],
            locator=chunk["locator"],
        )
        for chunk in corpus["chunks"]
    ]
