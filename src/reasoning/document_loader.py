"""Loads and chunks the traffic-regulation corpus for retrieval.

Chunking uses a simple fixed-size sliding window with overlap. This is a
reasonable default for short regulation paragraphs; if the real corpus
turns out to have long, deeply nested clauses, revisit with a
structure-aware splitter (e.g. by section/article number) instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    text: str
    source_file: str
    chunk_index: int


def load_regulation_documents(regulations_dir: str) -> list[tuple[str, str]]:
    """Load all .md/.txt files from the regulations directory.

    Returns a list of (filename, full_text) tuples. Raises FileNotFoundError
    if the directory doesn't exist, and returns an empty list (with no
    fabricated content) if the directory exists but has no documents.
    """
    dir_path = Path(regulations_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Regulations directory not found: {dir_path}")

    documents = []
    for path in sorted(dir_path.glob("*.md")) + sorted(dir_path.glob("*.txt")):
        documents.append((path.name, path.read_text(encoding="utf-8")))
    return documents


def chunk_text(
    text: str, source_file: str, chunk_size_chars: int, chunk_overlap_chars: int
) -> list[DocumentChunk]:
    """Split text into overlapping fixed-size chunks."""
    if chunk_overlap_chars >= chunk_size_chars:
        raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")

    chunks: list[DocumentChunk] = []
    start = 0
    index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size_chars, text_length)
        chunk_text_value = text[start:end].strip()
        if chunk_text_value:
            chunks.append(
                DocumentChunk(
                    text=chunk_text_value, source_file=source_file, chunk_index=index
                )
            )
            index += 1
        if end == text_length:
            break
        start = end - chunk_overlap_chars

    return chunks


def build_corpus_chunks(
    regulations_dir: str, chunk_size_chars: int, chunk_overlap_chars: int
) -> list[DocumentChunk]:
    """Load every document in regulations_dir and chunk it."""
    all_chunks: list[DocumentChunk] = []
    for filename, text in load_regulation_documents(regulations_dir):
        all_chunks.extend(
            chunk_text(text, filename, chunk_size_chars, chunk_overlap_chars)
        )
    return all_chunks
