"""One-off script: rebuild the Chroma vector index from the corpus artifact.

Usage:
    python scripts/build_vector_index.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reasoning.document_loader import (
    DocumentChunk,
    compute_corpus_fingerprint,
    load_regulation_corpus,
)
from src.reasoning.ollama_client import OllamaClient
from src.reasoning.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_index() -> None:
    reasoning_cfg = load_config("reasoning")
    rag_cfg = reasoning_cfg["rag"]

    corpus = load_regulation_corpus(rag_cfg["corpus_file"])
    source_files = {
        source["document_id"]: source["raw_file"] for source in corpus["sources"]
    }
    chunks = [
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
    if not chunks:
        raise ValueError("Cannot build a vector index from an empty corpus")

    schema_version = corpus["manifest"]["schema_version"]

    embedding_model = reasoning_cfg["ollama"]["embedding_model"]
    fingerprint = compute_corpus_fingerprint(
        corpus, schema_version, embedding_model
    )
    ollama_client = OllamaClient(reasoning_cfg["ollama"])
    ollama_client.embed("")
    vector_store = VectorStore(rag_cfg, ollama_client)
    vector_store.rebuild(
        chunks,
        fingerprint=fingerprint,
        schema_version=schema_version,
        embedding_model=embedding_model,
    )
    logger.info(
        "Rebuilt legal index: documents=%d chunks=%d fingerprint=%s "
        "schema_version=%s embedding_model=%s",
        len(corpus["sources"]),
        len(chunks),
        fingerprint,
        schema_version,
        embedding_model,
    )


def main() -> None:
    argparse.ArgumentParser().parse_args()
    build_index()


if __name__ == "__main__":
    main()
