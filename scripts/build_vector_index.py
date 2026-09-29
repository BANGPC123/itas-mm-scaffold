"""One-off script: rebuild the Chroma vector index from canonical JSON.

Usage:
    python scripts/build_vector_index.py
"""
from __future__ import annotations

from src.reasoning.document_loader import (
    build_legal_chunks,
    compute_corpus_fingerprint,
    load_canonical_documents,
)
from src.reasoning.ollama_client import OllamaClient
from src.reasoning.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_index() -> None:
    reasoning_cfg = load_config("reasoning")
    rag_cfg = reasoning_cfg["rag"]

    documents = load_canonical_documents(rag_cfg["normalized_dir"])
    if not documents:
        raise ValueError("Cannot build a vector index from an empty corpus")

    chunks = [chunk for document in documents for chunk in build_legal_chunks(document)]
    if not chunks:
        raise ValueError("Cannot build a vector index with no legal chunks")

    schema_versions = {document.schema_version for document in documents}
    if len(schema_versions) != 1:
        raise ValueError("Canonical documents must share one schema version")
    schema_version = schema_versions.pop()

    embedding_model = reasoning_cfg["ollama"]["embedding_model"]
    fingerprint = compute_corpus_fingerprint(
        documents, schema_version, embedding_model
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
        len(documents),
        len(chunks),
        fingerprint,
        schema_version,
        embedding_model,
    )


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
