"""One-off script: (re)builds the Chroma vector index from
data/regulations/*.md|*.txt.

Run this after adding/updating regulation documents, and any time the
embedding model changes (embeddings from different models are not
comparable).

Usage:
    python scripts/build_vector_index.py
"""
from __future__ import annotations

from src.reasoning.document_loader import build_corpus_chunks
from src.reasoning.ollama_client import OllamaClient
from src.reasoning.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    reasoning_cfg = load_config("reasoning")
    rag_cfg = reasoning_cfg["rag"]

    chunks = build_corpus_chunks(
        regulations_dir=rag_cfg["regulations_dir"],
        chunk_size_chars=rag_cfg["chunk_size_chars"],
        chunk_overlap_chars=rag_cfg["chunk_overlap_chars"],
    )
    logger.info("Loaded %d chunk(s) from %s", len(chunks), rag_cfg["regulations_dir"])

    ollama_client = OllamaClient(reasoning_cfg["ollama"])
    vector_store = VectorStore(rag_cfg, ollama_client)
    vector_store.index_chunks(chunks)


if __name__ == "__main__":
    main()
