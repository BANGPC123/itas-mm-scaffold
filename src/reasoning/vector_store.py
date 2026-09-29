"""Vector store for retrieval: Chroma (local, embedded) with embeddings
supplied by Ollama via OllamaClient.

Chroma was chosen over a hosted vector DB because it runs fully locally
with no additional service to stand up, which matches the "local LLM,
no API cost" decision for this project.
"""
from __future__ import annotations

from typing import Any

import chromadb

from src.reasoning.document_loader import DocumentChunk
from src.reasoning.ollama_client import OllamaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, rag_config: dict[str, Any], ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        self._client = chromadb.PersistentClient(path=rag_config["vector_store_dir"])
        self._collection_name = rag_config["collection_name"]
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Embed and upsert a list of chunks. Idempotent by chunk id."""
        if not chunks:
            logger.warning(
                "No chunks to index. Check that data/regulations/ contains "
                "real corpus files before relying on RAG output."
            )
            return

        ids = [self._chunk_id(chunk) for chunk in chunks]
        embeddings = [self.ollama_client.embed(c.text) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._metadata(chunk) for chunk in chunks]

        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        logger.info("Indexed %d chunks into the vector store.", len(chunks))

    @staticmethod
    def _chunk_id(chunk: DocumentChunk) -> str:
        return (
            f"{chunk.document_id}::{chunk.locator_type}::{chunk.locator}::"
            f"{chunk.chunk_index}"
        )

    @staticmethod
    def _metadata(chunk: DocumentChunk) -> dict[str, str | int]:
        return {
            "document_id": chunk.document_id,
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "locator_type": chunk.locator_type,
            "locator": chunk.locator,
        }

    def rebuild(
        self,
        chunks: list[DocumentChunk],
        *,
        fingerprint: str,
        schema_version: str,
        embedding_model: str,
    ) -> None:
        """Replace the derived collection after every candidate embeds successfully."""
        if not chunks:
            raise ValueError("Cannot rebuild a vector store from an empty corpus")

        embeddings = [self.ollama_client.embed(chunk.text) for chunk in chunks]
        ids = [self._chunk_id(chunk) for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._metadata(chunk) for chunk in chunks]
        metadata = {
            "corpus_fingerprint": fingerprint,
            "schema_version": schema_version,
            "embedding_model": embedding_model,
        }

        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata=metadata
        )
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Rebuilt vector store with %d chunks.", len(chunks))

    def query(self, query_text: str, top_k: int) -> list[DocumentChunk]:
        """Return the top_k most relevant chunks with source metadata.

        Returns an empty list if the collection is empty rather than
        raising, so callers can handle "no regulation found" gracefully.
        """
        if self._collection.count() == 0:
            logger.warning(
                "Vector store is empty. Run the indexing script before "
                "querying. See scripts/build_vector_index.py."
            )
            return []

        query_embedding = self.ollama_client.embed(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )
        if not results["documents"]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        return [
            DocumentChunk(
                text=text,
                source_file=metadata.get("source_file", ""),
                chunk_index=metadata.get("chunk_index", 0),
                document_id=metadata.get("document_id", ""),
                locator_type=metadata.get("locator_type", ""),
                locator=metadata.get("locator", ""),
            )
            for text, metadata in zip(documents, metadatas, strict=True)
        ]
