from __future__ import annotations

from unittest.mock import MagicMock

from src.reasoning.document_loader import DocumentChunk
from src.reasoning.vector_store import VectorStore


def _store_with(collection: MagicMock, ollama_client: MagicMock) -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store._collection = collection
    store.ollama_client = ollama_client
    return store


def test_query_empty_collection_returns_empty_without_embedding():
    collection = MagicMock()
    collection.count.return_value = 0
    ollama_client = MagicMock()
    store = _store_with(collection, ollama_client)

    assert store.query("speed limit", top_k=3) == []
    ollama_client.embed.assert_not_called()


def test_query_returns_document_chunks_with_metadata():
    collection = MagicMock()
    collection.count.return_value = 1
    collection.query.return_value = {
        "documents": [["Urban speed rule"]],
        "metadatas": [[{"source_file": "law.md", "chunk_index": 4}]],
    }
    ollama_client = MagicMock()
    ollama_client.embed.return_value = [0.1, 0.2]
    store = _store_with(collection, ollama_client)

    result = store.query("speed limit", top_k=3)

    assert result == [
        DocumentChunk(text="Urban speed rule", source_file="law.md", chunk_index=4)
    ]


def test_query_preserves_result_order_and_metadata_pairing():
    collection = MagicMock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "documents": [["First rule", "Second rule"]],
        "metadatas": [[
            {"source_file": "first.md", "chunk_index": 1},
            {"source_file": "second.md", "chunk_index": 7},
        ]],
    }
    ollama_client = MagicMock()
    ollama_client.embed.return_value = [0.3, 0.4]
    store = _store_with(collection, ollama_client)

    result = store.query("lane rule", top_k=5)

    assert result == [
        DocumentChunk(text="First rule", source_file="first.md", chunk_index=1),
        DocumentChunk(text="Second rule", source_file="second.md", chunk_index=7),
    ]
