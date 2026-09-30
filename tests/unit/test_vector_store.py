from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.reasoning.document_loader import DocumentChunk
from src.reasoning.vector_store import VectorStore


def _store_with(collection: MagicMock, ollama_client: MagicMock) -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store._collection = collection
    store._client = MagicMock()
    store._collection_name = "traffic_regulations"
    store.ollama_client = ollama_client
    return store


def test_query_empty_collection_returns_empty_without_embedding():
    collection = MagicMock()
    collection.count.return_value = 0
    ollama_client = MagicMock()
    store = _store_with(collection, ollama_client)

    assert store.query("speed limit", top_k=3) == []
    ollama_client.embed.assert_not_called()


def test_query_round_trips_full_legal_metadata():
    collection = MagicMock()
    collection.count.return_value = 1
    collection.query.return_value = {
        "documents": [["Urban speed rule"]],
        "metadatas": [[{
            "document_id": "law-36-2024-qh15",
            "source_file": "law.json",
            "chunk_index": 4,
            "locator_type": "point",
            "locator": "Article 6 Clause 1 Point a",
        }]],
    }
    ollama_client = MagicMock()
    ollama_client.embed.return_value = [0.1, 0.2]
    store = _store_with(collection, ollama_client)

    result = store.query("speed limit", top_k=3)

    assert result == [
        DocumentChunk(
            text="Urban speed rule",
            source_file="law.json",
            chunk_index=4,
            document_id="law-36-2024-qh15",
            locator_type="point",
            locator="Article 6 Clause 1 Point a",
        )
    ]
    ollama_client.embed.assert_called_once_with("speed limit")
    ollama_client.embed_many.assert_not_called()


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


def test_index_chunks_rejects_incomplete_legal_metadata_before_embedding_or_upsert():
    collection = MagicMock()
    store = _store_with(collection, MagicMock())
    legacy_chunk = DocumentChunk("Legacy rule", "legacy.md", 0)

    with pytest.raises(ValueError, match="legal identity"):
        store.index_chunks([legacy_chunk])

    store.ollama_client.embed_many.assert_not_called()
    collection.upsert.assert_not_called()


def test_index_chunks_uses_batch_embeddings_and_preserves_chunk_order():
    collection = MagicMock()
    ollama_client = MagicMock()
    ollama_client.embed_many.return_value = [[0.1], [0.2]]
    store = _store_with(collection, ollama_client)
    chunks = [
        DocumentChunk(
            text="First rule",
            source_file="first.json",
            chunk_index=1,
            document_id="law-a",
            locator_type="article",
            locator="Article 1",
        ),
        DocumentChunk(
            text="Second rule",
            source_file="second.json",
            chunk_index=2,
            document_id="law-b",
            locator_type="clause",
            locator="Article 2 Clause 1",
        ),
    ]

    store.index_chunks(chunks)

    ollama_client.embed_many.assert_called_once_with(["First rule", "Second rule"])
    ollama_client.embed.assert_not_called()
    assert collection.upsert.call_args.kwargs == {
        "ids": ["law-a::article::Article 1::1", "law-b::clause::Article 2 Clause 1::2"],
        "embeddings": [[0.1], [0.2]],
        "documents": ["First rule", "Second rule"],
        "metadatas": [
            {
                "document_id": "law-a",
                "source_file": "first.json",
                "chunk_index": 1,
                "locator_type": "article",
                "locator": "Article 1",
            },
            {
                "document_id": "law-b",
                "source_file": "second.json",
                "chunk_index": 2,
                "locator_type": "clause",
                "locator": "Article 2 Clause 1",
            },
        ],
    }


def test_rebuild_uses_deterministic_chunk_ids():
    previous = MagicMock()
    candidate = MagicMock()
    ollama_client = MagicMock()
    ollama_client.embed_many.return_value = [[0.1, 0.2]]
    store = _store_with(previous, ollama_client)
    store._client.get_or_create_collection.return_value = candidate
    chunks = [
        DocumentChunk(
            text="Rule text",
            source_file="raw/law/source.json",
            chunk_index=2,
            document_id="law-36-2024-qh15",
            locator_type="point",
            locator="Article 6 Clause 1 Point a",
        )
    ]

    store.rebuild(
        chunks,
        fingerprint="fingerprint",
        schema_version="1",
        embedding_model="nomic-embed-text",
    )

    store._client.delete_collection.assert_called_once_with(name="traffic_regulations")
    store._client.get_or_create_collection.assert_called_once_with(
        name="traffic_regulations",
        metadata={
            "corpus_fingerprint": "fingerprint",
            "schema_version": "1",
            "embedding_model": "nomic-embed-text",
        },
    )
    candidate.upsert.assert_called_once_with(
        ids=["law-36-2024-qh15::point::Article 6 Clause 1 Point a::2"],
        embeddings=[[0.1, 0.2]],
        documents=["Rule text"],
        metadatas=[{
            "document_id": "law-36-2024-qh15",
            "source_file": "raw/law/source.json",
            "chunk_index": 2,
            "locator_type": "point",
            "locator": "Article 6 Clause 1 Point a",
        }],
    )
    ollama_client.embed_many.assert_called_once_with(["Rule text"])
    ollama_client.embed.assert_not_called()


def test_embed_failure_keeps_existing_collection():
    previous = MagicMock()
    store = _store_with(previous, MagicMock())
    store.ollama_client.embed_many.side_effect = RuntimeError("embedding unavailable")
    chunk = DocumentChunk(
        text="Rule text",
        source_file="raw/law/source.json",
        chunk_index=0,
        document_id="law-36-2024-qh15",
        locator_type="article",
        locator="Article 1",
    )

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        store.rebuild(
            [chunk],
            fingerprint="fingerprint",
            schema_version="1",
            embedding_model="nomic-embed-text",
        )

    store._client.delete_collection.assert_not_called()
    assert store._collection is previous
    store.ollama_client.embed.assert_not_called()


def test_successful_rebuild_removes_stale_chunks():
    previous = MagicMock()
    replacement = MagicMock()
    ollama_client = MagicMock()
    ollama_client.embed_many.return_value = [[0.1]]
    store = _store_with(previous, ollama_client)
    store._client.get_or_create_collection.return_value = replacement
    chunk = DocumentChunk(
        text="Current rule",
        source_file="raw/law/source.json",
        chunk_index=0,
        document_id="law-36-2024-qh15",
        locator_type="article",
        locator="Article 1",
    )

    store.rebuild(
        [chunk],
        fingerprint="fingerprint",
        schema_version="1",
        embedding_model="nomic-embed-text",
    )

    store._client.delete_collection.assert_called_once_with(name="traffic_regulations")
    assert store._collection is replacement
    assert replacement.upsert.call_args.kwargs["documents"] == ["Current rule"]
    ollama_client.embed_many.assert_called_once_with(["Current rule"])
    ollama_client.embed.assert_not_called()
