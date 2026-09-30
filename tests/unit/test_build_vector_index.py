from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts import build_vector_index


def _corpus() -> dict:
    return {
        "manifest": {"schema_version": "1"},
        "sources": [{"document_id": "law-36-2024-qh15", "raw_file": "raw/law.json"}],
        "chunks": [
            {
                "text": "A traffic rule.",
                "document_id": "law-36-2024-qh15",
                "ordinal": 2,
                "locator_type": "article",
                "locator": "Article 1",
            }
        ],
    }


def _configure_build(monkeypatch, corpus: dict) -> tuple[MagicMock, MagicMock]:
    ollama_client = MagicMock()
    vector_store = MagicMock()
    monkeypatch.setattr(
        build_vector_index,
        "load_config",
        lambda _: {
            "ollama": {"embedding_model": "nomic-embed-text"},
            "rag": {"corpus_file": "data/regulations/corpus.json"},
        },
    )
    monkeypatch.setattr(build_vector_index, "load_regulation_corpus", MagicMock(return_value=corpus))
    monkeypatch.setattr(build_vector_index, "compute_corpus_fingerprint", MagicMock(return_value="fingerprint"))
    monkeypatch.setattr(build_vector_index, "OllamaClient", MagicMock(return_value=ollama_client))
    monkeypatch.setattr(build_vector_index, "VectorStore", MagicMock(return_value=vector_store))
    return ollama_client, vector_store


def test_build_index_loads_configured_corpus_file(monkeypatch):
    _configure_build(monkeypatch, _corpus())

    build_vector_index.build_index()

    build_vector_index.load_regulation_corpus.assert_called_once_with(
        "data/regulations/corpus.json"
    )


def test_invalid_corpus_does_not_call_rebuild(monkeypatch):
    _, vector_store = _configure_build(monkeypatch, _corpus())
    monkeypatch.setattr(
        build_vector_index,
        "load_regulation_corpus",
        MagicMock(side_effect=ValueError("invalid corpus")),
    )

    with pytest.raises(ValueError, match="invalid corpus"):
        build_vector_index.build_index()

    vector_store.rebuild.assert_not_called()


def test_empty_corpus_does_not_call_rebuild(monkeypatch):
    corpus = _corpus()
    corpus["chunks"] = []
    _, vector_store = _configure_build(monkeypatch, corpus)

    with pytest.raises(ValueError, match="empty corpus"):
        build_vector_index.build_index()

    vector_store.rebuild.assert_not_called()


def test_build_index_passes_fingerprint_schema_and_embedding_model_to_rebuild(monkeypatch):
    _, vector_store = _configure_build(monkeypatch, _corpus())

    build_vector_index.build_index()

    vector_store.rebuild.assert_called_once_with(
        [
            build_vector_index.DocumentChunk(
                text="A traffic rule.",
                source_file="raw/law.json",
                chunk_index=2,
                document_id="law-36-2024-qh15",
                locator_type="article",
                locator="Article 1",
            )
        ],
        fingerprint="fingerprint",
        schema_version="1",
        embedding_model="nomic-embed-text",
    )
