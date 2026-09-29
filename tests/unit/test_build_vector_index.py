from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scripts import build_vector_index


def test_invalid_corpus_does_not_call_rebuild(monkeypatch):
    vector_store = MagicMock()
    monkeypatch.setattr(
        build_vector_index,
        "load_config",
        lambda _: {"ollama": {}, "rag": {"normalized_dir": "normalized"}},
    )
    monkeypatch.setattr(
        build_vector_index,
        "load_canonical_documents",
        MagicMock(side_effect=ValueError("invalid canonical document")),
    )
    monkeypatch.setattr(build_vector_index, "VectorStore", vector_store)

    with pytest.raises(ValueError, match="invalid canonical document"):
        build_vector_index.build_index()

    vector_store.assert_not_called()
