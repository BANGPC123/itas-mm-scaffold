from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.reasoning.document_loader import DocumentChunk
from src.reasoning.rag_chain import RagChain


def _chunk(
    text: str, document_id: str, locator_type: str, locator: str
) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        source_file=f"{document_id}.json",
        chunk_index=0,
        document_id=document_id,
        locator_type=locator_type,
        locator=locator,
    )


def _chain_with(
    retrieved_chunks: list[DocumentChunk], guidance_text: str = "Grounded guidance"
) -> tuple[RagChain, MagicMock, MagicMock]:
    vector_store = MagicMock()
    vector_store.query.return_value = retrieved_chunks
    ollama_client = MagicMock()
    ollama_client.generate.return_value = guidance_text
    config = {
        "rag": {"top_k": 3},
        "prompt": {"system_prompt": "Use only retrieved regulations."},
    }
    return RagChain(config, vector_store, ollama_client), vector_store, ollama_client


def test_answer_returns_fail_safe_without_generation_when_no_chunks_found():
    chunks: list[DocumentChunk] = []
    chain, vector_store, ollama_client = _chain_with(chunks)

    result = chain.answer("No sign detected", "Urban")

    assert result.guidance_text == (
        "Không tìm thấy quy định phù hợp trong cơ sở dữ liệu hiện có."
    )
    assert result.retrieved_chunks is chunks
    assert result.zone_label == "Urban"
    vector_store.query.assert_called_once_with("No sign detected", 3)
    ollama_client.generate.assert_not_called()


def test_prompt_labels_each_chunk_with_deterministic_source_locator():
    chunks = [
        _chunk("First regulation", "decree-168-2024-nd-cp", "point", "Điều 6 Khoản 1 Điểm a"),
        _chunk("Second regulation", "law-36-2024-qh15", "article", "Điều 12"),
    ]
    chain, _, ollama_client = _chain_with(chunks)

    chain.answer("Detected lane event", "Highway")

    _, user_prompt = ollama_client.generate.call_args.args
    assert "[S1] decree-168-2024-nd-cp — Điều 6 Khoản 1 Điểm a\nFirst regulation" in user_prompt
    assert "[S2] law-36-2024-qh15 — Điều 12\nSecond regulation" in user_prompt
    assert "only the provided retrieved evidence" in user_prompt


def test_reasoning_retains_exact_retrieved_evidence():
    chunks = [
        _chunk("Rule A", "law-36-2024-qh15", "article", "Điều 2"),
        _chunk("Rule B", "decree-168-2024-nd-cp", "clause", "Điều 3 Khoản 1"),
    ]
    chain, _, _ = _chain_with(chunks)

    result = chain.answer("Detected speed sign", "Urban")

    assert result.retrieved_chunks is chunks
    assert result.retrieved_chunks[0] is chunks[0]
    assert result.retrieved_chunks[1] is chunks[1]


def test_unknown_generated_source_alias_is_rejected():
    chain, _, _ = _chain_with(
        [_chunk("Rule A", "law-36-2024-qh15", "article", "Điều 2")],
        "Use [S2].",
    )

    with pytest.raises(
        ValueError, match="^Generated guidance cited evidence outside retrieval set$"
    ):
        chain.answer("Detected speed sign", "Urban")


def test_known_generated_source_alias_is_allowed():
    chain, _, _ = _chain_with(
        [_chunk("Rule A", "law-36-2024-qh15", "article", "Điều 2")],
        "Use [S1] and [S1].",
    )

    result = chain.answer("Detected speed sign", "Urban")

    assert result.guidance_text == "Use [S1] and [S1]."
