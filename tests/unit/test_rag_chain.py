from __future__ import annotations

from unittest.mock import MagicMock

from src.reasoning.document_loader import DocumentChunk
from src.reasoning.rag_chain import RagChain


def _chain_with(
    retrieved_chunks: list[DocumentChunk],
) -> tuple[RagChain, MagicMock, MagicMock]:
    vector_store = MagicMock()
    vector_store.query.return_value = retrieved_chunks
    ollama_client = MagicMock()
    ollama_client.generate.return_value = "Grounded guidance"
    config = {
        "rag": {"top_k": 3},
        "prompt": {"system_prompt": "Use only retrieved regulations."},
    }
    return RagChain(config, vector_store, ollama_client), vector_store, ollama_client


def test_answer_returns_fail_safe_without_generation_when_no_chunks_found():
    chain, vector_store, ollama_client = _chain_with([])

    result = chain.answer("No sign detected", "Urban")

    assert result.guidance_text == (
        "Không tìm thấy quy định phù hợp trong cơ sở dữ liệu hiện có."
    )
    assert result.retrieved_chunks == []
    assert result.zone_label == "Urban"
    vector_store.query.assert_called_once_with("No sign detected", 3)
    ollama_client.generate.assert_not_called()


def test_answer_retains_retrieved_document_chunks():
    chunks = [
        DocumentChunk("Rule A", "law-a.md", 2),
        DocumentChunk("Rule B", "law-b.md", 5),
    ]
    chain, _, ollama_client = _chain_with(chunks)

    result = chain.answer("Detected speed sign", "Urban")

    assert result.retrieved_chunks == chunks
    assert result.zone_label == "Urban"
    assert result.guidance_text == "Grounded guidance"
    ollama_client.generate.assert_called_once()


def test_answer_builds_prompt_from_chunk_text_in_retrieval_order():
    chunks = [
        DocumentChunk("First regulation", "first.md", 1),
        DocumentChunk("Second regulation", "second.md", 3),
    ]
    chain, _, ollama_client = _chain_with(chunks)

    chain.answer("Detected lane event", "Highway")

    _, user_prompt = ollama_client.generate.call_args.args
    assert user_prompt.index("First regulation") < user_prompt.index("Second regulation")
    assert "Highway" in user_prompt
    assert "Detected lane event" in user_prompt
