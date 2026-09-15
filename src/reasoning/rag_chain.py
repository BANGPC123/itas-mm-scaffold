"""RAG chain: retrieves regulation chunks relevant to the current
perception + zone context, then asks the LLM to ground its answer in
exactly those chunks (per the project's "no open-domain generation"
requirement).
"""
from __future__ import annotations

from typing import Any

from src.reasoning.ollama_client import OllamaClient
from src.reasoning.vector_store import VectorStore
from src.utils.logger import get_logger
from src.utils.types import ReasoningResult

logger = get_logger(__name__)


class RagChain:
    def __init__(
        self,
        reasoning_config: dict[str, Any],
        vector_store: VectorStore,
        ollama_client: OllamaClient,
    ):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.top_k = reasoning_config["rag"]["top_k"]
        self.system_prompt = reasoning_config["prompt"]["system_prompt"]

    def answer(self, query_text: str, zone_label: str) -> ReasoningResult:
        """Retrieve relevant regulation chunks and generate a grounded
        answer. If no chunks are found, returns a clear "not found" message
        instead of letting the LLM improvise — per project rule "không
        được tự bịa ... business requirement".
        """
        retrieved_chunks = self.vector_store.query(query_text, self.top_k)

        if not retrieved_chunks:
            return ReasoningResult(
                guidance_text=(
                    "Không tìm thấy quy định phù hợp trong cơ sở dữ liệu hiện có."
                ),
                retrieved_chunks=[],
                zone_label=zone_label,
            )

        context_block = "\n\n".join(retrieved_chunks)
        user_prompt = (
            f"Khu vực hiện tại: {zone_label}\n\n"
            f"Sự kiện được phát hiện: {query_text}\n\n"
            f"Văn bản quy định liên quan:\n{context_block}\n\n"
            "Hãy đưa ra hướng dẫn ngắn gọn cho tài xế dựa trên thông tin trên."
        )

        guidance_text = self.ollama_client.generate(self.system_prompt, user_prompt)

        return ReasoningResult(
            guidance_text=guidance_text,
            retrieved_chunks=retrieved_chunks,
            zone_label=zone_label,
        )
