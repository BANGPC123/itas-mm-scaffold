"""Thin HTTP client for a local Ollama server.

Kept separate from rag_chain.py so the RAG logic can be unit-tested with a
mocked client instead of requiring a running Ollama instance.
"""
from __future__ import annotations

from typing import Any

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaConnectionError(Exception):
    """Raised when the Ollama server cannot be reached."""


class OllamaClient:
    def __init__(self, ollama_config: dict[str, Any]):
        self.base_url = ollama_config["base_url"].rstrip("/")
        self.generation_model = ollama_config["generation_model"]
        self.embedding_model = ollama_config["embedding_model"]
        self.timeout_s = ollama_config["request_timeout_s"]

    def embed(self, text: str) -> list[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Failed to reach Ollama at {self.base_url} for embeddings. "
                f"Is `ollama serve` running? Original error: {exc}"
            ) from exc

        return response.json()["embedding"]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts in fixed-size Ollama batches."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            try:
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.embedding_model, "input": batch},
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                raise OllamaConnectionError(
                    f"Failed to reach Ollama at {self.base_url} for embeddings. "
                    f"Is `ollama serve` running? Original error: {exc}"
                ) from exc

            try:
                response_body = response.json()
            except ValueError as exc:
                raise ValueError(
                    "Ollama returned malformed embeddings for a batch"
                ) from exc
            batch_embeddings = (
                response_body.get("embeddings")
                if isinstance(response_body, dict)
                else None
            )
            if (
                not isinstance(batch_embeddings, list)
                or len(batch_embeddings) != len(batch)
                or any(
                    not isinstance(embedding, list)
                    or any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        for value in embedding
                    )
                    for embedding in batch_embeddings
                )
            ):
                raise ValueError("Ollama returned malformed embeddings for a batch")

            embeddings.extend(batch_embeddings)

        return embeddings

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.generation_model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                },
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Failed to reach Ollama at {self.base_url} for generation. "
                f"Is `ollama serve` running and is model "
                f"{self.generation_model!r} pulled? Original error: {exc}"
            ) from exc

        return response.json()["response"]
