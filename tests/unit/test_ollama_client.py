from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import requests

from src.reasoning.ollama_client import OllamaClient, OllamaConnectionError


def _client() -> OllamaClient:
    return OllamaClient(
        {
            "base_url": "http://localhost:11434/",
            "generation_model": "llama3.1",
            "embedding_model": "nomic-embed-text",
            "request_timeout_s": 30,
        }
    )


def test_embed_many_batches_17_inputs_at_16_and_preserves_order():
    texts = [f"chunk {index}" for index in range(17)]
    first_response = MagicMock()
    first_response.json.return_value = {"embeddings": [[index] for index in range(16)]}
    second_response = MagicMock()
    second_response.json.return_value = {"embeddings": [[16]]}

    with patch(
        "src.reasoning.ollama_client.requests.post",
        side_effect=[first_response, second_response],
    ) as post:
        embeddings = _client().embed_many(texts)

    assert embeddings == [[index] for index in range(17)]
    assert post.call_args_list == [
        call(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": texts[:16]},
            timeout=30,
        ),
        call(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text", "input": texts[16:]},
            timeout=30,
        ),
    ]


def test_embed_many_empty_input_does_not_make_http_request():
    with patch("src.reasoning.ollama_client.requests.post") as post:
        assert _client().embed_many([]) == []

    post.assert_not_called()


@pytest.mark.parametrize(
    "response_body",
    [
        {"embeddings": [[0.1]]},
        {"embeddings": [[0.1], "malformed"]},
        {"embeddings": "malformed"},
        [],
    ],
)
def test_embed_many_rejects_incomplete_or_malformed_batch_responses(response_body):
    response = MagicMock()
    response.json.return_value = response_body

    with patch("src.reasoning.ollama_client.requests.post", return_value=response):
        with pytest.raises(ValueError, match="embedding"):
            _client().embed_many(["first", "second"])


def test_embed_many_wraps_request_failures():
    with patch(
        "src.reasoning.ollama_client.requests.post",
        side_effect=requests.RequestException("unavailable"),
    ):
        with pytest.raises(OllamaConnectionError, match="Failed to reach Ollama"):
            _client().embed_many(["chunk"])
