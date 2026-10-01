"""Integration test for the full pipeline.

Perception and Context run for real (no external service needed). The
Reasoning stage's Ollama calls and the Interaction stage's TTS call are
mocked, since they depend on services (a running Ollama server, a system
TTS backend) that are not guaranteed to be present in a CI environment.
This test verifies the stages are wired together correctly, not that the
external services themselves work — that is covered by manual smoke
testing per README.md "Running the backend".
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.orchestrator import Pipeline
from src.reasoning.document_loader import DocumentChunk
from src.utils.config import load_config as load_real_config


@pytest.fixture
def pipeline_with_mocked_externals(tmp_path):
    def isolated_load_config(name):
        config = load_real_config(name)
        if name != "reasoning":
            return config
        return {
            **config,
            "rag": {
                **config["rag"],
                "vector_store_dir": str(tmp_path / "chroma_db"),
                "collection_name": "pipeline_e2e",
            },
        }

    with patch(
        "src.pipeline.orchestrator.load_config", side_effect=isolated_load_config
    ), patch("src.reasoning.ollama_client.OllamaClient.embed") as mock_embed, patch(
        "src.reasoning.ollama_client.OllamaClient.embed_many"
    ) as mock_embed_many, patch(
        "src.reasoning.ollama_client.OllamaClient.generate"
    ) as mock_generate, patch(
        "src.interaction.tts_engine.TtsEngine.synthesize"
    ) as mock_tts:
        mock_embed.return_value = [0.0] * 8
        mock_embed_many.side_effect = lambda texts: [[0.0] * 8 for _ in texts]
        mock_generate.return_value = "Hãy giảm tốc độ do đang ở khu vực đô thị."
        mock_tts.return_value = "data/processed/tts_output/guidance_test.wav"

        pipeline = Pipeline()
        yield pipeline, mock_generate, mock_tts


def test_pipeline_runs_end_to_end_with_no_signs_detected(pipeline_with_mocked_externals):
    pipeline, mock_generate, mock_tts = pipeline_with_mocked_externals
    pipeline.rag_chain.vector_store.index_chunks(
        [
            DocumentChunk(
                text="Drive safely.",
                source_file="raw/test/source.json",
                chunk_index=0,
                document_id="test-regulation",
                locator_type="article",
                locator="Article 1",
            )
        ]
    )

    # Blank image: no weights configured, so no signs will be detected —
    # this exercises the "no detections" path through Reasoning.
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.run(blank_image, latitude=10.7769, longitude=106.7009)

    assert result.perception.signs == []
    assert result.context.zone_label in {"Urban", "Suburban", "Highway"}
    assert result.reasoning.guidance_text  # some text was produced
    assert result.audio_path is not None
    mock_tts.assert_called_once()


def test_pipeline_skips_tts_when_synthesize_audio_is_false(pipeline_with_mocked_externals):
    pipeline, mock_generate, mock_tts = pipeline_with_mocked_externals
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.run(
        blank_image, latitude=10.7769, longitude=106.7009, synthesize_audio=False
    )

    assert result.audio_path is None
    mock_tts.assert_not_called()


def test_pipeline_uses_urban_zone_for_urban_coordinates(pipeline_with_mocked_externals):
    pipeline, _, _ = pipeline_with_mocked_externals
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Coordinates fall inside the placeholder Urban zone from configs/context.yaml.
    result = pipeline.run(blank_image, latitude=10.7769, longitude=106.7009)

    assert result.context.zone_label == "Urban"


def test_pipeline_handles_empty_image_gracefully(pipeline_with_mocked_externals):
    pipeline, _, _ = pipeline_with_mocked_externals
    empty_image = np.zeros((0, 0, 3), dtype=np.uint8)

    # Should not raise — lane detector returns no lines for an empty frame.
    result = pipeline.run(empty_image, latitude=10.7769, longitude=106.7009)

    assert result.perception.lane.lane_lines == []
