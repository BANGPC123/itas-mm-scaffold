"""Pipeline orchestrator: wires Perception -> Context -> Reasoning ->
Interaction into a single callable, per the architecture diagram.

Each stage is constructed independently and can be swapped or mocked in
tests without touching the others — this is the main payoff of keeping
the stages decoupled (see README "Architecture").
"""
from __future__ import annotations

import numpy as np

from src.context.zone_classifier import ZoneClassifier
from src.interaction.tts_engine import TtsEngine
from src.perception.lane_detector import LaneDetector
from src.perception.sign_detector import SignDetector
from src.reasoning.ollama_client import OllamaClient
from src.reasoning.rag_chain import RagChain
from src.reasoning.vector_store import VectorStore
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.types import PerceptionResult, PipelineResult

logger = get_logger(__name__)


class Pipeline:
    def __init__(self):
        perception_cfg = load_config("perception")
        context_cfg = load_config("context")
        reasoning_cfg = load_config("reasoning")
        interaction_cfg = load_config("interaction")

        self.sign_detector = SignDetector(perception_cfg)
        self.lane_detector = LaneDetector(perception_cfg)
        self.zone_classifier = ZoneClassifier(context_cfg)

        ollama_client = OllamaClient(reasoning_cfg["ollama"])
        vector_store = VectorStore(reasoning_cfg["rag"], ollama_client)
        self.rag_chain = RagChain(reasoning_cfg, vector_store, ollama_client)

        self.tts_engine = TtsEngine(interaction_cfg)

    def run(
        self,
        image: np.ndarray,
        latitude: float,
        longitude: float,
        synthesize_audio: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline on a single frame + GPS reading.

        Args:
            image: BGR image (as read by OpenCV) from the forward-facing camera.
            latitude, longitude: simulated GPS reading for this frame.
            synthesize_audio: if False, skips TTS (useful for tests/benchmarks).
        """
        signs = self.sign_detector.detect(image)
        lane = self.lane_detector.detect(image)
        perception_result = PerceptionResult(signs=signs, lane=lane)
        logger.debug("Perception: %d sign(s) detected", len(signs))

        zone_result = self.zone_classifier.classify(latitude, longitude)
        logger.debug("Context: zone=%s", zone_result.zone_label)

        query_text = self._build_query_text(perception_result)
        reasoning_result = self.rag_chain.answer(query_text, zone_result.zone_label)
        logger.debug("Reasoning: guidance=%r", reasoning_result.guidance_text[:80])

        audio_path = None
        if synthesize_audio and reasoning_result.guidance_text:
            audio_path = self.tts_engine.synthesize(reasoning_result.guidance_text)

        return PipelineResult(
            perception=perception_result,
            context=zone_result,
            reasoning=reasoning_result,
            audio_path=audio_path,
        )

    @staticmethod
    def _build_query_text(perception_result: PerceptionResult) -> str:
        """Turn detected signs/lane info into a short natural-language
        query for the RAG stage. Kept simple and explicit rather than
        clever — easy to unit test and to extend later.
        """
        if not perception_result.signs:
            return "Không phát hiện biển báo nào trong khung hình hiện tại."

        sign_descriptions = []
        for sign in perception_result.signs:
            if sign.is_compound:
                sign_descriptions.append(
                    f"biển báo hợp thành gồm: {', '.join(sign.panel_labels)}"
                )
            else:
                sign_descriptions.append(sign.label)

        return "Phát hiện: " + "; ".join(sign_descriptions)
