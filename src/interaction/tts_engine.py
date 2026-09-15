"""Text-to-speech: offline synthesis via pyttsx3.

pyttsx3 was chosen because it works fully offline with no API cost, which
matches the "hands-free, low-distraction" requirement without adding a
network dependency for the Interaction stage.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TtsEngine:
    def __init__(self, interaction_config: dict[str, Any]):
        cfg = interaction_config["tts"]
        self.rate = cfg["rate"]
        self.volume = cfg["volume"]
        self.voice_id = cfg.get("voice_id")
        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str) -> str:
        """Synthesize `text` to a .wav file and return its path.

        Raises RuntimeError with a clear message if pyttsx3 fails to
        initialize (commonly a missing system TTS backend on Linux).
        """
        try:
            import pyttsx3
        except ImportError as exc:
            raise ImportError(
                "pyttsx3 is required for TTS. Install it with "
                "`pip install pyttsx3` (Linux also needs the `espeak` or "
                "`espeak-ng` system package)."
            ) from exc

        try:
            engine = pyttsx3.init()
        except Exception as exc:  # pyttsx3 raises generic RuntimeError variants
            raise RuntimeError(
                "Failed to initialize a TTS backend. On Linux, install "
                "`espeak-ng` (e.g. `apt install espeak-ng`) and retry."
            ) from exc

        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        if self.voice_id:
            engine.setProperty("voice", self.voice_id)

        output_path = self.output_dir / f"guidance_{uuid.uuid4().hex[:8]}.wav"
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()

        logger.info("Synthesized guidance audio to %s", output_path)
        return str(output_path)
