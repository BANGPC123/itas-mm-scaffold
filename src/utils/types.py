"""Shared data types passed between pipeline stages.

Using explicit dataclasses (instead of raw dicts) between stages makes the
contract between modules checkable by type checkers and IDEs, and prevents
silent key-typos when one stage's output feeds the next stage's input.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignDetection:
    """A single detected sign panel, or a compound sign group."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    is_compound: bool = False
    panel_labels: list[str] = field(default_factory=list)  # populated if is_compound


@dataclass
class LaneInfo:
    """Result of the lane-detection stage for the current frame."""

    lane_lines: list[tuple[int, int, int, int]]  # list of (x1, y1, x2, y2) segments
    lane_center_offset_px: float | None = None  # signed offset from frame center


@dataclass
class PerceptionResult:
    signs: list[SignDetection]
    lane: LaneInfo


@dataclass
class ZoneResult:
    zone_label: str  # "Urban" | "Suburban" | "Highway"
    latitude: float
    longitude: float


@dataclass
class ReasoningResult:
    guidance_text: str
    retrieved_chunks: list[str]
    zone_label: str


@dataclass
class PipelineResult:
    perception: PerceptionResult
    context: ZoneResult
    reasoning: ReasoningResult
    audio_path: str | None = None
