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
    recognition_confidence: float | None = None
    panel_role: str = "unknown"
    panel_role_source: str = 'unresolved'  # geometric_candidate | reviewed | unresolved
    allowed_vehicles: list[str] = field(default_factory=list)
    overhead: bool | None = None
    # Independently reviewed placement and scope; never inferred from YOLO boxes.
    placement: str = 'unknown'  # roadside | overhead | unknown
    lane_scope: str = 'unknown'  # all | ego | other | unknown
    context_verified: bool = False
    label_verified: bool = False
    applicability: str = "uncertain"
    applicability_reason: str = "Context not established"
    track_id: int | None = None
    recognition_source: str = 'single_frame'
    crop_age_ms: float | None = None
    confirmation_delay_ms: float | None = None
    name_vi: str | None = None
    time_windows: list[tuple[str,str]] = field(default_factory=list)
    time_conditions: list[list[tuple[str,str]]] = field(default_factory=list)
    car_context: dict = field(default_factory=dict)


@dataclass
class LaneInfo:
    """Result of the lane-detection stage for the current frame."""

    lane_lines: list[tuple[int, int, int, int]]  # list of (x1, y1, x2, y2) segments
    lane_center_offset_px: float | None = None  # signed offset from frame center
    polylines: list[list[tuple[int, int]]] = field(default_factory=list)
    ego_boundaries: list[list[tuple[int, int]]] = field(default_factory=list)
    backend: str = "unavailable"
    rejection_reason: str | None = None
    display_polylines: list[list[tuple[int, int]]] | None = None
    camera_view: str = 'unknown'


@dataclass
class PerceptionResult:
    signs: list[SignDetection]
    lane: LaneInfo
    supplementary_candidates: list[dict] = field(default_factory=list)


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
    references: list[dict] = field(default_factory=list)
    speech_text: str = ''
    observation_text: str = ''  # AI hypothesis for on-screen display, not a legal instruction
    observation_speech_text: str = ''  # explicitly tentative demo narration
    car_summary: str = ''


@dataclass
class PipelineResult:
    perception: PerceptionResult
    context: ZoneResult
    reasoning: ReasoningResult
    audio_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    annotated_image: str | None = None
    verified_zone_status: str = 'unknown'  # sign-derived only, never mocked GPS
