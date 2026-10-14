"""Pydantic schemas for the FastAPI backend.

Kept separate from routers so the API contract is easy to read/review on
its own, and so schemas can be reused by future clients (e.g. tests) that
don't need the router code.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ZoneRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class ZoneResponse(BaseModel):
    zone_label: str
    latitude: float
    longitude: float


class SignDetectionResponse(BaseModel):
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    is_compound: bool
    panel_labels: list[str]
    recognition_confidence: float | None = None
    panel_role: str = 'unknown'
    panel_role_source: str = 'unresolved'
    placement: str = 'unknown'
    lane_scope: str = 'unknown'
    context_verified: bool = False
    label_verified: bool = False
    time_windows: list[tuple[str,str]] = Field(default_factory=list)
    time_conditions: list[list[tuple[str,str]]] = Field(default_factory=list)
    allowed_vehicles: list[str] = Field(default_factory=list)
    overhead: bool | None = None
    applicability: str = 'uncertain'
    applicability_reason: str = ''
    track_id: int | None = None
    recognition_source: str = 'single_frame'
    crop_age_ms: float | None = None
    confirmation_delay_ms: float | None = None
    name_vi: str | None = None
    car_context: dict = Field(default_factory=dict)


class LaneResponse(BaseModel):
    lane_lines: list[tuple[int,int,int,int]]
    lane_center_offset_px: float | None
    polylines: list[list[tuple[int,int]]]
    display_polylines: list[list[tuple[int,int]]] | None = None
    ego_boundaries: list[list[tuple[int,int]]]
    backend: str
    rejection_reason: str | None = None
    camera_view: str = 'unknown'


class PerceptionResponse(BaseModel):
    signs: list[SignDetectionResponse]
    lane: LaneResponse
    supplementary_candidates: list[dict] = Field(default_factory=list)
    lane_line_count: int
    lane_center_offset_px: float | None


class ReasoningResponse(BaseModel):
    guidance_text: str
    retrieved_chunks: list[str]
    retrieved_chunk_count: int
    zone_label: str
    references: list[dict] = Field(default_factory=list)
    speech_text: str = ''
    observation_text: str = ''
    observation_speech_text: str = ''
    car_summary: str = ''


class PipelineRunResponse(BaseModel):
    perception: PerceptionResponse
    context: ZoneResponse
    reasoning: ReasoningResponse
    audio_url: str | None
    warnings: list[str]
    timings_ms: dict[str,float]
    annotated_image: str | None
    frame_shape: tuple[int,int,int]
    verified_zone_status: str = 'unknown'
