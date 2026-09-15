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


class PerceptionResponse(BaseModel):
    signs: list[SignDetectionResponse]
    lane_line_count: int
    lane_center_offset_px: float | None


class ReasoningResponse(BaseModel):
    guidance_text: str
    retrieved_chunk_count: int
    zone_label: str


class PipelineRunResponse(BaseModel):
    perception: PerceptionResponse
    context: ZoneResponse
    reasoning: ReasoningResponse
    audio_path: str | None
