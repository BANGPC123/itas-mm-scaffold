from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.schemas import (
    PerceptionResponse,
    PipelineRunResponse,
    ReasoningResponse,
    SignDetectionResponse,
    ZoneResponse,
)
from src.pipeline.orchestrator import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# Constructed once at import time — model/vector-store loading happens here,
# not per-request, so a request failure doesn't silently reload everything.
_pipeline = Pipeline()


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
) -> PipelineRunResponse:
    contents = await image.read()
    np_array = np.frombuffer(contents, dtype=np.uint8)
    decoded_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if decoded_image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    try:
        result = _pipeline.run(decoded_image, latitude, longitude)
    except Exception as exc:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PipelineRunResponse(
        perception=PerceptionResponse(
            signs=[
                SignDetectionResponse(
                    label=s.label,
                    confidence=s.confidence,
                    bbox_xyxy=s.bbox_xyxy,
                    is_compound=s.is_compound,
                    panel_labels=s.panel_labels,
                )
                for s in result.perception.signs
            ],
            lane_line_count=len(result.perception.lane.lane_lines),
            lane_center_offset_px=result.perception.lane.lane_center_offset_px,
        ),
        context=ZoneResponse(
            zone_label=result.context.zone_label,
            latitude=result.context.latitude,
            longitude=result.context.longitude,
        ),
        reasoning=ReasoningResponse(
            guidance_text=result.reasoning.guidance_text,
            retrieved_chunk_count=len(result.reasoning.retrieved_chunks),
            zone_label=result.reasoning.zone_label,
        ),
        audio_path=result.audio_path,
    )
