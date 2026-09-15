from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import ZoneRequest, ZoneResponse
from src.context.zone_classifier import ZoneClassifier
from src.utils.config import load_config

router = APIRouter(prefix="/context", tags=["context"])

_classifier = ZoneClassifier(load_config("context"))


@router.post("/zone", response_model=ZoneResponse)
def classify_zone(request: ZoneRequest) -> ZoneResponse:
    result = _classifier.classify(request.latitude, request.longitude)
    return ZoneResponse(
        zone_label=result.zone_label,
        latitude=result.latitude,
        longitude=result.longitude,
    )
