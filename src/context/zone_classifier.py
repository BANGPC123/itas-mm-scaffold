"""Road-zone classification: maps simulated GPS coordinates to a road-zone
label (Urban / Suburban / Highway) using a simple radius-based rule lookup
defined in configs/context.yaml.

This is deliberately a simple, explainable rule-based module rather than a
learned model — the project spec does not require ML here, and a
transparent rule set is easier to validate and debug than a black-box
classifier for a safety-relevant context signal.
"""
from __future__ import annotations

import math
from typing import Any

from src.utils.types import ZoneResult

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


class ZoneClassifier:
    def __init__(self, context_config: dict[str, Any]):
        self.default_zone = context_config["default_zone"]
        self.zones = context_config["zones"]

    def classify(self, latitude: float, longitude: float) -> ZoneResult:
        """Return the first matching zone, or the configured default."""
        for zone in self.zones:
            distance = _haversine_km(
                latitude, longitude, zone["center_lat"], zone["center_lon"]
            )
            if distance <= zone["radius_km"]:
                return ZoneResult(
                    zone_label=zone["label"], latitude=latitude, longitude=longitude
                )

        return ZoneResult(
            zone_label=self.default_zone, latitude=latitude, longitude=longitude
        )
