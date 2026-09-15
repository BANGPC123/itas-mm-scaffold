from src.context.zone_classifier import ZoneClassifier, _haversine_km


def _sample_config():
    return {
        "default_zone": "Suburban",
        "zones": [
            {"label": "Urban", "center_lat": 10.0, "center_lon": 106.0, "radius_km": 5.0},
            {"label": "Highway", "center_lat": 20.0, "center_lon": 110.0, "radius_km": 3.0},
        ],
    }


def test_point_inside_urban_zone_returns_urban():
    classifier = ZoneClassifier(_sample_config())
    result = classifier.classify(latitude=10.01, longitude=106.01)
    assert result.zone_label == "Urban"


def test_point_outside_all_zones_returns_default():
    classifier = ZoneClassifier(_sample_config())
    result = classifier.classify(latitude=0.0, longitude=0.0)
    assert result.zone_label == "Suburban"


def test_point_on_zone_boundary_edge_case():
    # A point exactly at a zone's center must always match that zone.
    classifier = ZoneClassifier(_sample_config())
    result = classifier.classify(latitude=20.0, longitude=110.0)
    assert result.zone_label == "Highway"


def test_haversine_distance_same_point_is_zero():
    assert _haversine_km(10.0, 106.0, 10.0, 106.0) == 0.0


def test_haversine_distance_is_symmetric():
    d1 = _haversine_km(10.0, 106.0, 11.0, 107.0)
    d2 = _haversine_km(11.0, 107.0, 10.0, 106.0)
    assert abs(d1 - d2) < 1e-9


def test_invalid_missing_zones_key_raises_key_error():
    # Missing configuration should fail loudly, not silently default.
    import pytest

    with pytest.raises(KeyError):
        ZoneClassifier({"default_zone": "Suburban"})
