"""Tests for the public API payload and trip-scoped graph contracts."""

import pytest

from src.api.identity import arc_lookup_key
from src.api.schemas import MapMatchRequest, MapMatchResponse


def test_map_match_accepts_ordered_gps_points():
    request = MapMatchRequest.model_validate(
        {
            "trip_id": "trip-1",
            "points": [
                {"lat": 12.97, "lon": 77.59},
                {"lat": 12.98, "lon": 77.60, "timestamp_s": 2.0},
            ],
        }
    )
    assert len(request.points) == 2
    assert request.trip_id == "trip-1"


def test_map_match_response_is_an_arc_sequence():
    response = MapMatchResponse.model_validate(
        {"trip_id": "trip-1", "route_arc_ids": [10, 11, 12]}
    )
    assert response.route_arc_ids == [10, 11, 12]


def test_arc_lookup_key_scopes_ids_to_trip():
    assert arc_lookup_key("trip-a", 10) != arc_lookup_key("trip-b", 10)
    assert arc_lookup_key("trip-a", 10) == ("trip-a", "10")


@pytest.mark.parametrize(("trip_id", "arc_id"), [("", 1), ("trip-a", " ")])
def test_arc_lookup_key_rejects_empty_parts(trip_id, arc_id):
    with pytest.raises(ValueError):
        arc_lookup_key(trip_id, arc_id)
