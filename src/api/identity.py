"""Namespaced identifiers for per-trip road graph records."""


def arc_lookup_key(trip_id: str, arc_id: str | int) -> tuple[str, str]:
    """Return a collision-safe key for an arc in a trip-local graph."""
    normalized_trip = str(trip_id).strip()
    normalized_arc = str(arc_id).strip()
    if not normalized_trip or not normalized_arc:
        raise ValueError("trip_id and arc_id must be non-empty")
    return normalized_trip, normalized_arc
