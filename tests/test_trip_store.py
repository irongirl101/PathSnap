"""Tests for the supplied per-trip file layout and label isolation boundary."""

from src.api.trips import TripStore


def test_prediction_input_parses_indexes_and_never_requires_route(tmp_path):
    trip_dir = tmp_path / "000001"
    trip_dir.mkdir()
    (trip_dir / "000001.track").write_text("54.963519 56.251559 1.0\n54.963143 56.251523 2.0\n")
    (trip_dir / "000001.nodes").write_text("54.9 56.2\n55.0 56.3\n")
    (trip_dir / "000001.arcs").write_text("0 1\n1 0\n")

    trip = TripStore(tmp_path).load_prediction_input("000001")

    assert trip.points[0] == {"lat": 56.251559, "lon": 54.963519, "timestamp_s": 1.0}
    assert trip.arcs == [
        {"arc_id": 0, "from_node_id": 0, "to_node_id": 1},
        {"arc_id": 1, "from_node_id": 1, "to_node_id": 0},
    ]
    assert "route" not in trip.as_dict()


def test_reference_route_is_loaded_separately(tmp_path):
    trip_dir = tmp_path / "trip_a"
    trip_dir.mkdir()
    (trip_dir / "trip_a.route").write_text("5\n7\n5\n")

    assert TripStore(tmp_path).read_reference_route("trip_a") == [5, 7, 5]


def test_trip_ids_cannot_escape_dataset_root(tmp_path):
    store = TripStore(tmp_path)

    try:
        store.load_prediction_input("../outside")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal trip IDs must be rejected")
