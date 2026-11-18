import pandas as pd

from src.features.features import point_features
from src.features.loader import load_trip, to_local_xy


def test_loader_preserves_raw_longitude_latitude_order(tmp_path):
    trip_id = "00000001"
    trip_dir = tmp_path / trip_id
    trip_dir.mkdir()
    (trip_dir / f"{trip_id}.nodes").write_text(
        "44.0 40.0\n44.001 40.0\n", encoding="utf-8"
    )
    (trip_dir / f"{trip_id}.arcs").write_text("0 1\n", encoding="utf-8")
    (trip_dir / f"{trip_id}.track").write_text(
        "44.0005 40.0 1\n44.0006 40.0 2\n", encoding="utf-8"
    )

    _, nodes, _, track, _ = load_trip(trip_dir)
    nodes_xy, track_xy = to_local_xy(nodes, track)

    assert nodes.loc[0, "lon"] == 44.0
    assert nodes.loc[0, "lat"] == 40.0
    assert track.loc[0, "lon"] == 44.0005
    assert track.loc[0, "lat"] == 40.0
    assert nodes_xy.loc[1, "x"] > nodes_xy.loc[0, "x"]
    assert abs(nodes_xy.loc[1, "y"] - nodes_xy.loc[0, "y"]) < 1e-8
    assert track_xy.loc[0, "x"] > nodes_xy.loc[0, "x"]


def test_missing_time_column_stays_supported():
    nodes = pd.DataFrame({"node_id": [0], "lon": [44.0], "lat": [40.0]})
    track = pd.DataFrame({"lon": [44.0], "lat": [40.0]})

    nodes_xy, track_xy = to_local_xy(nodes, track)

    assert nodes_xy.loc[0, "x"] == 0
    assert nodes_xy.loc[0, "y"] == 0
    assert track_xy.loc[0, "x"] == 0
    assert track_xy.loc[0, "y"] == 0


def test_large_time_gaps_are_not_used_as_spatial_neighbors():
    features = point_features(pd.DataFrame({
        "x": [0.0, 10.0, 1000.0],
        "y": [0.0, 0.0, 0.0],
        "t": [0.0, 1.0, 100.0],
    }))

    assert pd.isna(features.loc[1, "step_next"])
    assert pd.isna(features.loc[2, "step_prev"])
    assert features.loc[1, "speed"] == 10.0
    assert pd.isna(features.loc[2, "speed"])
    assert features.loc[2, "dt_prev"] == 99.0
