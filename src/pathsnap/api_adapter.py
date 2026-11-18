"""Adapt the PathSnap candidate generator and HMM to the HTTP API.

The matcher consumes observation and graph data only. It never opens `.route`;
the API passes reference labels separately to `evaluate_trip` after prediction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.candidates import generate_candidates as generate_feature_candidates
from src.features.features import FEATURE_COLS, Network, add_features
from src.features.loader import to_local_xy
from src.pathsnap.candidates import generate_candidates as generate_geometric_candidates
from src.pathsnap.config import MatchConfig, load_config
from src.pathsnap.evaluate import (
    _reference_polyline,
    _snap_points_to_reference,
    edge_f1_from_sets,
    rmf_from_sets,
)
from src.pathsnap.graph import Graph
from src.pathsnap.hmm import Candidate, MatchResult, match_trip
from src.pathsnap.io import Trip


def _data_root() -> Path:
    configured = os.getenv("PATHSNAP_DATA_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "data"


def _find_trip_dir(trip_id: str, data_root: Path) -> Path | None:
    """Find a trip folder without opening its reference route."""
    if not trip_id or Path(trip_id).name != trip_id or trip_id in {".", ".."}:
        raise ValueError("invalid trip_id")
    for split in ("train", "student_test", "evaluator_test"):
        candidate = data_root / split / trip_id
        if candidate.is_dir():
            return candidate
    return None


def _load_network(request: dict[str, Any], trip_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Use graph arrays supplied by TripStore, or read only .nodes/.arcs."""
    if request.get("nodes") is not None and request.get("arcs") is not None:
        nodes = request["nodes"]
        arcs = request["arcs"]
        node_array = np.asarray(
            [[float(node["lon"]), float(node["lat"])] for node in nodes], dtype=np.float64
        )
        arc_array = np.asarray(
            [[int(arc["from_node_id"]), int(arc["to_node_id"])] for arc in arcs],
            dtype=np.int64,
        )
    else:
        trip_dir = _find_trip_dir(trip_id, _data_root())
        if trip_dir is None:
            raise FileNotFoundError(f"graph files not found for trip {trip_id}")
        node_array = np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=np.float64, ndmin=2)
        arc_array = np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    if node_array.ndim != 2 or node_array.shape[1] != 2:
        raise ValueError(f"trip {trip_id} has invalid node coordinates")
    if arc_array.ndim != 2 or arc_array.shape[1] != 2:
        raise ValueError(f"trip {trip_id} has invalid arc endpoints")
    return node_array, arc_array


def _request_points(request: dict[str, Any]) -> list[tuple[float, float, int]]:
    """Convert API lat/lon points to HMM lon/lat/index tuples.

    For trip files, `timestamp_s` preserves sampling gaps in the HMM's point
    index field. Manual traces without timestamps use their row positions.
    """
    points = []
    for row_index, point in enumerate(request.get("points", []) or []):
        lat = float(point["lat"])
        lon = float(point["lon"])
        point_index = int(round(float(point.get("timestamp_s", row_index))))
        points.append((lon, lat, point_index))
    return points


def _load_config_from_env() -> MatchConfig:
    config_path = os.getenv("PATHSNAP_CONFIG")
    if config_path and Path(config_path).is_file():
        return load_config(config_path)
    return MatchConfig()


def _result_to_response(
    trip_id: str,
    result: MatchResult,
    graph: Graph,
    candidate_source: str,
) -> dict[str, Any]:
    """Serialize per-observation selections and the stitched route separately."""
    point_rows = []
    for index, (matched, (lon, lat, _point_idx)) in enumerate(
        zip(result.matched, result.points)
    ):
        point_rows.append(
            {
                "input_index": index,
                "lat": float(lat),
                "lon": float(lon),
                "node_id": None,
                "arc_id": int(matched.edge_id) if matched is not None else None,
                "confidence": float(matched.emission) if matched is not None else None,
            }
        )

    route_arc_ids = result.stitched_edge_ids
    if not route_arc_ids:
        route_arc_ids = result.decoded_edge_ids
    distance_m = sum(float(graph._edge_length[edge_id]) for edge_id in route_arc_ids)

    return {
        "trip_id": trip_id,
        "route_arc_ids": [int(edge_id) for edge_id in route_arc_ids],
        "points": point_rows,
        "distance_m": distance_m,
        "metadata": {
            "num_points": result.num_points,
            "matched_count": result.matched_count,
            "unmatched_point_indices": result.unmatched_point_indices,
            "gaps": [list(gap) for gap in result.gaps],
            "breaks": list(result.breaks),
            "num_segments": len(result.segments),
            "num_stitched_edges": len(result.stitched_route),
            "total_score": float(result.total_score),
            "representation": "stitched" if result.stitched_route else "decoded",
            "candidate_source": candidate_source,
        },
    }


def _classifier_candidates(
    trip_id: str,
    points: list[tuple[float, float, int]],
    request: dict[str, Any],
    graph: Graph,
    classifier: Any,
) -> list[list[Candidate] | None]:
    """Build A's feature rows and score each trip-scoped arc with B's model."""
    nodes_df = pd.DataFrame(
        {
            "node_id": [int(node["node_id"]) for node in request["nodes"]],
            "lat": [float(node["lat"]) for node in request["nodes"]],
            "lon": [float(node["lon"]) for node in request["nodes"]],
        }
    )
    arcs_df = pd.DataFrame(
        {
            "arc_id": [int(arc["arc_id"]) for arc in request["arcs"]],
            "u": [int(arc["from_node_id"]) for arc in request["arcs"]],
            "v": [int(arc["to_node_id"]) for arc in request["arcs"]],
        }
    )
    track_df = pd.DataFrame(
        {
            "lat": [float(point["lat"]) for point in request["points"]],
            "lon": [float(point["lon"]) for point in request["points"]],
            "t": [float(point.get("timestamp_s", i)) for i, point in enumerate(request["points"])],
        }
    )
    nodes_xy, track_xy = to_local_xy(nodes_df, track_df)
    network = Network(nodes_xy, arcs_df)
    candidates = generate_feature_candidates(network, track_xy)
    result: list[list[Candidate] | None] = [None] * len(points)
    if candidates.empty:
        return result

    candidates = add_features(network, track_xy, candidates)
    feature_rows = candidates[FEATURE_COLS]
    if callable(getattr(classifier, "predict_proba", None)):
        class_ids = list(getattr(classifier, "classes_", [0, 1]))
        if 1 not in class_ids:
            raise ValueError("Loaded final model has no positive candidate class (label 1)")
        positive_col = class_ids.index(1)
        probabilities = classifier.predict_proba(feature_rows)[:, positive_col]
    elif callable(getattr(classifier, "predict", None)):
        scores = pd.Series(classifier.predict(feature_rows), index=candidates.index)
        shifted = scores - scores.groupby(candidates["pos"]).transform("max")
        exp_scores = np.exp(shifted)
        probabilities = (
            exp_scores / exp_scores.groupby(candidates["pos"]).transform("sum")
        ).to_numpy()
    else:
        raise TypeError("Loaded final model must provide predict_proba() or predict()")

    for row, probability in zip(candidates.to_dict("records"), probabilities):
        pos = int(row["pos"])
        edge_index = int(row["arc_idx"])
        edge_id = int(network.arc_ids[edge_index])
        u, v = int(network.u[edge_index]), int(network.v[edge_index])
        node_a, node_b = nodes_xy.iloc[u], nodes_xy.iloc[v]
        fraction = float(row["frac"])
        point_lon, point_lat, point_idx = points[pos]
        candidate = Candidate(
            trip_id=trip_id,
            point_idx=point_idx,
            lon=point_lon,
            lat=point_lat,
            cand_id=edge_id,
            edge_id=edge_id,
            proj_lon=float(node_a["lon"] + fraction * (node_b["lon"] - node_a["lon"])),
            proj_lat=float(node_a["lat"] + fraction * (node_b["lat"] - node_a["lat"])),
            offset_m=float(row["s"]),
            edge_len_m=float(graph._edge_length[edge_id]),
            emission=float(np.clip(probability, 1e-12, 1.0)),
            dist_m=float(row["dist"]),
        )
        if result[pos] is None:
            result[pos] = []
        result[pos].append(candidate)
    return result


def matcher(request_dict: dict[str, Any], runtime_dict: dict[str, Any]) -> dict[str, Any]:
    """Run candidate generation and HMM decoding for one trip.

    When B's final artifact is present, use A's feature generator and B's
    `predict_proba` scores as HMM emissions. Without that artifact, use C's
    documented geometric candidate baseline so the API can still be exercised.
    """
    trip_id = str(request_dict.get("trip_id") or "")
    if not trip_id:
        raise ValueError("trip_id is required")
    points = _request_points(request_dict)
    if len(points) < 2:
        raise ValueError("at least two GPS points are required")

    nodes, arcs = _load_network(request_dict, trip_id)
    feature_request = dict(request_dict)
    feature_request["nodes"] = [
        {"node_id": i, "lon": float(node[0]), "lat": float(node[1])}
        for i, node in enumerate(nodes)
    ]
    feature_request["arcs"] = [
        {"arc_id": i, "from_node_id": int(arc[0]), "to_node_id": int(arc[1])}
        for i, arc in enumerate(arcs)
    ]
    config = _load_config_from_env()
    graph = Graph(
        nodes=nodes,
        arcs=arcs,
        directed=config.routing.directed,
        max_route_distance_m=config.routing.max_route_distance_m,
    )
    classifier = runtime_dict.get("model")
    if classifier is None:
        candidates = generate_geometric_candidates(trip_id, points, graph)
        candidate_source = "geometric_standin"
    else:
        candidates = _classifier_candidates(trip_id, points, feature_request, graph, classifier)
        candidate_source = "trained_classifier"
    result = match_trip(trip_id, points, candidates, graph, config)
    return _result_to_response(trip_id, result, graph, candidate_source)


def evaluate_trip(
    *,
    trip_id: str,
    predicted_arc_ids: list[int | str],
    reference_arc_ids: list[int],
    trip_data: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Compute C's route metrics after the prediction has been produced."""
    nodes = np.asarray(
        [[float(node["lon"]), float(node["lat"])] for node in trip_data["nodes"]],
        dtype=np.float64,
    )
    arcs = np.asarray(
        [[int(arc["from_node_id"]), int(arc["to_node_id"])] for arc in trip_data["arcs"]],
        dtype=np.int64,
    )
    route = np.asarray([int(edge_id) for edge_id in reference_arc_ids], dtype=np.int64)
    track_points = trip_data["points"]
    track = np.asarray(
        [
            [float(point["lon"]), float(point["lat"]), float(point.get("timestamp_s", i))]
            for i, point in enumerate(track_points)
        ],
        dtype=np.float64,
    )
    trip = Trip(trip_id=trip_id, track=track, route=route, nodes=nodes, arcs=arcs)
    config = _load_config_from_env()
    graph = Graph(
        nodes=nodes,
        arcs=arcs,
        directed=config.routing.directed,
        max_route_distance_m=config.routing.max_route_distance_m,
    )

    predicted = [int(edge_id) for edge_id in predicted_arc_ids]
    edge_lengths = {int(edge_id): float(length) for edge_id, length in graph._edge_length.items()}
    rmf, ref_length, pred_length, omitted_length, extra_length, ref_count, pred_count = rmf_from_sets(
        predicted, route.tolist(), edge_lengths
    )
    edge_metrics = edge_f1_from_sets(
        predicted,
        route.tolist(),
        graph,
        directed=config.routing.directed,
    )

    points = [(float(row[0]), float(row[1]), i) for i, row in enumerate(track)]
    reference_segments, _ = _reference_polyline(trip)
    reference_by_point = _snap_points_to_reference(points, reference_segments)
    prediction_rows = prediction.get("points", [])
    predicted_by_point: dict[int, int] = {}
    for row in prediction_rows:
        if row.get("arc_id") is not None:
            predicted_by_point[int(row["input_index"])] = int(row["arc_id"])

    evaluable = correct = 0
    for index, reference_arc in enumerate(reference_by_point):
        if reference_arc is None:
            continue
        evaluable += 1
        if predicted_by_point.get(index) == int(reference_arc):
            correct += 1
    matched_count = len(predicted_by_point)
    point_accuracy = correct / evaluable if evaluable else 0.0
    coverage = matched_count / len(points) if points else 0.0

    return {
        "metrics": {
            "rmf": float(rmf),
            "edge_precision": float(edge_metrics["precision"]),
            "edge_recall": float(edge_metrics["recall"]),
            "edge_f1": float(edge_metrics["f1"]),
            "point_accuracy": float(point_accuracy),
            "coverage": float(coverage),
            "reference_length_m": float(ref_length),
            "predicted_length_m": float(pred_length),
            "omitted_length_m": float(omitted_length),
            "extra_length_m": float(extra_length),
            "reference_edge_count": int(ref_count),
            "predicted_edge_count": int(pred_count),
            "point_evaluable_count": int(evaluable),
        },
        "metadata": {"representation": "stitched"},
    }


__all__ = ["matcher", "evaluate_trip", "_data_root", "_find_trip_dir", "_request_points"]
