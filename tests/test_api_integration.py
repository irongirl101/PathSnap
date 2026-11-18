"""Integration coverage for C's matcher and route scorer."""

from types import SimpleNamespace

import numpy as np

from src.api.main import evaluate
from src.api.schemas import EvaluationRequest
from src.models.wrapper import PathSnapModel
from src.pathsnap.api_adapter import _load_config_from_env, evaluate_trip, matcher
from src.pathsnap.config import load_config


def test_api_uses_repository_default_config_when_no_override(monkeypatch):
    monkeypatch.delenv("PATHSNAP_CONFIG", raising=False)

    actual = _load_config_from_env()
    expected = load_config("config/default.yaml")

    assert actual.routing == expected.routing
    assert actual.emission == expected.emission


def test_matcher_and_route_scorer_share_the_api_contract():
    trip_data = {
        "trip_id": "demo",
        "points": [
            {"lat": 56.0, "lon": 54.0002, "timestamp_s": 1.0},
            {"lat": 56.0, "lon": 54.0018, "timestamp_s": 2.0},
        ],
        "nodes": [
            {"node_id": 0, "lon": 54.0, "lat": 56.0},
            {"node_id": 1, "lon": 54.001, "lat": 56.0},
            {"node_id": 2, "lon": 54.002, "lat": 56.0},
        ],
        "arcs": [
            {"arc_id": 0, "from_node_id": 0, "to_node_id": 1},
            {"arc_id": 1, "from_node_id": 1, "to_node_id": 2},
        ],
    }

    prediction = matcher(trip_data, runtime_dict={})
    evaluation = evaluate_trip(
        trip_id="demo",
        predicted_arc_ids=prediction["route_arc_ids"],
        reference_arc_ids=[0, 1],
        trip_data=trip_data,
        prediction=prediction,
    )

    assert prediction["route_arc_ids"] == [0, 1]
    assert evaluation["metrics"]["rmf"] == 0.0
    assert evaluation["metrics"]["edge_f1"] == 1.0
    assert evaluation["metrics"]["point_accuracy"] == 1.0


def test_matcher_uses_loaded_classifier_for_candidate_emissions():
    trip_data = {
        "trip_id": "demo",
        "points": [
            {"lat": 56.0, "lon": 54.0002, "timestamp_s": 1.0},
            {"lat": 56.0, "lon": 54.0018, "timestamp_s": 2.0},
        ],
        "nodes": [
            {"node_id": 0, "lon": 54.0, "lat": 56.0},
            {"node_id": 1, "lon": 54.001, "lat": 56.0},
            {"node_id": 2, "lon": 54.002, "lat": 56.0},
        ],
        "arcs": [
            {"arc_id": 0, "from_node_id": 0, "to_node_id": 1},
            {"arc_id": 1, "from_node_id": 1, "to_node_id": 2},
        ],
    }

    class PositiveCandidateModel:
        classes_ = np.asarray([0, 1])

        def predict_proba(self, features):
            return np.tile([0.2, 0.8], (len(features), 1))

    result = matcher(trip_data, runtime_dict={"model": PositiveCandidateModel()})

    assert result["metadata"]["candidate_source"] == "trained_classifier"
    assert result["route_arc_ids"] == [0, 1]


def test_matcher_accepts_a_candidate_ranker_for_emissions():
    trip_data = {
        "trip_id": "demo",
        "points": [
            {"lat": 56.0, "lon": 54.0002, "timestamp_s": 1.0},
            {"lat": 56.0, "lon": 54.0018, "timestamp_s": 2.0},
        ],
        "nodes": [
            {"node_id": 0, "lon": 54.0, "lat": 56.0},
            {"node_id": 1, "lon": 54.001, "lat": 56.0},
            {"node_id": 2, "lon": 54.002, "lat": 56.0},
        ],
        "arcs": [
            {"arc_id": 0, "from_node_id": 0, "to_node_id": 1},
            {"arc_id": 1, "from_node_id": 1, "to_node_id": 2},
        ],
    }

    class CandidateRanker:
        def predict(self, features):
            return -features["dist"].to_numpy()

    result = matcher(trip_data, runtime_dict={"model": CandidateRanker()})

    assert result["metadata"]["candidate_source"] == "trained_classifier"
    assert result["route_arc_ids"] == [0, 1]
    assert all(0.0 < point["confidence"] <= 1.0 for point in result["points"])


def test_matcher_accepts_the_deployable_ranker_wrapper():
    trip_data = {
        "trip_id": "demo",
        "points": [
            {"lat": 56.0, "lon": 54.0002, "timestamp_s": 1.0},
            {"lat": 56.0, "lon": 54.0018, "timestamp_s": 2.0},
        ],
        "nodes": [
            {"node_id": 0, "lon": 54.0, "lat": 56.0},
            {"node_id": 1, "lon": 54.001, "lat": 56.0},
            {"node_id": 2, "lon": 54.002, "lat": 56.0},
        ],
        "arcs": [
            {"arc_id": 0, "from_node_id": 0, "to_node_id": 1},
            {"arc_id": 1, "from_node_id": 1, "to_node_id": 2},
        ],
    }

    class CandidateRanker:
        def predict(self, features):
            return -features["dist"].to_numpy()

    model = PathSnapModel(
        CandidateRanker(), None, ["dist"], kind="ranker"
    )
    result = matcher(trip_data, runtime_dict={"model": model})

    assert result["metadata"]["candidate_source"] == "trained_classifier"
    assert result["route_arc_ids"] == [0, 1]


def test_api_evaluate_handler_reads_reference_after_matching(tmp_path):
    trip_id = "demo"
    trip_dir = tmp_path / trip_id
    trip_dir.mkdir()
    (trip_dir / f"{trip_id}.track").write_text(
        "54.0002 56.0 1.0\n54.0018 56.0 2.0\n", encoding="utf-8"
    )
    (trip_dir / f"{trip_id}.nodes").write_text(
        "54.0 56.0\n54.001 56.0\n54.002 56.0\n", encoding="utf-8"
    )
    (trip_dir / f"{trip_id}.arcs").write_text("0 1\n1 2\n", encoding="utf-8")
    (trip_dir / f"{trip_id}.route").write_text("0\n1\n", encoding="utf-8")

    app = SimpleNamespace(
        state=SimpleNamespace(
            matcher=matcher,
            evaluator=evaluate_trip,
            evaluation_data_dir=tmp_path,
            model=None,
            model_path=tmp_path / "final_model.joblib",
        )
    )
    request = SimpleNamespace(app=app)
    result = evaluate(EvaluationRequest(trip_id=trip_id), request)

    assert result["route_arc_ids"] == [0, 1]
    assert result["metrics"]["rmf"] == 0.0
    assert result["metrics"]["edge_f1"] == 1.0
    assert "reference_arc_ids" not in result
