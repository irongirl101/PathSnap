"""FastAPI adapter for the PathSnap HMM map-matcher.

Wires `src/pathsnap` into the committed API via the `PATHSNAP_MATCHER`
environment variable, as expected by `src/api/main.py`:

    matcher(request_dict: dict, runtime_dict: dict) -> dict

The committed `MapMatchRequest`/`MapMatchResponse` contracts use **(lat, lon)**
coordinate order. The internal `pathsnap` library uses **(lon, lat)** to match
the supplied data files (`.track`, `.nodes`). This adapter performs the
coordinate conversion at the boundary so the core matcher never needs to know
about the API contract.

The matcher callable signature is fixed by the API:
    (request_dict, runtime_dict) -> response_dict

Trip data discovery
-------------------
The adapter locates a trip's `.track`/`.nodes`/`.arcs`/`.route` files under a
data root, configurable via the `PATHSNAP_DATA_ROOT` environment variable
(default: `<project_root>/data`). It looks in both `<root>/train/<trip_id>/`
and `<root>/student_test/<trip_id>/`.

Because the supplied archive contains no classifier emission scores, the
adapter uses the documented stand-in candidate generator
(`pathsnap.candidates.generate_candidates`). When the classifier owner
provides real candidates/emissions, they can replace this call — the matcher
itself is unchanged.

Set the adapter with:

    export PATHSNAP_MATCHER=pathsnap.api_adapter:matcher
    export PATHSNAP_DATA_ROOT=/path/to/data/root   # optional
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

# Allow import both as `pathsnap.api_adapter` (package) and as a top-level
# module when src/pathsnap is on sys.path.
try:  # package import (src on path)
    from pathsnap.config import MatchConfig, load_config
    from pathsnap.graph import Graph
    from pathsnap.hmm import Candidate, MatchResult, match_trip
    from pathsnap.io import Trip, build_graph, load_trip
    from pathsnap.candidates import generate_candidates
except Exception:  # fallback: src/pathsnap on sys.path directly
    from config import MatchConfig, load_config  # type: ignore[no-redef]
    from graph import Graph  # type: ignore[no-redef]
    from hmm import Candidate, MatchResult, match_trip  # type: ignore[no-redef]
    from io import Trip, build_graph, load_trip  # type: ignore[no-redef]
    from candidates import generate_candidates  # type: ignore[no-redef]


def _data_root() -> Path:
    """Resolve the directory containing train/ and student_test/."""
    env = os.getenv("PATHSNAP_DATA_ROOT")
    if env:
        return Path(env)
    # Default: <project_root>/data, where project_root is this file's
    # parents[2] (PathSnap/src/pathsnap/api_adapter.py -> PathSnap).
    return Path(__file__).resolve().parents[2] / "data"


def _find_trip_dir(trip_id: str, data_root: Path) -> Optional[Path]:
    """Look for <trip_id> under train/ and student_test/."""
    for split in ("train", "student_test"):
        cand = data_root / split / trip_id
        if cand.is_dir():
            return cand
    return None


def _request_points(request: Dict[str, Any]) -> List[Tuple[float, float, int]]:
    """Convert API (lat, lon) points to internal (lon, lat, point_index).

    API contract: points[i] = {"lat": ..., "lon": ..., "timestamp": ...}
    Internal:     (lon, lat, index), index = position in the list (0-based).
    """
    points: List[Tuple[float, float, int]] = []
    for i, p in enumerate(request.get("points", []) or []):
        lat = float(p["lat"])
        lon = float(p["lon"])
        points.append((lon, lat, i))
    return points


def _result_to_response(
    trip_id: Optional[str],
    result: MatchResult,
) -> Dict[str, Any]:
    """Convert a MatchResult to a MapMatchResponse-shaped dict.

    Converts internal (lon, lat) back to API (lat, lon). `edge_id` and
    `confidence` come from the matched candidate; `node_id` is left None
    because the candidate schema snaps to edges, not nodes (the reference
    `.route` is an edge sequence, not per-point node labels).
    """
    route_arc_ids: List[Any] = []
    points_out: List[Dict[str, Any]] = []
    total_distance_m = 0.0

    for i in range(result.num_points):
        cand = result.matched[i]
        lon, lat, _pidx = result.points[i]
        if cand is not None:
            route_arc_ids.append(int(cand.edge_id))
            total_distance_m += float(cand.dist_m)
            points_out.append(
                {
                    "input_index": i,
                    "lat": float(lat),
                    "lon": float(lon),
                    "node_id": None,
                    "edge_id": int(cand.edge_id),
                    "confidence": float(cand.emission),
                }
            )
        else:
            points_out.append(
                {
                    "input_index": i,
                    "lat": float(lat),
                    "lon": float(lon),
                    "node_id": None,
                    "edge_id": None,
                    "confidence": None,
                }
            )

    # Stitched route edge ids (deduped, direction-preserved).
    if not route_arc_ids:
        route_arc_ids = [int(e.edge_id) for e in result.stitched_route]

    return {
        "trip_id": trip_id,
        "route_arc_ids": route_arc_ids,
        "points": points_out,
        "distance_m": float(total_distance_m) if total_distance_m > 0 else None,
        "metadata": {
            "num_points": result.num_points,
            "matched_count": result.matched_count,
            "unmatched_point_indices": result.unmatched_point_indices,
            "gaps": [list(g) for g in result.gaps],
            "breaks": list(result.breaks),
            "num_segments": len(result.segments),
            "num_stitched_edges": len(result.stitched_route),
            "total_score": float(result.total_score),
            "stitched_edge_sources": {
                "decoder": sum(1 for e in result.stitched_route if e.source == "decoder"),
                "stitched": sum(1 for e in result.stitched_route if e.source == "stitched"),
                "bridge": sum(1 for e in result.stitched_route if e.source == "bridge"),
            },
        },
    }


def _load_config_from_env() -> MatchConfig:
    """Load MatchConfig from PATHSNAP_CONFIG if set, else defaults."""
    config_path = os.getenv("PATHSNAP_CONFIG")
    if config_path and Path(config_path).is_file():
        return load_config(config_path)
    return MatchConfig()


def matcher(request_dict: Dict[str, Any], runtime_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Map-match a single GPS trace.

    This is the callable referenced by `PATHSNAP_MATCHER`. Signature is fixed
    by the committed API (`src/api/main.py`):

        matcher(request_dict, runtime_dict) -> dict

    Parameters
    ----------
    request_dict : dict
        `MapMatchRequest.model_dump(mode="json")`:
        {"trip_id": str|None, "points": [{"lat": float, "lon": float,
        "timestamp": str|None}, ...]}
    runtime_dict : dict
        `get_runtime(app)` output: {"model": ..., "model_path": ...}.
        Unused by this adapter (kept for signature compatibility); the model
        artifact is owned by the classifier member.

    Returns
    -------
    dict
        `MapMatchResponse`-shaped dict validated by the API endpoint.
    """
    trip_id = request_dict.get("trip_id")
    if not trip_id:
        raise ValueError("trip_id is required")

    data_root = _data_root()
    trip_dir = _find_trip_dir(str(trip_id), data_root)
    if trip_dir is None:
        raise ValueError(
            f"Trip data not found for '{trip_id}' under {data_root}/train "
            f"or {data_root}/student_test"
        )

    # Load trip + build graph.
    trip: Trip = load_trip(str(trip_id), str(trip_dir))
    config = _load_config_from_env()
    graph: Graph = build_graph(
        trip,
        directed=config.routing.directed,
        max_route_distance_m=config.routing.max_route_distance_m,
    )

    # Points: API (lat, lon) -> internal (lon, lat, index).
    points = _request_points(request_dict)
    if not points:
        raise ValueError("at least one point is required")

    # Candidates: stand-in generator (no classifier in the archive).
    # The classifier owner can substitute real candidates here.
    cands = generate_candidates(str(trip_id), points, graph)

    # Match.
    result: MatchResult = match_trip(str(trip_id), points, cands, graph, config)

    return _result_to_response(trip_id, result)


__all__ = ["matcher", "_data_root", "_find_trip_dir", "_request_points", "_result_to_response"]
