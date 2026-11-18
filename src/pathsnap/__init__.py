"""PathSnap Member-C component: HMM map-matching, routing, stitching, evaluation.

Public API
----------
    from pathsnap.config import MatchConfig, load_config, save_config
    from pathsnap.graph import Graph
    from pathsnap.hmm import Candidate, match_trip, classifier_only, MatchResult
    from pathsnap.evaluate import evaluate_match, rmf, edge_f1, point_accuracy
    from pathsnap.io import load_trip
    from pathsnap.pipeline import run_trip, run_split

See each module for full input/output schemas.
"""

from .config import MatchConfig, DEFAULT_CONFIG, load_config, save_config
from .graph import Graph, PathResult, NO_ROUTE, haversine_m
from .hmm import (
    Candidate,
    MatchResult,
    Segment,
    StitchedEdge,
    match_trip,
    classifier_only,
    detect_gaps,
)

__all__ = [
    "MatchConfig",
    "DEFAULT_CONFIG",
    "load_config",
    "save_config",
    "Graph",
    "PathResult",
    "NO_ROUTE",
    "haversine_m",
    "Candidate",
    "MatchResult",
    "Segment",
    "StitchedEdge",
    "match_trip",
    "classifier_only",
    "detect_gaps",
]
