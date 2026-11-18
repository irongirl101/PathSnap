"""Configuration for the PathSnap HMM map-matching component.

All values are tunable. Select settings with a trip-level training holdout.
The student_test partition is available for development checks; never use the
reserved evaluator_test partition to choose settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml


@dataclass
class RoutingConfig:
    """Bounded shortest-path (Dijkstra) settings."""
    # Transition-distance scale (metres) for the log-transition score
    #   log_transition = -abs(d_gc - d_route) / beta
    beta: float = 100.0
    # Maximum network distance (metres) a bounded Dijkstra search will explore.
    # Queries whose shortest path exceeds this are treated as unreachable.
    max_route_distance_m: float = 2000.0
    # Graph is directed. If False, arcs are treated as bidirectional.
    directed: bool = True
    # Hard plausibility cutoff: a transition is rejected as infeasible when
    # d_route > max(plausibility_factor * d_gc, min_route_gap_m). This prevents
    # a candidate on a far-away road from being selected just because emissions
    # are high. Set both plausibility_factor and min_route_gap_m to 0 to disable
    # this cutoff (the hard max_route_distance_m bound still remains).
    plausibility_factor: float = 3.0
    min_route_gap_m: float = 50.0


@dataclass
class GapConfig:
    """GPS-gap detection settings.

    The .track third column is elapsed time in seconds. Gaps are detected from
    elapsed-time discontinuities, with a distance-based fallback.
    """
    # Detect a gap when elapsed seconds between observations exceed this value
    # (1 = nominal 1 Hz sampling has no gap).
    index_gap_threshold: int = 1
    # Distance-based fallback: detect a gap when great-circle distance between
    # consecutive GPS observations exceeds this (metres). Set <= 0 to disable.
    distance_gap_threshold_m: float = 500.0
    # When a gap is detected, the bridge policy is applied between decoded
    # segments (see StitchingConfig).
    record_gaps: bool = True


@dataclass
class EmissionConfig:
    """Classifier-emission handling."""
    # Exponent applied to emission log-probability: alpha * log(emission).
    # alpha = 1.0 uses raw classifier log-probability; alpha = 0 ignores
    # classifier emissions (geometric/uniform emission ablation); alpha > 1
    # sharpens the classifier preference.
    alpha: float = 1.0
    # Emissions are expected to be probabilities in (0, 1]. Values <= 0 are
    # clipped to this floor before taking the log to avoid -inf.
    emission_floor: float = 1e-12
    # Validate that emissions are nonnegative.
    validate_nonnegative: bool = True


@dataclass
class StitchingConfig:
    """Route-stitching and bridge policy."""
    # Connect consecutive decoded candidate projections through shortest paths.
    enable_stitching: bool = True
    # Remove duplicate consecutive edges in the stitched route.
    dedupe_consecutive: bool = True
    # Bridge policy for lattice breaks / gaps: 'shortest_path' inserts the
    # bounded shortest path between segment endpoints and marks edges as
    # 'bridge'; 'none' leaves segments disconnected.
    bridge_policy: str = "shortest_path"  # 'shortest_path' | 'none'
    # Max distance for a bridge path (reuses RoutingConfig.max_route_distance_m
    # if None).
    bridge_max_distance_m: Optional[float] = None


@dataclass
class MatchConfig:
    """Top-level matcher configuration."""
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    # Deterministic tie-breaking in Viterbi: 'first' picks the lowest candidate
    # index on score ties; 'last' picks the highest.
    tie_break: str = "first"  # 'first' | 'last'
    # Random seed used for any train/validation split (kept for reproducibility).
    seed: int = 42


def load_config(path: str) -> MatchConfig:
    """Load a MatchConfig from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _dict_to_config(raw)


def save_config(config: MatchConfig, path: str) -> None:
    """Save a MatchConfig to a YAML file."""
    with open(path, "w") as f:
        yaml.safe_load  # noqa: B018  (placeholder kept for symmetry)
        yaml.safe_dump(_config_to_dict(config), f, sort_keys=False)


def _config_to_dict(c: MatchConfig) -> Dict[str, Any]:
    return {
        "routing": asdict(c.routing),
        "gap": asdict(c.gap),
        "emission": asdict(c.emission),
        "stitching": asdict(c.stitching),
        "tie_break": c.tie_break,
        "seed": c.seed,
    }


def _dict_to_config(raw: Dict[str, Any]) -> MatchConfig:
    routing = RoutingConfig(**raw.get("routing", {}))
    gap = GapConfig(**raw.get("gap", {}))
    emission = EmissionConfig(**raw.get("emission", {}))
    stitching = StitchingConfig(**raw.get("stitching", {}))
    return MatchConfig(
        routing=routing,
        gap=gap,
        emission=emission,
        stitching=stitching,
        tie_break=raw.get("tie_break", "first"),
        seed=raw.get("seed", 42),
    )


DEFAULT_CONFIG = MatchConfig()


def default_config_path() -> str:
    import os
    return os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")
