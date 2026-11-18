"""Pipeline orchestration: per-trip matching, train/validation tuning,
ablations, and metric aggregation.

All tuning uses training data only with a trip-level split (no leakage of
points across splits). Held-out student_test routes are never consulted
during configuration selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable
import itertools
import json
import math
import os
import random

import numpy as np
import yaml

from .config import MatchConfig, RoutingConfig, EmissionConfig, GapConfig, \
    StitchingConfig, load_config, save_config
from .graph import Graph
from .hmm import Candidate, match_trip, classifier_only
from .io import Trip, load_trip, build_graph, list_trips, trip_dir
from .candidates import generate_candidates
from .evaluate import evaluate_match, aggregate, TripMetrics


# ---------------------------------------------------------------------------
# Per-trip run
# ---------------------------------------------------------------------------

@dataclass
class TripRun:
    trip_id: str
    hmm_metrics: Optional[TripMetrics]
    clf_metrics: Optional[TripMetrics]
    hmm_result: Optional[object]  # MatchResult
    clf_result: Optional[object]  # MatchResult
    config: MatchConfig
    representation: str


def run_trip(
    trip: Trip,
    graph: Graph,
    config: MatchConfig,
    representation: str = "stitched",
    gen_cfg=None,
) -> TripRun:
    """Generate candidates, run classifier-only and HMM, evaluate both."""
    points = [(float(t[0]), float(t[1]), int(t[2])) for t in trip.track]
    cands = generate_candidates(trip.trip_id, points, graph, gen_cfg)

    hmm_res = match_trip(trip.trip_id, points, cands, graph, config)
    clf_res = classifier_only(trip.trip_id, points, cands)

    hmm_m = evaluate_match(hmm_res, trip, graph, config, representation)
    clf_m = evaluate_match(clf_res, trip, graph, config, representation)

    return TripRun(
        trip_id=trip.trip_id,
        hmm_metrics=hmm_m,
        clf_metrics=clf_m,
        hmm_result=hmm_res,
        clf_result=clf_res,
        config=config,
        representation=representation,
    )


# ---------------------------------------------------------------------------
# Train/validation split (trip-level, no leakage)
# ---------------------------------------------------------------------------

def train_val_split(
    train_trip_ids: List[str],
    val_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split training trip ids into train/validation at the trip level.

    All points of a trip stay in one split (prevents leakage). Returns
    (train_ids, val_ids).
    """
    rng = random.Random(seed)
    ids = list(train_trip_ids)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    n_val = min(n_val, len(ids) - 1) if len(ids) > 1 else n_val
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Objective for tuning
# ---------------------------------------------------------------------------

def objective_from_metrics(metrics: List[TripMetrics],
                           primary: str = "edge_f1") -> float:
    """Tuning objective: mean of a per-trip metric over the validation set.

    Defined in advance; same evaluation implementation used for all configs.
    """
    if not metrics:
        return -float("inf")
    vals = []
    for m in metrics:
        if primary == "edge_f1":
            vals.append(m.edge_f1)
        elif primary == "rmf":
            vals.append(-m.rmf)  # lower RMF better
        elif primary == "point_accuracy":
            vals.append(m.point_accuracy)
        elif primary == "coverage":
            vals.append(m.coverage)
        else:
            raise ValueError(f"unknown objective {primary}")
    return sum(vals) / len(vals)


def tune_beta_alpha(
    train_root: str,
    train_ids: List[str],
    val_ids: List[str],
    beta_grid: List[float],
    alpha_grid: List[float],
    base_config: Optional[MatchConfig] = None,
    representation: str = "stitched",
    primary: str = "edge_f1",
    val_fraction_note: str = "",
) -> Dict:
    """Grid search over (beta, alpha) on training data, validated on val split.

    Returns a dict with the best config, best score, and full grid results.
    Held-out test routes are never used.
    """
    if base_config is None:
        base_config = MatchConfig()

    # Pre-load trips + graphs for train+val (candidates regenerated per config
    # only via emission alpha, which doesn't change candidates; graphs cached).
    trips: Dict[str, Trip] = {}
    graphs: Dict[str, Graph] = {}
    for tid in train_ids + val_ids:
        td = trip_dir(train_root, tid)
        trips[tid] = load_trip(tid, td)
        graphs[tid] = build_graph(trips[tid],
                                  directed=base_config.routing.directed,
                                  max_route_distance_m=base_config.routing.max_route_distance_m)

    grid_results = []
    best_score = -float("inf")
    best_cfg = None
    best_params = None

    for beta in beta_grid:
        for alpha in alpha_grid:
            cfg = MatchConfig()
            cfg.routing = RoutingConfig(
                beta=beta,
                max_route_distance_m=base_config.routing.max_route_distance_m,
                directed=base_config.routing.directed,
                plausibility_factor=base_config.routing.plausibility_factor,
                min_route_gap_m=base_config.routing.min_route_gap_m,
            )
            cfg.emission = EmissionConfig(
                alpha=alpha,
                emission_floor=base_config.emission.emission_floor,
                validate_nonnegative=base_config.emission.validate_nonnegative,
            )
            cfg.gap = base_config.gap
            cfg.stitching = base_config.stitching
            cfg.tie_break = base_config.tie_break
            cfg.seed = base_config.seed

            # Evaluate on validation trips
            val_metrics: List[TripMetrics] = []
            for tid in val_ids:
                run = run_trip(trips[tid], graphs[tid], cfg, representation)
                val_metrics.append(run.hmm_metrics)
            score = objective_from_metrics(val_metrics, primary)

            # Also record train metrics for diagnosis
            train_metrics: List[TripMetrics] = []
            for tid in train_ids:
                run = run_trip(trips[tid], graphs[tid], cfg, representation)
                train_metrics.append(run.hmm_metrics)
            train_score = objective_from_metrics(train_metrics, primary)

            grid_results.append({
                "beta": beta, "alpha": alpha,
                "val_score": score,
                "train_score": train_score,
                "val_n": len(val_metrics),
            })
            if score > best_score:
                best_score = score
                best_params = (beta, alpha)
                best_cfg = cfg

    return {
        "best_params": {"beta": best_params[0], "alpha": best_params[1]},
        "best_score": best_score,
        "objective": primary,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "seed": base_config.seed,
        "representation": representation,
        "grid": grid_results,
    }


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------

def run_ablations(
    train_root: str,
    trip_ids: List[str],
    base_config: MatchConfig,
    representation: str = "stitched",
) -> List[Dict]:
    """Run ablation configurations over the given trips.

    Ablations:
      - classifier_only (baseline, no transitions)
      - hmm_full (tuned beta, alpha)
      - no_transition (beta -> very large so transitions flat / alpha-only)
      - no_emission (alpha = 0, geometric/uniform emission => pure routing)
    """
    results = []

    def eval_cfg(cfg: MatchConfig, label: str):
        per_trip = []
        for tid in trip_ids:
            td = trip_dir(train_root, tid)
            trip = load_trip(tid, td)
            graph = build_graph(trip, directed=cfg.routing.directed,
                                max_route_distance_m=cfg.routing.max_route_distance_m)
            run = run_trip(trip, graph, cfg, representation)
            per_trip.append(run.hmm_metrics)
        return per_trip

    # 1. classifier-only baseline (separate function, no transitions)
    clf_per_trip = []
    for tid in trip_ids:
        td = trip_dir(train_root, tid)
        trip = load_trip(tid, td)
        graph = build_graph(trip, directed=base_config.routing.directed,
                            max_route_distance_m=base_config.routing.max_route_distance_m)
        points = [(float(t[0]), float(t[1]), int(t[2])) for t in trip.track]
        cands = generate_candidates(trip.trip_id, points, graph)
        clf_res = classifier_only(trip.trip_id, points, cands)
        clf_per_trip.append(evaluate_match(clf_res, trip, graph, base_config, representation))
    results.append({
        "ablation": "classifier_only",
        "note": "independent per-point argmax emission, no transitions/stitching",
        "aggregate": asdict(aggregate(clf_per_trip)),
    })

    # 2. HMM full
    hmm_per_trip = eval_cfg(base_config, "hmm_full")
    results.append({
        "ablation": "hmm_full",
        "note": f"beta={base_config.routing.beta}, alpha={base_config.emission.alpha}",
        "aggregate": asdict(aggregate(hmm_per_trip)),
    })

    # 3. No transition term: set beta huge so transition log-score ~ 0 for all
    #    feasible routes (flat), emissions dominate. Keeps no-route handling.
    no_trans = MatchConfig()
    no_trans.routing = RoutingConfig(
        beta=1e6,
        max_route_distance_m=base_config.routing.max_route_distance_m,
        directed=base_config.routing.directed,
        plausibility_factor=0.0,  # disable plausibility cutoff
        min_route_gap_m=0.0,
    )
    no_trans.emission = base_config.emission
    no_trans.gap = base_config.gap
    no_trans.stitching = base_config.stitching
    nt_per_trip = eval_cfg(no_trans, "no_transition")
    results.append({
        "ablation": "no_transition",
        "note": "flat transition (beta=1e6, cutoff disabled); emissions only",
        "aggregate": asdict(aggregate(nt_per_trip)),
    })

    # 4. No classifier emission (alpha=0): pure geometric/routing emission.
    #    Use a geometric emission based on GPS->edge distance so it's not
    #    degenerate. Since generate_candidates normalizes emissions to sum 1,
    #    alpha=0 makes all emissions equal (uniform) => pure shortest-path.
    no_em = MatchConfig()
    no_em.routing = base_config.routing
    no_em.emission = EmissionConfig(alpha=0.0,
                                    emission_floor=base_config.emission.emission_floor,
                                    validate_nonnegative=False)
    no_em.gap = base_config.gap
    no_em.stitching = base_config.stitching
    ne_per_trip = eval_cfg(no_em, "no_emission")
    results.append({
        "ablation": "no_emission",
        "note": "alpha=0 (uniform emission); pure geometric routing, no classifier",
        "aggregate": asdict(aggregate(ne_per_trip)),
    })

    return results


# ---------------------------------------------------------------------------
# Sensitivity grid (compact heatmap data)
# ---------------------------------------------------------------------------

def sensitivity_grid(
    train_root: str,
    val_ids: List[str],
    beta_grid: List[float],
    alpha_grid: List[float],
    base_config: MatchConfig,
    representation: str = "stitched",
    primary: str = "edge_f1",
) -> List[Dict]:
    """Produce compact grid of val scores over (beta, alpha) for heatmap."""
    trips = {}
    graphs = {}
    for tid in val_ids:
        td = trip_dir(train_root, tid)
        trips[tid] = load_trip(tid, td)
        graphs[tid] = build_graph(trips[tid],
                                  directed=base_config.routing.directed,
                                  max_route_distance_m=base_config.routing.max_route_distance_m)

    rows = []
    for beta in beta_grid:
        for alpha in alpha_grid:
            cfg = MatchConfig()
            cfg.routing = RoutingConfig(
                beta=beta,
                max_route_distance_m=base_config.routing.max_route_distance_m,
                directed=base_config.routing.directed,
                plausibility_factor=base_config.routing.plausibility_factor,
                min_route_gap_m=base_config.routing.min_route_gap_m,
            )
            cfg.emission = EmissionConfig(alpha=alpha,
                                          emission_floor=base_config.emission.emission_floor)
            cfg.gap = base_config.gap
            cfg.stitching = base_config.stitching
            mets = []
            for tid in val_ids:
                run = run_trip(trips[tid], graphs[tid], cfg, representation)
                mets.append(run.hmm_metrics)
            rows.append({
                "beta": beta, "alpha": alpha,
                "val_score": objective_from_metrics(mets, primary),
                "mean_rmf": np.mean([m.rmf for m in mets]),
                "mean_edge_f1": np.mean([m.edge_f1 for m in mets]),
                "mean_point_accuracy": np.mean([m.point_accuracy for m in mets]),
                "n": len(mets),
            })
    return rows
