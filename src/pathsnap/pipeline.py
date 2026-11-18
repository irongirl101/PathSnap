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
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

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


# ---------------------------------------------------------------------------
# Bayesian optimization (Gaussian Process + Expected Improvement)
# ---------------------------------------------------------------------------

def _make_config_from_params(
    base_config: MatchConfig,
    beta: float,
    alpha: float,
    representation: str = "stitched",
) -> MatchConfig:
    cfg = MatchConfig()
    cfg.routing = RoutingConfig(
        beta=float(beta),
        max_route_distance_m=base_config.routing.max_route_distance_m,
        directed=base_config.routing.directed,
        plausibility_factor=base_config.routing.plausibility_factor,
        min_route_gap_m=base_config.routing.min_route_gap_m,
    )
    cfg.emission = EmissionConfig(
        alpha=float(alpha),
        emission_floor=base_config.emission.emission_floor,
        validate_nonnegative=base_config.emission.validate_nonnegative,
    )
    cfg.gap = base_config.gap
    cfg.stitching = base_config.stitching
    cfg.tie_break = base_config.tie_break
    cfg.seed = base_config.seed
    return cfg


def bayesian_optimize_beta_alpha(
    train_root: str,
    train_ids: List[str],
    val_ids: List[str],
    n_iterations: int = 30,
    n_initial_points: int = 8,
    base_config: Optional[MatchConfig] = None,
    representation: str = "stitched",
    primary: str = "edge_f1",
    seed: int = 42,
) -> Dict:
    """Bayesian optimization over (beta, alpha) using Gaussian Process +
    Expected Improvement acquisition.

    Beta and alpha are optimized in log-space. The same evaluation
    implementation and trip-level split are used as grid search, so results
    are directly comparable.

    Parameters
    ----------
    train_root : str
        Path to the training data root directory.
    train_ids : list[str]
        Training trip IDs (used for logging/diagnosis only; objective is
        computed on val_ids).
    val_ids : list[str]
        Validation trip IDs. The objective (mean edge F1 over val trips) is
        computed here.
    n_iterations : int
        Number of Bayesian optimization iterations after initial points.
    n_initial_points : int
        Number of initial random points before GP-based selection.
    base_config : MatchConfig | None
        Base configuration; if None, defaults are used.
    representation : str
        Route representation to evaluate ('stitched' or 'decoded').
    primary : str
        Primary metric for the objective ('edge_f1', 'rmf', etc.).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Best params, best score, optimization history, and comparison metadata.
    """
    if base_config is None:
        base_config = MatchConfig()

    rng = np.random.RandomState(seed)

    # Pre-load trips + graphs for validation trips (same objects reused across
    # all objective evaluations to avoid redundant I/O).
    trips: Dict[str, Trip] = {}
    graphs: Dict[str, Graph] = {}
    for tid in val_ids:
        td = trip_dir(train_root, tid)
        trips[tid] = load_trip(tid, td)
        graphs[tid] = build_graph(
            trips[tid],
            directed=base_config.routing.directed,
            max_route_distance_m=base_config.routing.max_route_distance_m,
        )

    def objective_logspace(x: np.ndarray) -> float:
        """Objective function in log-space: x = [log10(beta), log10(alpha)]."""
        beta = 10.0 ** x[0]
        alpha = 10.0 ** x[1]
        cfg = _make_config_from_params(base_config, beta, alpha, representation)
        mets: List[TripMetrics] = []
        for tid in val_ids:
            run = run_trip(trips[tid], graphs[tid], cfg, representation)
            mets.append(run.hmm_metrics)
        return objective_from_metrics(mets, primary)

    # Bounds in log10 space
    beta_log_bounds = (math.log10(10.0), math.log10(1000.0))   # beta in [10, 1000]
    alpha_log_bounds = (math.log10(0.1), math.log10(5.0))      # alpha in [0.1, 5]

    # Initial random design (uniform random in log-space)
    n_init = n_initial_points
    X_init = np.column_stack([
        rng.uniform(beta_log_bounds[0], beta_log_bounds[1], n_init),
        rng.uniform(alpha_log_bounds[0], alpha_log_bounds[1], n_init),
    ])
    y_init = np.array([objective_logspace(x) for x in X_init])

    X = X_init.copy()
    y = y_init.copy()

    history: List[Dict] = []
    for i in range(n_init):
        history.append({
            "iteration": i,
            "phase": "initial",
            "beta": float(10.0 ** X[i, 0]),
            "alpha": float(10.0 ** X[i, 1]),
            "val_score": float(y[i]),
        })

    # Gaussian Process surrogate
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )

    best_idx = int(np.argmax(y))
    best_x = X[best_idx].copy()
    best_y = float(y[best_idx])

    for it in range(n_iterations):
        # Fit GP
        gp.fit(X, y)

        # Expected Improvement acquisition
        def negative_ei(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            improvement = mu - best_y - 0.01  # small exploration constant
            z = improvement / sigma
            ei = improvement * _norm_cdf(z) + sigma * _norm_pdf(z)
            return -float(ei[0])

        # Optimize acquisition via multiple random restarts + L-BFGS-B
        acquisition_bounds = [beta_log_bounds, alpha_log_bounds]
        best_acq_x = None
        best_acq_val = float("inf")
        for _ in range(20):
            x0 = np.array([
                rng.uniform(beta_log_bounds[0], beta_log_bounds[1]),
                rng.uniform(alpha_log_bounds[0], alpha_log_bounds[1]),
            ])
            res = minimize(
                negative_ei, x0, method="L-BFGS-B", bounds=acquisition_bounds,
            )
            if res.fun < best_acq_val:
                best_acq_val = res.fun
                best_acq_x = res.x

        if best_acq_x is None:
            break

        x_next = best_acq_x
        y_next = objective_logspace(x_next)

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

        if y_next > best_y:
            best_y = float(y_next)
            best_x = x_next.copy()

        history.append({
            "iteration": n_init + it,
            "phase": "bayesian",
            "beta": float(10.0 ** x_next[0]),
            "alpha": float(10.0 ** x_next[1]),
            "val_score": float(y_next),
        })

    best_beta = float(10.0 ** best_x[0])
    best_alpha = float(10.0 ** best_x[1])

    # Evaluate best on training trips for diagnosis
    train_mets: List[TripMetrics] = []
    for tid in train_ids:
        td = trip_dir(train_root, tid)
        trip = load_trip(tid, td)
        graph = build_graph(
            trip,
            directed=base_config.routing.directed,
            max_route_distance_m=base_config.routing.max_route_distance_m,
        )
        cfg = _make_config_from_params(base_config, best_beta, best_alpha, representation)
        run = run_trip(trip, graph, cfg, representation)
        train_mets.append(run.hmm_metrics)
    train_score = objective_from_metrics(train_mets, primary)

    return {
        "method": "bayesian_optimization",
        "best_params": {"beta": best_beta, "alpha": best_alpha},
        "best_score": best_y,
        "train_score": train_score,
        "objective": primary,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "seed": seed,
        "n_iterations": n_iterations,
        "n_initial_points": n_initial_points,
        "representation": representation,
        "history": history,
    }


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(lambda v: math.erf(v / math.sqrt(2.0)))(z))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
