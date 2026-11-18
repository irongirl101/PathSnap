"""Compare Bayesian optimization vs grid search for beta/alpha tuning.

Runs both methods on the same train/validation split and compares:
- Best parameters found
- Best validation score
- Convergence curves (best score over iterations)
- Held-out test performance of each method's best config

Outputs:
    artifacts/comparison/bo_vs_grid.json  -- full comparison results
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pathsnap.config import MatchConfig, RoutingConfig, EmissionConfig, save_config
from pathsnap.io import list_trips, load_trip, build_graph, trip_dir
from pathsnap.candidates import generate_candidates
from pathsnap.hmm import match_trip, classifier_only
from pathsnap.evaluate import evaluate_match, aggregate
from pathsnap.pipeline import (
    train_val_split,
    tune_beta_alpha,
    bayesian_optimize_beta_alpha,
    objective_from_metrics,
    run_trip,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path(
    os.getenv("PATHSNAP_DATA_ROOT", REPO_ROOT.parent / "Default Project" / "Student folder")
) / "train"
TEST_ROOT = Path(
    os.getenv("PATHSNAP_DATA_ROOT", REPO_ROOT.parent / "Default Project" / "Student folder")
) / "student_test"

ARTIFACTS = REPO_ROOT / "artifacts"
COMPARISON_DIR = ARTIFACTS / "comparison"

BETA_GRID = [25.0, 50.0, 100.0, 200.0, 400.0]
ALPHA_GRID = [0.25, 0.5, 1.0, 2.0]

PRIMARY_OBJECTIVE = "edge_f1"
VAL_FRACTION = 0.25
SEED = 42
REPRESENTATION = "stitched"


def evaluate_config_on_test(
    test_trip_ids: list[str],
    config: MatchConfig,
    representation: str = "stitched",
) -> dict:
    """Evaluate a config on held-out test trips. Returns aggregate + per-trip."""
    hmm_mets = []
    clf_mets = []
    for tid in test_trip_ids:
        trip = load_trip(tid, str(trip_dir(TEST_ROOT, tid)))
        graph = build_graph(
            trip,
            directed=config.routing.directed,
            max_route_distance_m=config.routing.max_route_distance_m,
        )
        points = [(float(t[0]), float(t[1]), int(t[2])) for t in trip.track]
        cands = generate_candidates(trip.trip_id, points, graph)

        hmm_res = match_trip(trip.trip_id, points, cands, graph, config)
        clf_res = classifier_only(trip.trip_id, points, cands)

        hmm_mets.append(evaluate_match(hmm_res, trip, graph, config, representation))
        clf_mets.append(evaluate_match(clf_res, trip, graph, config, representation))

    hmm_agg = aggregate(hmm_mets)
    clf_agg = aggregate(clf_mets)
    return {
        "hmm": {
            "mean_edge_f1": hmm_agg.mean_edge_f1,
            "mean_rmf": hmm_agg.mean_rmf,
            "mean_point_accuracy": hmm_agg.mean_point_accuracy,
            "mean_coverage": hmm_agg.mean_coverage,
            "micro_edge_f1": hmm_agg.micro_edge_f1,
        },
        "classifier_only": {
            "mean_edge_f1": clf_agg.mean_edge_f1,
            "mean_rmf": clf_agg.mean_rmf,
            "mean_point_accuracy": clf_agg.mean_point_accuracy,
            "mean_coverage": clf_agg.mean_coverage,
            "micro_edge_f1": clf_agg.micro_edge_f1,
        },
        "per_trip_hmm": [
            {
                "trip_id": m.trip_id,
                "edge_f1": m.edge_f1,
                "rmf": m.rmf,
                "point_accuracy": m.point_accuracy,
                "coverage": m.coverage,
            }
            for m in hmm_mets
        ],
    }


def main() -> None:
    train_trip_ids = list_trips(str(TRAIN_ROOT))
    print(f"Found {len(train_trip_ids)} training trips")

    # Same train/val split for both methods (fair comparison)
    train_ids, val_ids = train_val_split(
        train_trip_ids, val_fraction=VAL_FRACTION, seed=SEED
    )
    print(f"Train: {len(train_ids)} trips, Val: {len(val_ids)} trips")

    test_trip_ids = list_trips(str(TEST_ROOT))
    print(f"Found {len(test_trip_ids)} held-out test trips")

    # ------------------------------------------------------------------
    # Grid search results (reuse saved grid_results.json if present)
    # ------------------------------------------------------------------
    grid_path = ARTIFACTS / "tuning" / "grid_results.json"
    if grid_path.is_file():
        with open(grid_path) as f:
            grid_result = json.load(f)
        print(f"\nLoaded existing grid search results: beta={grid_result['best_params']['beta']}, "
              f"alpha={grid_result['best_params']['alpha']}, val_score={grid_result['best_score']:.4f}")
    else:
        print("\nRunning grid search...")
        grid_result = tune_beta_alpha(
            train_root=str(TRAIN_ROOT),
            train_ids=train_ids,
            val_ids=val_ids,
            beta_grid=BETA_GRID,
            alpha_grid=ALPHA_GRID,
            base_config=MatchConfig(),
            representation=REPRESENTATION,
            primary=PRIMARY_OBJECTIVE,
        )
        print(f"Grid search best: beta={grid_result['best_params']['beta']}, "
              f"alpha={grid_result['best_params']['alpha']}, val_score={grid_result['best_score']:.4f}")

    # ------------------------------------------------------------------
    # Bayesian optimization
    # ------------------------------------------------------------------
    print("\nRunning Bayesian optimization...")
    bo_result = bayesian_optimize_beta_alpha(
        train_root=str(TRAIN_ROOT),
        train_ids=train_ids,
        val_ids=val_ids,
        n_iterations=25,
        n_initial_points=8,
        base_config=MatchConfig(),
        representation=REPRESENTATION,
        primary=PRIMARY_OBJECTIVE,
        seed=SEED,
    )
    print(f"BO best: beta={bo_result['best_params']['beta']:.3f}, "
          f"alpha={bo_result['best_params']['alpha']:.3f}, val_score={bo_result['best_score']:.4f}")

    # ------------------------------------------------------------------
    # Evaluate both best configs on held-out test trips
    # ------------------------------------------------------------------
    print("\nEvaluating grid-search best config on held-out test trips...")
    grid_cfg = MatchConfig()
    grid_cfg.routing = RoutingConfig(
        beta=grid_result["best_params"]["beta"],
        max_route_distance_m=2000.0,
        directed=True,
        plausibility_factor=3.0,
        min_route_gap_m=50.0,
    )
    grid_cfg.emission = EmissionConfig(
        alpha=grid_result["best_params"]["alpha"],
        emission_floor=1e-12,
        validate_nonnegative=True,
    )
    grid_test = evaluate_config_on_test(test_trip_ids, grid_cfg, REPRESENTATION)

    print("\nEvaluating BO best config on held-out test trips...")
    bo_cfg = MatchConfig()
    bo_cfg.routing = RoutingConfig(
        beta=bo_result["best_params"]["beta"],
        max_route_distance_m=2000.0,
        directed=True,
        plausibility_factor=3.0,
        min_route_gap_m=50.0,
    )
    bo_cfg.emission = EmissionConfig(
        alpha=bo_result["best_params"]["alpha"],
        emission_floor=1e-12,
        validate_nonnegative=True,
    )
    bo_test = evaluate_config_on_test(test_trip_ids, bo_cfg, REPRESENTATION)

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    comparison = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_trip_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_trip_ids,
        "primary_objective": PRIMARY_OBJECTIVE,
        "representation": REPRESENTATION,
        "grid_search": {
            "best_params": grid_result["best_params"],
            "best_val_score": grid_result["best_score"],
            "n_configs_evaluated": len(grid_result["grid"]),
            "test": grid_test,
        },
        "bayesian_optimization": {
            "best_params": bo_result["best_params"],
            "best_val_score": bo_result["best_score"],
            "n_configs_evaluated": len(bo_result["history"]),
            "history": bo_result["history"],
            "test": bo_test,
        },
        "comparison": {
            "val_score_improvement": bo_result["best_score"] - grid_result["best_score"],
            "beta_ratio": bo_result["best_params"]["beta"] / grid_result["best_params"]["beta"],
            "alpha_ratio": bo_result["best_params"]["alpha"] / grid_result["best_params"]["alpha"],
            "bo_better_on_val": bo_result["best_score"] > grid_result["best_score"],
            "test_edge_f1_diff": bo_test["hmm"]["mean_edge_f1"] - grid_test["hmm"]["mean_edge_f1"],
            "test_rmf_diff": bo_test["hmm"]["mean_rmf"] - grid_test["hmm"]["mean_rmf"],
            "test_point_acc_diff": bo_test["hmm"]["mean_point_accuracy"] - grid_test["hmm"]["mean_point_accuracy"],
            "bo_better_on_test_f1": bo_test["hmm"]["mean_edge_f1"] > grid_test["hmm"]["mean_edge_f1"],
        },
    }

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_DIR / "bo_vs_grid.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nSaved comparison to {COMPARISON_DIR / 'bo_vs_grid.json'}")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nGrid Search:")
    print(f"  Best params: beta={grid_result['best_params']['beta']}, alpha={grid_result['best_params']['alpha']}")
    print(f"  Best val score ({PRIMARY_OBJECTIVE}): {grid_result['best_score']:.4f}")
    print(f"  Configs evaluated: {len(grid_result['grid'])}")
    print(f"  Test HMM edge F1: {grid_test['hmm']['mean_edge_f1']:.4f}")
    print(f"  Test HMM RMF: {grid_test['hmm']['mean_rmf']:.4f}")
    print(f"  Test HMM point accuracy: {grid_test['hmm']['mean_point_accuracy']:.4f}")

    print(f"\nBayesian Optimization:")
    print(f"  Best params: beta={bo_result['best_params']['beta']:.3f}, alpha={bo_result['best_params']['alpha']:.3f}")
    print(f"  Best val score ({PRIMARY_OBJECTIVE}): {bo_result['best_score']:.4f}")
    print(f"  Configs evaluated: {len(bo_result['history'])}")
    print(f"  Test HMM edge F1: {bo_test['hmm']['mean_edge_f1']:.4f}")
    print(f"  Test HMM RMF: {bo_test['hmm']['mean_rmf']:.4f}")
    print(f"  Test HMM point accuracy: {bo_test['hmm']['mean_point_accuracy']:.4f}")

    print(f"\nComparison:")
    print(f"  BO val score improvement: {comparison['comparison']['val_score_improvement']:+.4f}")
    print(f"  BO better on val: {comparison['comparison']['bo_better_on_val']}")
    print(f"  BO better on test edge F1: {comparison['comparison']['bo_better_on_test_f1']}")
    print(f"  Test edge F1 diff: {comparison['comparison']['test_edge_f1_diff']:+.4f}")
    print(f"  Test RMF diff: {comparison['comparison']['test_rmf_diff']:+.4f}")
    print(f"  Test point acc diff: {comparison['comparison']['test_point_acc_diff']:+.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
