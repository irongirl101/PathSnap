"""β/α tuning, ablations, and held-out evaluation for PathSnap Member C.

Runs on training data only (80 trips). Held-out student_test routes are
never consulted during configuration selection.

Usage:
    # From PathSnap repo root:
    $env:PYTHONPATH="src"; python scripts/tune.py

Outputs (under artifacts/):
    tuning/split.json          -- train/val trip IDs and seed
    tuning/grid_results.json   -- full (beta, alpha) grid scores
    tuning/best_config.json    -- selected (beta, alpha)
    tuning/tuned.yaml          -- MatchConfig with tuned beta/alpha
    ablations/ablations.json   -- ablation results per config
    sensitivity/sensitivity_grid.json -- compact (beta, alpha) heatmap data
    evaluation/test_metrics.json -- held-out test trip metrics (HMM + classifier)
    evaluation/test_summary.json -- aggregate test metrics
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure src/pathsnap is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.config import MatchConfig, RoutingConfig, EmissionConfig, save_config
from pathsnap.io import list_trips, load_trip, build_graph, trip_dir
from pathsnap.candidates import generate_candidates
from pathsnap.hmm import match_trip, classifier_only
from pathsnap.evaluate import evaluate_match, aggregate
from pathsnap.pipeline import (
    train_val_split,
    tune_beta_alpha,
    run_ablations,
    sensitivity_grid,
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

# Tuning grid (compact)
BETA_GRID = [25.0, 50.0, 100.0, 200.0, 400.0]
ALPHA_GRID = [0.25, 0.5, 1.0, 2.0]

PRIMARY_OBJECTIVE = "edge_f1"
VAL_FRACTION = 0.25
SEED = 42
REPRESENTATION = "stitched"


def main() -> None:
    train_trip_ids = list_trips(str(TRAIN_ROOT))
    print(f"Found {len(train_trip_ids)} training trips")

    # --- Trip-level train/validation split (no point leakage) ---
    train_ids, val_ids = train_val_split(
        train_trip_ids, val_fraction=VAL_FRACTION, seed=SEED
    )
    print(f"Train: {len(train_ids)} trips, Val: {len(val_ids)} trips")

    (ARTIFACTS / "tuning").mkdir(parents=True, exist_ok=True)
    split_info = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "note": "All points from a trip stay in one split (no leakage).",
    }
    with open(ARTIFACTS / "tuning" / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("Saved tuning/split.json")

    # --- β/α grid search on validation trips ---
    base_config = MatchConfig()
    tuning = tune_beta_alpha(
        train_root=str(TRAIN_ROOT),
        train_ids=train_ids,
        val_ids=val_ids,
        beta_grid=BETA_GRID,
        alpha_grid=ALPHA_GRID,
        base_config=base_config,
        representation=REPRESENTATION,
        primary=PRIMARY_OBJECTIVE,
    )
    with open(ARTIFACTS / "tuning" / "grid_results.json", "w") as f:
        json.dump(tuning, f, indent=2, default=str)
    print("Saved tuning/grid_results.json")
    print(f"Best params: beta={tuning['best_params']['beta']}, alpha={tuning['best_params']['alpha']}, "
          f"val_score={tuning['best_score']:.4f}")

    # --- Save tuned config ---
    best_beta = tuning["best_params"]["beta"]
    best_alpha = tuning["best_params"]["alpha"]
    tuned_cfg = MatchConfig()
    tuned_cfg.routing = RoutingConfig(
        beta=best_beta,
        max_route_distance_m=base_config.routing.max_route_distance_m,
        directed=base_config.routing.directed,
        plausibility_factor=base_config.routing.plausibility_factor,
        min_route_gap_m=base_config.routing.min_route_gap_m,
    )
    tuned_cfg.emission = EmissionConfig(
        alpha=best_alpha,
        emission_floor=base_config.emission.emission_floor,
        validate_nonnegative=base_config.emission.validate_nonnegative,
    )
    tuned_cfg.gap = base_config.gap
    tuned_cfg.stitching = base_config.stitching
    tuned_cfg.tie_break = base_config.tie_break
    tuned_cfg.seed = SEED
    save_config(tuned_cfg, str(ARTIFACTS / "tuning" / "tuned.yaml"))
    # Also write a plain config/default.yaml for the repo
    save_config(tuned_cfg, str(REPO_ROOT / "config" / "default.yaml"))
    print(f"Saved tuning/tuned.yaml and config/default.yaml (beta={best_beta}, alpha={best_alpha})")

    # --- Ablations on all training trips ---
    ablation_results = run_ablations(
        train_root=str(TRAIN_ROOT),
        trip_ids=train_trip_ids,
        base_config=tuned_cfg,
        representation=REPRESENTATION,
    )
    (ARTIFACTS / "ablations").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "ablations" / "ablations.json", "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    print("Saved ablations/ablations.json")
    for ab in ablation_results:
        agg = ab["aggregate"]
        print(f"  {ab['ablation']}: edge_f1={agg.get('mean_edge_f1', 'n/a'):.4f}, "
              f"rmf={agg.get('mean_rmf', 'n/a'):.4f}, "
              f"point_acc={agg.get('mean_point_accuracy', 'n/a'):.4f}")

    # --- Sensitivity grid on validation trips ---
    sens = sensitivity_grid(
        train_root=str(TRAIN_ROOT),
        val_ids=val_ids,
        beta_grid=BETA_GRID,
        alpha_grid=ALPHA_GRID,
        base_config=tuned_cfg,
        representation=REPRESENTATION,
        primary=PRIMARY_OBJECTIVE,
    )
    (ARTIFACTS / "sensitivity").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "sensitivity" / "sensitivity_grid.json", "w") as f:
        json.dump(sens, f, indent=2, default=str)
    print("Saved sensitivity/sensitivity_grid.json")

    # --- Held-out test evaluation (routes untouched until now) ---
    test_trip_ids = list_trips(str(TEST_ROOT))
    print(f"\nFound {len(test_trip_ids)} held-out test trips")
    print("Evaluating HMM (tuned) and classifier-only on held-out test trips...")

    test_hmm_metrics = []
    test_clf_metrics = []
    for tid in test_trip_ids:
        trip = load_trip(tid, str(trip_dir(TEST_ROOT, tid)))
        graph = build_graph(
            trip,
            directed=tuned_cfg.routing.directed,
            max_route_distance_m=tuned_cfg.routing.max_route_distance_m,
        )
        points = [(float(t[0]), float(t[1]), int(t[2])) for t in trip.track]
        cands = generate_candidates(trip.trip_id, points, graph)

        hmm_res = match_trip(trip.trip_id, points, cands, graph, tuned_cfg)
        clf_res = classifier_only(trip.trip_id, points, cands)

        test_hmm_metrics.append(
            evaluate_match(hmm_res, trip, graph, tuned_cfg, REPRESENTATION)
        )
        test_clf_metrics.append(
            evaluate_match(clf_res, trip, graph, tuned_cfg, REPRESENTATION)
        )

    hmm_agg = aggregate(test_hmm_metrics)
    clf_agg = aggregate(test_clf_metrics)

    test_results = {
        "representation": REPRESENTATION,
        "config": {
            "beta": best_beta,
            "alpha": best_alpha,
        },
        "n_trips": len(test_trip_ids),
        "trip_ids": test_trip_ids,
        "hmm": {
            "aggregate": hmm_agg.__dict__ if hasattr(hmm_agg, "__dict__") else hmm_agg,
            "per_trip": [m.__dict__ for m in test_hmm_metrics],
        },
        "classifier_only": {
            "aggregate": clf_agg.__dict__ if hasattr(clf_agg, "__dict__") else clf_agg,
            "per_trip": [m.__dict__ for m in test_clf_metrics],
        },
    }

    # Make aggregate JSON-serializable
    def clean_agg(agg):
        return {
            "n_trips": agg.n_trips,
            "n_evaluable_trips": agg.n_evaluable_trips,
            "total_points": agg.total_points,
            "total_matched_points": agg.total_matched_points,
            "mean_coverage": agg.mean_coverage,
            "mean_rmf": agg.mean_rmf,
            "mean_edge_precision": agg.mean_edge_precision,
            "mean_edge_recall": agg.mean_edge_recall,
            "mean_edge_f1": agg.mean_edge_f1,
            "mean_point_accuracy": agg.mean_point_accuracy,
            "micro_edge_precision": agg.micro_edge_precision,
            "micro_edge_recall": agg.micro_edge_recall,
            "micro_edge_f1": agg.micro_edge_f1,
        }

    test_results["hmm"]["aggregate"] = clean_agg(hmm_agg)
    test_results["classifier_only"]["aggregate"] = clean_agg(clf_agg)

    (ARTIFACTS / "evaluation").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "evaluation" / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    print("Saved evaluation/test_metrics.json")

    summary = {
        "n_trips": len(test_trip_ids),
        "representation": REPRESENTATION,
        "config": {"beta": best_beta, "alpha": best_alpha},
        "hmm": clean_agg(hmm_agg),
        "classifier_only": clean_agg(clf_agg),
        "improvement": {
            "edge_f1": hmm_agg.mean_edge_f1 - clf_agg.mean_edge_f1,
            "rmf": clf_agg.mean_rmf - hmm_agg.mean_rmf,  # lower RMF is better
            "point_accuracy": hmm_agg.mean_point_accuracy - clf_agg.mean_point_accuracy,
            "coverage": hmm_agg.mean_coverage - clf_agg.mean_coverage,
        },
    }
    with open(ARTIFACTS / "evaluation" / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved evaluation/test_summary.json")
    print(f"\nHMM: edge_f1={hmm_agg.mean_edge_f1:.4f}, rmf={hmm_agg.mean_rmf:.4f}, "
          f"point_acc={hmm_agg.mean_point_accuracy:.4f}")
    print(f"CLF: edge_f1={clf_agg.mean_edge_f1:.4f}, rmf={clf_agg.mean_rmf:.4f}, "
          f"point_acc={clf_agg.mean_point_accuracy:.4f}")
    print(f"Improvement: edge_f1={summary['improvement']['edge_f1']:+.4f}, "
          f"rmf={summary['improvement']['rmf']:+.4f}, "
          f"point_acc={summary['improvement']['point_accuracy']:+.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
