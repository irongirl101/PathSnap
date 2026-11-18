"""Compare models with trip-grouped CV on training trips and the supplied student-test trips.

Run after generating the two feature tables:
    python -m src.features.build_dataset data/raw/train
    python -m src.features.build_dataset data/raw/student_test \
        --out data/processed/student_test_candidates.parquet \
        --summary-path reports/student_test/trip_summary.csv
    python -m src.models.evaluate_student_test
"""
import argparse
import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from src.features.features import FEATURE_COLS
from src.api.trips import TripStore
from src.pathsnap.api_adapter import (
    _classifier_candidates,
    _load_config_from_env,
    _load_network,
    _request_points,
)
from src.pathsnap.evaluate import evaluate_match
from src.pathsnap.graph import Graph
from src.pathsnap.hmm import classifier_only, match_trip
from src.pathsnap.io import Trip
from src.models.train_models import (
    EPS,
    INFO,
    argmax_pred,
    fit_model,
    get_models,
    labelled,
    metrics,
    score_model,
)
from src.models.wrapper import PathSnapModel


def _raw_point_counts(root):
    root = pathlib.Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {root}")
    counts = {}
    for trip_dir in sorted(root.iterdir()):
        if not trip_dir.is_dir():
            continue
        with (trip_dir / f"{trip_dir.name}.track").open(encoding="utf-8") as track:
            counts[trip_dir.name] = sum(1 for line in track if line.strip())
    return counts


def _load_candidates(path):
    frame = pd.read_parquet(path, columns=["trip_id", "point_idx", "arc_id", "label"] + FEATURE_COLS)
    frame["trip_id"] = frame["trip_id"].astype(str)
    for column in frame.select_dtypes(include="float64").columns:
        frame[column] = frame[column].astype("float32")
    return frame.reset_index(drop=True)


def _write_data_quality(train_root, test_root, output_dir, train, test):
    from collections import Counter

    from part_a.data import clean_track

    records = []
    for partition, root in (("train", train_root), ("student_test", test_root)):
        totals = Counter()
        trips = [
            path for path in sorted(pathlib.Path(root).iterdir())
            if path.is_dir() and not path.name.startswith(".")
        ]
        for trip_dir in trips:
            track_path = trip_dir / f"{trip_dir.name}.track"
            _, counts = clean_track(track_path)
            totals.update(counts)
        records.append({
            "partition": partition,
            "trips": len(trips),
            **{key: int(totals[key]) for key in (
                "raw_points", "clean_points", "missing_rows", "malformed_rows",
                "nonfinite_rows", "invalid_coordinates", "raw_time_reversals",
                "duplicate_observations", "duplicate_timestamps", "gaps",
            )},
        })
    pd.DataFrame(records).to_csv(pathlib.Path(output_dir) / "data_quality_summary.csv", index=False)
    missingness = []
    for name in FEATURE_COLS:
        train_missing = int(train[name].isna().sum())
        test_missing = int(test[name].isna().sum())
        missingness.append({
            "feature": name,
            "train_missing": train_missing,
            "train_missing_rate": train_missing / len(train),
            "student_test_missing": test_missing,
            "student_test_missing_rate": test_missing / len(test),
        })
    pd.DataFrame(missingness).to_csv(
        pathlib.Path(output_dir) / "feature_missingness.csv", index=False
    )


def evaluate(train_path, test_path, train_root, test_root, output_dir, folds=5,
             selected_model_path="artifacts/final_model.joblib", tuned_params_path=None):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_all = _load_candidates(train_path)
    test_all = _load_candidates(test_path)
    train = labelled(train_all).reset_index(drop=True)
    test = labelled(test_all).reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Both training and student-test data must contain labelled candidates")

    raw_train_counts = _raw_point_counts(train_root)
    raw_test_counts = _raw_point_counts(test_root)
    _write_data_quality(train_root, test_root, out, train_all, test_all)
    train_ids, test_ids = set(raw_train_counts), set(raw_test_counts)
    if train_ids & test_ids:
        raise ValueError("Training and student-test trip IDs overlap")
    if set(train.trip_id.unique()) - train_ids or set(test.trip_id.unique()) - test_ids:
        raise ValueError("Feature tables contain trip IDs absent from their raw partitions")

    splitter = GroupKFold(n_splits=folds)
    cv_folds = list(splitter.split(train, train["label"], train["trip_id"]))
    model_specs = get_models()
    if set(model_specs) != set(INFO) - {"lgbm_tuned"}:
        raise ValueError("Model definitions and report metadata do not match")
    tuned_params = None
    if tuned_params_path:
        params_record = json.loads(pathlib.Path(tuned_params_path).read_text())
        tuned_params = params_record["params"]
        from src.models.bayes import build_model

        model_specs["lgbm_tuned"] = build_model("lgbm", tuned_params)

    fold_rows, test_rows, confusion_rows, oof_by_model = [], [], [], {}
    mlp_history = None
    for name, prototype in model_specs.items():
        oof = np.full(len(train), np.nan, dtype=float)
        fold_inference = []
        started = time.perf_counter()
        for fold, (fit_idx, valid_idx) in enumerate(cv_folds):
            model = fit_model(name, clone(prototype), train.iloc[fit_idx])
            if name == "mlp" and fold == 0:
                estimator = model.named_steps["mlpclassifier"]
                mlp_history = {
                    "epoch": np.arange(1, len(estimator.loss_curve_) + 1),
                    "training_loss": np.asarray(estimator.loss_curve_, dtype=float),
                    "validation_accuracy": np.asarray(estimator.validation_scores_, dtype=float),
                }
            valid = train.iloc[valid_idx]
            t0 = time.perf_counter()
            scores = score_model(name, model, valid)
            fold_inference.append((time.perf_counter() - t0) * 1000 / len(valid) * 1000)
            oof[valid_idx] = scores
            result = metrics(valid, scores)
            raw_points = sum(raw_train_counts[t] for t in valid.trip_id.unique())
            fold_rows.append({
                "model": name,
                "fold": fold + 1,
                **{key: value for key, value in result.items() if key != "hits"},
                "top1_e2e": result["hits"] / raw_points,
                "inference_ms_per_1k_rows": fold_inference[-1],
            })
        if not np.isfinite(oof).all():
            raise RuntimeError(f"Cross-validation did not score every row for {name}")

        model = fit_model(name, clone(prototype), train)
        t0 = time.perf_counter()
        test_scores = score_model(name, model, test)
        inference_ms = (time.perf_counter() - t0) * 1000 / len(test) * 1000
        test_result = metrics(test, test_scores)
        raw_test_points = sum(raw_test_counts.values())
        test_rows.append({
            "model": name,
            "algorithm_family": INFO[name][0],
            "configuration": INFO[name][1],
            **{key: value for key, value in test_result.items() if key != "hits"},
            "top1_e2e": test_result["hits"] / raw_test_points,
            "inference_ms_per_1k_rows": inference_ms,
            "candidate_rows": len(test),
            "labelled_points": test.groupby(["trip_id", "point_idx"]).ngroups,
            "raw_points": raw_test_points,
        })
        tn, fp, fn, tp = confusion_matrix(
            test["label"], argmax_pred(test, test_scores), labels=[0, 1]
        ).ravel()
        confusion_rows.append(dict(model=name, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
        oof_by_model[name] = oof
        print(f"{name:11s} CV macro F1 {np.mean([r['macro_f1'] for r in fold_rows if r['model'] == name]):.3f} | "
              f"student-test Top-1 {test_result['top1']:.3f} "
              f"(end-to-end {test_result['hits'] / raw_test_points:.3f}) | "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    cv_frame = pd.DataFrame(fold_rows)
    cv_summary = cv_frame.groupby("model").agg(["mean", "std"])
    cv_summary.columns = [f"{metric}_{stat}" for metric, stat in cv_summary.columns]
    test_frame = pd.DataFrame(test_rows).set_index("model")
    table = pd.DataFrame({
        "Model": [f"{i + 1}. {name}" for i, name in enumerate(test_frame.index)],
        "Algorithm family": [test_frame.loc[name, "algorithm_family"] for name in test_frame.index],
        "Key configuration": [test_frame.loc[name, "configuration"] for name in test_frame.index],
        "CV macro F1": [
            f"{cv_summary.loc[name, 'macro_f1_mean']:.3f} ± {cv_summary.loc[name, 'macro_f1_std']:.3f}"
            for name in test_frame.index
        ],
        "CV Top-1 (all raw points)": [
            f"{cv_summary.loc[name, 'top1_e2e_mean']:.3f} ± {cv_summary.loc[name, 'top1_e2e_std']:.3f}"
            for name in test_frame.index
        ],
        "Student-test Top-1": [f"{test_frame.loc[name, 'top1']:.3f}" for name in test_frame.index],
        "Student-test end-to-end Top-1": [
            f"{test_frame.loc[name, 'top1_e2e']:.3f}" for name in test_frame.index
        ],
        "Student-test macro F1": [
            f"{test_frame.loc[name, 'macro_f1']:.3f}" for name in test_frame.index
        ],
        "Student-test log loss": [
            f"{test_frame.loc[name, 'log_loss']:.4f}" for name in test_frame.index
        ],
        "Inference ms / 1k rows": [
            f"{test_frame.loc[name, 'inference_ms_per_1k_rows']:.2f}" for name in test_frame.index
        ],
    })
    cv_frame.to_csv(out / "cv_results_per_fold.csv", index=False)
    test_frame.to_csv(out / "student_test_results.csv")
    table.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "student_test_confusion_matrices.csv", index=False)
    pd.concat(
        [train[["trip_id", "point_idx", "arc_id", "label"]].reset_index(drop=True),
         pd.DataFrame(oof_by_model)],
        axis=1,
    ).to_parquet(out / "cv_oof_predictions.parquet", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    names = list(test_frame.index)
    cv_means = [cv_summary.loc[name, "macro_f1_mean"] for name in names]
    axes[0].bar(names, cv_means, color="#35618f")
    axes[0].set_title("5-fold CV macro F1")
    x = np.arange(len(names))
    width = .25
    axes[1].bar(x - width, [test_frame.loc[name, "top1"] for name in names],
                width, label="Candidate Top-1")
    axes[1].bar(x, [test_frame.loc[name, "macro_f1"] for name in names],
                width, label="Macro F1")
    axes[1].bar(x + width, [test_frame.loc[name, "top1_e2e"] for name in names],
                width, label="End-to-end Top-1")
    axes[1].set_xticks(x, names, rotation=35, ha="right")
    axes[1].set_title("Actual student-test results")
    axes[1].legend()
    axes[2].bar(names, [test_frame.loc[name, "log_loss"] for name in names], color="#5a8d70")
    axes[2].set_title("Student-test log loss")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=.2)
    fig.savefig(out / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    if mlp_history is not None:
        pd.DataFrame(mlp_history).to_csv(out / "mlp_training_history.csv", index=False)
        history_fig, history_axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        history_axes[0].plot(mlp_history["epoch"], mlp_history["training_loss"])
        history_axes[0].set(xlabel="Epoch", ylabel="Training loss", title="MLP training loss")
        history_axes[1].plot(mlp_history["epoch"], mlp_history["validation_accuracy"])
        history_axes[1].set(xlabel="Epoch", ylabel="Validation accuracy",
                            title="MLP early-stopping validation")
        history_fig.savefig(out / "mlp_training_history.png", dpi=160, bbox_inches="tight")
        plt.close(history_fig)

    # Select only by training-partition CV, never by the student-test metrics.
    selected = cv_summary["macro_f1_mean"].idxmax()
    selected_prototype = clone(model_specs[selected])
    selected_model = fit_model(selected, selected_prototype, train)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    calibrator.fit(oof_by_model[selected], train["label"].to_numpy())
    fitted = PathSnapModel(
        selected_model,
        calibrator,
        FEATURE_COLS,
        kind="ranker" if selected == "lgbm_ranker" else "classifier",
        meta={
            "model": selected,
            "selection_metric": "training trip-grouped CV macro F1",
            "training_trips": sorted(train_ids),
            "student_test_trips": sorted(test_ids),
            "feature_contract": "artifacts/feature_columns.json",
            "params": tuned_params if selected == "lgbm_tuned" else None,
        },
    )
    pathlib.Path(selected_model_path).parent.mkdir(parents=True, exist_ok=True)
    fitted.save(selected_model_path)
    route_comparison = _evaluate_routes(fitted, test_root, test_ids, out)
    summary = {
        "training_partition": {"trips": len(train_ids), "raw_points": sum(raw_train_counts.values())},
        "student_test_partition": {
            "trips": len(test_ids),
            "raw_points": raw_test_points,
            "candidate_rows": len(test),
            "labelled_points": test.groupby(["trip_id", "point_idx"]).ngroups,
        },
        "selected_model": selected,
        "selection_metric": "mean five-fold trip-grouped CV macro F1",
        "artifact": selected_model_path,
        "tuned_params": tuned_params,
        "route_comparison": route_comparison,
    }
    (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Selected {selected} by training CV macro F1; saved deployable artifact to {selected_model_path}")
    return summary


def _evaluate_routes(model, test_root, trip_ids, output_dir):
    """Compare independent candidate picks with HMM/Viterbi on identical test trips."""
    store = TripStore(test_root)
    config = _load_config_from_env()
    rows = []
    for trip_id in sorted(trip_ids):
        data = store.load_prediction_input(trip_id).as_dict()
        points = _request_points(data)
        nodes, arcs = _load_network(data, trip_id)
        graph = Graph(nodes, arcs, directed=config.routing.directed,
                      max_route_distance_m=config.routing.max_route_distance_m)
        candidates = _classifier_candidates(trip_id, points, data, graph, model)
        independent = classifier_only(trip_id, points, candidates)
        decoded = match_trip(trip_id, points, candidates, graph, config)
        route = np.asarray(store.read_reference_route(trip_id), dtype=np.int64)
        track = np.asarray([
            [point["lon"], point["lat"], point.get("timestamp_s", index)]
            for index, point in enumerate(data["points"])
        ], dtype=float)
        trip = Trip(trip_id=trip_id, track=track, route=route, nodes=nodes, arcs=arcs)
        for method, result in (("classifier_only", independent), ("hmm_viterbi", decoded)):
            metrics_result = evaluate_match(result, trip, graph, config, representation="stitched")
            predicted_route = [edge.edge_id for edge in result.stitched_route]
            connected = [
                arcs[left, 1] == arcs[right, 0]
                for left, right in zip(predicted_route[:-1], predicted_route[1:])
            ]
            rows.append({
                "trip_id": trip_id,
                "method": method,
                "point_accuracy": metrics_result.point_accuracy,
                "coverage": metrics_result.coverage,
                "rmf": metrics_result.rmf,
                "edge_precision": metrics_result.edge_precision,
                "edge_recall": metrics_result.edge_recall,
                "edge_f1": metrics_result.edge_f1,
                "route_continuity": float(np.mean(connected)) if connected else 1.0,
                "disconnected_transitions": int(np.count_nonzero(~np.asarray(connected, dtype=bool))),
                "predicted_edges": metrics_result.pred_edges,
                "reference_edges": metrics_result.ref_edges,
            })
        print(f"route evaluation {trip_id}: {len(points)} observations", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(pathlib.Path(output_dir) / "route_metrics_by_trip.csv", index=False)
    summary = frame.groupby("method").mean(numeric_only=True).to_dict(orient="index")
    (pathlib.Path(output_dir) / "route_metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/processed/candidates.parquet")
    parser.add_argument("--test-path", default="data/processed/student_test_candidates.parquet")
    parser.add_argument("--train-root", default="data/raw/train")
    parser.add_argument("--test-root", default="data/raw/student_test")
    parser.add_argument("--output-dir", default="reports/student_test")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-path", default="artifacts/final_model.joblib")
    parser.add_argument("--tuned-params", help="JSON emitted by src.models.bayes")
    args = parser.parse_args()
    evaluate(args.train_path, args.test_path, args.train_root, args.test_root,
             args.output_dir, args.folds, args.model_path, args.tuned_params)
