"""Compare candidate scorers with trip-grouped CV on candidates.parquet.

usage: python -m src.models.train_models --exclude 00000054 [--models lgbm_ranker]
       [--save-model artifacts/final_model.joblib]
Writes reports/results.csv (mean/std over folds), reports/results_per_fold.csv, reports/oof_predictions.parquet
"""
import argparse, pathlib, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from src.features.features import FEATURE_COLS

SEED = 42
EPS = 1e-6


def get_models():
    # No class weighting on purpose: weighting inflates probabilities, and the HMM needs calibrated ones.
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
    return {
        "logreg": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "rf": make_pipeline(imp(), RandomForestClassifier(150, max_depth=16, min_samples_leaf=20,
                                                          n_jobs=-1, random_state=SEED)),
        "lgbm": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=SEED, verbose=-1),
        "lgbm_ranker": lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", n_estimators=500, learning_rate=0.05,
            num_leaves=31, min_child_samples=30, reg_lambda=1.0, random_state=SEED, verbosity=-1,
        ),
        "mlp": make_pipeline(imp(), StandardScaler(),
                             MLPClassifier((64, 32), early_stopping=True, max_iter=60, random_state=SEED)),
        "svm": make_pipeline(
            imp(), StandardScaler(),
            CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
                method="sigmoid", cv=3,
            ),
        ),
    }


def top1(d, p):
    """Share of points with candidates whose top-ranked candidate is labelled."""
    t = d[["trip_id", "point_idx", "label"]].reset_index(drop=True).assign(p=np.asarray(p))
    best = t.loc[t.groupby(["trip_id", "point_idx"])["p"].idxmax()]
    return best["label"].mean()


def top1_hits(d, p):
    """Count points whose top-ranked candidate is labelled."""
    t = d[["trip_id", "point_idx", "label"]].reset_index(drop=True).assign(p=np.asarray(p))
    best = t.loc[t.groupby(["trip_id", "point_idx"])["p"].idxmax()]
    return int(best["label"].sum())


def point_softmax(d, scores):
    """Convert per-point ranking scores into candidate probabilities."""
    groups = [d["trip_id"], d["point_idx"]]
    score = pd.Series(np.asarray(scores), index=d.index)
    shifted = score - score.groupby(groups).transform("max")
    exp_score = np.exp(shifted)
    return (exp_score / exp_score.groupby(groups).transform("sum")).to_numpy()


def metrics(d, p):
    p = np.clip(p, EPS, 1 - EPS)
    y = d["label"].to_numpy()
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                log_loss=log_loss(y, p), brier=brier_score_loss(y, p), top1=top1(d, p))


def load(path, exclude, frac, raw_root):
    if not 0 < frac <= 1:
        raise ValueError("frac must be greater than 0 and at most 1")
    df = pd.read_parquet(path)
    df["trip_id"] = df["trip_id"].astype(str)
    df = df[~df["trip_id"].isin(exclude)]
    raw_root = pathlib.Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw trip directory not found: {raw_root}")

    point_counts, selected_points = {}, []
    rng = np.random.default_rng(SEED)
    for trip_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        if trip_dir.name in exclude:
            continue
        track_path = trip_dir / f"{trip_dir.name}.track"
        if not track_path.is_file():
            raise FileNotFoundError(f"Track file not found: {track_path}")
        with track_path.open() as track_file:
            n_points = sum(1 for line in track_file if line.strip())
        indices = np.arange(n_points)
        if frac < 1:
            indices = indices[rng.random(n_points) < frac]
        point_counts[trip_dir.name] = len(indices)
        if len(indices):
            selected_points.append(pd.DataFrame({"trip_id": trip_dir.name, "point_idx": indices}))

    if selected_points:
        selected = pd.concat(selected_points, ignore_index=True)
        df = df.merge(selected, on=["trip_id", "point_idx"], how="inner", validate="many_to_one")
    else:
        df = df.iloc[0:0]
    if df.empty:
        raise ValueError("No candidate rows remain after applying the raw-trip point selection")
    return df.reset_index(drop=True), point_counts


def main(path, exclude, frac, splits, names, raw_root, output_dir="reports",
         save_model=None, save_model_name="lgbm_ranker"):
    df, point_counts = load(path, exclude, frac, raw_root)
    print(f"trips {df.trip_id.nunique()} | points {df.groupby(['trip_id','point_idx']).ngroups} | "
          f"raw points {sum(point_counts.values())} | rows {len(df)} | positive rate {df.label.mean():.3f}")
    X, y, g = df[FEATURE_COLS], df["label"], df["trip_id"]
    folds = list(GroupKFold(splits).split(X, y, g))
    fold_trip_ids = [set(df.iloc[va]["trip_id"].unique()) for _, va in folds]
    for trip_id in set(point_counts) - set.union(*fold_trip_ids):
        fold = min(
            range(len(fold_trip_ids)),
            key=lambda i: sum(point_counts[tid] for tid in fold_trip_ids[i]),
        )
        fold_trip_ids[fold].add(trip_id)
    models = {k: v for k, v in get_models().items() if k in names}
    rows, oof = [], {}
    for name, model in models.items():
        p = np.zeros(len(df)); cal = np.zeros(len(df))
        t0 = time.time()
        for tr, va in folds:
            if name == "lgbm_ranker":
                train = df.iloc[tr].sort_values(["trip_id", "point_idx"], kind="mergesort")
                model.fit(
                    train[FEATURE_COLS],
                    train["label"],
                    group=train.groupby(["trip_id", "point_idx"], sort=False).size().to_numpy(),
                )
                p[va] = model.predict(X.iloc[va])
            else:
                model.fit(X.iloc[tr], y.iloc[tr])
                p[va] = model.predict_proba(X.iloc[va])[:, 1]
        fit_s = time.time() - t0
        if name == "lgbm_ranker":
            p = point_softmax(df, p)
        for k, (tr, va) in enumerate(folds):          # cross-fitted isotonic calibration of the OOF scores
            iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p[tr], y.iloc[tr])
            cal[va] = iso.predict(p[va])
        for k, (tr, va) in enumerate(folds):
            val_trip_ids = fold_trip_ids[k]
            val_point_count = sum(point_counts[tid] for tid in val_trip_ids)
            top1_correct = top1_hits(df.iloc[va], p[va])
            top1_correct_cal = top1_hits(df.iloc[va], cal[va])
            m = metrics(df.iloc[va], p[va])
            m_cal = metrics(df.iloc[va], cal[va])
            rows.append(dict(model=name, fold=k, **m, log_loss_cal=m_cal["log_loss"],
                             brier_cal=m_cal["brier"], top1_cal=m_cal["top1"],
                             top1_e2e=top1_correct / val_point_count,
                             top1_e2e_cal=top1_correct_cal / val_point_count))
        oof[name], oof[name + "_cal"] = p, cal
        print(f"{name:7s} done in {fit_s:.0f}s")
    per_fold = pd.DataFrame(rows)
    cols = [c for c in per_fold.columns if c not in ("model", "fold")]
    res = per_fold.groupby("model")[cols].agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    out = pathlib.Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out / "results_per_fold.csv", index=False)
    res.to_csv(out / "results.csv")
    pd.concat([df[["trip_id", "point_idx", "arc_id", "label"]], pd.DataFrame(oof)], axis=1) \
        .to_parquet(out / "oof_predictions.parquet", index=False)
    show = res[[c for c in res.columns if c.endswith("_mean")]].round(4)
    print(show.to_string())
    if save_model:
        if save_model_name not in get_models():
            raise ValueError(f"Cannot save unknown model: {save_model_name}")
        final_model = clone(get_models()[save_model_name])
        if save_model_name == "lgbm_ranker":
            training = df.sort_values(["trip_id", "point_idx"], kind="mergesort")
            final_model.fit(
                training[FEATURE_COLS],
                training["label"],
                group=training.groupby(["trip_id", "point_idx"], sort=False).size().to_numpy(),
            )
        else:
            final_model.fit(X, y)
        import joblib

        model_path = pathlib.Path(save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_path)
        print(f"saved fitted {save_model_name} model to {model_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/processed/candidates.parquet")
    ap.add_argument("--exclude", nargs="*", default=["00000054"])
    ap.add_argument("--frac", type=float, default=1.0, help="share of points to use (speeds up experiments)")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--raw-root", default="data/raw/train", help="raw trip directory used for point counts")
    ap.add_argument("--models", nargs="*", default=["logreg", "rf", "lgbm", "mlp", "svm", "lgbm_ranker"])
    ap.add_argument("--output-dir", default="reports", help="directory for CV results and OOF predictions")
    ap.add_argument("--save-model", help="optional path to save a final model fitted on all loaded trips")
    ap.add_argument("--save-model-name", default="lgbm_ranker", help="model name to fit and save")
    a = ap.parse_args()
    main(a.path, a.exclude, a.frac, a.splits, a.models, a.raw_root, a.output_dir,
         a.save_model, a.save_model_name)