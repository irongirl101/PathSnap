"""Compare classifiers with trip-grouped CV on data/processed/candidates.parquet.

usage: python -m src.models.train_models --exclude 00000054 [--frac 0.3] [--models logreg svm lgbm]
Writes reports/results.csv (mean/std over folds), reports/results_per_fold.csv, reports/oof_predictions.parquet
"""
import argparse, pathlib, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
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
    """Share of points whose highest-scored candidate is the labelled arc."""
    t = d[["trip_id", "point_idx", "label"]].reset_index(drop=True).assign(p=np.asarray(p))
    best = t.loc[t.groupby(["trip_id", "point_idx"])["p"].idxmax()]
    return best["label"].mean()


def metrics(d, p):
    p = np.clip(p, EPS, 1 - EPS)
    y = d["label"].to_numpy()
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                log_loss=log_loss(y, p), brier=brier_score_loss(y, p), top1=top1(d, p))


def load(path, exclude, frac):
    df = pd.read_parquet(path)
    df["trip_id"] = df["trip_id"].astype(str)
    df = df[~df["trip_id"].isin(exclude)]
    has_pos = df.groupby(["trip_id", "point_idx"])["label"].transform("max") == 1
    df = df[has_pos]
    if frac < 1:                                   # subsample whole points, keeping all their candidates
        pts = df[["trip_id", "point_idx"]].drop_duplicates()
        pts = pts[np.random.default_rng(SEED).random(len(pts)) < frac]
        df = df.merge(pts, on=["trip_id", "point_idx"])
    return df.reset_index(drop=True)


def main(path, exclude, frac, splits, names):
    df = load(path, exclude, frac)
    print(f"trips {df.trip_id.nunique()} | points {df.groupby(['trip_id','point_idx']).ngroups} | "
          f"rows {len(df)} | positive rate {df.label.mean():.3f}")
    X, y, g = df[FEATURE_COLS], df["label"], df["trip_id"]
    folds = list(GroupKFold(splits).split(X, y, g))
    models = {k: v for k, v in get_models().items() if k in names}
    rows, oof = [], {}
    for name, model in models.items():
        p = np.zeros(len(df)); cal = np.zeros(len(df))
        t0 = time.time()
        for k, (tr, va) in enumerate(folds):
            model.fit(X.iloc[tr], y.iloc[tr])
            p[va] = model.predict_proba(X.iloc[va])[:, 1]
        fit_s = time.time() - t0
        for k, (tr, va) in enumerate(folds):          # cross-fitted isotonic calibration of the OOF scores
            iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p[tr], y.iloc[tr])
            cal[va] = iso.predict(p[va])
        for k, (tr, va) in enumerate(folds):
            m = metrics(df.iloc[va], p[va])
            m_cal = metrics(df.iloc[va], cal[va])
            rows.append(dict(model=name, fold=k, **m, log_loss_cal=m_cal["log_loss"],
                             brier_cal=m_cal["brier"], top1_cal=m_cal["top1"]))
        oof[name], oof[name + "_cal"] = p, cal
        print(f"{name:7s} done in {fit_s:.0f}s")
    per_fold = pd.DataFrame(rows)
    cols = [c for c in per_fold.columns if c not in ("model", "fold")]
    res = per_fold.groupby("model")[cols].agg(["mean", "std"])
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    out = pathlib.Path("reports"); out.mkdir(exist_ok=True)
    per_fold.to_csv(out / "results_per_fold.csv", index=False)
    res.to_csv(out / "results.csv")
    pd.concat([df[["trip_id", "point_idx", "arc_id", "label"]], pd.DataFrame(oof)], axis=1) \
        .to_parquet(out / "oof_predictions.parquet", index=False)
    show = res[[c for c in res.columns if c.endswith("_mean")]].round(4)
    print(show.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/processed/candidates.parquet")
    ap.add_argument("--exclude", nargs="*", default=["00000054"])
    ap.add_argument("--frac", type=float, default=1.0, help="share of points to use (speeds up experiments)")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--models", nargs="*", default=["logreg", "rf", "lgbm", "mlp", "svm"])
    a = ap.parse_args()
    main(a.path, a.exclude, a.frac, a.splits, a.models)