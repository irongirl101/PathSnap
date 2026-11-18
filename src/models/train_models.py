"""Compare candidate scorers: trip-grouped CV + a fixed held-out trip set ("student test").

usage: python -m src.models.train_models --exclude 00000054 [--frac 0.5] [--models lgbm lgbm_ranker]

Evaluation conventions
  * Models are trained on GPS points whose true arc is among the candidates (all-zero points would
    teach the classifiers that real candidates are negatives).
  * Candidate-level metrics (AUC, log-loss, F1, top1) are computed on those same labelled points.
  * top1_e2e divides correct points by ALL raw GPS points of the evaluated trips, so points with no
    candidate / no labelled candidate count as misses (the honest end-to-end number).
  * macro F1: highest-scored candidate of each point = predicted 1, others 0; mean of F1 over {0,1}.

Writes reports/: results.csv (CV mean/std), results_per_fold.csv (incl. test row), model_table.csv
(layout of the brief), confusion_matrices.csv, oof_predictions.parquet, splits.json
"""
import argparse, json, pathlib, time
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
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss, brier_score_loss,
                             f1_score, confusion_matrix)
from src.features.features import FEATURE_COLS

SEED, EPS = 42, 1e-6
KEYS = ["trip_id", "point_idx"]
REPORTS = pathlib.Path("reports")

INFO = {   # (algorithm family, key configuration) for the report table
    "logreg": ("Linear (logistic regression)", "median impute + scale, L2, C=1"),
    "rf": ("Bagged trees (random forest)", "60 trees, depth 10, min leaf 50"),
    "lgbm": ("Gradient boosting (LightGBM classifier)", "400 trees, lr 0.05, 63 leaves"),
    "lgbm_ranker": ("Learning-to-rank (LightGBM LambdaRank)", "500 trees, lr 0.05, 31 leaves, per-point softmax"),
    "mlp": ("Neural network (MLP)", "64-32 ReLU, early stopping"),
    "svm": ("Linear SVM (LinearSVC + sigmoid calibration)", "C=1, 3-fold Platt calibration"),
}


def get_models():
    # No class weighting on purpose: weighting inflates probabilities, and the HMM needs calibrated ones.
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
    return {
        "logreg": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "rf": make_pipeline(imp(), RandomForestClassifier(60, max_depth=10, min_samples_leaf=50,
                                                          n_jobs=-1, random_state=SEED)),
        "lgbm": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
        "lgbm_ranker": lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=500,
                                      learning_rate=0.05, num_leaves=31, min_child_samples=30,
                                      reg_lambda=1.0, random_state=SEED, verbosity=-1),
        "mlp": make_pipeline(imp(), StandardScaler(),
                             MLPClassifier((64, 32), early_stopping=True, max_iter=60, random_state=SEED)),
        "svm": make_pipeline(imp(), StandardScaler(),
                             CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
                                                    method="sigmoid", cv=3)),
    }


def labelled(d):
    """Keep only GPS points whose true arc is among the candidates."""
    return d[d.groupby(KEYS)["label"].transform("max") == 1]


def point_softmax(d, scores):
    """Per-point ranking scores -> candidate probabilities (sum to 1 within a GPS point)."""
    g = [d["trip_id"].to_numpy(), d["point_idx"].to_numpy()]
    s = pd.Series(np.asarray(scores, float))
    e = np.exp(s - s.groupby(g).transform("max"))
    return (e / e.groupby(g).transform("sum")).to_numpy()


def fit_model(name, model, d):
    """Fit on the labelled points of d. Rankers additionally need the group sizes."""
    d = labelled(d)
    if name == "lgbm_ranker":
        d = d.sort_values(KEYS, kind="mergesort")
        model.fit(d[FEATURE_COLS], d["label"], group=d.groupby(KEYS, sort=False).size().to_numpy())
    else:
        model.fit(d[FEATURE_COLS], d["label"])
    return model


def score_model(name, model, d):
    if name == "lgbm_ranker":
        return point_softmax(d, model.predict(d[FEATURE_COLS]))
    return model.predict_proba(d[FEATURE_COLS])[:, 1]


def timed_score(name, model, d):
    t0 = time.perf_counter(); p = score_model(name, model, d)
    return p, (time.perf_counter() - t0) / len(d) * 1e6            # ms per 1000 candidate rows


def argmax_pred(d, p):
    t = pd.DataFrame({"t": d["trip_id"].to_numpy(), "p": d["point_idx"].to_numpy(), "s": np.asarray(p)})
    return (t.groupby(["t", "p"])["s"].rank(method="first", ascending=False) == 1).to_numpy()


def metrics(d, p):
    """d must contain only labelled points."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = d["label"].to_numpy()
    pred = argmax_pred(d, p)
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                log_loss=log_loss(y, p), brier=brier_score_loss(y, p),
                top1=float(y[pred].mean()), macro_f1=f1_score(y, pred, average="macro"),
                hits=int(y[pred].sum()))


def load(path, exclude, frac, raw_root):
    """Candidate rows plus the number of raw GPS points per trip (after optional point subsampling)."""
    if not 0 < frac <= 1:
        raise ValueError("frac must be greater than 0 and at most 1")
    # read only the columns we use and store features as float32: ~half the memory on 1.6M rows
    df = pd.read_parquet(path, columns=KEYS + ["arc_id", "label"] + FEATURE_COLS)
    f64 = df.select_dtypes("float64").columns
    df[f64] = df[f64].astype("float32")
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
        with track_path.open() as f:
            n_points = sum(1 for line in f if line.strip())
        indices = np.arange(n_points)
        if frac < 1:
            indices = indices[rng.random(n_points) < frac]
        point_counts[trip_dir.name] = len(indices)
        if len(indices):
            selected_points.append(pd.DataFrame({"trip_id": trip_dir.name, "point_idx": indices}))
    if selected_points:
        df = df.merge(pd.concat(selected_points, ignore_index=True), on=KEYS, how="inner", validate="many_to_one")
    else:
        df = df.iloc[0:0]
    if df.empty:
        raise ValueError("No candidate rows remain after applying the raw-trip point selection")
    return df.reset_index(drop=True), point_counts


def get_splits(trips, exclude, test_frac):
    """Fixed trip-level split ('cv' / 'test'), written once so the whole team uses the same one."""
    f = REPORTS / "splits.json"
    if f.exists():
        return json.loads(f.read_text())
    t = sorted(trips); np.random.default_rng(SEED).shuffle(t)
    n_test = max(1, round(test_frac * len(t)))
    sp = {"test": sorted(t[:n_test]), "cv": sorted(t[n_test:]), "excluded": list(exclude)}
    REPORTS.mkdir(exist_ok=True); f.write_text(json.dumps(sp, indent=1))
    return sp


def main(path, exclude, frac, cv_splits, names, raw_root, test_frac, output_dir, save_model, save_model_name):
    out = pathlib.Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df, pcount = load(path, exclude, frac, raw_root)
    sp = get_splits(sorted(pcount), exclude, test_frac)
    cv_ids, te_ids = set(sp["cv"]), set(sp["test"])
    cv = df[df.trip_id.isin(cv_ids)].reset_index(drop=True)
    te = df[df.trip_id.isin(te_ids)].reset_index(drop=True)
    pos_cv = (cv.groupby(KEYS)["label"].transform("max") == 1).to_numpy()
    pos_te = (te.groupby(KEYS)["label"].transform("max") == 1).to_numpy()
    y = cv["label"].to_numpy()
    n_te_raw = sum(pcount.get(t, 0) for t in te_ids)
    print(f"CV: {len(cv_ids)} trips | student test: {len(te_ids)} trips | rows cv {len(cv)} test {len(te)} | "
          f"labelled points cv {cv[pos_cv].groupby(KEYS).ngroups} test {te[pos_te].groupby(KEYS).ngroups} "
          f"of {n_te_raw} raw test points")

    folds = list(GroupKFold(cv_splits).split(cv, y, cv["trip_id"]))
    fold_ids = [set(cv.iloc[va]["trip_id"].unique()) for _, va in folds]
    for tid in sorted(cv_ids - set.union(*fold_ids)):          # trips with raw points but no candidates
        i = min(range(len(fold_ids)), key=lambda j: sum(pcount.get(t, 0) for t in fold_ids[j]))
        fold_ids[i].add(tid)

    rows, cms, oof = [], [], {}
    for name in [n for n in get_models() if n in names]:
        model = get_models()[name]
        p, cal, inf_ms, t0 = np.zeros(len(cv)), np.zeros(len(cv)), [], time.time()
        for tr, va in folds:
            fit_model(name, model, cv.iloc[tr])
            p[va], ms = timed_score(name, model, cv.iloc[va]); inf_ms.append(ms)
        for tr, va in folds:                                    # cross-fitted isotonic calibration
            trp = tr[pos_cv[tr]]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p[trp], y[trp])
            cal[va] = iso.predict(p[va])
        for k, (tr, va) in enumerate(folds):
            vp = va[pos_cv[va]]
            m, mc = metrics(cv.iloc[vp], p[vp]), metrics(cv.iloc[vp], cal[vp])
            n_raw = sum(pcount.get(t, 0) for t in fold_ids[k])
            rows.append(dict(model=name, fold=k, **{a: b for a, b in m.items() if a != "hits"},
                             log_loss_cal=mc["log_loss"], brier_cal=mc["brier"], top1_cal=mc["top1"],
                             top1_e2e=m["hits"] / n_raw, top1_e2e_cal=mc["hits"] / n_raw, infer_ms_per_1k=inf_ms[k]))
        tn, fp, fn, tp = confusion_matrix(y[pos_cv], argmax_pred(cv[pos_cv], p[pos_cv]), labels=[0, 1]).ravel()
        cms.append(dict(model=name, split="cv_oof", tn=tn, fp=fp, fn=fn, tp=tp))
        # refit on all CV trips, score the held-out student-test trips
        fit_model(name, model, cv)
        pt, ms = timed_score(name, model, te)
        mt = metrics(te[pos_te], pt[pos_te])
        tn, fp, fn, tp = confusion_matrix(te["label"][pos_te], argmax_pred(te[pos_te], pt[pos_te]), labels=[0, 1]).ravel()
        cms.append(dict(model=name, split="student_test", tn=tn, fp=fp, fn=fn, tp=tp))
        rows.append(dict(model=name, fold="test", **{a: b for a, b in mt.items() if a != "hits"},
                         top1_e2e=mt["hits"] / n_te_raw, infer_ms_per_1k=ms))
        oof[name], oof[name + "_cal"] = p, cal
        cvt = np.mean([r["top1"] for r in rows if r["model"] == name and r["fold"] != "test"])
        print(f"{name:11s} {time.time() - t0:5.0f}s | CV top1 {cvt:.3f} | test top1 {mt['top1']:.3f} "
              f"(e2e {mt['hits'] / n_te_raw:.3f}) macroF1 {mt['macro_f1']:.3f}")

    pf = pd.DataFrame(rows)
    cvr, ter = pf[pf.fold != "test"].drop(columns="fold"), pf[pf.fold == "test"].set_index("model")
    agg = cvr.groupby("model").agg(["mean", "std"]); agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    pf.to_csv(out / "results_per_fold.csv", index=False)
    agg.to_csv(out / "results.csv")
    pd.DataFrame(cms).to_csv(out / "confusion_matrices.csv", index=False)
    pd.concat([cv[KEYS + ["arc_id", "label"]], pd.DataFrame(oof)], axis=1).to_parquet(out / "oof_predictions.parquet", index=False)
    ms_ = lambda m, c: f"{agg.loc[m, c + '_mean']:.3f} ± {agg.loc[m, c + '_std']:.3f}"
    tab = pd.DataFrame({
        "Model": [f"Model {i + 1} ({m})" for i, m in enumerate(agg.index)],
        "Algorithm family": [INFO[m][0] for m in agg.index],
        "Key configuration": [INFO[m][1] for m in agg.index],
        "CV macro F1": [ms_(m, "macro_f1") for m in agg.index],
        "Top-1 accuracy (CV, all raw points)": [ms_(m, "top1_e2e") for m in agg.index],
        "Top-1 (labelled points only)": [ms_(m, "top1") for m in agg.index],
        "Route continuity": ["(from HMM pipeline)"] * len(agg),
        "Student-test macro F1": [f"{ter.loc[m, 'macro_f1']:.3f}" for m in agg.index],
        "Log loss (CV)": [f"{agg.loc[m, 'log_loss_mean']:.4f}" for m in agg.index],
        "Inference ms / 1k rows": [f"{ter.loc[m, 'infer_ms_per_1k']:.1f}" for m in agg.index]})
    tab.to_csv(out / "model_table.csv", index=False)
    print(tab.drop(columns=["Key configuration"]).to_string(index=False))

    if save_model:                                               # raw, uncalibrated; final_model.py makes the official one
        import joblib
        if save_model_name not in get_models():
            raise ValueError(f"Cannot save unknown model: {save_model_name}")
        m = fit_model(save_model_name, clone(get_models()[save_model_name]), cv)
        pathlib.Path(save_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(m, save_model)
        print(f"saved {save_model_name} (fitted on CV trips only) to {save_model}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/processed/candidates.parquet")
    ap.add_argument("--exclude", nargs="*", default=["00000054"])
    ap.add_argument("--frac", type=float, default=1.0, help="share of raw GPS points used (quick experiments)")
    ap.add_argument("--splits", type=int, default=5, help="CV folds")
    ap.add_argument("--test-frac", type=float, default=0.15, dest="test_frac", help="share of trips held out")
    ap.add_argument("--raw-root", default="data/raw/train", help="raw trip directory used for point counts")
    ap.add_argument("--models", nargs="*", default=["logreg", "rf", "lgbm", "mlp", "svm", "lgbm_ranker"])
    ap.add_argument("--output-dir", default="reports")
    ap.add_argument("--save-model")
    ap.add_argument("--save-model-name", default="lgbm_ranker")
    a = ap.parse_args()
    main(a.path, a.exclude, a.frac, a.splits, a.models, a.raw_root, a.test_frac, a.output_dir,
         a.save_model, a.save_model_name)