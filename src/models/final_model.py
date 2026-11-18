"""Fit a scorer and calibrator using the training partition's internal holdout.

usage: python -m src.models.final_model --model lgbm          # Optuna-tuned classifier (needs best_params_lgbm.json)
       python -m src.models.final_model --model lgbm_ranker   # LambdaRank + per-point softmax (default config)

For the supplied student_test partition and final deployment artifact, use
src.models.evaluate_student_test instead.
"""
import argparse, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from src.features.features import FEATURE_COLS
from src.models.train_models import (load, get_splits, labelled, get_models, fit_model, score_model,
                                     metrics, EPS)
from src.models.wrapper import PathSnapModel


def main(name, frac, exclude, test_frac, raw_root, cv_splits=5):
    df, pcount = load("data/processed/candidates.parquet", exclude, frac, raw_root)
    sp = get_splits(sorted(pcount), exclude, test_frac)
    tr = labelled(df[df.trip_id.isin(sp["cv"])]).reset_index(drop=True)
    te_all = df[df.trip_id.isin(sp["test"])]
    te = labelled(te_all).reset_index(drop=True)
    n_te_raw = sum(pcount.get(t, 0) for t in sp["test"])

    ranker = name == "lgbm_ranker"
    if ranker:
        make, params, untuned = (lambda: get_models()["lgbm_ranker"]), {}, None
    else:
        from src.models.bayes import build_model, DEFAULTS
        params = json.load(open(f"artifacts/best_params_{name}.json"))["params"]
        make = lambda: build_model(name, params)
        untuned = lambda: build_model(name, DEFAULTS[name])
    key = name if ranker else "clf"                       # fit_model/score_model only branch on "lgbm_ranker"

    oof = np.zeros(len(tr))                               # out-of-fold scores -> calibrator
    for a, b in GroupKFold(cv_splits).split(tr, tr["label"], tr["trip_id"]):
        oof[b] = score_model(key, fit_model(key, make(), tr.iloc[a]), tr.iloc[b])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(oof, tr["label"])

    model = fit_model(key, make(), tr)
    final = PathSnapModel(model, iso, FEATURE_COLS, kind="ranker" if ranker else "classifier",
                          meta=dict(model=name, params=params, cv_trips=len(sp["cv"]), test_trips=sp["test"]))
    final.save("artifacts/model.joblib")

    import time
    raw = score_model(key, model, te)
    t0 = time.perf_counter(); cal = final.predict_proba(te); ms = (time.perf_counter() - t0) / len(te) * 1e6
    res = {"tuned_raw" if not ranker else "raw": metrics(te, raw),
           "calibrated": metrics(te, cal), "inference_ms_per_1k_rows": ms}
    if untuned:
        res["untuned"] = metrics(te, score_model(key, fit_model(key, untuned(), tr), te))
    for k in ("untuned", "tuned_raw", "raw", "calibrated"):
        if k in res:
            h = res[k].pop("hits"); res[k]["top1_e2e"] = h / n_te_raw
            print(f"{k:11s}", {m: round(float(v), 4) for m, v in res[k].items()})
    json.dump(res, open("reports/final_model_metrics.json", "w"), indent=2, default=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, p in (("raw", raw), ("calibrated", cal)):
        fr, mp = calibration_curve(te["label"], p, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title("Reliability (held-out test trips)"); ax.legend()
    fig.savefig("reports/figures/reliability_final.png", dpi=130, bbox_inches="tight")
    print("saved artifacts/model.joblib, reports/final_model_metrics.json, reports/figures/reliability_final.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "xgb", "lgbm_ranker"], default="lgbm")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--exclude", nargs="*", default=["00000054"])
    ap.add_argument("--test-frac", type=float, default=0.15, dest="test_frac")
    ap.add_argument("--raw-root", default="data/raw/train")
    a = ap.parse_args()
    main(a.model, a.frac, a.exclude, a.test_frac, a.raw_root)