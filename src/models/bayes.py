"""Bayesian hyper-parameter optimisation (Optuna TPE) with trip-grouped CV on the training trips only.

usage: python -m src.models.bayes --model lgbm --trials 40 --frac 0.3
The study is stored in reports/optuna/<model>.db, so an interrupted run can simply be restarted.
Objective: mean log-loss over CV folds (the HMM needs good probabilities, not just good ranking).
Trial 0 is the untuned default configuration, so tuned-vs-untuned is compared on identical folds.
"""
import argparse, json, pathlib
import numpy as np, pandas as pd, optuna
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from src.features.features import FEATURE_COLS
from src.models.train_models import load, get_splits, labelled, SEED, EPS

DEFAULTS = {
    "lgbm": dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3, reg_alpha=1e-3),
    "xgb": dict(n_estimators=400, learning_rate=0.05, max_depth=8, min_child_weight=1,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, gamma=1e-3),
}


def suggest(trial, name):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 800),
             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if name == "lgbm":
        p.update(num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
    else:
        p.update(max_depth=trial.suggest_int("max_depth", 3, 12),
                 min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                 gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True))
    return p


def build_model(name, params):
    if name == "lgbm":
        return lgb.LGBMClassifier(subsample_freq=1, random_state=SEED, verbose=-1, **params)
    import xgboost as xgb
    return xgb.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=SEED, **params)


def main(name, trials, frac, cv_splits, exclude, test_frac, raw_root):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df, pcount = load("data/processed/candidates.parquet", exclude, frac, raw_root)
    sp = get_splits(sorted(pcount), exclude, test_frac)
    df = labelled(df[df.trip_id.isin(sp["cv"])]).reset_index(drop=True)     # the test trips are never touched
    X, y = df[FEATURE_COLS], df["label"]
    folds = list(GroupKFold(cv_splits).split(X, y, df["trip_id"]))
    print(f"tuning {name} on {df.trip_id.nunique()} trips, {len(df)} rows, {cv_splits}-fold grouped CV")

    def objective(trial):
        params = suggest(trial, name)
        losses = []
        for k, (tr, va) in enumerate(folds):
            m = build_model(name, params).fit(X.iloc[tr], y.iloc[tr])
            p = np.clip(m.predict_proba(X.iloc[va])[:, 1], EPS, 1 - EPS)
            losses.append(log_loss(y.iloc[va], p))
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    out = pathlib.Path("reports/optuna"); out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=name, direction="minimize", load_if_exists=True,
                                storage=f"sqlite:///{out}/{name}.db",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULTS[name])                          # trial 0 = untuned baseline
    study.optimize(objective, n_trials=max(0, trials - len(study.trials)))

    t = study.trials_dataframe()
    t.to_csv(out / f"{name}_trials.csv", index=False)
    base = study.trials[0].value
    best = study.best_trial
    print(f"untuned log-loss {base:.5f} -> best {best.value:.5f} (trial {best.number}); "
          f"{sum(x.state.name == 'COMPLETE' for x in study.trials)} complete, "
          f"{sum(x.state.name == 'PRUNED' for x in study.trials)} pruned")
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    json.dump(dict(model=name, objective="mean grouped-CV log-loss", cv_splits=cv_splits, frac=frac,
                   untuned_logloss=base, tuned_logloss=best.value, params=best.params),
              open(f"artifacts/best_params_{name}.json", "w"), indent=2)

    done = [x for x in study.trials if x.value is not None]
    vals = [x.value for x in done]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x.number for x in done], vals, "o", alpha=.5, label="trial")
    ax.plot([x.number for x in done], np.minimum.accumulate(vals), "-", label="best so far")
    ax.axhline(base, ls="--", c="grey", label="untuned default")
    ax.set_xlabel("trial"); ax.set_ylabel("CV log-loss"); ax.legend(); ax.set_title(f"Optuna history ({name})")
    pathlib.Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"reports/figures/optuna_history_{name}.png", dpi=130, bbox_inches="tight")
    imp = optuna.importance.get_param_importances(study)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_xlabel("importance")
    ax.set_title(f"Hyper-parameter importance ({name})")
    fig.savefig(f"reports/figures/optuna_importance_{name}.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--raw-root", default="data/raw/train")
    ap.add_argument("--exclude", nargs="*", default=["00000054"])
    ap.add_argument("--test-frac", type=float, default=0.15, dest="test_frac")
    a = ap.parse_args()
    main(a.model, a.trials, a.frac, a.splits, a.exclude, a.test_frac, a.raw_root)