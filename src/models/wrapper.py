"""The single object saved to artifacts/model.joblib: classifier + calibrator + feature list.
Must stay importable as src.models.wrapper so joblib can unpickle it on Laptop 1."""
import platform
from importlib.metadata import PackageNotFoundError, version
import joblib
import numpy as np
import pandas as pd

EPS = 1e-6
KEYS = ["trip_id", "point_idx"]


class PathSnapModel:
    def __init__(self, model, calibrator, feature_columns, meta=None, kind="classifier"):
        self.model, self.calibrator, self.kind = model, calibrator, kind   # kind: "classifier" | "ranker"
        self.feature_columns, self.meta = list(feature_columns), meta or {}

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(candidate is the true arc), calibrated. One value per row of df."""
        X = df[self.feature_columns]
        if self.kind == "ranker":      # scores -> probabilities over each GPS point's candidates
            g = [df[k].to_numpy() for k in KEYS]
            sc = pd.Series(self.model.predict(X))
            e = np.exp(sc - sc.groupby(g).transform("max"))
            p = (e / e.groupby(g).transform("sum")).to_numpy()
        else:
            p = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, EPS, 1 - EPS)

    def predict_proba_normalised(self, df: pd.DataFrame, keys=("trip_id", "point_idx")) -> np.ndarray:
        """Same scores rescaled to sum to 1 over each GPS point's candidates (softmax-style emission)."""
        p = pd.Series(self.predict_proba(df), index=df.index)
        return (p / p.groupby([df[k] for k in keys]).transform("sum")).to_numpy()

    def save(self, path="artifacts/model.joblib"):
        self.meta.update(python=platform.python_version(), sklearn=version("scikit-learn"),
                         lightgbm=version("lightgbm"), pandas=pd.__version__,
                         numpy=np.__version__)
        try:
            self.meta["xgboost"] = version("xgboost")
        except PackageNotFoundError:
            self.meta.pop("xgboost", None)
        joblib.dump(self, path)

    @staticmethod
    def load(path="artifacts/model.joblib"):
        return joblib.load(path)