"""Optional loading of teammate-owned model and pipeline adapters."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable


def load_artifact(path: str | Path) -> Any | None:
    """Load a joblib artifact if configured and present; missing artifacts are OK."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return None
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError("Install joblib to load the configured model artifact") from exc
    return joblib.load(artifact_path)


def load_callable(spec: str | None, purpose: str) -> Callable[..., Any] | None:
    """Resolve a `some.module:callable` adapter configured through the environment."""
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeError(f"{purpose} must use the format 'package.module:callable'")
    module = importlib.import_module(module_name)
    callback = getattr(module, attribute, None)
    if not callable(callback):
        raise RuntimeError(f"Configured {purpose} '{spec}' is not callable")
    return callback


def configure_runtime(app: Any) -> None:
    """Populate FastAPI app state; kept separate for simple test injection."""
    root = Path(os.getenv("PATHSNAP_ROOT", Path(__file__).resolve().parents[2]))
    model_path = Path(os.getenv("PATHSNAP_MODEL_PATH", root / "artifacts" / "final_model.joblib"))
    app.state.project_root = root
    app.state.model_path = model_path
    app.state.model = load_artifact(model_path)
    app.state.matcher = load_callable(os.getenv("PATHSNAP_MATCHER"), "PATHSNAP_MATCHER")
    app.state.evaluator = load_callable(os.getenv("PATHSNAP_EVALUATOR"), "PATHSNAP_EVALUATOR")
    app.state.test_trips_path = Path(
        os.getenv("PATHSNAP_TEST_TRIPS", root / "data" / "splits" / "test_trips.json")
    )


def get_runtime(app: Any) -> dict[str, Any]:
    """Return loaded components to a teammate adapter without module globals."""
    return {"model": app.state.model, "model_path": app.state.model_path}
