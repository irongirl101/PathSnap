"""FastAPI service and static browser UI for PathSnap."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from src.api.runtime import configure_runtime
from src.api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    MapMatchRequest,
    MapMatchResponse,
)

UI_FILE = Path(__file__).resolve().parents[2] / "ui" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_runtime(app)
    yield


app = FastAPI(
    title="PathSnap API",
    description="GPS trace map matching and held-out evaluation service.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(UI_FILE, media_type="text/html")


@app.get("/healthz")
def health(request: Request) -> dict[str, Any]:
    """Readiness and artifact status for local and LAN deployments."""
    state = request.app.state
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "matcher_ready": state.matcher is not None,
        "evaluator_ready": state.evaluator is not None,
    }


@app.get("/api/v1/config")
def config(request: Request) -> dict[str, Any]:
    state = request.app.state
    return {
        "api_version": "v1",
        "model_loaded": state.model is not None,
        "matcher_ready": state.matcher is not None,
        "evaluator_ready": state.evaluator is not None,
    }


@app.post("/api/v1/map-match", response_model=MapMatchResponse)
def map_match(payload: MapMatchRequest, request: Request) -> dict[str, Any]:
    matcher = request.app.state.matcher
    if matcher is None:
        raise HTTPException(
            status_code=503,
            detail="Map matcher is not configured. Set PATHSNAP_MATCHER=package.module:function.",
        )
    try:
        result = matcher(payload.model_dump(mode="json"), get_runtime(request.app))
        return MapMatchResponse.model_validate(result).model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Map matching failed: {exc}") from exc


@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest, request: Request) -> dict[str, Any]:
    """Run evaluation on the server so held-out labels stay off the browser."""
    evaluator = request.app.state.evaluator
    if evaluator is None:
        raise HTTPException(
            status_code=503,
            detail="Evaluator is not configured. Set PATHSNAP_EVALUATOR=package.module:function.",
        )
    try:
        result = evaluator(
            trip_ids=payload.trip_ids,
            test_trips_path=request.app.state.test_trips_path,
            runtime=get_runtime(request.app),
        )
        return EvaluationResponse.model_validate(result).model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
