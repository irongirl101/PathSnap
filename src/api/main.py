"""FastAPI service and static browser UI for PathSnap."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from src.api.runtime import configure_runtime
from src.api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    MapMatchRequest,
    MapMatchResponse,
    TripListResponse,
)
from src.api.trips import TripStore

UI_FILE = Path(__file__).resolve().parents[2] / "ui" / "index.html"
logger = logging.getLogger("pathsnap.api")


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
        "evaluation_trip_count": len(TripStore(state.evaluation_data_dir).list_trip_ids()),
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


@app.get("/api/v1/trips", response_model=TripListResponse)
def list_evaluation_trips(request: Request) -> dict[str, Any]:
    """Expose selectable trip IDs only; route labels remain server-side."""
    store = TripStore(request.app.state.evaluation_data_dir)
    return {"trips": [{"trip_id": trip_id} for trip_id in store.list_trip_ids()]}


@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest, request: Request) -> dict[str, Any]:
    """Match one selected trip, then score it server-side against its route file."""
    state = request.app.state
    matcher = state.matcher
    evaluator = request.app.state.evaluator
    if matcher is None:
        raise HTTPException(
            status_code=503,
            detail="Map matcher is not configured. Set PATHSNAP_MATCHER=package.module:function.",
        )
    if evaluator is None:
        raise HTTPException(
            status_code=503,
            detail="Evaluator is not configured. Set PATHSNAP_EVALUATOR=package.module:function.",
        )
    try:
        store = TripStore(state.evaluation_data_dir)
        trip_input = store.load_prediction_input(payload.trip_id)
        started = perf_counter()
        prediction_result = matcher(trip_input.as_dict(), get_runtime(request.app))
        prediction = MapMatchResponse.model_validate(prediction_result)
        if prediction.trip_id not in (None, payload.trip_id):
            raise ValueError("Map matcher returned a different trip_id")
        if not prediction.route_arc_ids:
            raise ValueError("Map matcher returned an empty route_arc_ids sequence")
        known_arc_ids = {arc["arc_id"] for arc in trip_input.arcs}
        known_arc_ids_text = {str(arc_id) for arc_id in known_arc_ids}
        unknown_arc_ids = [
            arc_id for arc_id in prediction.route_arc_ids if str(arc_id) not in known_arc_ids_text
        ]
        if unknown_arc_ids:
            raise ValueError(f"Map matcher returned arc IDs missing from trip graph: {unknown_arc_ids[:5]}")

        # Keep reference labels out of the model/matcher call above.
        reference_arc_ids = store.read_reference_route(payload.trip_id)
        invalid_reference_ids = [arc_id for arc_id in reference_arc_ids if arc_id not in known_arc_ids]
        if invalid_reference_ids:
            raise ValueError(f"Reference route contains arc IDs missing from trip graph: {invalid_reference_ids[:5]}")
        evaluation = evaluator(
            trip_id=payload.trip_id,
            predicted_arc_ids=prediction.route_arc_ids,
            reference_arc_ids=reference_arc_ids,
        )
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        if not isinstance(evaluation, dict):
            raise ValueError("Evaluator adapter must return a dictionary")
        metrics = evaluation.get("metrics", evaluation)
        metadata = dict(evaluation.get("metadata", {})) if "metrics" in evaluation else {}
        metadata["inference_and_evaluation_ms"] = elapsed_ms
        logger.info("Evaluated trip %s in %.2f ms", payload.trip_id, elapsed_ms)
        return EvaluationResponse(
            trip_id=payload.trip_id,
            route_arc_ids=prediction.route_arc_ids,
            trace=trip_input.points,
            points=prediction.points,
            metrics=metrics,
            metadata=metadata,
        ).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Trip data file not found: {exc.filename}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
