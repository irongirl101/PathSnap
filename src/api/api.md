# API integration notes

The public API is implemented in `main.py`; wire formats live in `schemas.py`. Keep the HTTP contract independent of model internals so teams B/C can change implementation without breaking the UI.

## Routes

- `GET /healthz`: process readiness and model/matcher/evaluator availability.
- `GET /api/v1/config`: frontend capability flags.
- `POST /api/v1/map-match`: ordered GPS points in; `route_arc_ids` out.
- `POST /api/v1/evaluate`: server-side held-out evaluation only.
- `GET /docs`: generated OpenAPI documentation.

## Adapter signatures

Configure `PATHSNAP_MATCHER=package.module:function`. It receives `(payload: dict, runtime: dict)` where `payload` has `trip_id` and `points`; `runtime` has `model` and `model_path`. Return a dictionary accepted by `MapMatchResponse`, including `route_arc_ids`.

Configure `PATHSNAP_EVALUATOR=package.module:function`. It receives keyword arguments `trip_ids`, `test_trips_path`, and `runtime`. When `trip_ids` is omitted, the evaluator should read the configured manifest and evaluate every listed trip. Read each trip's ordered `.route` arc sequence as its truth. Return `evaluated_trips`, a `metrics` object, and optional `per_trip` details.

Store graph records under a tuple key returned by `arc_lookup_key(trip_id, arc_id)`. Arc IDs alone are not globally unique across these trip-local road graphs.

No adapter configured means HTTP 503 with a specific setup message. Malformed inputs receive FastAPI/Pydantic validation errors. Runtime exceptions are surfaced as HTTP 500 details for the development deployment.
