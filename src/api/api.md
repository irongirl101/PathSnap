# API integration notes

The public API is implemented in `main.py`; wire formats live in `schemas.py`. Keep the HTTP contract independent of model internals so teams B/C can change implementation without breaking the UI.

## Routes

- `GET /healthz`: process readiness and model/matcher/evaluator availability.
- `GET /api/v1/config`: frontend capability flags.
- `POST /api/v1/map-match`: custom ordered GPS points in; `route_arc_ids` out.
- `GET /api/v1/trips`: lists the configured evaluator trip IDs, never labels.
- `POST /api/v1/evaluate`: selected trip ID in; prediction plus server-side metrics out.
- `GET /docs`: generated OpenAPI documentation.

## Adapter signatures

The default matcher is `src.pathsnap.api_adapter:matcher`; override with `PATHSNAP_MATCHER=package.module:function` if needed. It receives `(payload: dict, runtime: dict)` where `runtime` has `model` and `model_path`. For custom `/map-match` calls, `payload` has `trip_id` and `points`; for selected-trip `/evaluate` calls, `payload` has `trip_id`, `points`, `nodes`, and `arcs`. Selected-trip payloads use the real file layout: `.track` is lon/lat/seconds, node IDs and arc IDs are their zero-based row indexes. Return a dictionary accepted by `MapMatchResponse`, including the stitched `route_arc_ids` and optional predicted per-point details.

The default scorer is `src.pathsnap.api_adapter:evaluate_trip`; override with `PATHSNAP_EVALUATOR=package.module:function`. It receives keyword arguments `trip_id`, `predicted_arc_ids`, `reference_arc_ids`, `trip_data`, and `prediction`, and returns `{"metrics": {"metric_name": scalar}, "metadata": {}}`. The API calls the matcher before opening `.route`, then supplies the reference sequence to this scoring callback only. The browser receives metrics and the prediction, never the reference route.

The evaluation UI lists trip folders under `PATHSNAP_EVAL_DATA_DIR` (default `data/evaluator_test`). The root is intentionally separate from `student_test`; the ten student-test trips are for development comparisons and are not the reserved final evaluation partition. `TripStore.load_prediction_input()` parses `.track`, `.nodes`, and `.arcs`; `TripStore.read_reference_route()` is called only after prediction. Store graph records under a tuple key returned by `arc_lookup_key(trip_id, arc_id)`. Arc IDs alone are not globally unique across these trip-local road graphs.

No adapter configured means HTTP 503 with a specific setup message. Malformed inputs receive FastAPI/Pydantic validation errors. Runtime exceptions are surfaced as HTTP 500 details for the development deployment.
