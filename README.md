# PathSnap

PathSnap is a student project for matching noisy GPS traces to a road network. The repository is organized so the data/feature, model, HMM, and API/UI work can be developed independently.

## Run the API and UI

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the browser UI and `http://localhost:8000/docs` for the API schema. For a second laptop on the same trusted LAN, use the server laptop's private IPv4 address, for example `http://192.168.1.20:8000`. Keep the host on a trusted network and allow inbound TCP port 8000 in the host firewall if Windows prompts.

The evaluator UI lists folders from `data/evaluator_test` by default. Mount the organizer-provided evaluator partition there on Laptop 1, preserving its per-trip folder and filename structure. For development-only checks using the supplied `Student folder.zip`, extract the `student_test` partition and point `PATHSNAP_EVAL_DATA_DIR` to that directory; switch it back to `data/evaluator_test` for the final evaluation. The API only returns trip IDs and predictions/metrics to Laptop 2, never `.route` reference sequences.

## API integration contract

`POST /api/v1/map-match` accepts a trip ID and ordered WGS84 points (`lat`, `lon`, optional `timestamp_s` in seconds since trip start). It returns the ordered matched `route_arc_ids` and optional predicted per-point details for display. Ground truth is an arc sequence from each trip's `.route` file, not a per-point label. The adapter must scope graph access by the composite key `(trip_id, arc_id)` so equal arc IDs from different trip graphs cannot collide. The API does not fabricate results: it returns HTTP 503 with setup guidance until the team supplies an adapter callable via `PATHSNAP_MATCHER=package.module:function`. That callable receives the request's JSON-compatible dictionary and runtime model state, and returns a dictionary matching `MapMatchResponse`.

The server loads `artifacts/final_model.joblib` by default (override with `PATHSNAP_MODEL_PATH`). Loading is optional so the API/UI can run before teammate B's model artifact exists. The adapter can import the loaded model through `src.api.runtime.get_runtime()` and combine candidate generation and HMM decoding. Set `PATHSNAP_MATCHER` to the adapter's `module:function` path. The matcher and scorer code are not in the current `main` branch yet; until both are configured, selected-trip evaluation returns HTTP 503 instead of fake metrics.

`GET /api/v1/trips` lists only trip IDs found under `PATHSNAP_EVAL_DATA_DIR`. The default is `data/evaluator_test`, reserved for final evaluation. `POST /api/v1/evaluate` accepts one selected trip ID, loads its `.track`, `.nodes`, and `.arcs` into the matcher, and only after prediction reads `.route` for the evaluator. Ground-truth IDs never enter the matcher payload or the UI response. The evaluator adapter (`PATHSNAP_EVALUATOR=package.module:function`) returns metric values; the API returns those metrics and the predicted arc sequence.

The provided `Student folder.zip` contains ten `student_test` trips for development-stage comparisons. Their IDs are recorded in `data/splits/student_test_trips.json`. They are distinct from the ten reserved `evaluator_test` trips, which must be mounted on the API laptop by the evaluation organizers and must not be used to select models or parameters.

## Repository layout

- `src/api/`: FastAPI app, request/response schemas, and runtime artifact/adapter loading.
- `ui/`: browser client served by FastAPI.
- `src/data/`, `src/models/`: team-owned data, feature, classifier, and HMM components.
- `artifacts/`: local model and graph artifacts; do not commit large generated binaries.
- `tests/`: API contract tests.
- `reports/`: generated evaluation plots and summaries.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PATHSNAP_MODEL_PATH` | `artifacts/final_model.joblib` | Classifier artifact loaded at startup when present |
| `PATHSNAP_MATCHER` | unset | `module:function` adapter for map matching |
| `PATHSNAP_EVALUATOR` | unset | `module:function` route-sequence scoring adapter |
| `PATHSNAP_EVAL_DATA_DIR` | `data/evaluator_test` | Server-only trip folders used by the evaluator UI |

The model/feature team must define how candidate arc labels are derived from training trips; the API does not invent candidate labels. Keep candidate/feature joins keyed by both trip and arc ID. Dataset rows are whitespace-separated without headers: `.track` is longitude, latitude, seconds since trip start; `.nodes` is longitude, latitude with row index as node ID; `.arcs` is from-node ID, to-node ID with row index as arc ID; `.route` is the chronological arc ID sequence.

Never commit credentials, private trip data, or generated model/graph binaries. See `.gitignore`.
