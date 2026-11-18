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

## API integration contract

`POST /api/v1/map-match` accepts a trip ID and ordered WGS84 points (`lat`, `lon`, optional ISO-8601 `timestamp`). It returns the ordered matched `route_arc_ids` and optional predicted per-point details for display. Ground truth is an arc sequence from each trip's `.route` file, not a per-point label. The adapter must scope graph access by the composite key `(trip_id, arc_id)` so equal arc IDs from different trip graphs cannot collide. The API does not fabricate results: it returns HTTP 503 with setup guidance until the team supplies an adapter callable via `PATHSNAP_MATCHER=package.module:function`. That callable receives the request's JSON-compatible dictionary and runtime model state, and returns a dictionary matching `MapMatchResponse`.

The server loads `artifacts/final_model.joblib` by default (override with `PATHSNAP_MODEL_PATH`). Loading is optional so the API/UI can run before teammate B's model artifact exists. The adapter can import the loaded model through `src.api.runtime.get_runtime()` and combine candidate generation and HMM decoding. Set `PATHSNAP_MATCHER` to the adapter's `module:function` path.

`POST /api/v1/evaluate` invokes the team-provided evaluator callable (`PATHSNAP_EVALUATOR=package.module:function`) with the held-out trip manifest. Its return value is a JSON object of metrics and optional per-trip results. Evaluation stays server-side; no test labels are sent to the browser. The supplied `Student folder.zip` contains ten `student_test` trips; their IDs are recorded in `data/splits/test_trips.json`. The endpoint is unavailable until that callable and the held-out data are ready.

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
| `PATHSNAP_EVALUATOR` | unset | `module:function` held-out evaluation adapter |
| `PATHSNAP_TEST_TRIPS` | `data/splits/test_trips.json` | Held-out trip manifest passed to evaluator |

The model/feature team must define how candidate arc labels are derived from the `.track` observations and route ground truth; the API accepts raw observations and does not invent candidate labels. Keep candidate/feature joins keyed by both trip and arc ID.

Never commit credentials, private trip data, or generated model/graph binaries. See `.gitignore`.
