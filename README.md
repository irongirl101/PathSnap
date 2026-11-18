# PathSnap

PathSnap matches GPS traces to the roads a vehicle likely travelled. The project brings together trip data, a candidate-road classifier, an HMM route decoder, and a small API and browser UI.

## Start the API and UI

Use Python 3.10 or newer. From the repository folder, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`. The API docs are at `http://localhost:8000/docs`.

For a two-laptop demo, run the API on Laptop 1 and open `http://<Laptop-1-private-IP>:8000` from Laptop 2 on the same trusted network. If Windows asks, allow inbound connections on port 8000.

## Trip data and evaluation

The dataset has three partitions: 80 training trips, 10 `student_test` trips for development comparisons, and 10 reserved `evaluator_test` trips for the final evaluation. The supplied `Student folder.zip` contains the training and student-test data; it does not include the reserved evaluator trips. The student-test trip IDs are in [`data/splits/student_test_trips.json`](data/splits/student_test_trips.json).

Keep the final evaluator data on Laptop 1 under `data/evaluator_test`, preserving the per-trip folders and filenames. To try the UI during development, you can point `PATHSNAP_EVAL_DATA_DIR` at your extracted `student_test` folder. Point it back at `data/evaluator_test` for the final evaluation. Don’t use results from the reserved evaluator trips to tune the model.

The API reads `.track`, `.nodes`, and `.arcs` to make a prediction. It opens the trip’s `.route` file only after prediction, for server-side scoring. The browser receives the trip ID, predicted route, and metrics; it never receives the reference route.

Each trip folder has four whitespace-separated files without headers:

- `.track`: longitude, latitude, seconds since the trip started.
- `.nodes`: longitude, latitude; the row number is the node ID.
- `.arcs`: from-node ID and to-node ID; the row number is the arc ID.
- `.route`: the chronological sequence of reference arc IDs.

An arc ID belongs to one trip’s graph. When joining arcs to features or graph records, use both `trip_id` and `arc_id`.

## API and model connection

The browser asks `GET /api/v1/trips` for the available trip IDs. When someone selects a trip, it sends that ID to `POST /api/v1/evaluate`. The API runs the matcher, compares its arc sequence with the reference route, and returns the prediction and metrics.

The matcher and route-scoring functions still need to be connected to the team’s candidate, model, and HMM code. Until they’re configured, the evaluation endpoint reports that the pipeline isn’t ready; it won’t return placeholder predictions or metrics.

The matcher is loaded from `PATHSNAP_MATCHER=package.module:function`. It receives a dictionary containing `trip_id`, ordered GPS points, nodes, and arcs, plus runtime state with the model artifact. It returns a `route_arc_ids` sequence and may include predicted points for the map preview.

The scorer is loaded from `PATHSNAP_EVALUATOR=package.module:function`. It receives the trip ID, predicted arc IDs, and reference arc IDs, and returns `{"metrics": {"metric_name": value}, "metadata": {}}`. The matcher never receives the reference sequence.

The API looks for `artifacts/final_model.joblib` at startup. Override that path if needed. It’s fine to start the API before the model artifact or adapters are ready; the health endpoint shows which pieces are configured.

## Configuration

| Variable | Default | What it controls |
| --- | --- | --- |
| `PATHSNAP_MODEL_PATH` | `artifacts/final_model.joblib` | Classifier artifact loaded at startup |
| `PATHSNAP_MATCHER` | unset | `module:function` that predicts an arc sequence |
| `PATHSNAP_EVALUATOR` | unset | `module:function` that scores the predicted sequence |
| `PATHSNAP_EVAL_DATA_DIR` | `data/evaluator_test` | Trip folders available in the evaluator UI |

## Project folders

- `src/api/` contains the API, trip-file reader, and request/response formats.
- `src/data/` and `src/models/` hold the data, feature, classifier, and HMM work.
- `ui/` contains the browser interface served by the API.
- `artifacts/` is for local model and graph files; large generated files are ignored by Git.
- `tests/` contains API and trip-reader tests. Run them from the repository folder with `pytest`.
- `reports/` is for evaluation summaries and plots.

Don’t commit credentials or local/private trip data. The repository’s `.gitignore` excludes common data and model artifacts.
