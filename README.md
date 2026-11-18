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

The dataset has three partitions: 80 training trips, 10 `student_test` trips for development comparisons, and 10 reserved `evaluator_test` trips for the final evaluation. The supplied `Student folder.zip` contains the training and student-test data; it does not include the reserved evaluator trips. The student-test trip IDs are in [`data/splits/test_trips.json`](data/splits/test_trips.json).

Keep the final evaluator data on Laptop 1 under `data/evaluator_test`, preserving the per-trip folders and filenames. To try the UI during development, you can point `PATHSNAP_DATA_ROOT` at your extracted `student_test` folder. Point it back at `data/evaluator_test` for the final evaluation. Don’t use results from the reserved evaluator trips to tune the model.

The API reads `.track`, `.nodes`, and `.arcs` to make a prediction. It opens the trip’s `.route` file only after prediction, for server-side scoring. The browser receives the trip ID, predicted route, and metrics; it never receives the reference route.

Each trip folder has four whitespace-separated files without headers:

- `.track`: longitude, latitude, point index (1-based sequential; gaps possible).
- `.nodes`: longitude, latitude; the row number is the node ID.
- `.arcs`: from-node ID and to-node ID; the row number is the arc ID (directed).
- `.route`: the chronological sequence of reference arc IDs (0-based).

An arc ID belongs to one trip’s graph. When joining arcs to features or graph records, use both `trip_id` and `arc_id`.

## API and model connection

The browser asks `GET /api/v1/trips` for the available trip IDs. When someone selects a trip, it sends that ID to `POST /api/v1/evaluate`. The API runs the matcher, compares its arc sequence with the reference route, and returns the prediction and metrics.

The API loads the HMM matcher and route scorer from `src.pathsnap.api_adapter`. The training code compares classifiers and a LambdaRank candidate ranker using trip-grouped cross-validation. To evaluate the ranker and save a final model fitted on the training trips, run:

```bash
python -m src.models.train_models --models lgbm_ranker --save-model artifacts/final_model.joblib
```

The adapter accepts either a classifier with `predict_proba` or a ranker with `predict`; ranker scores are normalized within each GPS point before they are used as HMM emissions. The current branch does not include a fitted model artifact, so predictions use the geometric candidate baseline until the training command is run. The health endpoint reports whether a model was loaded.

The matcher receives a dictionary containing `trip_id`, ordered GPS points, nodes, and arcs, plus runtime state with the optional model artifact. It returns a stitched `route_arc_ids` sequence and per-observation matches for the map preview. The model-backed path uses the feature generator and the loaded model’s probabilities or point-normalized ranking scores as HMM emissions. Override the default with `PATHSNAP_MATCHER=package.module:function` if the team changes the matcher adapter.

The scorer receives the trip ID, predicted arc IDs, reference arc IDs, trip graph/trace data, and per-observation predictions. It returns RMF, edge precision/recall/F1, point accuracy, and coverage. The matcher never receives the reference sequence. You can override the scorer with `PATHSNAP_EVALUATOR=package.module:function` if the evaluation implementation changes.

The API looks for `artifacts/final_model.joblib` at startup. Override that path if needed. It’s fine to start the API before the model artifact or adapters are ready; the health endpoint shows which pieces are configured.

## Configuration

| Variable | Default | What it controls |
| --- | --- | --- |
| `PATHSNAP_MODEL_PATH` | `artifacts/final_model.joblib` | Classifier artifact loaded at startup |
| `PATHSNAP_MATCHER` | `src.pathsnap.api_adapter:matcher` | `module:function` that predicts an arc sequence |
| `PATHSNAP_EVALUATOR` | `src.pathsnap.api_adapter:evaluate_trip` | `module:function` that scores the predicted sequence |
| `PATHSNAP_DATA_ROOT` | `<repo>/data` | Directory containing `train/` and `student_test/` trip folders |
| `PATHSNAP_CONFIG` | `config/default.yaml` | Optional path to a `MatchConfig` YAML file; this repository default is used when unset |
| `PATHSNAP_EVAL_DATA_DIR` | `data/evaluator_test` | Trip folders available in the evaluator UI |

## Member C — HMM map-matching component (`src/pathsnap/`)

The HMM map-matcher is a standalone library that converts GPS observations and
candidate-road probabilities into road routes. It is independent of the
classifier training code: it receives candidate emissions as input and never
retrains or modifies the classifier.

### Public interface

```python
from pathsnap import match_trip, classifier_only, MatchConfig, Graph, Candidate
match_trip(trip_id, points, candidates_by_point, graph, config) -> MatchResult
classifier_only(trip_id, points, candidates_by_point) -> MatchResult
```

### Key features

- **Bounded Dijkstra** with configurable distance cutoff and a deterministic
  cache keyed on `(source, target, max_distance_m)`; `clear_cache()` available
- **Log-space Viterbi** over variable-width candidate lattices with `α × log(emission)`
  and deterministic tie-breaking
- **Explicit no-route handling** — unreachable transitions return `-inf`, never
  silently converted to plausible scores
- **Gap detection** from point-index discontinuity or distance fallback; gaps
  and breaks recorded in the result
- **Route stitching** with edges labeled `decoder` / `stitched` / `bridge`;
  interpolated bridge edges flagged; direction preserved; duplicates removed
- **Classifier-only baseline** using the same candidates and eligible points
- **Evaluation**: RMF, directed edge precision/recall/F1, point accuracy,
  macro + micro aggregation

### Configuration

`config/default.yaml` sets β (transition-distance scale), α (emission weight),
transition cutoffs, gap policy, and route stitching. The current API settings
are **β=10, α=0.25, and plausibility factor=20**. They were selected on a
trip-level holdout from the training partition using the fitted ranker and the
same classifier-backed matcher used by the API. The `student_test` trips were
used only as a development check; the reserved evaluator partition was not
used to select these settings. See
[`reports/api_point_accuracy_tuning.md`](reports/api_point_accuracy_tuning.md)
for the search and validation results.

### Tuning and results

Run `scripts/tune.py` for the full β/α grid search, ablations, and held-out
test evaluation. Results are saved under `artifacts/`:

- `artifacts/tuning/` — train/val split, grid results, tuned config
- `artifacts/ablations/` — ablation study results
- `artifacts/sensitivity/` — β/α sensitivity grid
- `artifacts/evaluation/` — held-out test metrics (HMM vs classifier-only)
- `artifacts/figures/` — geographic success/failure figures

See [`reports/reports.md`](reports/reports.md) for the full methods, assumptions,
results, limitations, and failure cases.

### API integration

Set `PATHSNAP_MATCHER=src.pathsnap.api_adapter:matcher` to wire the matcher into
the FastAPI app. The adapter converts the API's `(lat, lon)` coordinate
contract to internal `(lon, lat)` at the boundary. Set
`PATHSNAP_EVALUATOR=src.pathsnap.api_adapter:evaluate_trip` to wire the
evaluator into `/api/v1/evaluate`.

## Project folders

- `src/api/` contains the API, trip-file reader, and request/response formats.
- `src/data/` and `src/models/` hold the data, feature, classifier, and HMM work.
- `src/pathsnap/` contains the Member C HMM map-matching component.
- `ui/` contains the browser interface served by the API.
- `artifacts/` is for local model and graph files; large generated files are ignored by Git.
- `scripts/` contains tuning, evaluation, and figure-generation scripts.
- `tests/` contains API and trip-reader tests. Run them from the repository folder with `pytest`.
- `reports/` is for evaluation summaries and plots.

Don’t commit credentials or local/private trip data. The repository’s `.gitignore` excludes common data and model artifacts.
