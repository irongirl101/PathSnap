# PathSnap Part A

This package audits and cleans the student dataset, builds trip-specific directed
road graphs, searches road candidates, evaluates their coverage and exports 24 model features in the order specified by `artifacts/feature_columns.json`.

## Setup and execution

Use Python 3.11 or later with SQLite compiled with R-tree support. Extract the
dataset ZIP so that `Student folder/train` and `Student folder/student_test`
are siblings. From this package directory run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part-a.txt
python -m unittest discover -s part_a/tests -v
python -m part_a.pipeline '/absolute/path/to/Student folder' ./part_a_run
python -m part_a.verify '/absolute/path/to/Student folder' ./part_a_run
```

Use a fresh output directory for each run. The full candidate and alignment
tables use about 112 MiB combined. NumPy and Matplotlib are needed for the
pipeline. NetworkX enables the graph adapter and Folium enables HTML maps.
SQLite uses Python's standard-library binding. Pandas enables ordered model input and
categorical conversion. `results/` includes all three compressed feature splits,
offline alignment metadata, audit summaries, samples and plots. To add features to
an existing A1–A6 run directory containing `split_manifest.csv` and `candidates.csv`, use
`python -m part_a.features '/absolute/path/to/Student folder' ./part_a_run`.
The feature exporter creates a fresh `features/` subdirectory.

| File | Meaning |
| --- | --- |
| `split_manifest.csv` | One record per trip. Retain zero padding on trip IDs. |
| `trip_eda.csv` | Input sizes, cleaning audit, track and network statistics. |
| `candidate_metrics_by_trip.csv` | Metric numerators and denominators for nine settings. |
| `candidate_metrics.csv` | Pooled ratios by supplied partition. |
| `candidates.csv` | One row per observation/candidate at radius 100 m and K=10. |
| `features/{train,validation,student_test}.csv.gz` | Full candidate features for the three disjoint splits. |
| `features/verification.json` | Row counts per trip and missing-value counts per feature for each split. |
| `alignment.csv` | Offline provisional alignment and quality status for every cleaned observation. |
| `map_*.png` and `map_*.html` | Representative overlays with GPS, network and reference. HTML requires Folium. |

The track row ID is zero-based and remains stable during cleaning. The same ID is named `point_id` in the audit tables and `point_idx` in the feature contract. Use
`(trip_id, point_idx)` to group feature observations and `(trip_id, arc_id)` to identify
roads. The CSV omits candidate rows for empty observations. Use the batch API
to retain those empty lists at inference.

## Team integration

**Member B.** Load ordered inputs through `load_model_input`. Only the two
declared categorical columns receive pandas categorical dtype, with categories
`[0, 1]` and missing values preserved. The contract gives no category levels,
imputation, or formulas. See `TECHNICAL_REPORT.md` for the implemented
semantics. Review those definitions against the training code before merging.
For estimators that need one-hot encoding or imputation, put those transforms
inside the model pipeline and fit them on the training fold.

```python
from pathlib import Path

import pandas as pd
from part_a.features import FeatureContract, load_model_input

contract = FeatureContract.load("artifacts/feature_columns.json")
base = Path("results")
alignment = pd.read_csv(base / "alignment.csv.gz", dtype={"trip_id": str})
accepted = (alignment.loc[alignment["status"].eq("accepted"),
                          ["trip_id", "point_id", "provisional_arc_id"]]
            .rename(columns={"point_id": "point_idx"}))

def supervised(split):
    ids, X = load_model_input(base / "features" / f"{split}.csv.gz", contract)
    keys = ["trip_id", "point_idx"]
    matches = ids.assign(row=X.index).merge(accepted, on=keys, validate="many_to_one")
    y = matches["arc_id"].eq(matches["provisional_arc_id"]).astype(int)
    return X.loc[matches["row"]].reset_index(drop=True), y

X_train, y_train = supervised("train")
X_val, y_val = supervised("validation")
```

The example includes retained candidate groups when the provisional label is
missing from the candidate set. Those groups have all-zero labels and should
also be counted in candidate-miss analysis, including accepted observations
with no candidate rows. Evaluate the student-test feature file separately.
Keep `alignment.csv.gz` and all targets outside prediction inputs.

The train/validation snapshot uses fold 0 as validation. For cross-validation,
use the supplied training partition and select each of the five fold IDs as
validation in turn. Fit all learned transforms within the relevant training fold.

**Member C.** Read the graph and candidate lists without the reference route.

```python
from pathlib import Path
from part_a.data import Trip
from part_a.inference import candidate_batches
from part_a.roads import RoadGraph

trip_id = "00000001"
track = Path("Student folder") / "train" / trip_id / (trip_id + ".track")
trip = Trip("train", trip_id, track)
roads = RoadGraph.load(trip)
graph = roads.to_networkx()

for observation, candidates in candidate_batches(track, roads, k=10, radius_m=100):
    states = [candidate.segment_id for candidate in candidates]
    # Pass states, directed endpoints, and distances to the decoder.
    # observation.gap marks a break in the sampling cadence.
```

**Member D.** Resolve trip IDs through a server-owned catalog and invoke the
same reference-free candidate path. For model scoring, call
`part_a.features.infer_features(track, roads, contract)` once per complete trip
and build a DataFrame from the ordered feature names. Cast the two categorical
fields using `CATEGORY_LEVELS` as in `load_model_input`. Temporal features
use up to two subsequent observations and need a completed trip or that much
lookahead. Keep offline alignment inputs server-side
when calculating permitted evaluation summaries. Predicted routes and the
deployed API are team integration work.

## GitHub handoff

Open the team checkout in Codex. Start from the team's default branch and run
`git switch -c feature/part-a-data-features` if the branch does not already
exist. Ask Codex to compare the extracted `part_a/` package with the existing
source and check Member B's input conventions. Run
`cmp /path/to/extracted/artifacts/feature_columns.json artifacts/feature_columns.json`
inside the checkout. Resolve any difference with the team before integrating.
Keep the repository contract at its current path. Copy the reviewed code,
tests and dependency file, and incorporate the documentation in the team's
README. Keep the raw data and full generated training exports local.

Run the setup, test, pipeline and verification commands above inside the
checkout before opening a PR from the feature branch. Suggested PR title
“Add PathSnap Part A data pipeline and contract-ordered features”. Use the
measured counts and documented feature semantics in the PR description.
Coordinate push and PR creation with the team.

## Evaluation preparation

Explain how row indices identify nodes and arcs, how trip-level folds avoid
sharing points from the same drive, and why projection gives metre-based
distances. Walk through one R-tree envelope lookup and its exact segment test.
Show how the offline alignment preserves route order, how large gaps and weak
alignment are flagged, and why candidate coverage and point-level recall have
different denominators. Reproduce the results and review the analysis before
using it in the consolidated team report. Disclose the AI assistance as required.
