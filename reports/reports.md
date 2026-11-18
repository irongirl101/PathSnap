# PathSnap — Member C: HMM Map-Matching Component

## Overview

This component turns noisy GPS observations and candidate-road probabilities into
plausible road routes using a Hidden Markov Model (HMM) with bounded
shortest-path transitions, Viterbi decoding, route stitching, and evaluation
metrics. It is compared against a classifier-only baseline (independent
per-point argmax emission).

## Input/output schemas

**`match_trip(trip_id, points, candidates_by_point, graph, config) -> MatchResult`**

- `points`: list of `(lon, lat, point_idx)` GPS observations in order
- `candidates_by_point`: list parallel to `points`; each entry is a list of
  `Candidate` objects or `None`/empty when a point has no candidates
- `graph`: `Graph` (directed, weighted road network for one trip)
- `config`: `MatchConfig` (β, α, cutoffs, gap policy, stitching, cache)
- Returns: `MatchResult` with per-point matches, gaps, breaks, segments,
  stitched route (edges labeled `decoder` / `stitched` / `bridge`), and total
  log-score

**`classifier_only(trip_id, points, candidates_by_point) -> MatchResult`**

Independent per-point highest-emission selection, same candidates and eligible
points as the HMM. No transitions, no stitching.

## Data formats (verified against the archive)

| File | Columns | Meaning |
|------|---------|---------|
| `.track` | `lon  lat  point_index` | GPS observations; index is 1-based sequential (gaps possible) |
| `.nodes` | `lon  lat` | Row index = node id (0-based) |
| `.arcs` | `src  dst` | Directed edges (0-based node ids) |
| `.route` | `edge_id` | 0-based arc/edge ids forming a connected reference path |

No candidate files or classifier emission scores exist in the supplied archive
(owned by other team members). A documented stand-in candidate generator
(`src/pathsnap/candidates.py`) snaps each GPS point to nearby edges with a
Gaussian emission; the matcher itself accepts any candidate set + emissions.

## Methods

### Transition model

For consecutive candidates `a` → `b`:

```
log_transition = -|d_gc - d_route| / β
```

- `d_gc`: great-circle (haversine) distance between the two GPS observations
- `d_route`: direction-aware network distance between projections, including
  projection offsets on the first and last edges:
  - same edge: `|offset_b - offset_a|`
  - different edges: `(len_a - offset_a) + shortest_path(dst(a), src(b)) + offset_b`

A configurable plausibility cutoff rejects transitions where
`d_route > max(plausibility_factor × d_gc, min_route_gap_m)`.
Unreachable routes return `-inf` (explicit no-route, never silently plausible).

### Bounded Dijkstra + cache

Bounded Dijkstra stops expanding once the frontier exceeds
`max_route_distance_m`. Results are cached per `Graph` instance keyed by
`(source, target, max_distance_m)`. Call `Graph.clear_cache()` to invalidate.

### Viterbi

Log-space Viterbi over variable-width candidate lattices. Emission scored as
`α × log(emission)` (emissions validated as nonnegative, clipped at
`emission_floor`). Deterministic tie-breaking (`first` = lowest candidate index
on exact score ties). One-point trips and empty candidate sets handled
deliberately.

### Gaps and lattice breaks

Gaps detected from point-index discontinuity (primary) or great-circle distance
(fallback). Lattice breaks (no candidates at a point) split decoding into
contiguous segments with a restart strategy. Gaps and breaks are recorded in
the result.

### Route stitching

Decoder-selected candidate edges are emitted as `source='decoder'`.
Shortest-path connections between consecutive projections are `source='stitched'`.
Gap/break bridges are `source='bridge'`, `interpolated=True`. Duplicate
consecutive edges removed; direction preserved.

## Evaluation metrics

### RMF (Route Mismatch Fraction)

`RMF = (omitted_length + extra_length) / reference_length`

- Omitted = reference edges not predicted; extra = predicted edges not in reference
- Zero-length reference: RMF = 0.0 if prediction empty, else 1.0
- Can exceed 1.0 when extra length is large

### Edge precision/recall/F1

Direction-sensitive by default (configurable undirected). Empty sets handled:
- precision = 1.0 if both empty, 0.0 if prediction empty but reference not
- recall = 1.0 if both empty, 0.0 if reference empty but prediction not
- F1 = 0 when precision + recall = 0

### Point accuracy

Reference assignment derived by snapping each GPS observation to the nearest
point on the reference polyline (documented assumption since `.route` is not
per-point labeled). Unmatched points and missing reference assignments
reported alongside accuracy (coverage included).

Macro (mean over trips) and micro (pooled) aggregation.

## Tuning

β (transition-distance scale) and α (emission weight) tuned on training data
only, with a trip-level train/validation split (60/20, seed 42) to prevent
point leakage. Held-out `student_test/` routes never consulted during
configuration selection.

Objective: mean edge F1 on validation set. Grid: β ∈ {25, 50, 100, 200, 400},
α ∈ {0.25, 0.5, 1.0, 2.0}.

Selected: **β=200.0, α=2.0** (val edge F1 = 0.5249).

## Ablations (training trips)

| Configuration | Edge F1 | RMF | Point Acc |
|---------------|---------|-----|-----------|
| classifier_only | 0.5430 | 0.9467 | 0.5338 |
| hmm_full (tuned) | 0.5273 | 1.5232 | 0.5831 |
| no_transition | 0.5175 | 1.5779 | 0.5338 |
| no_emission | 0.5263 | 1.5300 | 0.5780 |

## Held-out test results (10 student_test trips)

| Metric | HMM (tuned) | Classifier-only | Difference |
|--------|-------------|-----------------|------------|
| Edge F1 | 0.5425 | 0.5469 | -0.0044 |
| RMF | 1.5023 | 0.9567 | -0.5456 |
| Point accuracy | 0.5715 | 0.5194 | +0.0520 |
| Coverage | 0.9300 | 0.9300 | 0.0000 |

HMM achieves higher point accuracy (+5.2%) and higher recall (0.989 vs 0.583)
but lower precision (0.384 vs 0.519) and higher RMF, because it inserts more
stitched/bridge edges. This is an honest result given the stand-in candidate
generator (Gaussian emissions from GPS-to-edge distance, not a real classifier).

## Limitations

- Stand-in candidate generator is not a real classifier; results would improve
  with actual classifier emissions
- Point accuracy uses reference-polyline snapping (approximation, not per-point
  labels)
- No basemap/internet for geographic figures; equal-aspect lon/lat plots only
- HMM RMF is higher because it inserts more edges (stitched connectors and
  bridges); this may be acceptable or tunable via β/plausibility cutoff

## Failure cases

- `trip_00000065`: HMM RMF=3.115 (worst), CLF F1=0.667 vs HMM 0.333 — routing
  failure where HMM inserts many extra edges
- `trip_00000091`: CLF better on F1 and RMF
- `trip_00000000`: coverage only 0.636, 911 unmatched points — candidate
  generation/sparsity issue

## Reproducibility

- Tuning split: `artifacts/tuning/split.json` (60 train / 20 val, seed 42)
- Grid results: `artifacts/tuning/grid_results.json`
- Selected config: `config/default.yaml` (β=200.0, α=2.0)
- Ablations: `artifacts/ablations/ablations.json`
- Sensitivity grid: `artifacts/sensitivity/sensitivity_grid.json`
- Test metrics: `artifacts/evaluation/test_metrics.json`
- Test summary: `artifacts/evaluation/test_summary.json`
- Figures: `artifacts/figures/`

## Running

```powershell
# From PathSnap repo root
$env:PYTHONPATH = "src"
$env:PATHSNAP_DATA_ROOT = "<path to Student folder>"
$env:PATHSNAP_CONFIG = "config/default.yaml"

# Run full tuning + evaluation
python scripts/tune.py

# Generate geographic figures
python scripts/make_figures.py
```

## API integration

Set `PATHSNAP_MATCHER=pathsnap.api_adapter:matcher`. The adapter converts the
API's `(lat, lon)` contract to internal `(lon, lat)`, runs `match_trip`, and
returns a `MapMatchResponse`-shaped dict.
