# PathSnap Part A execution report

The run used the local `ml_hackthon.zip` and all five PDFs in
`PathSnap_student_materials.zip`. The two archives had SHA-256 digests
`4804b0399d1d56f4844b9c74e2f998aae0a281fb5947a1f47ad1bd013214d248` and
`db368793f68c76baa8eaa30d783d376df50841287c29c43f6911feb98cab4367`
respectively. The official dataset, training, deliverables, assessment and
main guidelines were reviewed. Evaluator-only data was outside the run.

## A1–A3. Dataset and cleaning

Each supplied trip has four headerless whitespace-delimited files.
`.track` stores longitude, latitude and time in seconds. `.nodes` stores
longitude and latitude, with zero-based row IDs. `.arcs` stores integer
directed from/to node IDs, with the zero-based row as the arc ID.
`.route` stores one integer arc ID per row in traversal order. The run
validated each arc endpoint and every reference arc ID across all 360 files.
The full route ID inventory is available in the split manifest.

| Measured quantity | Supplied training | Student testing |
| --- | ---: | ---: |
| Trips | 80 | 10 |
| Raw and retained GPS observations | 208,417 | 19,160 |
| Exactly 1 s intervals | 201,048 / 208,337 | 18,886 / 19,150 |
| Gaps over 5 s | 1,216 | 44 |
| Maximum time gap (s) | 153,064 | 210 |
| Steps over 55 m/s among intervals at most 5 s | 994 | 4 |
| Longitude range | -100.418538 to 137.237029 | -93.424337 to 77.271115 |
| Latitude range | -34.640291 to 61.267258 | 42.645529 to 61.699590 |
| Median GPS length (km) | 24.04 | 20.97 |
| Median GPS length ignoring gaps over 5 s (km) | 23.09 | 20.76 |
| Median reference polyline length (km) | 14.32 | 14.77 |
| Nodes / directed arcs | 1,933,485 / 3,876,904 | 90,689 / 185,041 |
| Weak components summed / isolated nodes | 3,121 / 0 | 227 / 0 |
| Zero-length arcs | 100 | 6 |
| Repeated directed endpoint pairs beyond the first | 1,775 | 128 |
| Reference entries / discontinuous trips | 31,336 / 10 | 3,477 / 1 |

There were zero missing, malformed, nonfinite, invalid-coordinate,
duplicate-timestamp or exact duplicate rows in either partition. The
reusable cleaner drops invalid rows, retains the first valid row at each
timestamp, and sorts by time while preserving row IDs. It flags intervals
above 5 s without filling or deleting them. Time is interpreted as elapsed
seconds at the observation cadence. Speed spikes, nearest-road offsets and
gap-sensitive distances indicate potential quality issues. Network offset
also reflects road geometry and parallel roads, so it cannot identify
sensor error by itself. The median of the per-trip median nearest offsets
within 100 m was 4.14 m for training and 4.31 m for student testing.

The seed-42 SHA-256 trip ordering assigns 16 trips to each validation fold.
The default snapshot contains 64 training and 16 validation trips. All ten
student-testing trips are held out from both. Training results in this report
refer to the full supplied training partition before that internal split.

## A4–A5. Graph and candidates

A compact directed multigraph representation retains original endpoint IDs,
parallel arcs and zero-length arcs. A local WGS84 tangent-plane approximation
converts coordinate differences to metres using the network-centre latitude.
This approximation assumes a regional network away from the poles and
antimeridian. Each road is represented by a straight segment because the arc
records contain endpoints only. The R-tree filters intersecting bounds.
Distance and offset are calculated by perpendicular projection clamped to the
segment, with a zero-length segment treated as a point. Results sort first
by distance and then by arc ID. Candidate generation uses only the track
and road graph. Empty lists and large gaps are visible to the decoder.

## A6. Offline reference alignment and coverage

The aligner uses dynamic programming over reference route positions. It
minimizes point-to-segment distances capped at 60 m subject to a
nondecreasing route position. The sequence may remain on a segment or skip
ahead over unsampled segments. Directed endpoint continuity is checked
before alignment. The whole trip is withheld from point-label recall if
its reference has a break. For continuous routes, a provisional point is
accepted if distance is at most 30 m and the best full alignment that forces
a different arc at that point costs at least 8 m more, when such an
alternative exists. These are fixed heuristic thresholds. This method
permits backwards motion within a single arc's geometry, and it does not
enforce travel time, speed or direction of GPS motion within the arc.

| Alignment status | Supplied training | Student testing |
| --- | ---: | ---: |
| Accepted | 110,812 | 12,202 |
| Ambiguous | 41,393 | 5,323 |
| More than 30 m from aligned segment | 20,437 | 149 |
| Discontinuous route | 35,775 | 1,486 |

Route coverage is the fraction of distinct reference arcs retrieved at least
once across the trip, pooled using trip-specific arc identities. The
point-any-reference rate asks whether any candidate appears anywhere in that
trip's reference. Provisional point recall counts exact arc hits among accepted
points. Miss rate is one minus that recall. Empty rate uses all retained points.

| Radius / K | Partition | Empty points | Route arc coverage | Provisional point miss |
| --- | --- | ---: | ---: | ---: |
| 25 m / 5 | Training | 10.26% | 97.53% | 1.42% |
| 25 m / 5 | Student testing | 1.17% | 98.48% | 1.64% |
| 50 m / 10 | Training | 8.75% | 99.34% | 0.18% |
| 50 m / 10 | Student testing | 0.00% | 99.88% | 0.02% |
| 100 m / 10 | Training | 0.67% | 99.47% | 0.18% |
| 100 m / 10 | Student testing | 0.00% | 99.91% | 0.02% |

See the CSV for all nine settings per partition. Alignment and candidate
retrieval both use geometric distance, and recall excludes uncertain points.
These choices can make reported recall optimistic for GPS observations with
poor alignment. The results describe candidate retrieval against a heuristic
label subset. They provide no classifier or decoder accuracy estimate.

## A7. Feature contract and reproducibility

A7 uses the exact uploaded `artifacts/feature_columns.json`. The export includes
all 24 feature columns in the supplied order, preceded by `trip_id`,
`point_idx` and `arc_id`. The model loader selects only the 24 named
features. `label` and `on_route` stay outside the feature tables. The export
uses cleaned GPS and the road graph with the A5 candidate tables (100 m,
K=10). The inference API can regenerate the candidate lists and features from
just GPS and the road network.

| Contract feature group | Implemented meaning |
| --- | --- |
| `dist`, `frac`, `arc_len` | Metre distance to the candidate, clamped projection fraction from its directed start, and metre segment length. |
| `dist_rank`, `n_cands` | One-based distance rank and count of returned candidates at the observation. |
| `dist_gap_best`, `dist_ratio_best` | Difference from the best distance and ratio with 0.001 m added to both distances. |
| `heading_diff`, `heading_cos` | Angle in degrees and cosine between directed arc and local GPS motion. GPS motion uses the previous-to-next chord, or the available one-sided step. |
| `along_ratio` | Signed projection of that motion along the road divided by arc length. This dimensionless value can exceed 1 or be negative. |
| `dist_p2`, `dist_p1`, `dist_n1`, `dist_n2` | Distances in metres from nearby GPS samples to the current candidate arc. |
| `win_mean_dist`, `win_max_dist` | Mean and maximum of the available distances in the five-observation window, including the current point. |
| `conn_prev`, `conn_next` | Category 1 if any adjacent-observation candidate is the same arc or has a directionally compatible shared node, else 0 when an adjacent candidate list is available. |
| `out_deg_end` | Directed outgoing arc count at the candidate's end node, counting parallel arcs. |
| `step_prev`, `step_next`, `speed`, `accel`, `dt_prev` | Distances to previous and next GPS samples in metres, backward speed in m/s, backward difference of speeds over current elapsed time in m/s², and previous elapsed time in seconds. |

Features requiring context use blank CSV fields for unavailable values at
trip edges, across GPS gaps over 5 s, or when heading is undefined due to zero
motion or zero arc length. `dt_prev` retains actual time intervals through
gaps when a predecessor exists. Missing neighbor candidate lists produce
missing connectivity categories. The heading and forward context require
batch inference or up to two observations of lookahead. Zero-based `point_idx`
retains the raw row index, including after filtering. An observation with an
empty candidate list contributes no feature row and remains represented in the
offline audit and candidate batch stream.

The contract specifies column names, order, exclusions and categorical fields.
It leaves units, formulas, category values and missing-value encoding open.
The table records our choices. Category values are fixed across splits and
cast to pandas categorical dtype. Any estimator-specific encoding or
imputation belongs in the training pipeline, fitted within training folds.
The repository remained unreachable, so Member B's actual estimator and
preprocessing code could not be checked.

| Export split | Trips | Candidate-feature rows |
| --- | ---: | ---: |
| Train | 64 | 1,306,859 |
| Validation | 16 | 306,757 |
| Student test | 10 | 157,851 |
| Total | 90 | 1,771,467 |

The run processed 227,577 observations. The verifier checked the original
810 trip/configuration metric rows, compared three sampled observations per
trip with exhaustive road search, and checked every feature row against its
split assignment, contract header and source candidate ID. It compared
the exported features with route-free inference on one trip per split.
All 12 unit and fixture integration tests passed, including the NetworkX
adapter. Folium was unavailable locally and the package includes geographic
PNG overlays and optional HTML generation code.

This package and report were produced with AI assistance. Reproduce the
run, review the independent analysis and disclose assistance as required
before using this material in the team submission.
