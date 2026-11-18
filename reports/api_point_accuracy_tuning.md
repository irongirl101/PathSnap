# API point-accuracy tuning

The API now uses the fitted LightGBM ranker with the HMM. I tuned the HMM
settings against a trip-level holdout from the training partition, using the
same candidate features, ranker, matcher, and point-accuracy calculation as the
API. The selected holdout trips are listed in `reports/splits.json` under
`test`; they were excluded from the ranker fit.

The first search covered 72 combinations of `beta` and `alpha`. I then varied
the plausibility factor and minimum route gap around the strongest
beta/alpha settings. The best mean point accuracy on the 12 training-holdout
trips was **92.06%**, with edge F1 **0.739**.

| Setting | Value |
| --- | ---: |
| `routing.beta` | 10 |
| `emission.alpha` | 0.25 |
| `routing.plausibility_factor` | 20 |
| `routing.min_route_gap_m` | 50 m |

As a development check, the fixed settings scored **90.23% mean point
accuracy** and **0.649 mean edge F1** across the 10 `student_test` trips. The
previous API defaults scored 88.07% point accuracy and 0.633 edge F1 on the
same trips. The 90% figure is an average across trips; several individual
trips remain below 90%.

The reserved evaluator trips were not used to choose settings. They had been
scored once with the earlier defaults before this tuning began, at the user's
request, so any later evaluator result should be treated as a final check with
that earlier exploratory score already seen. With the selected settings, the
API's post-selection check across all 10 evaluator trips scored **91.34% mean
point accuracy** and **0.697 mean edge F1** (precision 0.543, recall 0.994,
coverage 1.000). This check did not feed back into the configuration.

`src.pathsnap.api_adapter` now loads `config/default.yaml` when
`PATHSNAP_CONFIG` is unset. Set that environment variable to use a different
configuration file.

## Version note

The measurements above were collected before the latest `origin/main` merge,
which changed feature extraction and the model wrapper. They document the
selected HMM settings and their results on that earlier code version. The API
integration suite passes on the merged code, but a fresh end-to-end evaluation
of the full evaluator set did not finish in the local run; the earlier scores
should therefore not be read as a verified 90% result for the merged version.
