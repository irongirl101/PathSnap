"""Build a factual PDF evidence report from the saved model/evaluation outputs."""
import argparse
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


MODEL_NOTES = {
    "logreg": (
        "A linear probabilistic baseline tests whether the engineered geometry, "
        "heading, kinematic, and topology features separate the selected candidate "
        "from alternatives without nonlinear interactions.",
        "Fast scoring and a compact, interpretable decision boundary; it may miss "
        "nonlinear feature interactions.",
    ),
    "rf": (
        "A bagged tree ensemble tests nonlinear thresholds and feature interactions "
        "on the same candidate-level feature contract.",
        "Can model nonlinear patterns with little scaling; it is larger and slower "
        "than linear models, so training uses a row subsample and shallow trees.",
    ),
    "lgbm": (
        "Gradient-boosted trees test whether nonlinear feature interactions improve "
        "candidate classification over the linear baseline.",
        "Captures nonlinear effects efficiently; probabilities require calibration "
        "and the model can overfit if tree complexity is not controlled.",
    ),
    "lgbm_tuned": (
        "The tuned LightGBM variant tests the Bayesian-search configuration on the "
        "same training folds and student-test observations as the untuned model.",
        "The search targets grouped-CV log loss; optimization is computationally "
        "costlier and the test set remains the final comparison, not a tuning input.",
    ),
    "lgbm_ranker": (
        "LambdaRank optimizes relative candidate ordering within each GPS-observation "
        "group, matching the one-road-choice-per-observation decision.",
        "Directly models within-point ranking; its scores are not probabilities and "
        "must be normalized/calibrated before serving as HMM emissions.",
    ),
    "mlp": (
        "A feed-forward neural network tests nonlinear combinations of the engineered "
        "features, with early stopping based on an internal validation subset.",
        "Can represent complex feature interactions; optimization is less transparent "
        "and convergence depends on scaling and early stopping.",
    ),
    "svm": (
        "A linear support-vector classifier tests a maximum-margin boundary on the "
        "same scaled feature set; sigmoid calibration makes scores usable by the "
        "probability-based route decoder.",
        "Provides a regularized linear decision rule; calibration adds fit cost and "
        "the linear kernel cannot represent nonlinear boundaries.",
    ),
}
MODEL_OBJECTIVES = {
    "logreg": "Binary logistic log loss with L2 regularization.",
    "rf": "Gini-impurity splits in the random-forest trees.",
    "lgbm": "LightGBM binary classification objective (binary log loss).",
    "lgbm_tuned": "Binary log loss; Optuna minimizes its mean over grouped CV folds.",
    "lgbm_ranker": "LambdaRank pairwise ranking objective with NDCG evaluation.",
    "mlp": "Binary cross-entropy optimized by Adam.",
    "svm": "Linear hinge-loss margin classifier with sigmoid probability calibration.",
}


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _paragraph(text, style):
    return Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), style)


def _table(rows, widths=None, header=True, font_size=7):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#bcc6d1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    table.setStyle(TableStyle(commands))
    return table


def build_report(results_dir, output_path):
    root = Path(results_dir)
    summary = json.loads((root / "evaluation_summary.json").read_text(encoding="utf-8"))
    models = _read_csv(root / "model_comparison.csv")
    route_summary = json.loads((root / "route_metrics_summary.json").read_text(encoding="utf-8"))
    quality = _read_csv(root / "data_quality_summary.csv")
    missingness = _read_csv(root / "feature_missingness.csv")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER,
                              textColor=colors.HexColor("#17365d"), fontSize=25, leading=30))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="ModelTitle", parent=styles["Heading3"], spaceBefore=9))
    doc = SimpleDocTemplate(
        str(output_path), pagesize=landscape(A4), rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm, title="PathSnap Submission Evidence",
        author="PathSnap team",
    )
    story = [
        Spacer(1, 18 * mm),
        _paragraph("PathSnap", styles["CoverTitle"]),
        _paragraph("Technical Report and Experimental Evidence", styles["Heading2"]),
        Spacer(1, 7 * mm),
        _paragraph(
            "Generated directly from the repository's feature, training, evaluation, "
            "and route-visualization outputs. Reported model-selection metrics use "
            "training-only trip-grouped cross-validation; the separate student-test "
            "partition is reported for evaluation and was not used for model selection.",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
        _paragraph(
            f"Training: {summary['training_partition']['trips']} trips, "
            f"{summary['training_partition']['raw_points']:,} raw GPS points. "
            f"Student test: {summary['student_test_partition']['trips']} trips, "
            f"{summary['student_test_partition']['raw_points']:,} raw GPS points. "
            f"Selected estimator: {summary['selected_model']} (selection metric: "
            f"{summary['selection_metric']}).",
            styles["BodyText"],
        ),
        PageBreak(),
        _paragraph("1. Data, preprocessing, and features", styles["Heading1"]),
        _paragraph(
            "Inputs are headerless per-trip files: longitude/latitude/time observations, "
            "longitude/latitude road nodes, directed node-pair arcs, and an ordered "
            "reference arc sequence. Raw coordinates are preserved in longitude, "
            "latitude order. Training features are projected into local metres; "
            "candidate roads are searched within 100 m and capped at ten per observation. "
            "Large time gaps are not treated as adjacent spatial samples. The label "
            "selects the nearest on-reference candidate when one is present; groups "
            "without a positive candidate are excluded from candidate-classifier fitting "
            "and counted as end-to-end misses.",
            styles["BodyText"],
        ),
        Spacer(1, 3 * mm),
        _paragraph(
            "The 24-feature contract covers candidate distance and rank, road geometry, "
            "GPS heading, nearby-sample distances, directed topology, and kinematics. "
            "Identifiers and labels are excluded from model inputs. Missing context "
            "features remain missing and are handled by estimator pipelines where needed.",
            styles["BodyText"],
        ),
        Spacer(1, 3 * mm),
        _paragraph("Raw observation quality", styles["Heading2"]),
        _table(
            [["Partition", "Trips", "Raw points", "Retained", "Missing rows",
              "Malformed", "Nonfinite", "Invalid coords", "Gaps > 5 s"]]
            + [[row["partition"], row["trips"], f"{int(row['raw_points']):,}",
                f"{int(row['clean_points']):,}", row["missing_rows"], row["malformed_rows"],
                row["nonfinite_rows"], row["invalid_coordinates"], row["gaps"]]
               for row in quality],
            widths=[30 * mm, 16 * mm, 28 * mm, 25 * mm, 27 * mm,
                    23 * mm, 22 * mm, 26 * mm, 26 * mm],
        ),
        Spacer(1, 3 * mm),
        _paragraph(
            "Most-missing model features (candidate rows): "
            + "; ".join(
                f"{row['feature']}: train {float(row['train_missing_rate']):.1%}, "
                f"student test {float(row['student_test_missing_rate']):.1%}"
                for row in sorted(missingness,
                                  key=lambda item: float(item["train_missing_rate"]),
                                  reverse=True)[:5]
            ),
            styles["Small"],
        ),
        Spacer(1, 3 * mm),
        _paragraph("2. Model comparison", styles["Heading1"]),
    ]
    table_headers = [
        "Model", "Family / configuration", "CV macro F1", "CV end-to-end Top-1",
        "Test Top-1", "Test end-to-end Top-1", "Test macro F1", "Test log loss",
        "Inference ms / 1k rows",
    ]
    table_rows = [table_headers]
    for row in models:
        table_rows.append([
            row["Model"],
            Paragraph(f"{row['Algorithm family']}<br/>{row['Key configuration']}", styles["Small"]),
            row["CV macro F1"], row["CV Top-1 (all raw points)"],
            row["Student-test Top-1"], row["Student-test end-to-end Top-1"],
            row["Student-test macro F1"], row["Student-test log loss"],
            row["Inference ms / 1k rows"],
        ])
    story.extend([
        _table(table_rows, widths=[19 * mm, 60 * mm, 24 * mm, 29 * mm, 21 * mm,
                                  31 * mm, 24 * mm, 23 * mm, 25 * mm], font_size=6.5),
        Spacer(1, 3 * mm),
        _paragraph(
            "CV folds are grouped by trip, so observations from one trip do not cross "
            "between fold training and validation. Student-test candidate metrics include "
            "only observations with a positive candidate; end-to-end Top-1 divides correct "
            "matches by all raw observations, including candidate misses. Inference time "
            "is measured per 1,000 candidate rows and excludes feature generation.",
            styles["Small"],
        ),
        Spacer(1, 4 * mm),
        Image(str(root / "model_comparison.png"), width=250 * mm, height=78 * mm),
        PageBreak(),
        _paragraph("3. Model-specific results and interpretation", styles["Heading1"]),
    ])
    for row in models:
        name = row["Model"].split(". ", 1)[-1]
        fit, tradeoff = MODEL_NOTES[name]
        story.append(_paragraph(f"{name}: {row['Algorithm family']}", styles["ModelTitle"]))
        story.append(_paragraph(
            f"Configuration: {row['Key configuration']}. "
            f"Training objective: {MODEL_OBJECTIVES[name]} "
            f"Task fit: {fit} "
            f"Strength and limitation: {tradeoff} "
            f"Measured result: CV macro F1 {row['CV macro F1']}; actual "
            f"student-test Top-1 {row['Student-test Top-1']}, macro F1 "
            f"{row['Student-test macro F1']}, log loss {row['Student-test log loss']}. "
            "Compare CV and student-test results cautiously: the latter contains "
            "only ten trips and is an estimate, not proof of overfitting or generalization.",
            styles["BodyText"],
        ))
    history = root / "mlp_training_history.png"
    if history.is_file():
        story.extend([
            Spacer(1, 4 * mm),
            _paragraph("MLP training history (first CV fold)", styles["Heading2"]),
            Image(str(history), width=170 * mm, height=68 * mm),
        ])

    story.extend([
        PageBreak(),
        _paragraph("4. Bayesian hyperparameter optimization", styles["Heading1"]),
    ])
    params_file = Path("artifacts/best_params_lgbm.json")
    trials_file = Path("reports/optuna/lgbm_trials.csv")
    if params_file.is_file() and trials_file.is_file():
        params = json.loads(params_file.read_text(encoding="utf-8"))
        trials = _read_csv(trials_file)
        story.append(_paragraph(
            f"Optuna TPE minimizes mean trip-grouped cross-validation log loss. "
            f"The search evaluated {len(trials)} persisted trial records; the selected "
            f"parameters were {json.dumps(params['params'], sort_keys=True)}. "
            f"Untuned default CV log loss: {params.get('untuned_logloss', 'not recorded')}; "
            f"best search CV log loss: {params.get('tuned_logloss', 'not recorded')}. "
            "Tuned-vs-default results below use the same reported folds and held-out "
            "student-test observations.",
            styles["BodyText"],
        ))
    else:
        story.append(_paragraph(
            "No completed model-specific Bayesian study was found when this report was "
            "generated. Run the documented Optuna command before treating this section "
            "as complete.",
            styles["BodyText"],
        ))
    story.extend([
        Spacer(1, 4 * mm),
        _paragraph("5. HMM/Viterbi route reconstruction", styles["Heading1"]),
        _paragraph(
            "The matcher uses candidate road arcs as hidden states. Emissions come from "
            "the selected model and are combined with directed network-transition scores "
            "by log-space Viterbi decoding. Missing candidate sets split the lattice; "
            "time/distance gaps and disconnected transitions are handled by the configured "
            "matcher policy. The following comparison uses the same student-test trips, "
            "features, candidates, and reference routes.",
            styles["BodyText"],
        ),
    ])
    route_rows = [["Method", "Point accuracy", "Coverage", "RMF", "Edge precision",
                   "Edge recall", "Edge F1", "Route continuity"]]
    for method, row in route_summary.items():
        route_rows.append([
            method, f"{row['point_accuracy']:.4f}", f"{row['coverage']:.4f}",
            f"{row['rmf']:.4f}", f"{row['edge_precision']:.4f}",
            f"{row['edge_recall']:.4f}", f"{row['edge_f1']:.4f}",
            f"{row['route_continuity']:.4f}",
        ])
    story.extend([
        _table(route_rows, widths=[34 * mm, 25 * mm, 22 * mm, 20 * mm,
                                  27 * mm, 22 * mm, 20 * mm, 30 * mm]),
        Spacer(1, 4 * mm),
        _paragraph("6. Geographic and deployment evidence", styles["Heading1"]),
    ])
    map_image = root / "map_00000000.png"
    if map_image.is_file():
        story.append(_paragraph(
            "Representative student-test map with observed GPS, reference route, "
            "predicted route, and surrounding road network. The matching interactive "
            "Folium HTML is included beside this PDF.",
            styles["BodyText"],
        ))
        story.append(Image(str(map_image), width=225 * mm, height=112 * mm))
    ui_image = root / "ui_screenshot.png"
    if ui_image.is_file():
        story.append(PageBreak())
        story.append(_paragraph("Working evaluator UI", styles["Heading2"]))
        story.append(Image(str(ui_image), width=250 * mm, height=140 * mm))
    story.extend([
        Spacer(1, 3 * mm),
        _paragraph(
            "The FastAPI application serves the evaluator UI, trip listing, map-match "
            "endpoint, and server-side scoring endpoint. Raw data and reserved evaluator "
            "trips are not tracked; the saved calibrated model is versioned as a joblib "
            "artifact. Setup, training, evaluation, API, and deployment commands are "
            "documented in README.md.",
            styles["BodyText"],
        ),
        Spacer(1, 5 * mm),
        _paragraph(
            "Review note: these tables and plots are generated from saved experiment "
            "outputs. The team must verify the model rationale, discuss observed failure "
            "cases, follow the event's AI disclosure rules, and capture any required "
            "receiving-laptop terminal screenshot before final submission.",
            styles["Small"],
        ),
    ])

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#68788a"))
        canvas.drawString(16 * mm, 8 * mm, "PathSnap | generated from repository experiment outputs")
        canvas.drawRightString(landscape(A4)[0] - 16 * mm, 8 * mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(f"saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="reports/student_test")
    parser.add_argument("--output", default="reports/student_test/PathSnap_submission_evidence.pdf")
    args = parser.parse_args()
    build_report(args.results_dir, args.output)
