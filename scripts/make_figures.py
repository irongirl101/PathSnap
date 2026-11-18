"""Generate geographic success/failure figures for PathSnap Member C.

Produces PNG figures under artifacts/figures/ for selected test trips.
Each figure shows:
  - GPS observations (scatter)
  - Reference route (from .route file)
  - Classifier-only route
  - HMM route

All layers share the same CRS (lon/lat degrees, equal aspect).
Uses matplotlib only (no basemap/internet required).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.io import load_trip, build_graph, trip_dir
from pathsnap.candidates import generate_candidates
from pathsnap.hmm import match_trip, classifier_only
from pathsnap.evaluate import evaluate_match
from pathsnap.config import MatchConfig, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.getenv("PATHSNAP_DATA_ROOT", REPO_ROOT.parent / "Default Project" / "Student folder")
)
TEST_ROOT = DATA_ROOT / "student_test"
FIG_DIR = REPO_ROOT / "artifacts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Selected trips: success, failure, ambiguity/coverage
SELECTED_TRIPS = [
    ("00000056", "success"),   # HMM f1=0.758, best HMM F1
    ("00000007", "success"),   # HMM much better than CLF (0.629 vs 0.498)
    ("00000065", "failure"),   # HMM rmf=3.115 worst, CLF much better
    ("00000091", "failure"),   # CLF better f1 and rmf
    ("00000000", "ambiguity"), # low coverage 0.636, many unmatched
]


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def plot_trip(trip_id: str, case: str, config: MatchConfig):
    """Generate one figure for a test trip."""
    trip = load_trip(trip_id, str(trip_dir(TEST_ROOT, trip_id)))
    graph = build_graph(
        trip,
        directed=config.routing.directed,
        max_route_distance_m=config.routing.max_route_distance_m,
    )

    points = [(float(t[0]), float(t[1]), int(t[2])) for t in trip.track]
    cands = generate_candidates(trip.trip_id, points, graph)

    hmm_res = match_trip(trip.trip_id, points, cands, graph, config)
    clf_res = classifier_only(trip.trip_id, points, cands)

    hmm_m = evaluate_match(hmm_res, trip, graph, config, "stitched")
    clf_m = evaluate_match(clf_res, trip, graph, config, "stitched")

    # --- Build geometry ---
    # GPS points
    gps_lons = [p[0] for p in points]
    gps_lats = [p[1] for p in points]

    # Reference route from .route edge ids
    arc_endpoints = {}
    for eid, (src, dst) in enumerate(trip.arcs):
        arc_endpoints[int(eid)] = (int(src), int(dst))
    ref_lons, ref_lats = [], []
    for eid in trip.route:
        eid = int(eid)
        if eid not in arc_endpoints:
            continue
        src, dst = arc_endpoints[eid]
        a = trip.nodes[src]
        b = trip.nodes[dst]
        ref_lons.extend([float(a[0]), float(b[0]), None])
        ref_lats.extend([float(a[1]), float(b[1]), None])

    # Classifier-only route edges
    clf_edges = [int(e.edge_id) for e in clf_res.stitched_route]
    clf_lons, clf_lats = [], []
    for eid in clf_edges:
        if eid not in arc_endpoints:
            continue
        src, dst = arc_endpoints[eid]
        a = trip.nodes[src]
        b = trip.nodes[dst]
        clf_lons.extend([float(a[0]), float(b[0]), None])
        clf_lats.extend([float(a[1]), float(b[1]), None])

    # HMM route edges (stitched)
    hmm_edges = [int(e.edge_id) for e in hmm_res.stitched_route]
    hmm_lons, hmm_lats = [], []
    for eid in hmm_edges:
        if eid not in arc_endpoints:
            continue
        src, dst = arc_endpoints[eid]
        a = trip.nodes[src]
        b = trip.nodes[dst]
        hmm_lons.extend([float(a[0]), float(b[0]), None])
        hmm_lats.extend([float(a[1]), float(b[1]), None])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reference route (gray, thin, dashed)
    ax.plot(ref_lons, ref_lats, color="#7f7f7f", linewidth=1.5, linestyle="--",
            alpha=0.7, label="Reference route")

    # Classifier-only (orange)
    ax.plot(clf_lons, clf_lats, color="#ff7f0e", linewidth=2, linestyle="-",
            alpha=0.9, label="Classifier-only")

    # HMM route (blue, thick)
    ax.plot(hmm_lons, hmm_lats, color="#1f77b4", linewidth=2.5, linestyle="-",
            alpha=0.9, label="HMM (tuned)")

    # GPS observations (red dots, small)
    ax.scatter(gps_lons, gps_lats, color="#d62728", s=8, alpha=0.6,
               label="GPS observations", zorder=5)

    # Mark unmatched points (black x)
    unmatched = set(hmm_res.unmatched_point_indices)
    if unmatched:
        um_lons = [points[i][0] for i in unmatched]
        um_lats = [points[i][1] for i in unmatched]
        ax.scatter(um_lons, um_lats, color="#000000", marker="x", s=40,
                   linewidth=1.5, label="Unmatched (HMM)", zorder=6)

    # Mark gaps
    pidx_to_pos = {p[2]: i for i, p in enumerate(points)}
    for (prev_pidx, cur_pidx) in hmm_res.gaps:
        if prev_pidx in pidx_to_pos and cur_pidx in pidx_to_pos:
            ax.annotate("", xy=(points[pidx_to_pos[cur_pidx]][0], points[pidx_to_pos[cur_pidx]][1]),
                        xytext=(points[pidx_to_pos[prev_pidx]][0], points[pidx_to_pos[prev_pidx]][1]),
                        arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))
    if hmm_res.gaps:
        # dummy for legend
        ax.plot([], [], color="purple", lw=1.5, label="GPS gap")

    # Formatting
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    title = (f"Trip {trip_id} — {case.capitalize()} case\n"
             f"HMM: F1={hmm_m.edge_f1:.3f} RMF={hmm_m.rmf:.3f} Acc={hmm_m.point_accuracy:.3f} Cov={hmm_m.coverage:.3f} | "
             f"CLF: F1={clf_m.edge_f1:.3f} RMF={clf_m.rmf:.3f} Acc={clf_m.point_accuracy:.3f} Cov={clf_m.coverage:.3f} | "
             f"β={config.routing.beta}, α={config.emission.alpha}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / f"trip_{trip_id}_{case}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    config = load_config(str(REPO_ROOT / "config" / "default.yaml"))
    for trip_id, case in SELECTED_TRIPS:
        plot_trip(trip_id, case, config)
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
