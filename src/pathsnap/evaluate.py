"""Evaluation metrics: RMF, edge precision/recall/F1, point accuracy.

All metrics are computed per trip and aggregated (macro = mean over trips;
micro = pooled counts). Classifier-only and HMM are evaluated on the same
eligible trips and points for paired comparison.

Reference-route alignment
-------------------------
The supplied `.route` is a sequence of 0-based arc/edge ids forming the
reference path. It is NOT per-point labeled. For point accuracy we therefore
derive a reference edge per GPS observation by snapping each observation to the
nearest point on the reference polyline (the concatenation of reference-edge
geometries). This is a documented, defensible alignment with these assumptions:

  * the reference route is traversed in the listed arc order;
  * each GPS point's reference assignment = the reference edge whose geometry
    passes closest to the observation.

This alignment is only used for point accuracy; RMF and edge F1 compare edge
sets directly on the shared network representation (edge ids), so they are
comparable between classifier-only and HMM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
from scipy.spatial import cKDTree

from .config import MatchConfig
from .graph import Graph, haversine_m
from .hmm import Candidate, MatchResult
from .io import Trip


@dataclass
class TripMetrics:
    trip_id: str
    num_points: int
    matched_points: int
    unmatched_points: int
    coverage: float  # matched_points / num_points

    # RMF (route-length based)
    rmf: float
    ref_length_m: float
    pred_length_m: float
    omitted_length_m: float
    extra_length_m: float
    ref_edges: int
    pred_edges: int

    # Edge F1 (direction-sensitive by default)
    edge_precision: float
    edge_recall: float
    edge_f1: float
    edge_policy: str  # 'directed' | 'undirected'
    edge_tp: int
    edge_fp: int
    edge_fn: int

    # Point accuracy
    point_accuracy: float
    point_correct: int
    point_evaluable: int  # points with both prediction and reference assignment
    point_unmatched: int  # evaluable points with no prediction

    # metadata
    representation: str  # 'stitched' or 'decoded'
    has_valid_reference: bool


@dataclass
class AggregateMetrics:
    n_trips: int
    n_evaluable_trips: int
    total_points: int
    total_matched_points: int
    mean_coverage: float

    mean_rmf: float
    mean_edge_precision: float
    mean_edge_recall: float
    mean_edge_f1: float
    mean_point_accuracy: float

    # micro / pooled
    micro_edge_precision: float
    micro_edge_recall: float
    micro_edge_f1: float

    per_trip: List[TripMetrics] = field(default_factory=list)


def _edge_lengths(graph: Graph) -> Dict[int, float]:
    return {int(eid): float(length) for eid, length in graph._edge_length.items()}


def _edge_key(eid: int, graph: Graph, directed: bool) -> Tuple:
    src, dst = graph._edge_endpoints[int(eid)]
    if directed:
        return ("d", int(eid))
    return ("u", min(int(src), int(dst)), max(int(src), int(dst)))


def rmf_from_sets(
    pred_edges: List[int],
    ref_edges: List[int],
    edge_lengths: Dict[int, float],
) -> Tuple[float, float, float, float, int, int]:
    """Route Mismatch Fraction.

    RMF = (omitted_length + extra_length) / reference_length
    where omitted = ref edges not predicted, extra = predicted edges not ref.

    Can exceed 1.0 when extra length is large relative to the reference.

    Zero-length reference (no ref edges):
        RMF = 0.0 if prediction also empty, else 1.0.
    """
    ref_set = set(ref_edges)
    pred_set = set(pred_edges)
    ref_len = sum(edge_lengths.get(e, 0.0) for e in ref_set)
    pred_len = sum(edge_lengths.get(e, 0.0) for e in pred_set)
    omitted = ref_set - pred_set
    extra = pred_set - ref_set
    omitted_len = sum(edge_lengths.get(e, 0.0) for e in omitted)
    extra_len = sum(edge_lengths.get(e, 0.0) for e in extra)

    if ref_len <= 0.0:
        rmf = 0.0 if len(pred_set) == 0 else 1.0
    else:
        rmf = (omitted_len + extra_len) / ref_len

    return rmf, ref_len, pred_len, omitted_len, extra_len, len(ref_set), len(pred_set)


def edge_f1_from_sets(
    pred_edges: List[int],
    ref_edges: List[int],
    graph: Graph,
    directed: bool = True,
) -> Dict[str, float]:
    """Direction-sensitive or -insensitive edge precision/recall/F1."""
    ref_set = set(ref_edges)
    pred_set = set(pred_edges)

    if directed:
        tp_set = pred_set & ref_set
        tp = len(tp_set)
    else:
        # canonicalize by undirected node pair
        ref_keys = {_edge_key(e, graph, False) for e in ref_set}
        pred_keys = {_edge_key(e, graph, False) for e in pred_set}
        tp_set = pred_keys & ref_keys
        tp = len(tp_set)

    fp = len(pred_set) - tp
    fn = len(ref_set) - tp

    if len(pred_set) == 0:
        precision = 1.0 if len(ref_set) == 0 else 0.0
    else:
        precision = tp / len(pred_set)

    if len(ref_set) == 0:
        recall = 1.0 if len(pred_set) == 0 else 0.0
    else:
        recall = tp / len(ref_set)

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }


def _reference_polyline(
    trip: Trip,
) -> Tuple[List[Tuple[float, float, float, float, int]], float]:
    """Build reference polyline segments from route edge ids.

    Returns list of (a_lon,a_lat,b_lon,b_lat,edge_id) and total length.
    The route is connected: arc[i].dst == arc[i+1].src (verified in most trips;
    if a break occurs we still treat each listed arc as its own segment).
    """
    segs = []
    total = 0.0
    arc_endpoints = {}
    for eid, (src, dst) in enumerate(trip.arcs):
        arc_endpoints[int(eid)] = (int(src), int(dst))
    for eid in trip.route:
        eid = int(eid)
        if eid not in arc_endpoints:
            continue  # invalid reference edge id: skip
        src, dst = arc_endpoints[eid]
        a = trip.nodes[src]
        b = trip.nodes[dst]
        segs.append((float(a[0]), float(a[1]), float(b[0]), float(b[1]), eid))
        total += haversine_m(a[0], a[1], b[0], b[1])
    return segs, total


def _snap_points_to_reference(
    points: List[Tuple[float, float, int]],
    ref_segs: List[Tuple[float, float, float, float, int]],
) -> List[Optional[int]]:
    """For each GPS point, return the nearest reference edge id (or None).

    Uses a KDTree on reference segment midpoints to limit exact projections to
    nearby segments.
    """
    if not ref_segs:
        return [None] * len(points)
    mids = np.array([[(s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0] for s in ref_segs],
                    dtype=np.float64)
    tree = cKDTree(mids)
    # search radius in degrees (generous); then exact-project
    mean_lat = float(np.mean([s[1] for s in ref_segs] + [s[3] for s in ref_segs]))
    m_per_deg_lon = 111132.0 * math.cos(math.radians(mean_lat))
    radius_deg = 500.0 / m_per_deg_lon if m_per_deg_lon > 0 else 0.005

    results: List[Optional[int]] = []
    for (lon, lat, _pidx) in points:
        idxs = tree.query_ball_point([lon, lat], r=radius_deg)
        if not idxs:
            # fall back to all segments
            idxs = list(range(len(ref_segs)))
        best_edge = None
        best_dist = float("inf")
        for idx in idxs:
            a_lon, a_lat, b_lon, b_lat, eid = ref_segs[int(idx)]
            proj_lon, proj_lat, _off, dist = _project_to_segment(
                lon, lat, a_lon, a_lat, b_lon, b_lat)
            if dist < best_dist:
                best_dist = dist
                best_edge = eid
        results.append(best_edge)
    return results


def _project_to_segment(lon, lat, a_lon, a_lat, b_lon, b_lat):
    lat_rad = math.radians((a_lat + b_lat) / 2.0)
    m_lat = 111132.0
    m_lon = 111132.0 * math.cos(lat_rad)
    ax, ay = a_lon * m_lon, a_lat * m_lat
    bx, by = b_lon * m_lon, b_lat * m_lat
    px, py = lon * m_lon, lat * m_lat
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 == 0.0:
        return a_lon, a_lat, 0.0, haversine_m(lon, lat, a_lon, a_lat)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    proj_lon = a_lon + (proj_x - ax) / m_lon
    proj_lat = a_lat + (proj_y - ay) / m_lat
    return proj_lon, proj_lat, t * math.sqrt(L2), math.hypot(px - proj_x, py - proj_y)


def _predicted_edges(result: MatchResult, representation: str) -> List[int]:
    if representation == "stitched":
        return [e.edge_id for e in result.stitched_route]
    if representation == "decoded":
        return result.decoded_edge_ids
    raise ValueError(f"unknown representation: {representation}")


def evaluate_match(
    result: MatchResult,
    trip: Trip,
    graph: Graph,
    config: MatchConfig,
    representation: str = "stitched",
) -> TripMetrics:
    """Evaluate one match result against the trip reference route.

    representation: 'stitched' (full stitched route, default) or 'decoded'
    (decoder-selected candidate edges only). The evaluator must know which
    representation it is scoring.
    """
    edge_lengths = _edge_lengths(graph)
    ref_edges = [int(e) for e in trip.route]
    pred_edges = _predicted_edges(result, representation)

    rmf, ref_len, pred_len, omitted_len, extra_len, n_ref, n_pred = rmf_from_sets(
        pred_edges, ref_edges, edge_lengths)

    f1 = edge_f1_from_sets(pred_edges, ref_edges, graph,
                           directed=(config.routing.directed))

    # Point accuracy via reference alignment
    ref_segs, _ = _reference_polyline(trip)
    ref_assign = _snap_points_to_reference(result.points, ref_segs)

    n_points = result.num_points
    n_matched = result.matched_count
    n_unmatched = len(result.unmatched_point_indices)
    coverage = (n_matched / n_points) if n_points else 0.0

    correct = 0
    evaluable = 0
    point_unmatched = 0
    for pidx in range(n_points):
        ref_edge = ref_assign[pidx]
        if ref_edge is None:
            continue  # no reference assignment for this point
        pred = result.matched[pidx]
        evaluable += 1
        if pred is None:
            point_unmatched += 1
            continue
        if int(pred.edge_id) == int(ref_edge):
            correct += 1

    point_accuracy = (correct / evaluable) if evaluable else 0.0

    return TripMetrics(
        trip_id=result.trip_id,
        num_points=n_points,
        matched_points=n_matched,
        unmatched_points=n_unmatched,
        coverage=coverage,
        rmf=rmf,
        ref_length_m=ref_len,
        pred_length_m=pred_len,
        omitted_length_m=omitted_len,
        extra_length_m=extra_len,
        ref_edges=n_ref,
        pred_edges=n_pred,
        edge_precision=f1["precision"],
        edge_recall=f1["recall"],
        edge_f1=f1["f1"],
        edge_policy="directed" if config.routing.directed else "undirected",
        edge_tp=f1["tp"],
        edge_fp=f1["fp"],
        edge_fn=f1["fn"],
        point_accuracy=point_accuracy,
        point_correct=correct,
        point_evaluable=evaluable,
        point_unmatched=point_unmatched,
        representation=representation,
        has_valid_reference=len(ref_edges) > 0,
    )


def aggregate(metrics: List[TripMetrics]) -> AggregateMetrics:
    """Macro-average (mean over trips) plus micro/pooled edge F1."""
    n = len(metrics)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, [])

    def mean(attr):
        vals = [getattr(m, attr) for m in metrics]
        return sum(vals) / len(vals)

    total_tp = sum(m.edge_tp for m in metrics)
    total_fp = sum(m.edge_fp for m in metrics)
    total_fn = sum(m.edge_fn for m in metrics)
    mic_p = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    mic_r = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
    mic_f1 = (2 * mic_p * mic_r / (mic_p + mic_r)) if (mic_p + mic_r) else 0.0

    return AggregateMetrics(
        n_trips=n,
        n_evaluable_trips=sum(1 for m in metrics if m.has_valid_reference),
        total_points=sum(m.num_points for m in metrics),
        total_matched_points=sum(m.matched_points for m in metrics),
        mean_coverage=mean("coverage"),
        mean_rmf=mean("rmf"),
        mean_edge_precision=mean("edge_precision"),
        mean_edge_recall=mean("edge_recall"),
        mean_edge_f1=mean("edge_f1"),
        mean_point_accuracy=mean("point_accuracy"),
        micro_edge_precision=mic_p,
        micro_edge_recall=mic_r,
        micro_edge_f1=mic_f1,
        per_trip=list(metrics),
    )
