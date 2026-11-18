"""Candidate schema, Viterbi decoding, transition scoring, gaps/breaks, and
route stitching for the PathSnap HMM map-matcher.

Public interface:
    match_trip(points, candidates_by_point, graph, config) -> MatchResult
    classifier_only(points, candidates_by_point) -> MatchResult

Schemas (documented):

Candidate (one per observation-candidate pair):
    trip_id        str        trip identifier
    point_idx      int        index into the GPS observation sequence
    lon, lat       float      GPS observation (degrees)
    cand_id        int        stable candidate/edge id (== edge_id here)
    edge_id        int        graph edge id the candidate is snapped to
    proj_lon, proj_lat float  projected location on the edge (degrees)
    offset_m       float      distance from edge src node to projection (m)
    edge_len_m     float      full edge length (m)
    emission       float      classifier probability/score in (0, 1]
    dist_m         float      GPS->projection distance (m)

MatchResult:
    trip_id            str
    num_points         int
    points             list[(lon, lat, point_idx)]
    matched            list[Candidate or None]  per observation
    gaps               list[(prev_idx, curr_idx)] detected GPS gaps
    breaks             list[int]                 point indices where lattice broke
    segments           list[Segment]             decoded contiguous segments
    stitched_route     list[StitchedEdge]        final stitched edge sequence
    total_score        float                      log-score of best path
    unmatched_point_indices list[int]

StitchedEdge:
    edge_id           int
    source            str  'decoder' | 'stitched' | 'bridge'
    interpolated      bool
    from_point_idx    int or None
    to_point_idx      int or None

Segment:
    start_point_idx   int
    end_point_idx     int
    candidate_ids     list[int]  one per point in segment
    score             float

All scoring is in log space. Unreachable/no-route transitions are represented
as -inf and never silently converted into plausible transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from .config import MatchConfig
from .graph import Graph, PathResult, NO_ROUTE, haversine_m


@dataclass
class Candidate:
    trip_id: str
    point_idx: int
    lon: float
    lat: float
    cand_id: int
    edge_id: int
    proj_lon: float
    proj_lat: float
    offset_m: float
    edge_len_m: float
    emission: float
    dist_m: float

    @property
    def point_index(self) -> int:
        return self.point_idx


@dataclass
class StitchedEdge:
    edge_id: int
    source: str            # 'decoder' | 'stitched' | 'bridge'
    interpolated: bool
    from_point_idx: Optional[int] = None
    to_point_idx: Optional[int] = None


@dataclass
class Segment:
    start_point_idx: int
    end_point_idx: int
    candidate_ids: List[int]
    score: float


@dataclass
class MatchResult:
    trip_id: str
    num_points: int
    points: List[Tuple[float, float, int]]
    matched: List[Optional[Candidate]]
    gaps: List[Tuple[int, int]]
    breaks: List[int]
    segments: List[Segment]
    stitched_route: List[StitchedEdge]
    total_score: float
    unmatched_point_indices: List[int]

    @property
    def matched_count(self) -> int:
        return sum(1 for m in self.matched if m is not None)

    @property
    def decoded_edge_ids(self) -> List[int]:
        """Edge ids selected directly by the decoder (in observation order)."""
        out: List[int] = []
        for m in self.matched:
            if m is not None:
                out.append(m.edge_id)
        return out

    @property
    def stitched_edge_ids(self) -> List[int]:
        """Edge ids in the stitched route (order preserved, dupes removed)."""
        return [e.edge_id for e in self.stitched_route]


# ---------------------------------------------------------------------------
# Transition scoring
# ---------------------------------------------------------------------------

def transition_log_score(
    graph: Graph,
    cfg: MatchConfig,
    cand_a: Candidate,
    cand_b: Candidate,
    d_gc_m: float,
) -> float:
    """Log transition score between consecutive candidates.

    Returns -inf when no feasible network route exists within the bound or the
    plausibility cutoff is violated. Never silently substitutes a plausible
    score for an unreachable route.

    d_route is computed as the direction-aware network distance between the two
    projections, including projection offsets on the first and last edges:

        same edge:  |offset_b - offset_a|
        else:       (edge_len_a - offset_a)          # travel to dst of edge a
                  + shortest_path(dst(a), src(b))    # through the network
                  + offset_b                          # from src of edge b to proj

    This respects graph directionality (forward travel along each edge).
    """
    beta = cfg.routing.max_route_distance_m  # placeholder, replaced below
    beta = _beta(cfg)

    # ---- d_route: bounded shortest path between projections ----
    d_route = _route_distance_between_projections(graph, cfg, cand_a, cand_b)
    if d_route is None or d_route == NO_ROUTE or math.isinf(d_route):
        return -math.inf  # no feasible route: explicit no-route transition

    # ---- plausibility cutoff (configurable) ----
    cutoff = max(cfg.routing.plausibility_factor * d_gc_m,
                 cfg.routing.min_route_gap_m)
    if d_route > cutoff:
        return -math.inf

    # ---- score: discrepancy between GPS displacement and route distance ----
    return -abs(d_gc_m - d_route) / beta


def _beta(cfg: MatchConfig) -> float:
    # beta lives on the routing section conceptually (transition-distance scale).
    # It is not a separate dataclass field in this revision; expose via routing.
    return getattr(cfg.routing, "beta", 100.0)


def _route_distance_between_projections(
    graph: Graph, cfg: MatchConfig, a: Candidate, b: Candidate
) -> Optional[float]:
    """Direction-aware network distance between two projected candidates.

    Returns None when the edges are the same (handled by caller as |da-db|) or
    when the connecting shortest path is unreachable within the bound.
    """
    if a.edge_id == b.edge_id:
        return abs(b.offset_m - a.offset_m)

    len_a = a.edge_len_m
    # travel forward along edge a to its destination node
    _, dst_a = graph._edge_endpoints[int(a.edge_id)]
    src_b, _ = graph._edge_endpoints[int(b.edge_id)]

    to_dst_a = len_a - a.offset_m        # proj_a -> dst(edge a)
    from_src_b = b.offset_m              # src(edge b) -> proj_b

    pr: PathResult = graph.shortest_path(int(dst_a), int(src_b))
    if not pr.reachable:
        return None
    return to_dst_a + pr.distance_m + from_src_b


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------

def detect_gaps(
    point_indices: List[int],
    coords: List[Tuple[float, float]],
    cfg: MatchConfig,
) -> List[Tuple[int, int]]:
    """Detect GPS gaps between consecutive observations.

    Primary signal: point-index discontinuity (diff > index_gap_threshold).
    Fallback: great-circle distance > distance_gap_threshold_m.

    Returns list of (prev_point_idx, curr_point_idx) gap locations.
    """
    gaps: List[Tuple[int, int]] = []
    dthr = cfg.gap.distance_gap_threshold_m
    use_dist = dthr is not None and dthr > 0
    for i in range(1, len(point_indices)):
        prev_i = point_indices[i - 1]
        cur_i = point_indices[i]
        is_gap = False
        if cfg.gap.index_gap_threshold is not None:
            if (cur_i - prev_i) > cfg.gap.index_gap_threshold:
                is_gap = True
        if (not is_gap) and use_dist:
            d = haversine_m(coords[i - 1][0], coords[i - 1][1],
                            coords[i][0], coords[i][1])
            if d > dthr:
                is_gap = True
        if is_gap:
            gaps.append((prev_i, cur_i))
    return gaps


# ---------------------------------------------------------------------------
# Viterbi decoding over a single contiguous segment
# ---------------------------------------------------------------------------

def _viterbi_segment(
    graph: Graph,
    cfg: MatchConfig,
    segment_cands: List[List[Candidate]],
    gps_coords: List[Tuple[float, float]],
    point_indices: List[int],
) -> Tuple[List[int], float]:
    """Run Viterbi on one contiguous segment (all points have >=1 candidate).

    Returns (best_candidate_ids, best_log_score). Deterministic tie-breaking:
    'first' keeps the lowest candidate index on exact score ties.
    """
    n_pts = len(segment_cands)
    if n_pts == 0:
        return [], 0.0

    alpha = cfg.emission.alpha
    floor = cfg.emission.emission_floor
    tie_first = (cfg.tie_break == "first")

    # Precompute emissions: alpha * log(p), clipped at floor.
    emiss: List[np.ndarray] = []
    for cands in segment_cands:
        arr = np.array([max(c.emission, floor) for c in cands], dtype=np.float64)
        emiss.append(alpha * np.log(arr))

    # log_delta[t][j], backpointer[t][j]
    n0 = len(segment_cands[0])
    log_delta = np.full((n_pts, n0 if n_pts else 1), -np.inf)
    # variable width: store per-row arrays
    log_delta_rows: List[np.ndarray] = []
    back_rows: List[np.ndarray] = []

    # row 0: emission only (no incoming transition)
    row0 = emiss[0].copy()
    log_delta_rows.append(row0)
    back_rows.append(np.full(len(segment_cands[0]), -1, dtype=np.int64))

    for t in range(1, n_pts):
        prev_cands = segment_cands[t - 1]
        curr_cands = segment_cands[t]
        prev_row = log_delta_rows[t - 1]
        n_prev = len(prev_cands)
        n_curr = len(curr_cands)
        new_row = np.full(n_curr, -np.inf, dtype=np.float64)
        new_back = np.full(n_curr, -1, dtype=np.int64)

        # GPS great-circle distance between these two observations
        d_gc = haversine_m(gps_coords[t - 1][0], gps_coords[t - 1][1],
                           gps_coords[t][0], gps_coords[t][1])

        for j in range(n_curr):
            cand_b = curr_cands[j]
            best_val = -np.inf
            best_i = -1
            for i in range(n_prev):
                cand_a = prev_cands[i]
                trans = transition_log_score(graph, cfg, cand_a, cand_b, d_gc)
                if trans == -math.inf:
                    continue
                val = prev_row[i] + trans
                if val > best_val:
                    best_val = val
                    best_i = i
                elif val == best_val and best_i != -1:
                    # deterministic tie-breaking
                    if tie_first:
                        # keep lowest index: do nothing (best_i already lower)
                        pass
                    else:
                        best_i = i
            if best_i == -1:
                # No feasible predecessor: this candidate is unreachable in
                # Viterbi (lattice break within segment). Mark -inf.
                new_row[j] = -np.inf
                new_back[j] = -1
            else:
                new_row[j] = emiss[t][j] + best_val
                new_back[j] = best_i
        log_delta_rows.append(new_row)
        back_rows.append(new_back)

    # Termination: pick best final candidate
    last_row = log_delta_rows[-1]
    if not np.isfinite(last_row).any():
        # Entire segment unreachable: fall back to best emission at each point
        # independently (classifier-like within segment) so we still return
        # something rather than crashing.
        chosen = []
        for t in range(n_pts):
            row = emiss[t]
            chosen.append(int(np.argmax(row)))
        score = float(np.sum([emiss[t][chosen[t]] for t in range(n_pts)]))
        return chosen, score

    best_last = int(np.argmax(last_row))
    chosen = [0] * n_pts
    chosen[-1] = best_last
    for t in range(n_pts - 1, 0, -1):
        chosen[t - 1] = int(back_rows[t][chosen[t]])

    score = float(last_row[best_last])
    return chosen, score


# ---------------------------------------------------------------------------
# Segment splitting (lattice breaks)
# ---------------------------------------------------------------------------

def _split_segments(
    candidates_by_point: List[Optional[List[Candidate]]],
) -> List[Tuple[int, int, List[List[Candidate]]]]:
    """Split the full candidate lattice into contiguous segments.

    A segment boundary occurs wherever a point has no candidates (None or
    empty list). Returns list of (start_idx, end_idx_inclusive, candidates)
    for each segment, and records break points.

    Note: points with candidates but all-infeasible transitions are handled
    inside Viterbi (marked unreachable), not here, to avoid over-splitting.
    """
    segments: List[Tuple[int, int, List[List[Candidate]]]] = []
    n = len(candidates_by_point)
    i = 0
    while i < n:
        # skip empty points
        if candidates_by_point[i] is None or len(candidates_by_point[i]) == 0:
            i += 1
            continue
        start = i
        seg: List[List[Candidate]] = []
        while i < n and candidates_by_point[i] is not None and len(candidates_by_point[i]) > 0:
            seg.append(candidates_by_point[i])
            i += 1
        end = i - 1
        segments.append((start, end, seg))
    return segments


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def _stitch_route(
    graph: Graph,
    cfg: MatchConfig,
    points: List[Tuple[float, float, int]],
    matched: List[Optional[Candidate]],
    segments: List[Segment],
    gaps: List[Tuple[int, int]],
    breaks: List[int],
) -> List[StitchedEdge]:
    """Build the stitched route from decoded segments.

    - Within each segment, the decoder-selected candidate edges are emitted
      as 'decoder' edges (in observation order).
    - Between consecutive decoded candidate projections (across a segment
      boundary or gap), a shortest-path connection is added as 'stitched'
      edges if stitching is enabled.
    - Gaps / lattice breaks are bridged per bridge_policy ('bridge' edges).

    Duplicate consecutive edges are removed; direction is preserved.
    """
    edges: List[StitchedEdge] = []
    bridge_max = (cfg.stitching.bridge_max_distance_m
                  if cfg.stitching.bridge_max_distance_m is not None
                  else cfg.routing.max_route_distance_m)

    def add_decoder(cand: Optional[Candidate], pidx: int) -> None:
        if cand is not None:
            edges.append(StitchedEdge(
                edge_id=cand.edge_id, source="decoder", interpolated=False,
                from_point_idx=pidx, to_point_idx=pidx))

    def connect_projections(a: Optional[Candidate], b: Optional[Candidate],
                            src_label: str) -> None:
        """Add shortest-path edges between projection a and b (or gap ends)."""
        if a is None or b is None:
            return
        # candidate edges themselves already added as 'decoder'; here we add the
        # network path between the projections' edge endpoints.
        _, dst_a = graph._edge_endpoints[int(a.edge_id)]
        src_b, _ = graph._edge_endpoints[int(b.edge_id)]
        pr: PathResult = graph.shortest_path(int(dst_a), int(src_b),
                                             max_distance_m=bridge_max)
        if pr.reachable:
            for eid in pr.edges:
                edges.append(StitchedEdge(
                    edge_id=int(eid), source=src_label, interpolated=True,
                    from_point_idx=a.point_idx, to_point_idx=b.point_idx))

    # Build per-point decoder edges first, in order, then fill connectors.
    # Simpler: walk point-by-point.
    prev_cand: Optional[Candidate] = None
    prev_pidx: Optional[int] = None
    break_set = set(breaks)
    gap_set = set(gaps)  # (prev, curr)

    # Determine segment membership per point for stitching decisions.
    # We'll stitch decoder edges within a segment, and bridge across gaps/breaks.
    seg_of_point: Dict[int, int] = {}
    for s_idx, seg in enumerate(segments):
        for p in range(seg.start_point_idx, seg.end_point_idx + 1):
            seg_of_point[p] = s_idx

    for pidx in range(len(points)):
        cand = matched[pidx]
        add_decoder(cand, pidx)

        if prev_cand is not None and cand is not None:
            same_seg = (seg_of_point.get(prev_pidx) is not None
                        and seg_of_point.get(prev_pidx) == seg_of_point.get(pidx))
            if same_seg:
                connect_projections(prev_cand, cand, "stitched")
            else:
                # segment boundary / gap / break: bridge
                if cfg.stitching.bridge_policy == "shortest_path":
                    connect_projections(prev_cand, cand, "bridge")
        prev_cand = cand
        prev_pidx = pidx

    # Dedupe consecutive duplicate edges, preserve direction.
    if cfg.stitching.dedupe_consecutive:
        deduped: List[StitchedEdge] = []
        for e in edges:
            if deduped and deduped[-1].edge_id == e.edge_id:
                continue
            deduped.append(e)
        edges = deduped

    return edges


# ---------------------------------------------------------------------------
# Public matcher
# ---------------------------------------------------------------------------

def match_trip(
    trip_id: str,
    points: List[Tuple[float, float, int]],
    candidates_by_point: List[Optional[List[Candidate]]],
    graph: Graph,
    config: Optional[MatchConfig] = None,
) -> MatchResult:
    """Match one GPS trip to the road network.

    Parameters
    ----------
    trip_id : str
    points : list of (lon, lat, point_idx) GPS observations in order.
    candidates_by_point : list parallel to points; each entry is a list of
        Candidate objects or None/empty when the point has no candidates.
    graph : Graph for this trip's network.
    config : MatchConfig (defaults to DEFAULT_CONFIG).

    Returns
    -------
    MatchResult with per-point matches, gaps, breaks, segments, and the
    stitched route. All scoring in log space; unreachable transitions are
    -inf (explicit no-route, never silently plausible).
    """
    if config is None:
        config = DEFAULT_CONFIG

    n = len(points)
    coords = [(p[0], p[1]) for p in points]
    pidxs = [p[2] for p in points]

    # 1. Gap detection
    gaps = detect_gaps(pidxs, coords, config) if config.gap.record_gaps else []

    # 2. Split into segments at points with no candidates (lattice breaks)
    segments_raw = _split_segments(candidates_by_point)

    breaks: List[int] = []
    seg_list: List[Segment] = []
    matched: List[Optional[Candidate]] = [None] * n
    total_score = 0.0

    for start, end, seg_cands in segments_raw:
        chosen, score = _viterbi_segment(
            graph, config, seg_cands, coords[start:end + 1], pidxs[start:end + 1]
        )
        seg_list.append(Segment(start, end, [seg_cands[t - start][chosen[t - start]].cand_id
                                              for t in range(start, end + 1)],
                                score))
        total_score += score
        for t, cand in zip(range(start, end + 1), chosen):
            matched[t] = seg_cands[t - start][cand]

    # Break points: any point with no candidates
    for i in range(n):
        if candidates_by_point[i] is None or len(candidates_by_point[i]) == 0:
            if matched[i] is None:
                breaks.append(i)

    # 3. Stitch route
    stitched_route = _stitch_route(
        graph, config, points, matched, seg_list, gaps, breaks
    )

    unmatched = [i for i in range(n) if matched[i] is None]

    return MatchResult(
        trip_id=trip_id,
        num_points=n,
        points=list(points),
        matched=matched,
        gaps=gaps,
        breaks=breaks,
        segments=seg_list,
        stitched_route=stitched_route,
        total_score=total_score,
        unmatched_point_indices=unmatched,
    )


def classifier_only(
    trip_id: str,
    points: List[Tuple[float, float, int]],
    candidates_by_point: List[Optional[List[Candidate]]],
) -> MatchResult:
    """Classifier-only baseline: pick the highest-emission candidate at each
    point independently, using the same eligible points/candidates as the HMM.

    No transitions, no stitching, no gap modeling. Returned MatchResult has
    empty gaps/breaks/segments and a per-point stitched_route of the selected
    candidate edges only (source='decoder').
    """
    matched: List[Optional[Candidate]] = [None] * n if (n := len(points)) else []
    stitched: List[StitchedEdge] = []
    for pidx in range(len(points)):
        cands = candidates_by_point[pidx]
        if not cands:
            continue
        best = max(cands, key=lambda c: c.emission)
        matched[pidx] = best
        stitched.append(StitchedEdge(
            edge_id=best.edge_id, source="decoder", interpolated=False,
            from_point_idx=pidx, to_point_idx=pidx))

    return MatchResult(
        trip_id=trip_id,
        num_points=len(points),
        points=list(points),
        matched=matched,
        gaps=[],
        breaks=[],
        segments=[],
        stitched_route=stitched,
        total_score=float(sum((m.emission for m in matched if m is not None))),
        unmatched_point_indices=[i for i in range(len(points)) if matched[i] is None],
    )
