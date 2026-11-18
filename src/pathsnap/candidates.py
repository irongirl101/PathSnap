"""Stand-in candidate generation.

The supplied archive contains GPS tracks, reference routes, and the road
network, but NO per-observation candidate sets or classifier emission scores
(those are owned by the features/model members). This module therefore provides
a clearly-documented stand-in: for each GPS observation it snaps to nearby
graph edges (within a search radius) and derives a Gaussian emission from the
GPS-to-edge distance.

The matcher itself (match_trip / classifier_only) is independent of this
module: it accepts any candidate list + emission probabilities. The classifier
owner can drop in real candidates/emissions by constructing Candidate objects
and passing them to match_trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import numpy as np
from scipy.spatial import cKDTree

from .config import MatchConfig
from .graph import Graph, haversine_m
from .hmm import Candidate


@dataclass
class CandidateGenConfig:
    search_radius_m: float = 100.0      # max GPS->edge distance for a candidate
    max_candidates_per_point: int = 5   # keep nearest N edges
    emission_sigma_m: float = 25.0      # Gaussian sigma for emission = exp(-d^2/2s^2)
    min_emission: float = 1e-6          # floor so emissions stay in (0,1]


def _edge_midpoints(graph: Graph) -> np.ndarray:
    mids = []
    for eid, (src, dst) in graph._edge_endpoints.items():
        mids.append(((graph.nodes[src, 0] + graph.nodes[dst, 0]) / 2.0,
                     (graph.nodes[src, 1] + graph.nodes[dst, 1]) / 2.0))
    return np.array(mids, dtype=np.float64) if mids else np.empty((0, 2))


def _project_point_to_segment(
    lon: float, lat: float,
    a_lon: float, a_lat: float,
    b_lon: float, b_lat: float,
) -> Tuple[float, float, float, float]:
    """Project (lon,lat) onto segment a->b (equirectangular approx, metres).

    Returns (proj_lon, proj_lat, offset_m_from_a, dist_m).
    Uses a local equirectangular approximation around the segment midpoint
    which is accurate over the short edge lengths in these networks.
    """
    # Local scale: metres per degree at segment latitude
    lat_rad = math.radians((a_lat + b_lat) / 2.0)
    m_per_deg_lat = 111132.0
    m_per_deg_lon = 111132.0 * math.cos(lat_rad)

    def to_m(l, la):
        return (l * m_per_deg_lon, la * m_per_deg_lat)

    ax, ay = to_m(a_lon, a_lat)
    bx, by = to_m(b_lon, b_lat)
    px, py = to_m(lon, lat)

    vx, vy = bx - ax, by - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0.0:
        return a_lon, a_lat, 0.0, haversine_m(lon, lat, a_lon, a_lat)
    t = ((px - ax) * vx + (py - ay) * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    # convert back
    proj_lon = a_lon + (proj_x - ax) / m_per_deg_lon
    proj_lat = a_lat + (proj_y - ay) / m_per_deg_lat
    offset_m = t * math.sqrt(seg_len2)
    dist_m = math.hypot(px - proj_x, py - proj_y)
    return proj_lon, proj_lat, offset_m, dist_m


def generate_candidates(
    trip_id: str,
    points: List[Tuple[float, float, int]],
    graph: Graph,
    cfg: Optional[CandidateGenConfig] = None,
) -> List[Optional[List[Candidate]]]:
    """Generate stand-in candidates for every GPS observation.

    Returns a list parallel to `points`; each entry is a list of Candidate
    objects (sorted by distance) or None/empty if no edge within radius.
    """
    if cfg is None:
        cfg = CandidateGenConfig()

    mids = _edge_midpoints(graph)
    if len(mids) == 0:
        return [None] * len(points)

    tree = cKDTree(mids)
    # KDTree works in degrees; convert radius to degrees (conservative: use
    # lon scaling at mean latitude)
    mean_lat = float(np.mean(graph.nodes[:, 1])) if len(graph.nodes) else 0.0
    m_per_deg_lon = 111132.0 * math.cos(math.radians(mean_lat))
    radius_deg = cfg.search_radius_m / m_per_deg_lon if m_per_deg_lon > 0 else 0.0

    out: List[Optional[List[Candidate]]] = []
    for (lon, lat, pidx) in points:
        idxs = tree.query_ball_point([lon, lat], r=radius_deg)
        cands: List[Candidate] = []
        for eid in idxs:
            eid = int(eid)
            src, dst = graph._edge_endpoints[eid]
            a = graph.nodes[src]
            b = graph.nodes[dst]
            proj_lon, proj_lat, offset_m, dist_m = _project_point_to_segment(
                lon, lat, a[0], a[1], b[0], b[1])
            if dist_m > cfg.search_radius_m:
                continue
            cands.append(Candidate(
                trip_id=trip_id, point_idx=pidx, lon=lon, lat=lat,
                cand_id=eid, edge_id=eid,
                proj_lon=proj_lon, proj_lat=proj_lat,
                offset_m=offset_m, edge_len_m=float(graph._edge_length[eid]),
                emission=1.0,  # placeholder; normalized below
                dist_m=dist_m,
            ))
        # keep nearest N
        cands.sort(key=lambda c: c.dist_m)
        cands = cands[: cfg.max_candidates_per_point]
        if not cands:
            out.append(None)
            continue
        # Gaussian emission, normalized per point to sum to 1 (valid probs)
        sigma = cfg.emission_sigma_m
        dists = np.array([c.dist_m for c in cands], dtype=np.float64)
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        weights = np.maximum(weights, cfg.min_emission)
        weights = weights / weights.sum()
        for c, w in zip(cands, weights):
            c.emission = float(w)
        out.append(cands)
    return out


def candidate_table(cands_by_point: List[Optional[List[Candidate]]]) -> List[dict]:
    """Flatten candidates into a list of dicts (useful for CSV/records export)."""
    rows = []
    for point_entry in cands_by_point:
        if not point_entry:
            continue
        for c in point_entry:
            rows.append({
                "trip_id": c.trip_id,
                "point_idx": c.point_idx,
                "cand_id": c.cand_id,
                "edge_id": c.edge_id,
                "lon": c.lon, "lat": c.lat,
                "proj_lon": c.proj_lon, "proj_lat": c.proj_lat,
                "offset_m": c.offset_m, "edge_len_m": c.edge_len_m,
                "emission": c.emission, "dist_m": c.dist_m,
            })
    return rows
