"""Road network, candidate generation and label derivation.

Input schema (local metres):
  nodes : node_id, x, y
  arcs  : arc_id, u, v          (directed arc u -> v, drawn as a straight segment)
  track : x, y, [t]
  route : list of arc_id        (ground truth, optional)
Candidates = the MAX_CANDS nearest arcs within RADIUS_M of each GPS point.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

RADIUS_M = 100.0     # candidate search radius
MAX_CANDS = 10       # keep k nearest arcs per point


class Network:
    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame):
        nid = {n: i for i, n in enumerate(nodes["node_id"].to_numpy())}
        arcs = arcs[arcs["u"].isin(nid) & arcs["v"].isin(nid)].reset_index(drop=True)
        nx_, ny_ = nodes["x"].to_numpy(float), nodes["y"].to_numpy(float)
        self.arc_ids = arcs["arc_id"].to_numpy()
        self.u = arcs["u"].map(nid).to_numpy()
        self.v = arcs["v"].map(nid).to_numpy()
        self.ax, self.ay = nx_[self.u], ny_[self.u]
        self.bx, self.by = nx_[self.v], ny_[self.v]
        dx, dy = self.bx - self.ax, self.by - self.ay
        self.length = np.hypot(dx, dy)
        self.bearing = np.degrees(np.arctan2(dx, dy)) % 360.0
        self.out_deg = np.bincount(self.u, minlength=len(nodes))
        self.id_to_idx = {a: i for i, a in enumerate(self.arc_ids)}
        coords = np.stack([np.column_stack([self.ax, self.ay]),
                           np.column_stack([self.bx, self.by])], axis=1)
        self.tree = STRtree(shapely.linestrings(coords))

    def project(self, px, py, arc_idx):
        """Distance and along-arc position (m) of points onto given arcs. NaN-safe."""
        ax, ay = self.ax[arc_idx], self.ay[arc_idx]
        dx, dy = self.bx[arc_idx] - ax, self.by[arc_idx] - ay
        l2 = dx * dx + dy * dy
        t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        dist = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
        return dist, t * self.length[arc_idx], t


def generate_candidates(net: Network, track: pd.DataFrame) -> pd.DataFrame:
    pts = shapely.points(track["x"].to_numpy(float), track["y"].to_numpy(float))
    pi, ai = net.tree.query(pts, predicate="dwithin", distance=RADIUS_M)
    if len(pi) == 0:
        return pd.DataFrame(columns=["pos", "arc_idx", "dist", "s", "frac"])
    px, py = track["x"].to_numpy(float)[pi], track["y"].to_numpy(float)[pi]
    d, s, frac = net.project(px, py, ai)
    c = pd.DataFrame(dict(pos=pi, arc_idx=ai, dist=d, s=s, frac=frac))
    c = c.sort_values(["pos", "dist"], kind="mergesort").reset_index(drop=True)
    c["dist_rank"] = c.groupby("pos").cumcount() + 1
    return c[c["dist_rank"] <= MAX_CANDS].reset_index(drop=True)


def add_labels(net: Network, c: pd.DataFrame, route_ids) -> pd.DataFrame:
    """label=1 for the on-route candidate nearest to the GPS point; all others 0.
    Points with no on-route candidate get no positive (counts against candidate recall)."""
    route_idx = {net.id_to_idx[a] for a in route_ids if a in net.id_to_idx}
    c["on_route"] = c["arc_idx"].isin(route_idx).astype(int)
    c["label"] = 0
    on = c[c["on_route"] == 1]
    if len(on):
        win = on.groupby("pos")["dist"].idxmin()
        c.loc[win.to_numpy(), "label"] = 1
    return c