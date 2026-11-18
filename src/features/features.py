"""Feature engineering (candidate generation and labels live in candidates.py).

Input schema (local metres):
  nodes : node_id, x, y
  arcs  : arc_id, u, v          (directed arc u -> v, straight segment)
  track : x, y, [t]             (rows in time order)
  route : list of arc_id        (ground truth, optional)
Output: one row per (GPS point, candidate arc).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from src.features.candidates import (Network, generate_candidates, add_labels,  # noqa: F401
                                     RADIUS_M, MAX_CANDS)

MIN_MOVE_M = 5.0     # below this the heading is treated as unknown

FEATURE_COLS = [
    # geometry
    "dist", "frac", "arc_len", "dist_rank", "n_cands", "dist_gap_best", "dist_ratio_best",
    # heading / direction
    "heading_diff", "heading_cos", "along_ratio",
    # context: same arc seen from neighbouring points
    "dist_p2", "dist_p1", "dist_n1", "dist_n2", "win_mean_dist", "win_max_dist",
    # topology: link to best candidate of previous / next point
    "conn_prev", "conn_next", "out_deg_end",
    # kinematics
    "step_prev", "step_next", "speed", "accel", "dt_prev",
]
CATEGORICAL_COLS = ["conn_prev", "conn_next"]
NON_FEATURE_COLS = ["trip_id", "point_idx", "arc_id", "on_route", "label"]  # never fed to a model


def _shift(a, k):
    """out[i] = a[i-k]; NaN where undefined."""
    out = np.full(len(a), np.nan)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _ang_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.where(d > 180, 360 - d, d)


def point_features(track: pd.DataFrame) -> pd.DataFrame:
    x, y = track["x"].to_numpy(float), track["y"].to_numpy(float)
    t = track["t"].to_numpy(float) if "t" in track else np.full(len(track), np.nan)
    xp, yp, tp = _shift(x, 1), _shift(y, 1), _shift(t, 1)
    xn, yn, tn = _shift(x, -1), _shift(y, -1), _shift(t, -1)
    dt_prev = t - tp
    dt_next = tn - t
    if "t" in track:
        prev_ok = np.isfinite(tp) & ((t - tp) <= 5.0)
        next_ok = np.isfinite(tn) & ((tn - t) <= 5.0)
        xp, yp, tp = (np.where(prev_ok, a, np.nan) for a in (xp, yp, tp))
        xn, yn, tn = (np.where(next_ok, a, np.nan) for a in (xn, yn, tn))
        dt_prev_context = np.where(prev_ok, dt_prev, np.nan)
        dt_next_context = np.where(next_ok, dt_next, np.nan)
    else:
        dt_prev_context, dt_next_context = dt_prev, dt_next
    x0, y0, t0 = np.where(np.isnan(xp), x, xp), np.where(np.isnan(yp), y, yp), np.where(np.isnan(tp), t, tp)
    x1, y1, t1 = np.where(np.isnan(xn), x, xn), np.where(np.isnan(yn), y, yn), np.where(np.isnan(tn), t, tn)
    mdx, mdy = x1 - x0, y1 - y0
    move = np.hypot(mdx, mdy)
    bearing = np.where(move >= MIN_MOVE_M, np.degrees(np.arctan2(mdx, mdy)) % 360.0, np.nan)
    step_prev, step_next = np.hypot(x - xp, y - yp), np.hypot(xn - x, yn - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        span = t1 - t0
        speed = np.where(span > 0, move / span, np.nan)
        sp_prev = np.where(dt_prev_context > 0, step_prev / dt_prev_context, np.nan)
        sp_next = np.where(dt_next_context > 0, step_next / dt_next_context, np.nan)
        accel = (sp_next - sp_prev) / ((dt_prev_context + dt_next_context) / 2)
    return pd.DataFrame(dict(move_bearing=bearing, step_prev=step_prev, step_next=step_next,
                             speed=speed, accel=accel, dt_prev=dt_prev))


def add_features(net: Network, track: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    n = len(track)
    x, y = track["x"].to_numpy(float), track["y"].to_numpy(float)
    pf = point_features(track)
    c = c.join(pf, on="pos")
    arc = c["arc_idx"].to_numpy()
    g = c.groupby("pos")["dist"]
    c["n_cands"] = g.transform("size")
    best = g.transform("min")
    c["dist_gap_best"] = c["dist"] - best
    c["dist_ratio_best"] = c["dist"] / (best + 1.0)
    c["arc_len"] = net.length[arc]
    # heading
    c["heading_diff"] = _ang_diff(net.bearing[arc], c["move_bearing"].to_numpy())
    c["heading_cos"] = np.cos(np.radians(c["heading_diff"]))
    # same arc as seen from neighbouring points
    pos = c["pos"].to_numpy()
    timestamps = track["t"].to_numpy(float) if "t" in track else np.full(n, np.nan)
    breaks = np.zeros(n, dtype=bool)
    if n > 1:
        intervals = np.diff(timestamps)
        breaks[1:] = np.isfinite(intervals) & (intervals > 5.0)
    groups = np.cumsum(breaks)
    ss, xy = {}, {}
    for name, k in [("p2", -2), ("p1", -1), ("n1", 1), ("n2", 2)]:
        npos = pos + k
        in_range = (npos >= 0) & (npos < n)
        ok = in_range & (groups[np.clip(npos, 0, n - 1)] == groups[pos])
        px, py = np.full(len(c), np.nan), np.full(len(c), np.nan)
        px[ok], py[ok] = x[npos[ok]], y[npos[ok]]
        d, s, _ = net.project(px, py, arc)
        c[f"dist_{name}"] = d
        ss[name], xy[name] = s, (px, py)
    alld = np.column_stack([c["dist"], c["dist_p2"], c["dist_p1"], c["dist_n1"], c["dist_n2"]])
    c["win_mean_dist"] = np.nanmean(alld, axis=1)
    c["win_max_dist"] = np.nanmax(alld, axis=1)
    # signed progress along the arc between previous and next point, relative to actual movement
    mv = np.hypot(xy["n1"][0] - xy["p1"][0], xy["n1"][1] - xy["p1"][1])
    with np.errstate(divide="ignore", invalid="ignore"):
        c["along_ratio"] = np.where(mv > MIN_MOVE_M, (ss["n1"] - ss["p1"]) / mv, np.nan)
    # topology vs best candidate of neighbouring points
    best_arc = np.full(n, -1)
    first = c.groupby("pos")["arc_idx"].first()
    best_arc[first.index.to_numpy()] = first.to_numpy()
    def conn(other_pos, forward):
        in_range = (other_pos >= 0) & (other_pos < n)
        ok = in_range & (groups[np.clip(other_pos, 0, n - 1)] == groups[pos])
        oa = np.where(ok, best_arc[np.clip(other_pos, 0, n - 1)], -1)
        same = oa == arc
        if forward:   # previous best arc ends where this one starts
            link = net.v[np.where(oa >= 0, oa, 0)] == net.u[arc]
        else:         # this arc ends where next best arc starts
            link = net.v[arc] == net.u[np.where(oa >= 0, oa, 0)]
        out = np.where(same | link, 1.0, 0.0)
        return np.where(oa >= 0, out, np.nan)
    c["conn_prev"] = conn(pos - 1, True)
    c["conn_next"] = conn(pos + 1, False)
    c["out_deg_end"] = net.out_deg[net.v[arc]]
    return c


def build_trip_table(trip_id, nodes, arcs, track, route=None, net=None) -> pd.DataFrame:
    net = net or Network(nodes, arcs)
    c = generate_candidates(net, track)
    if c.empty:
        return pd.DataFrame()
    c = add_features(net, track, c)
    if route is not None:
        c = add_labels(net, c, route)
    c["trip_id"] = trip_id
    c["point_idx"] = c["pos"]
    c["arc_id"] = net.arc_ids[c["arc_idx"].to_numpy()]
    keep = ["trip_id", "point_idx", "arc_id"] + (["on_route", "label"] if route is not None else []) + FEATURE_COLS
    return c[keep]


def candidate_recall(df: pd.DataFrame) -> float:
    """Share of GPS points that have a positive candidate (ceiling for top-1 accuracy)."""
    per_pt = df.groupby(["trip_id", "point_idx"])["label"].max()
    return float(per_pt.mean())


def save_feature_spec(path="artifacts/feature_columns.json"):
    with open(path, "w") as f:
        json.dump({"feature_columns": FEATURE_COLS, "categorical": CATEGORICAL_COLS,
                   "never_use_as_features": NON_FEATURE_COLS}, f, indent=2)