"""Loader for the per-trip folders (<id>.nodes / .arcs / .track / .route).

!! The column layouts below are ASSUMPTIONS. Open one trip, look at the first
lines of each file, and edit ONLY the *_COLS / flags in this block. !!
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

# ---- Raw dataset schema ----------------------------------------------------
NODES_COLS = ["lon", "lat"]                     # .nodes: node id = row number (0-based)
ARCS_COLS  = ["u", "v"]                         # .arcs : arc id  = row number (0-based)
TRACK_COLS = ["lon", "lat", "t"]                # .track: longitude, latitude, time
COORDS_ARE_LATLON = True
HEADER = None                                   # no header rows
# -----------------------------------------------------------------------------

def _read(path, names):
    df = pd.read_csv(path, sep=r"\s+", header=HEADER, engine="python")
    if HEADER is None:
        df = df.iloc[:, : len(names)]
        df.columns = names[: df.shape[1]]
    return df


def inspect_trip(trip_dir, n=5):
    """Print the raw first lines of every file in a trip folder."""
    for f in sorted(pathlib.Path(trip_dir).iterdir()):
        lines = f.read_text().splitlines()
        print(f"=== {f.name} | {len(lines)} lines")
        for ln in lines[:n]:
            print("  ", repr(ln))


def load_trip(trip_dir):
    """Return (trip_id, nodes, arcs, track, route_ids) as DataFrames/list."""
    trip_dir = pathlib.Path(trip_dir)
    tid = trip_dir.name
    nodes = _read(trip_dir / f"{tid}.nodes", NODES_COLS)
    arcs = _read(trip_dir / f"{tid}.arcs", ARCS_COLS)
    track = _read(trip_dir / f"{tid}.track", TRACK_COLS)
    nodes.insert(0, "node_id", np.arange(len(nodes)))   # ids are implicit row numbers
    arcs.insert(0, "arc_id", np.arange(len(arcs)))
    route_path = trip_dir / f"{tid}.route"
    route = None
    if route_path.exists():
        route = [int(tok) for tok in route_path.read_text().split()]
    return tid, nodes, arcs, track, route


def to_local_xy(nodes, track):
    """lat/lon -> local metres (equirectangular, origin = node centroid)."""
    if not COORDS_ARE_LATLON:
        nodes = nodes.rename(columns={"lat": "x", "lon": "y"})
        track = track.rename(columns={"lat": "x", "lon": "y"})
        return nodes, track
    R = 6371000.0
    lat0, lon0 = nodes["lat"].mean(), nodes["lon"].mean()
    def proj(df):
        df = df.copy()
        df["x"] = np.radians(df["lon"] - lon0) * R * np.cos(np.radians(lat0))
        df["y"] = np.radians(df["lat"] - lat0) * R
        return df
    return proj(nodes), proj(track)