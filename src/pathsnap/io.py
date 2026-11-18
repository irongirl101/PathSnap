"""Loading PathSnap trip files.

File formats (verified against the supplied archive):

    .track  : TSV  lon  lat  point_index   (1-based sequential index; gaps possible)
    .route  : TSV  edge_id                  (0-based arc/edge id; forms a connected path = reference route)
    .nodes  : TSV  lon  lat                 (row index == node id, 0-based)
    .arcs   : TSV  src  dst                 (node ids; directed graph)

All four files share the trip id as their basename.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import os

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray        # (P,3) [lon, lat, point_index]
    route: np.ndarray        # (R,) int64 reference edge ids (0-based arc ids)
    nodes: np.ndarray        # (N,2) [lon, lat]; row index == node id
    arcs: np.ndarray         # (M,2) int64 [src, dst] node ids

    @property
    def num_points(self) -> int:
        return len(self.track)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)


def load_trip(trip_id: str, trip_dir: str) -> Trip:
    """Load a single trip's four files."""
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")

    track = np.loadtxt(track_path, delimiter="\t", dtype=np.float64)
    route = np.loadtxt(route_path, delimiter="\t", dtype=np.int64)
    nodes = np.loadtxt(nodes_path, delimiter="\t", dtype=np.float64)
    arcs = np.loadtxt(arcs_path, delimiter="\t", dtype=np.int64)

    # Guard against empty route files (single value loadtxt returns scalar).
    if route.ndim == 0:
        route = np.array([int(route)], dtype=np.int64)
    return Trip(trip_id=trip_id, track=track, route=route, nodes=nodes, arcs=arcs)


def build_graph(trip: Trip, directed: bool = True,
                max_route_distance_m: float = 2000.0) -> Graph:
    """Build a Graph from a Trip."""
    from .graph import Graph
    return Graph(
        nodes=trip.nodes,
        arcs=trip.arcs,
        directed=directed,
        max_route_distance_m=max_route_distance_m,
    )


def list_trips(root_dir: str) -> List[str]:
    """Return sorted trip ids (directory names) under root_dir."""
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


def trip_dir(root_dir: str, trip_id: str) -> str:
    return os.path.join(root_dir, trip_id)
