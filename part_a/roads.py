"""Directed trip roads and a SQLite R-tree over projected segment envelopes."""
from dataclasses import dataclass
from math import isfinite
import sqlite3

import numpy as np

from .data import read_matrix


@dataclass(frozen=True)
class Candidate:
    segment_id: tuple[str, int]
    from_node: int
    to_node: int
    distance_m: float
    offset_fraction: float
    rank: int

    @property
    def arc_id(self):
        return self.segment_id[1]


class RoadGraph:
    def __init__(self, trip_id, lonlat, arcs):
        self.trip_id = trip_id
        self.lonlat = np.asarray(lonlat, dtype=float)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        if not len(self.lonlat) or not np.isfinite(self.lonlat).all() or (abs(self.lonlat) > [180, 90]).any():
            raise ValueError(f"{trip_id}: invalid network coordinates")
        if not len(self.arcs) or self.arcs.min() < 0 or self.arcs.max() >= len(self.lonlat):
            raise ValueError(f"{trip_id}: invalid or missing arc endpoints")
        self.origin = self.lonlat.mean(axis=0)
        # Local WGS84 tangent-plane linearization using ellipsoidal curvature.
        latitude = np.radians(self.origin[1])
        e2 = 0.0066943799901413165
        w = np.sqrt(1 - e2 * np.sin(latitude)**2)
        a = 6378137.0
        self.scale = np.array([a / w * np.cos(latitude), a * (1 - e2) / w**3])
        self.xy = self.project(self.lonlat)
        self.start = self.xy[self.arcs[:, 0]]
        self.end = self.xy[self.arcs[:, 1]]
        self.vector = self.end - self.start
        self.length2 = np.einsum("ij,ij->i", self.vector, self.vector)
        self.length_m = np.sqrt(self.length2)
        self.in_degree = np.bincount(self.arcs[:, 1], minlength=len(self.xy))
        self.out_degree = np.bincount(self.arcs[:, 0], minlength=len(self.xy))

    @classmethod
    def load(cls, trip):
        return cls(trip.trip_id, read_matrix(trip.file("nodes"), 2, float),
                   read_matrix(trip.file("arcs"), 2, np.int64))

    def project(self, lonlat):
        delta = np.asarray(lonlat, dtype=float) - self.origin
        return np.radians(delta) * self.scale

    def distance(self, point_xy, ids):
        offset = point_xy - self.start[ids]
        length2 = self.length2[ids]
        dot = np.einsum("ij,ij->i", offset, self.vector[ids])
        along = np.clip(np.divide(dot, length2, out=np.zeros_like(dot), where=length2 > 0), 0, 1)
        delta = offset - along[:, None] * self.vector[ids]
        return np.hypot(delta[:, 0], delta[:, 1]), along

    def weak_components(self):
        parent = list(range(len(self.xy)))
        sizes = [1] * len(parent)
        def root(i):
            while i != parent[i]:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for u, v in self.arcs.tolist():
            u, v = root(u), root(v)
            if u != v:
                if sizes[u] < sizes[v]:
                    u, v = v, u
                parent[v] = u
                sizes[u] += sizes[v]
        return [size for i, size in enumerate(sizes) if parent[i] == i]

    def to_networkx(self):
        """Directed multigraph with original node IDs and arc IDs as edge keys."""
        import networkx as nx
        graph = nx.MultiDiGraph(trip_id=self.trip_id, crs="EPSG:4326")
        for i, (lon, lat) in enumerate(self.lonlat.tolist()):
            graph.add_node(i, x=lon, y=lat)
        for arc_id, (u, v) in enumerate(self.arcs.tolist()):
            graph.add_edge(u, v, key=arc_id, arc_id=arc_id,
                           segment_id=(self.trip_id, arc_id), length=float(self.length_m[arc_id]))
        return graph


class RoadIndex:
    def __init__(self, graph):
        self.graph = graph
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE segments USING rtree(id, minx, maxx, miny, maxy)")
        low = np.minimum(graph.start, graph.end)
        high = np.maximum(graph.start, graph.end)
        bounds = np.column_stack((low[:, 0], high[:, 0], low[:, 1], high[:, 1]))
        self.db.executemany("INSERT INTO segments VALUES (?, ?, ?, ?, ?)",
                            ((i, *row) for i, row in enumerate(bounds.tolist())))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()

    def query(self, point_xy, radius_m, k=None):
        if not isfinite(radius_m) or radius_m < 0 or (k is not None and k < 1):
            raise ValueError("Expected finite nonnegative radius and positive k")
        x, y = point_xy
        rows = self.db.execute(
            "SELECT id FROM segments WHERE minx<=? AND maxx>=? AND miny<=? AND maxy>=?",
            (x + radius_m, x - radius_m, y + radius_m, y - radius_m))
        ids = np.fromiter((row[0] for row in rows), dtype=np.int64)
        dist, along = self.graph.distance(point_xy, ids)
        keep = np.flatnonzero(dist <= radius_m)
        order = keep[np.lexsort((ids[keep], dist[keep]))][:k]
        result = []
        for rank, j in enumerate(order, 1):
            arc = int(ids[j])
            u, v = map(int, self.graph.arcs[arc])
            result.append(Candidate((self.graph.trip_id, arc), u, v,
                                    float(dist[j]), float(along[j]), rank))
        return result
