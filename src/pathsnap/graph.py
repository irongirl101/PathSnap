"""Routable graph + bounded Dijkstra shortest-path search with caching.

Graph representation (per trip):
    nodes : (N, 2) float array of [lon, lat]; row index == node id (0-based).
    arcs  : (M, 2) int array of [src, dst] node ids; directed edges.

Edge length = great-circle (haversine) distance between endpoint coordinates.

The shortest-path cache is keyed by (source_node, target_node, max_distance)
and is scoped to a single Graph instance (one graph == one trip's network).
Call Graph.clear_cache() to invalidate (e.g. after graph edits or between
independent runs where you want a fresh cache). Results are only valid while
the graph topology/edge lengths are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
import math

import numpy as np


# Sentinel for "no route / unreachable".
NO_ROUTE = float("inf")


@dataclass
class PathResult:
    """Result of a bounded shortest-path query."""
    reachable: bool
    distance_m: float           # inf when unreachable / beyond bound
    nodes: List[int]            # node id sequence from source to target
    edges: List[int]            # arc id sequence (parallel to nodes[1:])
    expanded: int               # number of nodes popped from the priority queue


@dataclass
class Graph:
    """A directed, weighted road network for one trip.

    Parameters
    ----------
    nodes : (N,2) array [lon, lat] in degrees.
    arcs : (M,2) array [src, dst] node ids.
    directed : if False, every arc is treated as bidirectional.
    max_route_distance_m : hard bound for bounded Dijkstra.
    """

    nodes: np.ndarray
    arcs: np.ndarray
    directed: bool = True
    max_route_distance_m: float = 2000.0

    # ---- internal (built in __post_init__) ----
    _adj: Dict[int, List[Tuple[int, int, float]]] = field(init=False, repr=False)
    _edge_length: Dict[int, float] = field(init=False, repr=False)
    _edge_endpoints: Dict[int, Tuple[int, int]] = field(init=False, repr=False)
    _cache: Dict[Tuple[int, int, float], PathResult] = field(
        init=False, repr=False, default_factory=dict
    )
    _cache_hits: int = field(init=False, repr=False, default_factory=int)
    _cache_misses: int = field(init=False, repr=False, default_factory=int)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.arcs = np.asarray(self.arcs, dtype=np.int64)
        self._adj = {}
        self._edge_length = {}
        self._edge_endpoints = {}
        n = len(self.nodes)
        for nid in range(n):
            self._adj[nid] = []
        for eid, (src, dst) in enumerate(self.arcs):
            src = int(src); dst = int(dst)
            length = _haversine_m(
                self.nodes[src, 0], self.nodes[src, 1],
                self.nodes[dst, 0], self.nodes[dst, 1],
            )
            self._adj[src].append((dst, eid, length))
            self._edge_length[eid] = length
            self._edge_endpoints[eid] = (src, dst)
            if not self.directed:
                self._adj[dst].append((src, eid, length))
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    # -- cache API -----------------------------------------------------
    def clear_cache(self) -> None:
        """Invalidate the shortest-path cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_stats(self) -> Dict[str, int]:
        return {
            "entries": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    # -- core search ---------------------------------------------------
    def shortest_path(
        self,
        source: int,
        target: int,
        max_distance_m: Optional[float] = None,
    ) -> PathResult:
        """Bounded Dijkstra from source to target.

        Stops expanding once the smallest tentative distance in the priority
        queue exceeds max_distance_m. Results are cached keyed by
        (source, target, max_distance_m).

        Returns a PathResult with reachable=False when no path exists within
        the bound (including source == target edge cases handled below).
        """
        if max_distance_m is None:
            max_distance_m = self.max_route_distance_m
        max_distance_m = float(max_distance_m)

        key = (int(source), int(target), max_distance_m)
        self._cache_misses += 1
        if key in self._cache:
            self._cache_hits += 1
            self._cache_misses -= 1  # don't count a hit as a miss
            return self._cache[key]

        result = self._dijkstra(source, target, max_distance_m)
        self._cache[key] = result
        return result

    def _dijkstra(self, source: int, target: int, bound: float) -> PathResult:
        source = int(source); target = int(target)

        if source == target:
            return PathResult(
                reachable=True, distance_m=0.0,
                nodes=[source], edges=[], expanded=0,
            )

        if source not in self._adj or target not in self._adj:
            return PathResult(False, NO_ROUTE, [], [], 0)

        dist: Dict[int, float] = {source: 0.0}
        prev_node: Dict[int, int] = {}
        prev_edge: Dict[int, int] = {}
        pq: List[Tuple[float, int]] = [(0.0, source)]
        visited = set()
        expanded = 0

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            expanded += 1

            # Bounded stop: smallest remaining frontier distance exceeds bound.
            if d > bound:
                break

            if u == target:
                break

            for v, eid, w in self._adj.get(u, []):
                if v in visited:
                    continue
                nd = d + w
                if nd > bound:
                    continue
                if nd < dist.get(v, NO_ROUTE):
                    dist[v] = nd
                    prev_node[v] = u
                    prev_edge[v] = eid
                    heapq.heappush(pq, (nd, v))

        if target not in dist:
            return PathResult(False, NO_ROUTE, [], [], expanded)

        # Reconstruct path.
        nodes: List[int] = []
        edges: List[int] = []
        cur = target
        while cur != source:
            nodes.append(cur)
            edges.append(prev_edge[cur])
            cur = prev_node[cur]
        nodes.append(source)
        nodes.reverse()
        edges.reverse()
        return PathResult(
            reachable=True,
            distance_m=dist[target],
            nodes=nodes,
            edges=edges,
            expanded=expanded,
        )


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two lon/lat points (degrees)."""
    R = 6371000.0  # mean Earth radius, metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    return 2.0 * R * math.asin(math.sqrt(min(1.0, a)))


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Public great-circle distance in metres (degrees input)."""
    return _haversine_m(lon1, lat1, lon2, lat2)
