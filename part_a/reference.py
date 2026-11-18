"""Offline reference diagnostics. Inference does not import this module."""
import numpy as np

from .data import read_matrix


def load_reference(trip, graph):
    route = read_matrix(trip.file("route"), 1, np.int64).ravel()
    if not len(route) or route.min() < 0 or route.max() >= len(graph.arcs):
        raise ValueError(f"{trip.trip_id}: missing or invalid reference arc IDs")
    edges = graph.arcs[route]
    breaks = np.flatnonzero(edges[:-1, 1] != edges[1:, 0]).tolist()
    return route, breaks


def align(graph, xy, route, breaks, max_distance=30, min_margin=8):
    """Fit a nondecreasing sequence of route positions by capped distance loss."""
    if not len(xy):
        return []
    if breaks:
        return [(None, None, None, "discontinuous_route")] * len(xy)
    distances = np.array([graph.distance(point, route)[0] for point in xy], dtype=np.float32)
    loss = np.minimum(distances, 60)
    forward = np.empty_like(loss)
    forward[0] = loss[0]
    for i in range(1, len(xy)):
        forward[i] = loss[i] + np.minimum.accumulate(forward[i-1])
    path = [0] * len(xy)
    path[-1] = int(np.argmin(forward[-1]))
    for i in range(len(xy)-2, -1, -1):
        path[i] = int(np.argmin(forward[i, :path[i+1]+1]))
    result = []
    suffix = np.zeros(len(route), dtype=np.float32)
    for i in range(len(xy)-1, -1, -1):
        j = path[i]
        arc = int(route[j])
        alternative = route != arc
        cost = forward[i] + suffix
        margin = float(cost[alternative].min() - cost[j]) if alternative.any() else None
        distance = float(distances[i, j])
        status = "accepted"
        if distance > max_distance:
            status = "far_from_reference"
        elif margin is not None and margin < min_margin:
            status = "ambiguous"
        result.append((arc, j, distance, status))
        suffix = np.minimum.accumulate((suffix + loss[i])[::-1])[::-1]
    return result[::-1]
