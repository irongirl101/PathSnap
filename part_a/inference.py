"""Shared candidate interface for the classifier, decoder and API."""
from .data import clean_track
from .roads import RoadGraph, RoadIndex


def candidate_batches(track, graph, *, k=10, radius_m=100, gap_s=5):
    """Yield every retained GPS observation and its directed candidates."""
    points, _ = clean_track(track, gap_s)
    with RoadIndex(graph) as index:
        for point in points:
            xy = graph.project((point.lon, point.lat))
            yield point, index.query(xy, radius_m, k)
