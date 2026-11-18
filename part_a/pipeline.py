"""Run reproducible Part A audits and offline candidate evaluation."""
import argparse
import csv
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .data import clean_track, discover, manifest
from .reference import align, load_reference
from .roads import RoadGraph, RoadIndex

RADII = (25, 50, 100)
COUNTS = (1, 5, 10)
CLEAN_KEYS = ("raw_points", "clean_points", "missing_rows", "malformed_rows", "nonfinite_rows",
              "invalid_coordinates", "raw_time_reversals", "duplicate_observations",
              "duplicate_timestamps", "gaps")


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trip_stats(trip, graph, route, breaks, points, xy, cleaning):
    if not points:
        raise ValueError(f"{trip.trip_id}: no valid observations")
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.array([p.dt_s for p in points[1:]])
    speed = steps / dt
    components = graph.weak_components()
    coordinates = np.array([(p.lon, p.lat) for p in points])
    return dict(trip_id=trip.trip_id, partition=trip.partition,
                **{key: cleaning[key] for key in CLEAN_KEYS},
                **{key + "_bytes": trip.file(key).stat().st_size for key in ("track","nodes","arcs","route")},
                lon_min=coordinates[:, 0].min(), lon_max=coordinates[:, 0].max(),
                lat_min=coordinates[:, 1].min(), lat_max=coordinates[:, 1].max(),
                nodes=len(graph.xy), arcs=len(graph.arcs), route_arcs=len(route),
                unique_route_arcs=len(set(route)), route_breaks=len(breaks),
                isolates=np.count_nonzero(graph.out_degree + graph.in_degree == 0),
                weak_components=len(components), largest_component=max(components),
                self_loops=np.count_nonzero(graph.arcs[:, 0] == graph.arcs[:, 1]),
                zero_length_arcs=np.count_nonzero(graph.length2 == 0),
                duplicate_directed_pairs=len(graph.arcs) - len(np.unique(graph.arcs, axis=0)),
                duration_s=points[-1].timestamp_s-points[0].timestamp_s,
                one_second_intervals=np.count_nonzero(dt == 1),
                median_dt_s=np.median(dt) if len(dt) else None,
                max_dt_s=max(dt, default=None), gps_length_m=steps.sum(),
                gps_contiguous_length_m=steps[dt <= 5].sum(), reference_length_m=graph.length_m[route].sum(),
                network_length_m=graph.length_m.sum(), speed_over_55_mps=np.count_nonzero((speed > 55) & (dt <= 5)))


def evaluate_trip(trip, graph, route, points, xy, labels, candidate_writer, label_writer):
    settings = [(radius, k) for radius in RADII for k in COUNTS]
    stats = {key: Counter() for key in settings}
    union = {key: set() for key in settings}
    reference = set(map(int, route))
    nearest = []
    statuses = Counter()
    with RoadIndex(graph) as index:
        for point, position, label in zip(points, xy, labels, strict=True):
            arc, route_pos, distance, status = label
            statuses[status] += 1
            candidates = index.query(position, max(RADII), max(COUNTS))
            if candidates:
                nearest.append(candidates[0].distance_m)
            for radius, k in settings:
                chosen = {c.arc_id for c in candidates[:k] if c.distance_m <= radius}
                count = stats[radius, k]
                count["empty"] += not chosen
                count["any_ref"] += bool(chosen & reference)
                count["labeled"] += status == "accepted"
                count["hit"] += status == "accepted" and arc in chosen
                union[radius, k].update(chosen & reference)
            label_writer.writerow((trip.trip_id, point.point_id, arc, route_pos, distance, status))
            for c in candidates:
                candidate_writer.writerow((trip.trip_id, point.point_id, c.arc_id, c.from_node, c.to_node,
                                           c.rank, c.distance_m, c.offset_fraction))
    rows = []
    for (radius, k), c in stats.items():
        rows.append(dict(trip_id=trip.trip_id, partition=trip.partition, radius_m=radius, k=k,
                         points=len(points), empty=c["empty"], point_any_reference=c["any_ref"],
                         unique_reference_count=len(reference), unique_reference_hit=len(union[radius, k]),
                         accepted_points=c["labeled"], accepted_hits=c["hit"]))
    return rows, nearest, statuses


def run(root, out, seed, feature_contract=None):
    from .eda import map_trip, summarize

    trips = discover(root)
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "split_manifest.csv", manifest(trips, seed))
    summaries, evaluations = [], []
    with ExitStack() as stack:
        cand = csv.writer(stack.enter_context((out / "candidates.csv").open("w", newline="")))
        lab = csv.writer(stack.enter_context((out / "alignment.csv").open("w", newline="")))
        cand.writerow(("trip_id","point_id","arc_id","from_node","to_node","rank","distance_m","offset_fraction"))
        lab.writerow(("trip_id","point_id","provisional_arc_id","route_position","distance_m","status"))
        for number, trip in enumerate(trips, 1):
            graph = RoadGraph.load(trip)
            route, breaks = load_reference(trip, graph)
            points, cleaning = clean_track(trip.track)
            xy = graph.project([(p.lon, p.lat) for p in points])
            row = trip_stats(trip, graph, route, breaks, points, xy, cleaning)
            labels = align(graph, xy, route, breaks)
            scores, nearest, statuses = evaluate_trip(trip, graph, route, points, xy, labels, cand, lab)
            row.update(nearest_within_100m=len(nearest),
                       median_nearest_capped_m=np.median(nearest) if nearest else None,
                       **{key: statuses[key] for key in
                          ("accepted", "far_from_reference", "ambiguous", "discontinuous_route")})
            summaries.append(row)
            evaluations.extend(scores)
            if number == 1 or trip.partition != trips[number-2].partition:
                map_trip(out, trip, graph, route, points)
            print(f"{number}/{len(trips)} {trip.partition}/{trip.trip_id}", flush=True)
    write_rows(out / "trip_eda.csv", summaries)
    write_rows(out / "candidate_metrics_by_trip.csv", evaluations)
    summarize(out, summaries, evaluations)
    if feature_contract is not None:
        from .features import export
        export(root, out, feature_contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    from .features import DEFAULT_CONTRACT

    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()
    run(args.dataset_root, args.output, args.seed, None if args.skip_features else args.contract)
