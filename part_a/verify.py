"""Check generated tables against the source and brute-force sampled lookups."""
import argparse
from collections import Counter
import csv
from contextlib import ExitStack
import gzip
from itertools import groupby
from pathlib import Path

import numpy as np

from .data import clean_track, discover, manifest
from .reference import load_reference
from .roads import RoadGraph, RoadIndex
from .features import DEFAULT_CONTRACT, FeatureContract, SPLITS, _csv_rows, infer_features, verify_feature_file


def read_csv(path):
    with path.open(newline="") as source:
        yield from csv.DictReader(source)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, out):
    trips = discover(root)
    split = list(read_csv(out / "split_manifest.csv"))
    seed = int(split[0]["seed"])
    expected = [(r["trip_id"], r["supplied_partition"], r["split"],
                 "" if r["cv_fold"] is None else str(r["cv_fold"]), str(seed))
                for r in manifest(trips, seed)]
    actual = [(r["trip_id"], r["supplied_partition"], r["split"], r["cv_fold"], r["seed"]) for r in split]
    require(actual == expected, "Split manifest mismatch")
    counts = Counter()
    trip_metrics = {}
    for r in read_csv(out / "candidate_metrics_by_trip.csv"):
        key = (r["trip_id"], int(r["radius_m"]), int(r["k"]))
        require(key not in trip_metrics, f"Duplicate metric key {key}")
        trip_metrics[key] = r
    candidates = groupby(read_csv(out / "candidates.csv"), key=lambda r: r["trip_id"])
    labels = groupby(read_csv(out / "alignment.csv"), key=lambda r: r["trip_id"])
    for trip in trips:
        tid = trip.trip_id
        graph = RoadGraph.load(trip)
        route, breaks = load_reference(trip, graph)
        points, _ = clean_track(trip.track)
        point_ids = [p.point_id for p in points]
        label_tid, label_rows = next(labels)
        label_rows = list(label_rows)
        require(label_tid == tid and [int(r["point_id"]) for r in label_rows] == point_ids,
                f"{tid}: alignment keys")
        valid_positions = [int(r["route_position"]) for r in label_rows if r["route_position"]]
        require(valid_positions == sorted(valid_positions), f"{tid}: route ordering")
        for row in label_rows:
            require(bool(breaks) == (row["status"] == "discontinuous_route"), f"{tid}: break status")
            if row["route_position"]:
                require(int(route[int(row["route_position"])]) == int(row["provisional_arc_id"]),
                        f"{tid}: provisional ID mismatch")
        accepted = {int(r["point_id"]): int(r["provisional_arc_id"])
                    for r in label_rows if r["status"] == "accepted"}
        cand_tid, cand_rows = next(candidates)
        require(cand_tid == tid, f"{tid}: candidate trip mismatch")
        by_point = {}
        for raw_id, rows in groupby(cand_rows, key=lambda r: r["point_id"]):
            pid = int(raw_id)
            require(pid not in by_point, f"{tid}: repeated candidate batch for {pid}")
            records = list(rows)
            arcs = [int(r["arc_id"]) for r in records]
            distances = [float(r["distance_m"]) for r in records]
            require(len(records) <= 10 and len(set(arcs)) == len(arcs)
                    and [int(r["rank"]) for r in records] == list(range(1, len(records)+1))
                    and list(zip(distances, arcs)) == sorted(zip(distances, arcs))
                    and all(0 <= d <= 100 for d in distances),
                    f"{tid}: invalid ranking at point {pid}")
            for row, arc in zip(records, arcs):
                require(0 <= arc < len(graph.arcs) and
                        tuple(graph.arcs[arc]) == (int(row["from_node"]), int(row["to_node"])),
                        f"{tid}: candidate direction mismatch")
            by_point[pid] = list(zip(distances, arcs))
            counts["candidate_rows"] += len(records)
        require(set(by_point) <= set(point_ids), f"{tid}: unknown candidate point")
        ref = set(map(int, route))
        for radius in (25, 50, 100):
            for k in (1, 5, 10):
                score = trip_metrics[tid, radius, k]
                seen = set()
                empty = any_ref = hits = 0
                for pid in point_ids:
                    hit_arcs = {arc for dist, arc in by_point.get(pid, [])[:k] if dist <= radius}
                    empty += not hit_arcs
                    any_ref += bool(hit_arcs & ref)
                    hits += accepted.get(pid) in hit_arcs
                    seen.update(hit_arcs & ref)
                actual = (len(points), empty, any_ref, len(ref), len(seen), len(accepted), hits)
                expected = tuple(int(score[key]) for key in
                                 ("points", "empty", "point_any_reference", "unique_reference_count",
                                  "unique_reference_hit", "accepted_points", "accepted_hits"))
                require(actual == expected, f"{tid}: metric disagreement at {radius}/{k}")
        ids = np.arange(len(graph.arcs))
        for i in sorted({0, len(points)//2, len(points)-1}):
            point = points[i]
            xy = graph.project((point.lon, point.lat))
            distances, _ = graph.distance(xy, ids)
            close = np.flatnonzero(distances <= 100)
            ranked = sorted((float(distances[j]), int(j)) for j in close)[:10]
            require(ranked == by_point.get(point.point_id, []), f"{tid}: brute-force mismatch")
            counts["brute_force_checks"] += 1
        counts["trips"] += 1
        counts["points"] += len(points)
    require(next(labels, None) is None and next(candidates, None) is None
            and len(trip_metrics) == len(trips) * 9, "Unexpected extra rows")
    print(dict(counts), flush=True)


def verify_features(root, run, contract_path):
    contract = FeatureContract.load(contract_path)
    trips = {trip.trip_id: trip for trip in discover(root, require_reference=False)}
    split = {row["trip_id"]: row["split"] for row in read_csv(run / "split_manifest.csv")}
    counts = Counter()
    sampled = set()
    with ExitStack() as stack:
        streams = {}
        for name in SPLITS:
            path = run / "features" / f"{name}.csv.gz"
            allowed = {tid for tid, group in split.items() if group == name}
            counts[name] = verify_feature_file(path, contract, allowed)["rows"]
            source = stack.enter_context(gzip.open(path, "rt", newline="", encoding="utf-8"))
            streams[name] = csv.DictReader(source)
        for trip_id, rows in groupby(read_csv(run / "candidates.csv"), key=lambda row: row["trip_id"]):
            name = split[trip_id]
            direct = None
            if name not in sampled:
                trip = trips[trip_id]
                table = infer_features(trip.track, RoadGraph.load(trip), contract)
                direct = iter(_csv_rows(table, contract))
            for row in rows:
                exported = next(streams[name], None)
                require(exported is not None and
                        (exported["trip_id"], exported["point_idx"], exported["arc_id"]) ==
                        (trip_id, row["point_id"], row["arc_id"]),
                        f"{trip_id}: feature/candidate join keys differ")
                require(exported["dist"] == format(float(row["distance_m"]), ".9g") and
                        exported["frac"] == format(float(row["offset_fraction"]), ".9g") and
                        exported["dist_rank"] == row["rank"],
                        f"{trip_id}: feature/candidate values differ")
                if direct is not None:
                    expected = next(direct, None)
                    require(expected is not None and [str(value) for value in expected] ==
                            list(exported.values()), f"{trip_id}: direct inference values differ")
            if direct is not None:
                require(next(direct, None) is None, f"{trip_id}: extra direct candidate rows")
                sampled.add(name)
        for name, stream in streams.items():
            require(next(stream, None) is None, f"{name}: extra feature rows")
    print({"validated_feature_rows": dict(counts), "route_free_inference_trips": len(sampled)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    verify(args.dataset_root, args.results)
    if (args.results / "features").is_dir():
        verify_features(args.dataset_root, args.results, args.contract)
