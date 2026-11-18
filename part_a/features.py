"""Route-free candidate features, with ordered CSV and pandas model inputs."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import gzip
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from .data import clean_track, discover
from .inference import candidate_batches
from .roads import Candidate, RoadGraph

DEFAULT_CONTRACT = Path(__file__).resolve().parent.parent / "artifacts" / "feature_columns.json"
IDS = ("trip_id", "point_idx", "arc_id")
SPLITS = ("train", "validation", "student_test")
CATEGORIES = ("conn_prev", "conn_next")
CATEGORY_LEVELS = [0, 1]


@dataclass(frozen=True)
class FeatureContract:
    columns: tuple
    categorical: tuple
    excluded: tuple

    @classmethod
    def load(cls, path=DEFAULT_CONTRACT):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        contract = cls(tuple(data["feature_columns"]), tuple(data["categorical"]),
                       tuple(data["never_use_as_features"]))
        if (len(set(contract.columns)) != len(contract.columns)
                or len(contract.categorical) != len(CATEGORIES)
                or set(contract.categorical) != set(CATEGORIES)
                or not set(CATEGORIES) <= set(contract.columns)
                or not set((*IDS, "on_route", "label")) <= set(contract.excluded)
                or set(contract.columns) & set(contract.excluded)):
            raise ValueError("Invalid feature contract or unsupported categorical definitions")
        return contract

    @property
    def header(self):
        return IDS + self.columns


def _shift_valid(positions, groups, offset):
    shifted = positions + offset
    in_range = (shifted >= 0) & (shifted < len(groups))
    safe = np.clip(shifted, 0, len(groups) - 1)
    return safe, in_range & (groups[safe] == groups[positions])


def _connections(current, adjacent, previous):
    if not adjacent:
        return [np.nan] * len(current)
    arc_ids = {c.arc_id for c in adjacent}
    if previous:
        nodes = {c.to_node for c in adjacent}
        return [int(c.arc_id in arc_ids or c.from_node in nodes) for c in current]
    nodes = {c.from_node for c in adjacent}
    return [int(c.arc_id in arc_ids or c.to_node in nodes) for c in current]


def compute_features(graph, batches, contract):
    """Return ordered column arrays for one complete trip, including offline lookahead.

    A batch pairs a cleaned Point and its candidate list. An observation with
    zero candidates still participates in the temporal window.
    """
    batches = list(batches)
    if not batches:
        return {name: np.array([]) for name in contract.header}
    points, candidates = zip(*batches)
    xy = graph.project([(p.lon, p.lat) for p in points])
    groups = np.cumsum([p.gap for p in points])
    counts = np.array([len(cs) for cs in candidates], dtype=int)
    pos = np.repeat(np.arange(len(points)), counts)
    flat = [c for cs in candidates for c in cs]
    if not flat:
        return {name: np.array([]) for name in contract.header}
    ids = np.array([c.arc_id for c in flat])
    dist = np.array([c.distance_m for c in flat])
    length = graph.length_m[ids]
    best = np.repeat([cs[0].distance_m if cs else np.nan for cs in candidates], counts)
    cols = dict(dist=dist, frac=np.array([c.offset_fraction for c in flat]),
                arc_len=length, dist_rank=np.array([c.rank for c in flat]),
                n_cands=counts[pos], dist_gap_best=dist-best,
                dist_ratio_best=(dist+0.001)/(best+0.001),
                out_deg_end=graph.out_degree[graph.arcs[ids, 1]])

    for offset, name in [(-2, "dist_p2"), (-1, "dist_p1"), (1, "dist_n1"), (2, "dist_n2")]:
        shifted, valid = _shift_valid(pos, groups, offset)
        distances, _ = graph.distance(xy[shifted], ids)
        cols[name] = np.where(valid, distances, np.nan)
    window = [dist, cols["dist_p2"], cols["dist_p1"], cols["dist_n1"], cols["dist_n2"]]
    cols["win_mean_dist"] = np.nanmean(window, axis=0)
    cols["win_max_dist"] = np.nanmax(window, axis=0)

    prev, valid_prev = _shift_valid(pos, groups, -1)
    nxt, valid_next = _shift_valid(pos, groups, 1)
    left = np.where(valid_prev, prev, pos)
    right = np.where(valid_next, nxt, pos)
    motion = xy[right] - xy[left]
    motion_norm = np.linalg.norm(motion, axis=1)
    dot = np.einsum("ij,ij->i", motion, graph.vector[ids])
    norm_product = motion_norm * length
    cosine = np.divide(dot, norm_product, out=np.full(len(ids), np.nan), where=norm_product > 0)
    cols["heading_cos"] = np.clip(cosine, -1, 1)
    cols["heading_diff"] = np.degrees(np.arccos(cols["heading_cos"]))
    cols["along_ratio"] = np.divide(dot, length**2, out=np.full(len(ids), np.nan), where=norm_product > 0)

    step = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)]
    step[np.array([p.gap for p in points], dtype=bool)] = np.nan
    dt = np.array([p.dt_s if p.dt_s is not None else np.nan for p in points])
    speed = step / dt
    next_step = np.r_[step[1:], np.nan]
    accel = (speed - np.r_[np.nan, speed[:-1]]) / dt
    cols.update(step_prev=step[pos], step_next=next_step[pos], dt_prev=dt[pos],
                speed=speed[pos], accel=accel[pos])
    for offset, name in [(-1, "conn_prev"), (1, "conn_next")]:
        conn = []
        for i, cs in enumerate(candidates):
            j = i + offset
            adjacent = candidates[j] if 0 <= j < len(points) and groups[i] == groups[j] else ()
            conn.extend(_connections(cs, adjacent, offset < 0))
        cols[name] = np.asarray(conn)

    if set(cols) != set(contract.columns):
        raise ValueError(f"Feature implementation differs from contract: {set(cols) ^ set(contract.columns)}")
    metadata = {"trip_id": np.full(len(ids), graph.trip_id),
                "point_idx": np.array([p.point_id for p in points])[pos], "arc_id": ids}
    return {**metadata, **{name: cols[name] for name in contract.columns}}


def infer_features(track, graph, contract, *, k=10, radius_m=100):
    return compute_features(graph, candidate_batches(track, graph, k=k, radius_m=radius_m), contract)


def load_model_input(path, contract):
    """Return identifiers and the ordered model frame with fixed categorical types."""
    import pandas as pd

    frame = pd.read_csv(path, dtype={"trip_id": str})
    if tuple(frame.columns) != contract.header:
        raise ValueError("CSV columns differ from the supplied feature contract")
    for column in contract.categorical:
        if not frame[column].dropna().isin(CATEGORY_LEVELS).all():
            raise ValueError(f"Unexpected category in {column}")
        frame[column] = pd.Categorical(frame[column], categories=CATEGORY_LEVELS)
    return frame.loc[:, list(IDS)], frame.loc[:, list(contract.columns)]


def _candidate_batches(trip, graph, rows):
    points, _ = clean_track(trip.track)
    by_point = {int(point_id): list(batch) for point_id, batch in
                groupby(rows, key=lambda row: row["point_id"])}
    for point in points:
        candidates = []
        for expected_rank, row in enumerate(by_point.pop(point.point_id, ()), 1):
            arc = int(row["arc_id"])
            u, v = int(row["from_node"]), int(row["to_node"])
            rank = int(row["rank"])
            if not 0 <= arc < len(graph.arcs) or rank != expected_rank or tuple(graph.arcs[arc]) != (u, v):
                raise ValueError(f"Candidate arc mismatch at {trip.trip_id}/{point.point_id}/{arc}")
            candidates.append(Candidate((trip.trip_id, arc), u, v,
                                        float(row["distance_m"]), float(row["offset_fraction"]), rank))
        yield point, candidates
    if by_point:
        raise ValueError(f"Candidate table contains unknown point IDs in {trip.trip_id}")


def _csv_rows(columns, contract):
    for row in zip(*columns.values(), strict=True):
        numeric = row[len(IDS):]
        yield (*row[:len(IDS)], *("" if np.isnan(value) else
               str(int(value)) if name in contract.categorical else format(value, ".9g")
               for name, value in zip(contract.columns, numeric, strict=True)))


def _trip_groups(candidates):
    """Yield one materialized trip at a time and skip zero-candidate trips."""
    for trip_id, rows in groupby(csv.DictReader(candidates), key=lambda row: row["trip_id"]):
        yield trip_id, list(rows)


def verify_feature_file(path, contract, expected_trip_ids):
    """Stream all rows and reject schema, ID, value, ordering or category errors."""
    count = 0
    previous = None
    previous_rank = 0
    seen_arcs = set()
    seen_points = set()
    rows_by_trip = Counter()
    missing = Counter()
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if tuple(next(reader)) != contract.header:
            raise ValueError(f"{path}: invalid ordered feature header")
        rank_idx = contract.columns.index("dist_rank")
        for row in reader:
            if len(row) != len(contract.header) or row[0] not in expected_trip_ids:
                raise ValueError(f"{path}: invalid row width or trip membership")
            point = (row[0], int(row[1]))
            arc, rank = int(row[2]), int(row[3+rank_idx])
            if point != previous:
                if point in seen_points:
                    raise ValueError(f"{path}: repeated point group")
                seen_points.add(point)
                seen_arcs.clear()
                previous_rank = 0
            if min(point[1], arc) < 0 or arc in seen_arcs or rank != previous_rank + 1:
                raise ValueError(f"{path}: invalid candidate ID/rank")
            for name, value in zip(contract.columns, row[3:], strict=True):
                if value == "":
                    missing[name] += 1
                elif name in contract.categorical and value not in {"0", "1"}:
                    raise ValueError(f"{path}: unexpected {name} category {value}")
                elif not np.isfinite(float(value)):
                    raise ValueError(f"{path}: nonfinite {name}")
            previous, previous_rank = point, rank
            seen_arcs.add(arc)
            rows_by_trip[point[0]] += 1
            count += 1
    return {"rows": count, "rows_by_trip": dict(rows_by_trip), "missing": dict(missing)}


def export(root, run_dir, contract_path=DEFAULT_CONTRACT):
    """Add feature tables to an A1–A6 run without reading routes or labels."""
    contract = FeatureContract.load(contract_path)
    trips = discover(root, require_reference=False)
    with (run_dir / "split_manifest.csv").open(newline="", encoding="utf-8") as source:
        split_rows = list(csv.DictReader(source))
    split = {row["trip_id"]: row["split"] for row in split_rows}
    if len(split) != len(split_rows) or set(split) != {trip.trip_id for trip in trips}:
        raise ValueError("Manifest must contain each trip exactly once")
    for trip in trips:
        permitted = {"student_test"} if trip.partition == "student_test" else {"train", "validation"}
        if split[trip.trip_id] not in permitted:
            raise ValueError(f"Invalid split membership: {trip.trip_id}")
    target = run_dir / "features"
    target.mkdir(exist_ok=False)
    count = Counter()
    with ExitStack() as stack:
        writers = {}
        for name in SPLITS:
            handle = stack.enter_context(gzip.open(target / f"{name}.csv.gz", "wt", newline="",
                                                   encoding="utf-8", compresslevel=3))
            writers[name] = csv.writer(handle)
            writers[name].writerow(contract.header)
        candidates = stack.enter_context((run_dir / "candidates.csv").open(newline="", encoding="utf-8"))
        groups = iter(_trip_groups(candidates))
        current = next(groups, None)
        for trip in trips:
            if current is None or current[0] != trip.trip_id:
                continue
            graph = RoadGraph.load(trip)
            columns = compute_features(graph, _candidate_batches(trip, graph, current[1]), contract)
            writers[split[trip.trip_id]].writerows(_csv_rows(columns, contract))
            count[trip.trip_id] = len(columns["arc_id"])
            if count[trip.trip_id] != len(current[1]):
                raise ValueError(f"Candidate count changed: {trip.trip_id}")
            current = next(groups, None)
        if current is not None:
            raise ValueError(f"Unexpected candidate trip order or ID: {current[0]}")
    report = {}
    for name in SPLITS:
        expected = {tid for tid, label in split.items() if label == name}
        report[name] = verify_feature_file(target / f"{name}.csv.gz", contract, expected)
        if report[name]["rows_by_trip"] != {tid: count[tid] for tid in expected if count[tid]}:
            raise ValueError(f"Export does not preserve candidate counts for {name}")
    (target / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print({name: report[name]["rows"] for name in SPLITS}, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    export(args.dataset_root, args.run_dir, args.contract)
