"""PathSnap raw schemas and non-destructive observation cleaning."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trip:
    partition: str
    trip_id: str
    track: Path

    def file(self, suffix):
        return self.track.with_suffix("." + suffix)


def discover(root, *, require_reference=True):
    trips = []
    for partition in ("train", "student_test"):
        folder = Path(root) / partition
        if not folder.is_dir():
            raise ValueError(f"Missing partition: {folder}")
        for directory in sorted(folder.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            trip = Trip(partition, directory.name, directory / (directory.name + ".track"))
            suffixes = ("track", "nodes", "arcs", "route") if require_reference else ("track", "nodes", "arcs")
            for suffix in suffixes:
                if not trip.file(suffix).is_file():
                    raise FileNotFoundError(trip.file(suffix))
            trips.append(trip)
    ids = [trip.trip_id for trip in trips]
    actual = Counter(trip.partition for trip in trips)
    if len(set(ids)) != len(ids) or actual != {"train": 80, "student_test": 10}:
        raise ValueError(f"Unexpected partitions or overlapping trip IDs: {actual}")
    return trips


def manifest(trips, seed=42, folds=5):
    """Fixed hash ordering is independent of filesystem and Python RNG version."""
    train = [trip.trip_id for trip in trips if trip.partition == "train"]
    ordered = sorted(train, key=lambda tid: (sha256(f"{seed}:{tid}".encode()).digest(), tid))
    fold = {tid: i % folds for i, tid in enumerate(ordered)}
    return [
        dict(trip_id=t.trip_id, supplied_partition=t.partition, seed=seed,
             split=("student_test" if t.partition == "student_test"
                    else "validation" if fold[t.trip_id] == 0 else "train"),
             cv_fold=fold.get(t.trip_id))
        for t in trips
    ]


def read_matrix(path, width, dtype):
    """Strict network parsing preserves physical row IDs and fails on corrupt input."""
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if len(line.split()) != width:
                raise ValueError(f"{path}:{line_no} expected {width} fields")
    return np.loadtxt(path, dtype=dtype, ndmin=2, comments=None)


@dataclass(frozen=True)
class Point:
    point_id: int
    lon: float
    lat: float
    timestamp_s: float
    dt_s: float | None
    gap: bool


def clean_track(path, gap_s=5):
    """Keep first valid row at a timestamp, then sort by time."""
    counts = Counter()
    unique = {}
    seen = set()
    previous_raw_t = None
    with Path(path).open(encoding="utf-8") as source:
        for point_id, line in enumerate(source):
            counts["raw_points"] += 1
            tokens = line.split()
            if len(tokens) < 3:
                counts["missing_rows"] += 1
                continue
            if len(tokens) != 3:
                counts["malformed_rows"] += 1
                continue
            try:
                lon, lat, timestamp = map(float, tokens)
            except ValueError:
                key = "missing_rows" if any(t.lower() in {"na", "null", "none"} for t in tokens) else "malformed_rows"
                counts[key] += 1
                continue
            if not all(map(isfinite, (lon, lat, timestamp))):
                counts["nonfinite_rows"] += 1
                continue
            if abs(lon) > 180 or abs(lat) > 90:
                counts["invalid_coordinates"] += 1
                continue
            if previous_raw_t is not None and timestamp < previous_raw_t:
                counts["raw_time_reversals"] += 1
            previous_raw_t = timestamp
            row = (lon, lat, timestamp)
            if row in seen:
                counts["duplicate_observations"] += 1
            seen.add(row)
            if timestamp in unique:
                counts["duplicate_timestamps"] += 1
                continue
            unique[timestamp] = (point_id, lon, lat)
    points = []
    previous = None
    for timestamp, (point_id, lon, lat) in sorted(unique.items()):
        dt = None if previous is None else timestamp - previous
        gap = dt is not None and dt > gap_s
        counts["gaps"] += int(gap)
        points.append(Point(point_id, lon, lat, timestamp, dt, gap))
        previous = timestamp
    counts["clean_points"] = len(points)
    return points, counts
