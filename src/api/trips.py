"""Reader for the supplied per-trip text format.

Prediction input deliberately excludes `.route`; callers must request the
reference sequence separately after the matcher has returned its prediction.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRIP_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class TripInput:
    trip_id: str
    points: list[dict[str, float]]
    nodes: list[dict[str, float | int]]
    arcs: list[dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "points": self.points,
            "nodes": self.nodes,
            "arcs": self.arcs,
        }


class TripStore:
    """Read `<root>/<trip_id>/<trip_id>.<track|nodes|arcs|route>` records."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _trip_dir(self, trip_id: str) -> Path:
        if not _TRIP_ID.fullmatch(trip_id):
            raise ValueError("trip_id may contain only letters, digits, '_' and '-'")
        folder = (self.root / trip_id).resolve()
        if folder.parent != self.root.resolve():
            raise ValueError("trip_id is outside the configured dataset")
        return folder

    def list_trip_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        trip_ids = []
        for folder in self.root.iterdir():
            if not folder.is_dir() or not _TRIP_ID.fullmatch(folder.name):
                continue
            if all((folder / f"{folder.name}{suffix}").is_file() for suffix in (".track", ".nodes", ".arcs", ".route")):
                trip_ids.append(folder.name)
        return sorted(trip_ids)

    def _rows(self, trip_id: str, suffix: str, columns: int) -> list[list[str]]:
        path = self._trip_dir(trip_id) / f"{trip_id}{suffix}"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != columns:
                    raise ValueError(f"{path.name}:{line_number} expected {columns} columns, found {len(fields)}")
                rows.append(fields)
        return rows

    @staticmethod
    def _number(value: str, path: str, line_number: int) -> float:
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} contains a non-numeric value") from exc
        if not math.isfinite(result):
            raise ValueError(f"{path}:{line_number} contains a non-finite value")
        return result

    def load_prediction_input(self, trip_id: str) -> TripInput:
        """Parse only observations and graph data; never open the `.route` file."""
        track_rows = self._rows(trip_id, ".track", 3)
        points = []
        for i, row in enumerate(track_rows, start=1):
            lon = self._number(row[0], f"{trip_id}.track", i)
            lat = self._number(row[1], f"{trip_id}.track", i)
            timestamp_s = self._number(row[2], f"{trip_id}.track", i)
            if not -180 <= lon <= 180 or not -90 <= lat <= 90:
                raise ValueError(f"{trip_id}.track:{i} has out-of-range coordinates")
            points.append({"lat": lat, "lon": lon, "timestamp_s": timestamp_s})

        node_rows = self._rows(trip_id, ".nodes", 2)
        nodes = []
        for i, row in enumerate(node_rows):
            lon = self._number(row[0], f"{trip_id}.nodes", i + 1)
            lat = self._number(row[1], f"{trip_id}.nodes", i + 1)
            if not -180 <= lon <= 180 or not -90 <= lat <= 90:
                raise ValueError(f"{trip_id}.nodes:{i + 1} has out-of-range coordinates")
            nodes.append({"node_id": i, "lon": lon, "lat": lat})

        arc_rows = self._rows(trip_id, ".arcs", 2)
        arcs = []
        for arc_id, row in enumerate(arc_rows):
            try:
                from_node, to_node = int(row[0]), int(row[1])
            except ValueError as exc:
                raise ValueError(f"{trip_id}.arcs:{arc_id + 1} must contain integer node IDs") from exc
            if not (0 <= from_node < len(nodes) and 0 <= to_node < len(nodes)):
                raise ValueError(f"{trip_id}.arcs:{arc_id + 1} references a missing node")
            arcs.append({"arc_id": arc_id, "from_node_id": from_node, "to_node_id": to_node})

        if len(points) < 2:
            raise ValueError(f"{trip_id} must contain at least two GPS observations")
        return TripInput(trip_id=trip_id, points=points, nodes=nodes, arcs=arcs)

    def read_reference_route(self, trip_id: str) -> list[int]:
        """Read the chronological arc sequence for server-side evaluation only."""
        rows = self._rows(trip_id, ".route", 1)
        try:
            route = [int(row[0]) for row in rows]
        except ValueError as exc:
            raise ValueError(f"{trip_id}.route must contain integer arc IDs") from exc
        if not route:
            raise ValueError(f"{trip_id}.route contains no reference arcs")
        return route
