import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from part_a.data import Trip, clean_track, manifest, read_matrix
from part_a.inference import candidate_batches
from part_a.reference import align, load_reference
from part_a.roads import RoadGraph, RoadIndex


class TestPartA(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.trip = Trip("train", "00000001", self.folder / "00000001.track")
        for suffix, data in {
            "track": "0.0005 0 1\n0.0015 0 2\n",
            "nodes": "0 0\n0.001 0\n0.002 0\n",
            "arcs": "0 1\n1 2\n1 0\n",
            "route": "0\n1\n",
        }.items():
            self.trip.file(suffix).write_text(data)

    def test_parse_ids_and_direction(self):
        graph = RoadGraph.load(self.trip)
        np.testing.assert_array_equal(graph.arcs, [[0, 1], [1, 2], [1, 0]])
        np.testing.assert_array_equal(graph.out_degree, [1, 2, 0])
        np.testing.assert_array_equal(graph.in_degree, [1, 1, 1])
        self.assertEqual(graph.weak_components(), [3])
        route, breaks = load_reference(self.trip, graph)
        np.testing.assert_array_equal(route, [0, 1])
        self.assertEqual(breaks, [])

    def test_strict_parser_and_bad_endpoint(self):
        self.trip.file("arcs").write_text("0 1\n\n1 2\n")
        with self.assertRaisesRegex(ValueError, r":2 expected"):
            read_matrix(self.trip.file("arcs"), 2, int)
        self.trip.file("arcs").write_text("0 4\n")
        with self.assertRaisesRegex(ValueError, "invalid or missing arc endpoints"):
            RoadGraph.load(self.trip)

    def test_clean_track_audit_and_large_gap(self):
        self.trip.track.write_text("1 2 12\n1 2 1\n1 2 1\n2 3 1\n181 2 2\n"
                                   "1 2\n1 x 4\n1 NaN 5\n1 NA 6\n")
        points, counts = clean_track(self.trip.track)
        self.assertEqual([p.point_id for p in points], [1, 0])
        self.assertEqual(points[1].dt_s, 11)
        self.assertTrue(points[1].gap)
        self.assertEqual([counts[key] for key in ("raw_points", "duplicate_timestamps",
                         "duplicate_observations", "raw_time_reversals", "invalid_coordinates",
                         "missing_rows", "malformed_rows", "nonfinite_rows")],
                         [9, 2, 1, 1, 1, 2, 1, 1])

    def test_distance_projection_and_rtree_agree_with_brute_force(self):
        graph = RoadGraph.load(self.trip)
        ids = np.arange(len(graph.arcs))
        np.testing.assert_allclose(graph.length_m, 111.31949, atol=.0001)
        with RoadIndex(graph) as index:
            for lon, lat, radius in [(0.0005, .0001, 12), (.0011, 0, 10), (0, 1, 30),
                                     (.0005, 0, 0)]:
                position = graph.project((lon, lat))
                distance, offset = graph.distance(position, ids)
                expected = sorted((int(i) for i in ids if distance[i] <= radius),
                                  key=lambda i: (distance[i], i))
                candidates = index.query(position, radius)
                self.assertEqual([c.arc_id for c in candidates], expected)
                for c in candidates:
                    self.assertEqual(c.segment_id, (self.trip.trip_id, c.arc_id))
                    self.assertAlmostEqual(c.distance_m, distance[c.arc_id])
                    self.assertAlmostEqual(c.offset_fraction, offset[c.arc_id])
            c = index.query(graph.project((.0005, .0001)), 12, 1)[0]
            self.assertAlmostEqual(c.distance_m, 11.0574276, places=5)
            self.assertAlmostEqual(c.offset_fraction, .5)
            self.assertEqual(c.rank, 1)
            with self.assertRaises(ValueError):
                index.query((0, 0), -1)

    def test_alignment_marks_ambiguity_and_breaks(self):
        graph = RoadGraph.load(self.trip)
        xy = graph.project([[.0005, 0], [.0015, 0]])
        labels = align(graph, xy, np.array([0, 1]), [])
        self.assertEqual([row[:2] for row in labels], [(0, 0), (1, 1)])
        self.assertEqual([row[3] for row in labels], ["accepted", "accepted"])
        self.assertEqual(align(graph, xy[:1], np.array([0, 2]), [0])[0][3], "discontinuous_route")
        at_joint = graph.project([[.001, 0]])
        self.assertEqual(align(graph, at_joint, np.array([0, 1]), [])[0][3], "ambiguous")

    def test_route_free_candidates_and_empty_observations(self):
        graph = RoadGraph.load(self.trip)
        self.trip.file("route").unlink()
        self.trip.track.write_text("0.0005 0 1\n0 1 2\n")
        rows = list(candidate_batches(self.trip.track, graph, k=1, radius_m=10))
        self.assertEqual(rows[0][1][0].arc_id, 0)
        self.assertEqual(rows[1][1], [])
        self.assertEqual(len(rows), 2)

    @unittest.skipUnless(importlib.util.find_spec("networkx"), "NetworkX not installed")
    def test_networkx_adapter(self):
        graph = RoadGraph.load(self.trip).to_networkx()
        self.assertTrue(graph.is_directed())
        self.assertEqual(graph[1][0][2]["segment_id"], ("00000001", 2))
        self.assertFalse(graph.has_edge(2, 1))

    def test_manifest_integrity(self):
        trips = [Trip("train" if i < 80 else "student_test", str(i), Path())
                 for i in range(90)]
        rows = manifest(trips)
        by_id = {r["trip_id"]: r for r in rows}
        self.assertEqual(by_id, {r["trip_id"]: r for r in manifest(list(reversed(trips)))})
        self.assertEqual(len(by_id), 90)
        self.assertEqual(sum(r["split"] == "validation" for r in rows), 16)
        for fold in range(5):
            self.assertEqual(sum(r["cv_fold"] == fold for r in rows), 16)
        self.assertEqual(sum(r["cv_fold"] is None and r["split"] == "student_test" for r in rows), 10)


if __name__ == "__main__":
    unittest.main()
