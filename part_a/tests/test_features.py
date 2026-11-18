import csv
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from part_a.data import Point
from part_a.features import (
    FeatureContract, IDS, _csv_rows, compute_features, infer_features,
    load_model_input, verify_feature_file,
)
from part_a.roads import Candidate, RoadGraph


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.contract = FeatureContract.load()
        self.graph = RoadGraph("00000001", [[0, 0], [.001, 0], [.002, 0], [.003, 0]],
                               [[0, 1], [1, 2], [2, 3], [2, 1]])
        self.points = [
            Point(4, .0005, 0, 1, None, False),
            Point(5, .0015, 0, 2, 1, False),
            Point(6, .0025, 0, 3, 1, False),
            Point(8, .0015, 0, 22, 19, True),
            Point(9, .0025, 0, 24, 2, False),
        ]

    def candidate(self, arc, rank=1, distance=0, frac=.5):
        u, v = map(int, self.graph.arcs[arc])
        return Candidate((self.graph.trip_id, arc), u, v, distance, frac, rank)

    def test_directed_connection_heading_and_motion(self):
        batches = zip(self.points, [
            [self.candidate(0)], [self.candidate(1), self.candidate(3, 2)], [],
            [self.candidate(1)], [self.candidate(2)],
        ])
        table = compute_features(self.graph, batches, self.contract)
        self.assertEqual(tuple(table), self.contract.header)
        self.assertEqual(list(table["point_idx"]), [4, 5, 5, 8, 9])
        self.assertEqual(list(table["dist_rank"]), [1, 1, 2, 1, 1])
        np.testing.assert_allclose(table["heading_diff"], [0, 0, 180, 0, 0])
        np.testing.assert_allclose(table["heading_cos"], [1, 1, -1, 1, 1])
        np.testing.assert_allclose(table["along_ratio"], [1, 2, -2, 1, 1])
        np.testing.assert_allclose(table["conn_prev"], [np.nan, 1, 0, np.nan, 1])
        np.testing.assert_allclose(table["conn_next"], [1, np.nan, np.nan, 1, np.nan])
        np.testing.assert_allclose(table["speed"], [np.nan, 111.31949, 111.31949, np.nan, 55.659745],
                                   rtol=1e-6)
        self.assertEqual(table["dt_prev"][3], 19)
        self.assertTrue(np.isnan(table["dist_n2"][1]))
        self.assertTrue(np.isnan(table["dist_p1"][3]))
        self.assertAlmostEqual(table["dist_n1"][1], 55.659745, places=5)
        self.assertAlmostEqual(table["win_mean_dist"][1], 2 * 55.659745 / 3, places=5)

    def test_route_free_csv_roundtrip_and_pandas_categories(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "sample.track"
            source.write_text("0.0005 0 1\n0.0015 0 2\n0.0025 0 3\n")
            table = infer_features(source, self.graph, self.contract, k=3, radius_m=60)
            out = Path(temp) / "sample.csv.gz"
            with gzip.open(out, "wt", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(self.contract.header)
                writer.writerows(_csv_rows(table, self.contract))
            check = verify_feature_file(out, self.contract, {"00000001"})
            self.assertEqual(check["rows"], len(table["arc_id"]))
            ids, features = load_model_input(out, self.contract)
            self.assertEqual(tuple(ids), IDS)
            self.assertTrue(ids["trip_id"].eq("00000001").all())
            self.assertEqual(tuple(features), self.contract.columns)
            self.assertFalse(set(features) & set(self.contract.excluded))
            for col in self.contract.categorical:
                self.assertEqual(features[col].cat.categories.tolist(), [0, 1])
            with self.assertRaisesRegex(ValueError, "trip membership"):
                verify_feature_file(out, self.contract, {"other"})

    def test_all_empty(self):
        table = compute_features(self.graph, [(self.points[0], [])], self.contract)
        self.assertEqual(tuple(table), self.contract.header)
        self.assertEqual(len(table["arc_id"]), 0)

    def test_best_ratio_is_finite_at_zero_distance(self):
        table = compute_features(self.graph, [(self.points[0], [
            self.candidate(0), self.candidate(1, 2, distance=1)
        ])], self.contract)
        np.testing.assert_allclose(table["dist_ratio_best"], [1, 1001])
        np.testing.assert_allclose(table["dist_gap_best"], [0, 1])
        np.testing.assert_allclose(table["n_cands"], [2, 2])

if __name__ == "__main__":
    unittest.main()
