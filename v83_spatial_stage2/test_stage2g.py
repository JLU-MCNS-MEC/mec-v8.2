"""CPU-only tests for the Stage-2g label, feature and metric helpers."""

import math
import unittest

import numpy as np

from v83_spatial_stage2.stage2g_features import (
    _check_feature_names,
    feature_names,
)
from v83_spatial_stage2.stage2g_labels import target_labels
from v83_spatial_stage2.stage2g_probe import (
    assign_episode_folds,
    auc_rank,
    average_precision,
    brier_score,
    confusion,
    expected_calibration_error,
    negative_operating_points,
    scalar_feature_columns,
)


class TestTargetLabels(unittest.TestCase):
    def test_invisible_has_no_label(self):
        self.assertEqual(target_labels(0, float("nan")), (0, 0))

    def test_visible_but_too_small_is_not_localizable(self):
        self.assertEqual(target_labels(1, 5.0), (1, 0))
        self.assertEqual(target_labels(10, 15.0), (1, 0))

    def test_visible_and_large_enough_is_localizable(self):
        self.assertEqual(target_labels(20, 10.0), (1, 1))
        self.assertEqual(target_labels(400, 40.0), (1, 1))


class TestMetrics(unittest.TestCase):
    def test_auc_rank(self):
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(
            auc_rank(labels, np.array([1.0, 2.0, 3.0, 4.0])), 1.0
        )
        self.assertAlmostEqual(
            auc_rank(labels, np.array([4.0, 3.0, 2.0, 1.0])), 0.0
        )
        self.assertAlmostEqual(
            auc_rank(np.array([0, 1, 0, 1]),
                     np.array([1.0, 1.0, 1.0, 1.0])),
            0.5,
        )

    def test_average_precision(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        self.assertAlmostEqual(
            average_precision(labels, scores), (1.0 + 2.0 / 3.0) / 2.0
        )
        self.assertAlmostEqual(
            average_precision(np.ones(3, dtype=int), np.array([0.1, 0.2, 0.3])),
            1.0,
        )

    def test_brier_and_ece(self):
        labels = np.array([0, 1])
        probabilities = np.array([0.0, 1.0])
        self.assertAlmostEqual(brier_score(labels, probabilities), 0.0)
        self.assertAlmostEqual(expected_calibration_error(labels, probabilities), 0.0)
        self.assertAlmostEqual(
            brier_score(labels, np.array([0.5, 0.5])), 0.25
        )
        self.assertAlmostEqual(
            expected_calibration_error(labels, np.array([0.9, 0.9])), 0.4
        )

    def test_confusion(self):
        result = confusion(
            np.array([1, 0, 1, 0]),
            np.array([0.9, 0.1, 0.6, 0.4]),
            0.5,
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"],
                          result["tn"]), (2, 0, 0, 2))
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["specificity"], 1.0)
        self.assertAlmostEqual(result["balanced_accuracy"], 1.0)

    def test_negative_operating_point(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])
        points = negative_operating_points(labels, probabilities)
        recall_point = points["at_negative_recall"]["0.5"]
        self.assertAlmostEqual(recall_point["recall"], 0.5)
        self.assertAlmostEqual(recall_point["precision"], 1.0)
        self.assertAlmostEqual(recall_point["threshold"], 0.1)


class TestFoldsAndLeakage(unittest.TestCase):
    def test_episode_folds_do_not_split_episodes(self):
        episodes = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        bands = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        folds = assign_episode_folds(episodes, bands, 2, seed=0)
        for episode in np.unique(episodes):
            self.assertEqual(len(np.unique(folds[episodes == episode])), 1)

    def test_scalar_features_exclude_true_labels(self):
        columns = [
            "sim_max", "goal_visible", "distance_m", "heatmap0_top1",
            "m0_arrival_probability", "band", "coherent_peak_error_px",
            "temporal_valid",
        ]
        selected = scalar_feature_columns(columns)
        self.assertEqual(
            selected, ["sim_max", "heatmap0_top1", "temporal_valid"]
        )

    def test_feature_names_have_no_forbidden_terms(self):
        names = feature_names()
        _check_feature_names(names)
        with self.assertRaises(ValueError):
            _check_feature_names(["distance_proxy"])
        with self.assertRaises(ValueError):
            _check_feature_names(["goal_visible"])


if __name__ == "__main__":
    unittest.main()
