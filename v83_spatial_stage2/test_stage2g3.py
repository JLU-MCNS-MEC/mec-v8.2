"""CPU-only tests for Stage-2g.3 real-rollout probe helpers."""

import unittest

import numpy as np

from v83_spatial_stage2.stage2g3_probe import (
    build_input,
    episode_stats,
    filter_moves,
    real_metrics,
    select_thresholds,
)
from v83_spatial_stage2.stage2g3_collect import derive_history


def synthetic_data(n=6):
    return {
        "episode_id": np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        "band": np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        "stop": np.array([0, 0, 1, 0, 0, 1], dtype=np.int64),
        "history_valid": np.array([0, 1, 1, 0, 1, 1], dtype=np.int64),
        "current_features": np.arange(n * 2858, dtype=np.float32).reshape(
            n, 2858
        ),
        "history_features": np.zeros((n, 2, 74), dtype=np.float32),
        "history_actions": np.zeros((n, 2), dtype=np.int64),
        "history_dt": np.ones((n, 2), dtype=np.float32),
        "distance_m": np.array([50, 50, 50, 200, 50, 200], dtype=np.float32),
        "target_visible": np.array([0, 1, 1, 0, 0, 1], dtype=np.int64),
        "target_localizable": np.array([0, 1, 1, 0, 0, 1], dtype=np.int64),
        "near_target_invisible": np.array([1, 0, 0, 0, 1, 0], dtype=np.int64),
    }


class TestHistory(unittest.TestCase):
    def test_order_and_validity(self):
        features = np.arange(4 * 74, dtype=np.float32).reshape(4, 74)
        actions = np.array([0, 1, 2, 3], dtype=np.int64)
        history, hist_actions, hist_dt, valid = derive_history(
            features, actions, np.arange(74)
        )
        self.assertEqual(history.shape, (4, 2, 74))
        np.testing.assert_allclose(history[2, 0], features[1])
        np.testing.assert_allclose(history[2, 1], features[0])
        self.assertEqual(hist_actions[2].tolist(), [1, 0])
        np.testing.assert_allclose(hist_dt[2], [1.0, 1.0])
        self.assertEqual(valid.tolist(), [0, 0, 1, 1])


class TestFilterAndInputs(unittest.TestCase):
    def test_filter_and_dimensions(self):
        data = synthetic_data()
        filtered, mask = filter_moves(data)
        self.assertEqual(mask.tolist(), [False, True, False, False, True, False])
        self.assertEqual(filtered["current_features"].shape, (2, 2858))
        self.assertEqual(build_input(filtered, 0).shape, (2, 2858))
        self.assertEqual(build_input(filtered, 1).shape, (2, 3016))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        # Keep only the two valid move states (indices 1 and 4) plus two
        # controls by constructing a custom valid subset.
        self.data = {
            key: value[[0, 1, 2, 3, 4, 5]]
            for key, value in self.data.items()
        }
        self.score = np.array([0.9, 0.1, 0.1, 0.1, 0.5, 0.2])

    def test_real_metrics_composition(self):
        metrics = real_metrics(self.data, self.score, 0.5)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(sum(metrics["composition"].values()), 1.0)

    def test_threshold_selection(self):
        selected = select_thresholds(self.data, self.score)
        self.assertIsNotNone(selected["f1_joint_far5"])
        self.assertIsNotNone(selected["recall_0.5_far5"])
        self.assertGreaterEqual(
            selected["recall_0.5_far5"]["recall"], 0.5
        )

    def test_episode_stats(self):
        stats = episode_stats(self.data, 0.5)
        self.assertEqual(stats["episodes"], 2)
        self.assertGreaterEqual(stats["episodes_with_near_invisible"], 1)
        self.assertIn("short", stats["bands"])
        self.assertIn("20-50", stats["distance_bins"])


if __name__ == "__main__":
    unittest.main()
