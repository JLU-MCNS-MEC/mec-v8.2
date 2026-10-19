"""CPU-only tests for the Stage-2g.1 near-target supplement helpers."""

import unittest

import numpy as np

from v83_spatial_stage2.stage2g1_common import (
    COMPACT_FEATURE_NAMES,
    COMPACT_DIM,
    GOAL_DIM,
    build_far_history,
    compact_indices,
    distance_band,
    goal_indices,
    near_invisible_label,
    one_hot_actions,
)
from v83_spatial_stage2.stage2g1_probe import (
    build_input,
    select_thresholds,
    trigger_metrics,
)
from v83_spatial_stage2.stage2g_features import feature_names


def synthetic_probe_data():
    return {
        "source": np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
        "variant": np.array([1, 1, 0, 0, -1, -1, -1, -1, -1, -1]),
        "distance_m": np.array([50, 50, 50, 50, 200, 200, 200, 200, 200, 200]),
        "target_visible": np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
        "target_localizable": np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
        "near_target_invisible": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        "history_origin": np.array([2, 2, 2, 2, 0, 0, 0, 1, 1, 1]),
        "episode_id": np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
        "band": np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
        "current_features": np.zeros((10, 2858), dtype=np.float32),
        "history_features": np.zeros((10, 3, 74), dtype=np.float32),
        "history_actions": np.zeros((10, 3), dtype=np.int64),
        "history_dt": np.ones((10, 3), dtype=np.float32),
        "feature_names": feature_names(),
    }


class TestLabelsAndFeatures(unittest.TestCase):
    def test_near_invisible_boundary(self):
        self.assertEqual(near_invisible_label(99.0, 0), 1)
        self.assertEqual(near_invisible_label(100.0, 0), 0)
        self.assertEqual(near_invisible_label(50.0, 1), 0)

    def test_distance_band(self):
        self.assertEqual(distance_band(5), "d_0_20")
        self.assertEqual(distance_band(20), "d_20_50")
        self.assertEqual(distance_band(99), "d_50_100")
        self.assertEqual(distance_band(100), "d_100_200")
        self.assertEqual(distance_band(400), "d_400_plus")

    def test_compact_and_goal_indices(self):
        names = feature_names()
        compact = compact_indices(names)
        goal = goal_indices(names)
        self.assertEqual(len(compact), COMPACT_DIM)
        self.assertEqual(len(set(compact.tolist())), COMPACT_DIM)
        self.assertEqual(len(goal), GOAL_DIM)
        for name in COMPACT_FEATURE_NAMES:
            self.assertNotIn(name, ("distance_m", "goal_visible"))

    def test_one_hot_actions(self):
        one_hot = one_hot_actions(np.array([0, 1, 2, -1]))
        np.testing.assert_array_equal(one_hot[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(one_hot[1], [0, 1, 0, 0])
        np.testing.assert_array_equal(one_hot[2], [0, 0, 1, 0])
        np.testing.assert_array_equal(one_hot[3], [0, 0, 0, 0])


class TestHistory(unittest.TestCase):
    def test_build_far_history_order_and_validity(self):
        n = 6
        features = np.arange(n * COMPACT_DIM, dtype=np.float32).reshape(
            n, COMPACT_DIM
        )
        data = {
            "episode_id": np.zeros(n, dtype=np.int64),
            "step_id": np.arange(n, dtype=np.int64),
            "sample_type": np.zeros(n, dtype=np.int64),
            "features": features,
        }
        actions = {(0, step): step % 3 for step in range(n)}
        history = build_far_history(
            data, actions, np.arange(COMPACT_DIM)
        )
        self.assertEqual(history["history_valid"].tolist(), [0, 0, 0, 1, 1, 1])
        # t=5 uses t-1=4, t-2=3, t-3=2.
        np.testing.assert_allclose(
            history["history_features"][5, 0], features[4]
        )
        np.testing.assert_allclose(
            history["history_features"][5, 1], features[3]
        )
        np.testing.assert_allclose(
            history["history_features"][5, 2], features[2]
        )
        self.assertEqual(history["history_actions"][5].tolist(), [1, 0, 2])

    def test_hard_state_has_no_actual_history(self):
        n = 4
        data = {
            "episode_id": np.zeros(n, dtype=np.int64),
            "step_id": np.arange(n, dtype=np.int64),
            "sample_type": np.array([0, 0, 0, 1], dtype=np.int64),
            "features": np.zeros((n, COMPACT_DIM), dtype=np.float32),
        }
        actions = {(0, step): 0 for step in range(n)}
        history = build_far_history(data, actions, np.arange(COMPACT_DIM))
        self.assertEqual(history["history_valid"].tolist(), [0, 0, 0, 0])


class TestP1Inputs(unittest.TestCase):
    def test_dimensions(self):
        data = synthetic_probe_data()
        self.assertEqual(build_input(data, 0).shape, (10, 2858))
        self.assertEqual(build_input(data, 1).shape, (10, 2858 + 79))
        self.assertEqual(build_input(data, 2).shape, (10, 2858 + 2 * 79))
        self.assertEqual(build_input(data, 3).shape, (10, 2858 + 3 * 79))

    def test_no_true_label_inputs(self):
        data = synthetic_probe_data()
        names = list(data["feature_names"]) + list(COMPACT_FEATURE_NAMES)
        forbidden = ("distance", "visible", "oracle", "half_size", "heading")
        for name in names:
            lowered = name.lower()
            for term in forbidden:
                self.assertNotIn(term, lowered)


class TestTrigger(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_probe_data()
        self.score = np.array([
            0.90, 0.80, 0.20, 0.30, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35
        ])

    def test_trigger_composition_sums_to_one(self):
        metrics = trigger_metrics(self.data, self.score, 0.5)
        self.assertAlmostEqual(sum(metrics["flag_composition"].values()), 1.0)
        self.assertAlmostEqual(
            metrics["near_target_invisible_precision"], 1.0
        )
        self.assertAlmostEqual(
            metrics["near_target_invisible_recall"], 1.0
        )

    def test_recall_threshold_is_highest_with_target_recall(self):
        selected = select_thresholds(self.data, self.score)
        entry = selected["recall_0.5"]
        self.assertAlmostEqual(entry["near_target_invisible_recall"], 0.5)
        self.assertAlmostEqual(
            entry["near_target_invisible_precision"], 1.0
        )
        self.assertGreaterEqual(entry["threshold"], 0.9)

    def test_precision_selection_ignores_zero_flag_threshold(self):
        selected = select_thresholds(self.data, self.score)
        entry = selected["precision_0.8"]
        self.assertIsNotNone(entry)
        self.assertGreater(entry["flags"], 0)
        self.assertGreaterEqual(entry["near_target_invisible_precision"], 0.8)


if __name__ == "__main__":
    unittest.main()
