"""CPU-only tests for the Stage-2g.4 event/long-history/memory helpers."""

import unittest

import numpy as np

from v83_spatial_stage2.stage2g4_data import (
    COMPACT_IDX,
    CONF_FEATURES,
    FEATURE_INDEX,
    MEMORY_NAMES,
    build_history,
    build_memory,
    confidence_stats,
    event_labels,
)
from v83_spatial_stage2.stage2g4_metrics import (
    event_metrics,
    fast_auc,
    fast_average_precision,
    per_episode_table,
    select_event_thresholds,
    stratified_event_vs_sustained,
)
from v83_spatial_stage2.stage2g_probe import (
    auc_rank,
    average_precision,
)


def make_data(n=40, episodes=2):
    episode_id = np.repeat(
        np.arange(episodes, dtype=np.int64), n // episodes
    )
    steps = np.tile(np.arange(n // episodes, dtype=np.int64), episodes)
    actions = np.zeros(n, dtype=np.int64)
    current = np.zeros((n, 2858), dtype=np.float32)
    # Give observable confidence columns a deterministic nonzero pattern.
    for index, name in enumerate(CONF_FEATURES):
        current[:, FEATURE_INDEX[name]] = np.linspace(
            0.0, 1.0 + index, n
        )
    compact = current[:, COMPACT_IDX].copy()
    localizable = (steps % 10 == 2).astype(np.int64)
    visible = localizable.copy()
    near = np.ones(n, dtype=np.int64)
    since = np.full(n, -1, dtype=np.int64)
    for episode in range(episodes):
        mask = episode_id == episode
        last = -1
        for index in np.flatnonzero(mask):
            if localizable[index]:
                since[index] = 0
                last = index
            elif last >= 0:
                since[index] = index - last
    return {
        "episode_id": episode_id,
        "band": np.zeros(n, dtype=np.int64),
        "step": steps,
        "action": actions,
        "stop": np.zeros(n, dtype=np.int64),
        "distance_m": np.full(n, 50.0),
        "target_visible": visible,
        "target_localizable": localizable,
        "near_target_invisible": near,
        "current_features": current,
        "compact_features": compact,
        "since_localizable": since,
        "until_next_localizable": np.full(n, -1, dtype=np.int64),
        "common_mask": np.ones(n, dtype=bool),
        "close_mask": np.ones(n, dtype=bool),
        "events": {
            k: event_labels(near, since, k) for k in (4, 8, 16)
        },
    }


class TestEventLabel(unittest.TestCase):
    def test_causal_recent_window(self):
        # localizable at t=2; near-invisible from t=3 onward.
        since = np.array([-1, -1, 0, 1, 2, 3, 4, 5])
        near = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        k4 = event_labels(near, since, 4)
        k2 = event_labels(near, since, 2)
        self.assertEqual(
            k4.tolist(), [False, False, False, True, True, True, True, False]
        )
        self.assertEqual(
            k2.tolist(), [False, False, False, True, True, False, False, False]
        )

    def test_future_localizable_does_not_leak(self):
        since = np.array([-1, 0, 1, 2, 3])
        near = np.array([0, 0, 1, 1, 1])
        labels = event_labels(near, since, 2)
        # t=2 is 1 step after t=1, t=3 is 2 steps after t=1.
        self.assertEqual(labels.tolist(), [False, False, True, True, False])


class TestHistory(unittest.TestCase):
    def test_order_and_horizon(self):
        data = make_data(n=40, episodes=2)
        history, actions, dt = build_history(data, 16)
        self.assertEqual(history.shape, (40, 16, 74))
        self.assertEqual(actions.shape, (40, 16))
        # First episode, step 18: frame 0 is step 17, frame 15 is step 2.
        index = 18
        self.assertTrue(np.array_equal(
            history[index, 0], data["compact_features"][17]
        ))
        self.assertTrue(np.array_equal(
            history[index, 15], data["compact_features"][2]
        ))
        self.assertEqual(dt[index].tolist(), [1.0] * 16)


class TestMemory(unittest.TestCase):
    def test_memory_does_not_read_labels(self):
        first = make_data()
        second = make_data()
        # Change every label column while keeping observables identical.
        second["target_visible"] = 1 - second["target_visible"]
        second["target_localizable"] = 1 - second["target_localizable"]
        second["near_target_invisible"] = 1 - second["near_target_invisible"]
        stats = confidence_stats(first)
        memory_a = build_memory(first, stats)
        memory_b = build_memory(second, stats)
        self.assertEqual(memory_a.shape, (len(first["step"]), len(MEMORY_NAMES)))
        self.assertTrue(np.array_equal(memory_a, memory_b))

    def test_memory_is_causal(self):
        data = make_data()
        stats = confidence_stats(data)
        memory = build_memory(data, stats)
        self.assertGreaterEqual(memory[:, 2].min(), 0.0)
        self.assertLessEqual(memory[:, 2].max(), 64.0)


class TestThresholds(unittest.TestCase):
    def test_event_metrics_and_constraints(self):
        data = make_data(n=80, episodes=4)
        # Make event positives score high and negatives low.
        scores = np.zeros(len(data["step"]))
        event = data["events"][16]
        scores[event] = 0.8
        scores[~event] = 0.1
        metrics = event_metrics(data, 16, scores, 0.5)
        self.assertEqual(metrics["tp"], int(event.sum()))
        self.assertEqual(metrics["fp"], 0)
        selected = select_event_thresholds(data, 16, scores)
        self.assertIsNotNone(selected["f1_joint"])
        self.assertIsNotNone(selected["precision_0.5"])
        self.assertGreaterEqual(selected["f1_joint"]["precision"], 0.5)
        stratified = stratified_event_vs_sustained(data, 16, scores)
        self.assertIn("0-20", stratified)
        self.assertIn("macro_auroc", stratified)
        self.assertAlmostEqual(
            fast_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
            auc_rank([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        )
        self.assertAlmostEqual(
            fast_average_precision(
                [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
            ),
            average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]),
        )
        table = per_episode_table(data, 16, scores, 0.5)
        self.assertEqual(len(table), len(np.unique(data["episode_id"])))
        self.assertGreaterEqual(table[0]["fp"], table[-1]["fp"])


if __name__ == "__main__":
    unittest.main()
