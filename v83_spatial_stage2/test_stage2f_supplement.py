"""CPU-only tests for the Stage-2f supplementary analysis helpers."""

import unittest

import numpy as np
import pandas as pd

from v83_spatial_stage2.stage2f_supplement import add_history, auc_rank


class TestAucRank(unittest.TestCase):
    def test_perfect_and_reversed(self):
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(
            auc_rank(np.array([1.0, 2.0, 3.0, 4.0]), labels), 1.0
        )
        self.assertAlmostEqual(
            auc_rank(np.array([4.0, 3.0, 2.0, 1.0]), labels), 0.0
        )

    def test_ties_are_average_rank(self):
        labels = np.array([0, 1, 0, 1])
        self.assertAlmostEqual(
            auc_rank(np.array([1.0, 1.0, 1.0, 1.0]), labels), 0.5
        )


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "episode_index": [0, 0, 0, 0],
            "step": [0, 1, 2, 3],
            "heatmap_max_top1": [0.30, 0.05, 0.25, 0.04],
            "heatmap_max_entropy": [0.50, 0.95, 0.60, 0.96],
            "owner_arrival_probability": [0.80, 0.60, 0.60, 0.60],
            "owner_move_action": [0, 0, 2, 0],
            "oracle_move_action": [0, 1, 2, 2],
            "goal_visible": [1, 1, 0, 0],
            "executed_action": [0, 0, 1, 2],
        })

    def test_history_and_offline_labels(self):
        frame = add_history(self.frame)
        self.assertEqual(frame["steps_since_high"].tolist(), [0, 1, 0, 1])
        self.assertEqual(
            frame["last_high_top1"].tolist(), [0.30, 0.30, 0.25, 0.25]
        )
        self.assertEqual(frame["prev_visible"].tolist()[1:], [1.0, 1.0, 0.0])
        self.assertEqual(
            frame["prev_action"].tolist()[1:], [0.0, 0.0, 1.0]
        )
        self.assertEqual(frame["next_vis_2"].tolist(), [1, 0, 0, 0])
        self.assertEqual(frame["move_state"].tolist(), [0, 1, 1, 1])
        self.assertEqual(frame["move_error"].tolist(), [0, 1, 0, 1])

    def test_high_state_resets_recency(self):
        frame = add_history(self.frame)
        high = frame["heatmap_max_top1"] >= 0.20
        low_entropy = frame["heatmap_max_entropy"] <= 0.85
        high = high & low_entropy
        self.assertTrue(
            (frame.loc[high, "steps_since_high"] == 0).all()
        )


if __name__ == "__main__":
    unittest.main()
