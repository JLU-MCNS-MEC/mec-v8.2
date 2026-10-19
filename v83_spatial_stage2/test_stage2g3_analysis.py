"""CPU-only tests for Stage-2g.3 post-hoc analysis helpers."""

import unittest

import numpy as np

from v83_spatial_stage2.stage2g3_analyze import (
    concentration,
    context_table,
    contiguous_runs,
    count_bins,
    precision_recall,
    quantile_summary,
    sign_test,
)


class TestRunHelpers(unittest.TestCase):
    def test_contiguous_runs(self):
        runs = contiguous_runs([0, 1, 1, 0, 1, 0, 0, 1])
        self.assertEqual(runs, [(1, 2), (4, 1), (7, 1)])

    def test_count_bins_closed_last_edge(self):
        bins = count_bins([0, 20, 50, 99.9, 100], [0, 20, 50, 100],
                          ["0-20", "20-50", "50-100"])
        self.assertEqual(bins, {"0-20": 1, "20-50": 1, "50-100": 3})

    def test_quantile_summary_empty(self):
        self.assertEqual(quantile_summary([]), {"n": 0})
        summary = quantile_summary([1, 2, 3])
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["quantiles"]["0.5"], 2.0)


class TestConcentration(unittest.TestCase):
    def test_single_episode_dominates(self):
        result = concentration([10, 0, 0, 0])
        self.assertAlmostEqual(result["top1_share"], 1.0)
        self.assertEqual(result["episodes_for_50pct"], 1)
        self.assertEqual(result["episodes_for_80pct"], 1)

    def test_even_distribution(self):
        result = concentration([5, 5, 5, 5])
        self.assertAlmostEqual(result["top1_share"], 0.25)
        self.assertAlmostEqual(result["top5_share"], 1.0)


class TestMetrics(unittest.TestCase):
    def test_precision_recall(self):
        result = precision_recall(
            [0, 1, 1, 0], [0.9, 0.8, 0.2, 0.1], 0.5
        )
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_sign_test(self):
        result = sign_test([1.0, 1.0, -1.0, 0.0])
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)


class TestContextTable(unittest.TestCase):
    def test_since_localizable_groups(self):
        data = {
            "episode_id": np.array([0, 0, 0], dtype=np.int64),
            "step": np.array([0, 1, 2], dtype=np.int64),
            "band": np.array([0, 0, 0], dtype=np.int64),
            "near_target_invisible": np.array([0, 1, 1], dtype=np.int64),
            "action": np.array([0, 0, 0], dtype=np.int64),
        }
        context = {
            (0, 0): {
                "since_localizable": 0, "run_length": 0, "run_start": False,
                "history_action_t1": -1, "history_action_t2": -1,
            },
            (0, 1): {
                "since_localizable": 2, "run_length": 1, "run_start": True,
                "history_action_t1": 0, "history_action_t2": -1,
            },
            (0, 2): {
                "since_localizable": 10, "run_length": 2, "run_start": False,
                "history_action_t1": 0, "history_action_t2": 0,
            },
        }
        scores = {
            "P0": np.array([0.1, 0.2, 0.3]),
            "P1-2": np.array([0.1, 0.3, 0.4]),
        }
        table = context_table(
            context, data, scores, {"P0": 0.5, "P1-2": 0.5}
        )
        self.assertEqual(
            table["since_localizable"]["2"]["P0"]["n"], 1
        )
        self.assertEqual(
            table["since_localizable"]["10-19"]["P0"]["n"], 1
        )
        self.assertEqual(table["run_length"]["1"]["P0"]["n"], 1)
        self.assertEqual(
            table["run_start_vs_interior"]["run_start"]["P0"]["n"], 1
        )


if __name__ == "__main__":
    unittest.main()
