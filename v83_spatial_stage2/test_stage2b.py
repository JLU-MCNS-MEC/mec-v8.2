"""CPU tests for Stage-2b A0/A1 masks and arrival metrics."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2b_arrival_ablation import (
    _brier,
    _ece,
    _f1,
    build_a1_mask,
)
from v83_spatial_stage2.stage2b_never_stop_diagnostics import (
    summarize_never,
    summarize_stop_events,
)


class ArrivalMaskTest(unittest.TestCase):
    def _train(self):
        n = 120
        sample_type = np.zeros(n, dtype=np.int64)
        label = np.zeros(n, dtype=np.float32)
        distance = np.full(n, 100.0, dtype=np.float32)
        # 2 positives, 4 near negatives, 100 far negatives, 14 hard samples.
        label[:2] = 1.0; distance[:2] = 10.0
        distance[2:6] = 30.0
        distance[6:106] = 100.0
        sample_type[106:] = 1
        distance[106:120] = 100.0
        mask = np.zeros(n, dtype=bool)
        mask[:6] = True
        mask[106:] = True
        mask[6:9] = True  # 3% of far negatives (3/100)
        return {
            "sample_type": sample_type, "arrival_label": label,
            "distance": distance, "arrival_mask": mask,
        }

    def test_a1_is_nested_and_protected_entries_unchanged(self):
        train = self._train()
        a0, a1, stats = build_a1_mask(train)
        self.assertTrue(np.all(a1[a0]))
        self.assertTrue(np.all(a1[:6]))
        self.assertTrue(np.all(a1[106:]))
        self.assertEqual(int(a1[6:106].sum()), 10)
        self.assertEqual(stats["A0"]["far_included"], 3)
        self.assertEqual(stats["A1"]["far_included"], 10)
        self.assertAlmostEqual(stats["A1"]["far_included"] / 100.0, 0.10)

    def test_pos_weight(self):
        train = self._train()
        _, _, stats = build_a1_mask(train)
        for name in ("A0", "A1"):
            self.assertAlmostEqual(
                stats[name]["pos_weight_neg_over_pos"],
                stats[name]["negatives"] / stats[name]["positives"],
            )


class ArrivalMetricsTest(unittest.TestCase):
    def test_basic_metrics(self):
        y = np.array([0, 1, 1, 0], dtype=float)
        p = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
        self.assertAlmostEqual(_brier(y, p), (0.01 + 0.01 + 0.04 + 0.04) / 4)
        self.assertEqual(_f1(y, p >= 0.5), 1.0)
        self.assertAlmostEqual(_ece(np.array([0.0, 1.0]), np.array([0.0, 1.0])), 0.0)

    def test_never_and_stop_summaries(self):
        rows = [
            {"mode": "M0", "band": "long", "osr": 0, "timeout": 1, "min_distance_m": 100.0, "steps": 400},
            {"mode": "M0", "band": "long", "osr": 1, "timeout": 1, "min_distance_m": 10.0, "steps": 400},
            {"mode": "M1", "band": "short", "osr": 1, "timeout": 1, "min_distance_m": 5.0, "steps": 400},
        ]
        s = summarize_never(rows)
        self.assertEqual(len(s), 2)
        events = [
            {"mode": "M1", "band": "long", "distance_m": 100.0,
             "arrival_probability": 0.9, "goal_visible": 1, "goal_center_inside": 1},
            {"mode": "M1", "band": "long", "distance_m": 10.0,
             "arrival_probability": 0.9, "goal_visible": 1, "goal_center_inside": 1},
        ]
        e = summarize_stop_events(events)
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0]["far_gt40"], 1)
        self.assertAlmostEqual(e[0]["goal_visible_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
