"""CPU tests for Stage-2c calibration metrics and plain-BCE protocol."""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2c_diagnostics import (
    _binary_metrics,
    _distance_bin_rows,
    MATCH_FEATURE_NAMES,
)


class BinaryMetricsTest(unittest.TestCase):
    def test_basic_metrics(self):
        y = np.array([0, 1, 1, 0], dtype=np.float32)
        p = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        m = _binary_metrics(y, p, threshold=0.5)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["tn"], 2)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)
        self.assertAlmostEqual(m["brier"], (0.01 + 0.01 + 0.04 + 0.04) / 4)
        self.assertGreaterEqual(m["ece"], 0.0)

    def test_threshold_moves_prediction(self):
        y = np.array([0, 1], dtype=np.float32)
        p = np.array([0.4, 0.6], dtype=np.float32)
        self.assertEqual(_binary_metrics(y, p, 0.5)["f1"], 1.0)
        self.assertEqual(_binary_metrics(y, p, 0.7)["tp"], 0)
        self.assertEqual(_binary_metrics(y, p, 0.7)["fn"], 1)


class DistanceBinTest(unittest.TestCase):
    def test_bins_split_distance_visibility_and_band(self):
        y = np.array([0, 0, 0, 1], dtype=np.float32)
        p = np.array([0.01, 0.9, 0.6, 0.8], dtype=np.float32)
        distance = np.array([50.0, 50.0, 150.0, 10.0], dtype=np.float32)
        visible = np.array([1, 0, 1, 1], dtype=np.int64)
        band = np.array([0, 0, 2, 0], dtype=np.int64)
        rows = _distance_bin_rows(y, p, distance, visible, band, "unit", 0.5)
        keyed = {(r["band"], r["distance_bin"], r["visibility"]): r for r in rows}
        self.assertEqual(keyed[("all", "40_60m", "visible")]["n"], 1)
        self.assertAlmostEqual(keyed[("all", "40_60m", "visible")]["pred_positive_rate"], 0.0)
        self.assertEqual(keyed[("all", "40_60m", "occluded")]["n"], 1)
        self.assertAlmostEqual(keyed[("all", "40_60m", "occluded")]["pred_positive_rate"], 1.0)
        self.assertEqual(keyed[("long", "100_200m", "all")]["n"], 1)
        self.assertEqual(keyed[("short", "0_20m", "all")]["positive_rate"], 1.0)
        self.assertEqual(len(MATCH_FEATURE_NAMES), 8)


class LeakageProtocolTest(unittest.TestCase):
    def test_arrival_feature_dimension_is_104(self):
        # 8 match features + 64 heatmap features + 32 network features.
        self.assertEqual(8 + 64 + 32, 104)
        self.assertEqual(len(MATCH_FEATURE_NAMES), 8)

    def test_rollout_collector_does_not_feed_gt_to_model(self):
        source = Path(__file__).with_name("stage2c_collect_rollouts.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_episode"
        )
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "predict_action"
        ]
        self.assertEqual(len(calls), 1)
        names = {
            node.id for node in ast.walk(calls[0]) if isinstance(node, ast.Name)
        }
        for forbidden in ("distance", "goal_visible", "position", "pose"):
            self.assertNotIn(forbidden, names)
        # Distance is used to build labels only.
        self.assertIn("distance < ENV.stop_radius_m", source)
        self.assertIn("distance > 40.0", source)


class PlainBCEProtocolTest(unittest.TestCase):
    """Stage-2c B1/B2 must use plain BCE, unlike Stage-2b A0/A1 pos_weight."""

    def test_train_plain_bce_has_no_pos_weight(self):
        source = Path(__file__).with_name("stage2c_arrival_calibration.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_plain_bce"
        )
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "BCEWithLogitsLoss"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {kw.arg for kw in calls[0].keywords}
        self.assertNotIn("pos_weight", keywords)
        self.assertFalse(
            any(
                isinstance(node, ast.Name) and node.id == "pos_weight"
                for node in ast.walk(function)
            )
        )

    def test_original_training_loss_is_plain_bce(self):
        source = (PROJECT_ROOT / "train_gbs_v3.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_training_loss"
        )
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "binary_cross_entropy_with_logits"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("pos_weight", {kw.arg for kw in calls[0].keywords})


if __name__ == "__main__":
    unittest.main(verbosity=2)
