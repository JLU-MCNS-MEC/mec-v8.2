"""CPU-only tests for Stage-2f features and the recovery policy."""

from __future__ import annotations

import unittest

import torch

from env import FORWARD, LEFT, RIGHT
from v83_spatial_stage2.stage2f_features import (
    extract_observable_features,
    feature_names,
)
from v83_spatial_stage2.stage2f_policy import (
    FORBIDDEN_KEYS,
    RecoveryConfig,
    RecoveryPolicy,
    _get,
)


def _features(**overrides):
    base = {
        "heatmap_max_top1": 0.05,
        "heatmap_max_entropy": 0.95,
        "baseline_centroid_x": 0.0,
        "coherent_centroid_x": 0.0,
        "temporal_valid": 0.0,
        "temporal_consistency": float("nan"),
    }
    base.update(overrides)
    return base


class TestFeatureLeakage(unittest.TestCase):
    def test_observable_keys_are_disjoint_from_forbidden(self):
        torch.manual_seed(0)
        current = torch.randn(1, 576, 384)
        goal = torch.randn(1, 3, 384)
        heatmap = torch.randn(1, 3, 8, 8)
        baseline = torch.randn(1, 24, 24)
        coherent = torch.randn(1, 24, 24)
        move_logits = torch.randn(1, 3)
        arrival_logit = torch.randn(1, 1)
        features = extract_observable_features(
            current, goal, heatmap, baseline, coherent,
            None, None, move_logits, arrival_logit,
        )
        self.assertEqual(set(features), set(feature_names()))
        self.assertFalse(set(features) & FORBIDDEN_KEYS)

    def test_get_rejects_true_labels(self):
        for key in ("goal_visible", "distance_m", "oracle_move_action"):
            with self.assertRaises(ValueError):
                _get({"goal_visible": 1}, key)


class TestRecoveryPolicy(unittest.TestCase):
    def test_low_confidence_run_triggers_and_scans(self):
        policy = RecoveryPolicy(RecoveryConfig(trigger_k=2))
        action, recovering = policy.act(_features(), FORWARD, 0)
        self.assertEqual(action, FORWARD)
        self.assertFalse(recovering)
        action, recovering = policy.act(_features(), FORWARD, 1)
        self.assertTrue(recovering)
        self.assertEqual(action, LEFT)  # deterministic scan starts left
        self.assertEqual(policy.trigger_count, 1)

    def test_last_high_side_is_used(self):
        policy = RecoveryPolicy(RecoveryConfig(trigger_k=1, use_side=True))
        high_left = _features(
            heatmap_max_top1=0.5, heatmap_max_entropy=0.5,
            baseline_centroid_x=-0.3,
        )
        policy.act(high_left, FORWARD, 0)  # stores last high side = left
        action, recovering = policy.act(_features(), FORWARD, 1)
        self.assertTrue(recovering)
        self.assertEqual(action, LEFT)

    def test_reacquire_exits_recovery(self):
        policy = RecoveryPolicy(RecoveryConfig(trigger_k=1, reacquire_k=1))
        policy.act(_features(), FORWARD, 0)
        self.assertTrue(policy.recovering)
        high = _features(
            heatmap_max_top1=0.5, heatmap_max_entropy=0.5,
            baseline_centroid_x=0.2,
        )
        action, recovering = policy.act(high, FORWARD, 1)
        self.assertFalse(recovering)
        self.assertEqual(action, FORWARD)

    def test_scan_only_ignores_side_history(self):
        policy = RecoveryPolicy(
            RecoveryConfig(trigger_k=1, use_side=False)
        )
        high_left = _features(
            heatmap_max_top1=0.5, heatmap_max_entropy=0.5,
            baseline_centroid_x=-0.3,
        )
        policy.act(high_left, FORWARD, 0)
        action, recovering = policy.act(_features(), FORWARD, 1)
        self.assertTrue(recovering)
        self.assertEqual(action, LEFT)


if __name__ == "__main__":
    unittest.main()
