"""CPU tests for V8.3 visual-spatial diagnostic utilities."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AUX, ENCODER
from env import ToyUAVEnv
from visual_grounding_diagnostic import (
    center_patch,
    consistency_metrics,
    distance_bin,
    goal_mask_24,
    localization_metrics,
    old_heatmap_24,
    _r2,
)


class VisualSpatialDiagnosticTest(unittest.TestCase):
    def test_localization_metrics_hit_and_iou(self):
        m = np.zeros((24, 24), dtype=np.float32)
        goal_patches = [(5, 7), (5, 8), (6, 7), (6, 8)]
        # Make the goal-center patch uniquely top-1 while keeping the other three
        # goal patches in the top-4 candidate set.
        m[5, 7] = 4.0
        for i, j in goal_patches[1:]:
            m[i, j] = 1.0
        mask = np.zeros((24, 24), dtype=bool)
        for i, j in goal_patches:
            mask[i, j] = True
        center = 5 * 24 + 7
        r1, r4, iou = localization_metrics(m, center, mask, k=4)
        self.assertEqual(r1, 1.0)
        self.assertEqual(r4, 1.0)
        self.assertAlmostEqual(iou, 1.0)

    def test_localization_metrics_miss(self):
        m = np.zeros((24, 24), dtype=np.float32)
        m[0, 0] = 1.0
        r1, r4, _ = localization_metrics(m, 5 * 24 + 7, None, k=4)
        self.assertEqual(r1, 0.0)
        self.assertEqual(r4, 0.0)

    def test_consistency_metrics(self):
        a = np.random.default_rng(3).normal(size=(24, 24))
        same = consistency_metrics(a, a)
        self.assertAlmostEqual(same["pearson"], 1.0, places=6)
        self.assertAlmostEqual(same["cosine"], 1.0, places=6)
        self.assertAlmostEqual(same["mse"], 0.0, places=8)
        self.assertAlmostEqual(same["top10_jaccard"], 1.0, places=6)
        shifted = np.roll(a, 1, axis=1)
        self.assertLess(consistency_metrics(a, shifted)["pearson"], 1.0)

    def test_old_heatmap_upsample_shape(self):
        c = torch.randn(1, ENCODER.num_tokens, ENCODER.token_dim)
        g = torch.randn(1, AUX.goal_token_count, ENCODER.token_dim)
        out = old_heatmap_24(c, g)
        self.assertEqual(tuple(out.shape), (1, 24, 24))
        self.assertTrue(torch.isfinite(out).all())

    def test_goal_mask_and_center(self):
        env = ToyUAVEnv()
        env.reset(seed=900001, render_observation=False)
        proj = env.goal_projection()
        self.assertIsNotNone(proj)
        mask = goal_mask_24(env, proj)
        self.assertEqual(mask.shape, (24, 24))
        center = center_patch(proj)
        self.assertIsNotNone(center)
        self.assertGreaterEqual(center, 0)
        self.assertLess(center, 24 * 24)

    def test_distance_bins(self):
        self.assertEqual(distance_bin(0.0), "0-20")
        self.assertEqual(distance_bin(19.9), "0-20")
        self.assertEqual(distance_bin(20.0), "20-40")
        self.assertEqual(distance_bin(60.0), "60-100")
        self.assertEqual(distance_bin(250.0), "200+")

    def test_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(_r2(y, y), 1.0)
        self.assertAlmostEqual(_r2(y, np.full_like(y, y.mean())), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
