"""CPU tests for the V8.3 Depth probe pipeline and metric implementations."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from depth_probe_experiment import (
    ProbeMLP,
    auroc,
    average_precision,
    brier,
    choose_threshold,
    ece,
    threshold_metrics,
    _prepare_inputs,
)


class ProbePipelineTest(unittest.TestCase):
    def test_probe_dimensions(self):
        for dim in (106, 112):
            model = ProbeMLP(dim)
            out = model(torch.zeros(3, dim))
            self.assertEqual(tuple(out.shape), (3,))

    def test_auroc_average_precision_perfect(self):
        y = np.array([0, 0, 1, 1], dtype=float)
        score = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        self.assertAlmostEqual(auroc(y, score), 1.0)
        self.assertAlmostEqual(average_precision(y, score), 1.0)

    def test_average_precision_known_value(self):
        y = np.array([1, 0, 1], dtype=float)
        score = np.array([0.9, 0.8, 0.7], dtype=float)
        self.assertAlmostEqual(average_precision(y, score), 5.0 / 6.0, places=6)

    def test_brier_ece_threshold(self):
        y = np.array([0, 1], dtype=float)
        prob = np.array([0.0, 1.0], dtype=float)
        self.assertAlmostEqual(brier(y, prob), 0.0)
        self.assertAlmostEqual(ece(y, prob), 0.0, places=6)
        p, r = threshold_metrics(y, prob, 0.5)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        t = choose_threshold(y, np.array([0.1, 0.9]))
        self.assertGreater(t, 0.1)
        self.assertLessEqual(t, 0.9)

    def test_metrics_are_finite_on_random_data(self):
        rng = np.random.default_rng(7)
        y = (rng.random(200) < 0.2).astype(float)
        score = rng.random(200)
        self.assertTrue(np.isfinite(auroc(y, score)))
        self.assertTrue(np.isfinite(average_precision(y, score)))
        self.assertTrue(np.isfinite(brier(y, score)))
        self.assertTrue(np.isfinite(ece(y, score)))

    def test_p2_shuffles_depth_only_and_preserves_marginal(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(40, 106)).astype(np.float32)
        d = rng.normal(size=(40, 6)).astype(np.float32)
        xv = rng.normal(size=(10, 106)).astype(np.float32)
        dv = rng.normal(size=(10, 6)).astype(np.float32)
        xh = rng.normal(size=(8, 106)).astype(np.float32)
        dh = rng.normal(size=(8, 6)).astype(np.float32)
        p1_tr, p1_vn, p1_vh, _ = _prepare_inputs(x, d, xv, dv, xh, dh, "P1", 20261018)
        p2_tr, p2_vn, p2_vh, _ = _prepare_inputs(x, d, xv, dv, xh, dh, "P2", 20261018)
        # RGB columns are identical.
        np.testing.assert_allclose(p1_tr[:, :106], p2_tr[:, :106])
        np.testing.assert_allclose(p1_vn[:, :106], p2_vn[:, :106])
        # Depth columns are a permutation (same marginal), and not row-aligned.
        np.testing.assert_allclose(np.sort(p1_tr[:, 106:], axis=0), np.sort(p2_tr[:, 106:], axis=0))
        self.assertFalse(np.allclose(p1_tr[:, 106:], p2_tr[:, 106:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
