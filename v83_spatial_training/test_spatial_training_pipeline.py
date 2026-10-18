"""CPU tests for V8.3 spatial-aware M0/M1/M2 wrappers and cache lookup."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gbs_model_v3 import build_gbs_model
from spatial_data import SpatialCache
from spatial_mec import SpatialAwareMEC, SpatialFusionHead, parameter_count
from train_spatial_mec import set_seed


class DummyBase(nn.Module):
    def forward_components(self, current_tokens, goal_tokens, heatmap, network_state,
                           return_attention=False):
        b = current_tokens.shape[0]
        return {
            "move_logits": torch.zeros(b, 3),
            "arrival_logit": torch.zeros(b),
            "arrival_features": torch.zeros(b, 104),
            "attention": {},
        }


class SpatialWrapperTest(unittest.TestCase):
    def _maps(self, b=2):
        maps = torch.randn(b, 4, 24, 24)
        valid = torch.zeros(b, 1, 1, 1)
        return maps, valid

    def test_fusion_head_zero_init_and_shape(self):
        head = SpatialFusionHead(2)
        out = head(torch.randn(3, 2, 24, 24))
        self.assertEqual(tuple(out.shape), (3, 3))
        self.assertTrue(torch.count_nonzero(out).item() == 0)

    def test_wrapper_modes_forward_and_predict(self):
        tokens = torch.randn(2, 576, 384)
        goal = torch.randn(2, 3, 384)
        heat = torch.randn(2, 3, 8, 8)
        net = torch.randn(2, 5)
        maps, valid = self._maps(2)
        for mode in ("M0", "M1", "M2"):
            model = SpatialAwareMEC(DummyBase(), mode)
            comp = model.forward_components(
                tokens, goal, heat, net, spatial_maps=maps, spatial_valid=valid
            )
            self.assertEqual(tuple(comp["move_logits"].shape), (2, 3))
            logp = model(tokens, goal, heat, net, spatial_maps=maps, spatial_valid=valid)
            self.assertEqual(tuple(logp.shape), (2, 4))
            self.assertTrue(torch.isfinite(logp).all())
            decision = model.predict_action(
                tokens, goal, heat, net, spatial_maps=maps, spatial_valid=valid
            )
            self.assertEqual(tuple(decision["action"].shape), (2,))

    def test_m2_validity_masks_temporal_channels(self):
        model = SpatialAwareMEC(DummyBase(), "M2")
        maps = torch.ones(2, 4, 24, 24)
        valid = torch.zeros(2, 1, 1, 1)
        x = model._spatial_input(maps, valid)
        self.assertEqual(tuple(x.shape), (2, 5, 24, 24))
        self.assertEqual(torch.count_nonzero(x[:, 2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(x[:, :2]).item(), 2 * 2 * 24 * 24)
        valid1 = torch.ones(2, 1, 1, 1)
        x1 = model._spatial_input(maps, valid1)
        self.assertEqual(torch.count_nonzero(x1[:, 4]).item(), 2 * 24 * 24)

    def test_m0_m1_base_init_identical(self):
        set_seed(12345)
        m0 = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0")
        set_seed(12345)
        m1 = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M1")
        s0 = m0.base.state_dict(); s1 = m1.base.state_dict()
        max_diff = 0.0
        for k in s0:
            max_diff = max(max_diff, float((s0[k] - s1[k]).abs().max().item()))
        self.assertEqual(max_diff, 0.0)
        self.assertGreater(parameter_count(m1), parameter_count(m0))


class SpatialCacheTest(unittest.TestCase):
    def test_lookup_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.npz"
            n = 3
            np.savez(
                p,
                episode_id=np.arange(n, dtype=np.int64),
                step_id=np.arange(n, dtype=np.int64),
                baseline=np.zeros((n, 24, 24), np.float16),
                coherent=np.ones((n, 24, 24), np.float16),
                temporal=np.ones((n, 24, 24), np.float16),
                innovation=np.zeros((n, 24, 24), np.float16),
                valid=np.asarray([0, 1, 0], np.uint8),
            )
            cache = SpatialCache(p)
            maps, valid = cache.lookup(torch.tensor([0, 2]), torch.tensor([0, 2]))
            self.assertEqual(tuple(maps.shape), (2, 4, 24, 24))
            self.assertEqual(tuple(valid.shape), (2, 1, 1, 1))
            self.assertEqual(float(valid[0, 0, 0, 0]), 0.0)
            self.assertEqual(float(valid[1, 0, 0, 0]), 0.0)
            with self.assertRaises(KeyError):
                cache.lookup(torch.tensor([9]), torch.tensor([9]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
