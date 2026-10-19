"""CPU-only tests for the Stage-2e diagnostic code.

No dataset, checkpoint, GPU or closed-loop run is required. The tests pin the
two properties that the report depends on:
  * the current SpatialFusionHead with AdaptiveAvgPool2d(1) is mirror-invariant;
  * a 2x2 pooled version of the same conv trunk is direction-sensitive.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from v83_spatial_stage2.stage2e_analyze import (
    fusion_bin_stats,
    movement_stats,
    outcome_partition,
)
from v83_spatial_stage2.stage2e_m1_no_fusion_eval import ZeroFusion
from v83_spatial_stage2.stage2e_movement_replay import mirrored_tokens
from v83_spatial_training.spatial_mec import SpatialFusionHead


def _blob_pair(grid=24):
    yy, xx = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32),
        torch.arange(grid, dtype=torch.float32),
        indexing="ij",
    )
    maps = torch.zeros(2, 2, grid, grid)
    for index, center_x in enumerate((6.0, 17.0)):
        blob = torch.exp(-0.5 * (((xx - center_x) / 1.8) ** 2
                                 + ((yy - 12.0) / 3.0) ** 2))
        maps[index, 0] = blob
        maps[index, 1] = blob
    return maps


class TestMirrorProbes(unittest.TestCase):
    def test_mirrored_tokens_flip_x_axis(self):
        grid = 4
        tokens = torch.zeros(1, grid * grid, 1)
        for y in range(grid):
            for x in range(grid):
                tokens[0, y * grid + x, 0] = float(x)
        mirrored = mirrored_tokens(tokens)
        for y in range(grid):
            for x in range(grid):
                self.assertAlmostEqual(
                    float(mirrored[0, y * grid + x, 0]), float(grid - 1 - x)
                )

    def test_global_pool_head_is_mirror_invariant(self):
        torch.manual_seed(7)
        head = SpatialFusionHead(2)
        head.eval()
        maps = _blob_pair()
        with torch.no_grad():
            left = head(maps[:1])
            right = head(maps[1:])
        # The convolutional trunk is mirror-equivariant, and 1x1 average
        # pooling discards the remaining spatial arrangement.
        self.assertLess(float((left - right).abs().max()), 1e-4)

    def test_pool2x2_preserves_left_right_difference(self):
        torch.manual_seed(7)
        conv = SpatialFusionHead(2).net[:4]
        maps = _blob_pair()
        with torch.no_grad():
            features = conv(maps)
            pooled_1x1 = nn.functional.adaptive_avg_pool2d(features, 1)
            pooled_2x2 = nn.functional.adaptive_avg_pool2d(features, (2, 2))
        self.assertLess(
            float((pooled_1x1[0] - pooled_1x1[1]).abs().max()), 1e-3
        )
        self.assertGreater(
            float((pooled_2x2[0] - pooled_2x2[1]).abs().max()), 1e-2
        )

    def test_zero_fusion_returns_zeros(self):
        maps = torch.randn(3, 2, 24, 24)
        residual = ZeroFusion()(maps)
        self.assertEqual(tuple(residual.shape), (3, 3))
        self.assertEqual(float(residual.abs().sum()), 0.0)


class TestOutcomePartition(unittest.TestCase):
    @staticmethod
    def _row(**overrides):
        row = {
            "success": 0, "osr": 0, "premature_stop": 0,
            "max_steps_termination": 0,
        }
        row.update(overrides)
        return row

    def test_partition_labels(self):
        self.assertEqual(outcome_partition(self._row(success=1, osr=1)), "success")
        self.assertEqual(
            outcome_partition(self._row(premature_stop=1, osr=0)),
            "premature_far",
        )
        self.assertEqual(
            outcome_partition(self._row(premature_stop=1, osr=1)),
            "premature_reached",
        )
        self.assertEqual(
            outcome_partition(self._row(max_steps_termination=1, osr=0)),
            "timeout_far",
        )
        self.assertEqual(
            outcome_partition(self._row(max_steps_termination=1, osr=1)),
            "timeout_reached",
        )


class TestMovementLabels(unittest.TestCase):
    def test_error_types(self):
        base = {
            "oracle_move_action": 1,
        }
        rows = [
            {**base, "action": 1},  # correct left
            {**base, "action": 0},  # missed turn
            {**base, "action": 2},  # wrong direction
            {"oracle_move_action": 0, "action": 2},  # spurious turn
        ]
        stats = movement_stats(rows, "action", "test")
        self.assertAlmostEqual(stats["accuracy"], 0.25)
        self.assertAlmostEqual(stats["missed_turn_rate"], 0.25)
        self.assertAlmostEqual(stats["wrong_turn_rate"], 0.25)
        self.assertAlmostEqual(stats["spurious_turn_rate"], 0.25)


class TestFusionBenefitLabel(unittest.TestCase):
    @staticmethod
    def _row(base, fused, oracle):
        logits = {
            0: (1.0, 0.0, 0.0),
            1: (0.0, 1.0, 0.0),
            2: (0.0, 0.0, 1.0),
        }
        base_logit = logits[base]
        fused_logit = logits[fused]
        residual = tuple(f - b for f, b in zip(fused_logit, base_logit))
        return {
            "distance_m": 10.0,
            "m1_arrival_probability": 0.1,
            "m1_base_action": base,
            "m1_fused_action": fused,
            "oracle_move_action": oracle,
            "m1_res_logit_forward": residual[0],
            "m1_res_logit_left": residual[1],
            "m1_res_logit_right": residual[2],
            "m1_base_logit_forward": base_logit[0],
            "m1_base_logit_left": base_logit[1],
            "m1_base_logit_right": base_logit[2],
        }

    def test_beneficial_and_harmful(self):
        rows = [
            self._row(base=0, fused=1, oracle=1),  # beneficial
            self._row(base=1, fused=0, oracle=1),  # harmful
            self._row(base=0, fused=0, oracle=0),  # unchanged correct
            self._row(base=2, fused=1, oracle=0),  # changed, both wrong
        ]
        stats = fusion_bin_stats(rows, "M1", "all", "0-20")
        self.assertEqual(stats["beneficial_count"], 1)
        self.assertEqual(stats["harmful_count"], 1)
        self.assertEqual(stats["neutral_count"], 1)
        self.assertAlmostEqual(stats["action_change_rate"], 0.75)
        self.assertAlmostEqual(stats["base_accuracy"], 0.5)
        self.assertAlmostEqual(stats["fused_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
