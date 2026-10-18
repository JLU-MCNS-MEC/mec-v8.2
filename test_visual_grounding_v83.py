"""CPU-only smoke tests for V8.3 visual grounding. No checkpoints/GPU."""
import unittest
import torch

from config import ENCODER, AUX
from visual_grounding_v83 import (
    static_grounding, visual_geometry, yaw_warp_previous, VisualBeliefTracker,
)


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.c = torch.randn(1, ENCODER.num_tokens, ENCODER.token_dim)
        self.g = torch.randn(1, AUX.goal_token_count, ENCODER.token_dim)

    def test_shapes_finite(self):
        b, c = static_grounding(self.c, self.g)
        self.assertEqual(b.shape, (1, 24, 24))
        self.assertEqual(c.shape, (1, 24, 24))
        result = visual_geometry(self.c, self.g)
        self.assertEqual(result.as_channels().shape, (1, 5, 24, 24))
        self.assertFalse(bool(result.temporal_valid[0]))
        self.assertTrue(torch.isfinite(result.as_channels()).all())

    def test_zero_yaw_preserves_map_on_turn(self):
        m = torch.randn(1, 24, 24)
        a, valid = yaw_warp_previous(m, torch.tensor([1]), turn_step_deg=0)
        self.assertTrue(bool(valid[0]))
        self.assertTrue(torch.allclose(a, m, atol=1e-5))

    def test_turn_moves_left_appearing_target_to_right(self):
        # A target at screen center before a LEFT turn appears on the RIGHT.
        m = torch.zeros(1, 24, 24)
        m[0, 12, 12] = 1
        warped, valid = yaw_warp_previous(m, torch.tensor([1]))
        self.assertTrue(bool(valid[0]))
        index = int(warped[0, 12].argmax())
        self.assertGreater(index, 12)

    def test_forward_motion_no_fake_warp(self):
        m = torch.randn(1, 24, 24)
        warped, valid = yaw_warp_previous(m, torch.tensor([0]))
        self.assertFalse(bool(valid[0]))
        self.assertEqual(torch.count_nonzero(warped).item(), 0)
        g = visual_geometry(
            self.c, self.g, previous_coherent=m, previous_action=torch.tensor([0])
        )
        self.assertTrue(torch.allclose(g.temporal, g.coherent))
        self.assertEqual(torch.count_nonzero(g.innovation).item(), 0)

    def test_tracker_episode_reset(self):
        tracker = VisualBeliefTracker()
        first = tracker.track(self.c, self.g)
        turn = tracker.track(self.c, self.g, previous_action=[1])
        self.assertTrue(bool(turn.temporal_valid[0]))
        tracker.reset()
        reset = tracker.track(self.c, self.g)
        self.assertFalse(bool(reset.temporal_valid[0]))
        self.assertTrue(torch.allclose(first.coherent, reset.coherent))


if __name__ == "__main__":
    unittest.main(verbosity=2)
