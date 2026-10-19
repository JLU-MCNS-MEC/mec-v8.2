"""CPU-only tests for the Stage-2g.2 bounded recovery policy."""

import unittest

import numpy as np

from env import FORWARD, LEFT, RIGHT, STOP
from v83_spatial_stage2.stage2g2_policy import (
    ProbeRecoveryConfig,
    ProbeRecoveryPolicy,
    build_probe_input,
)


class TestProbeInput(unittest.TestCase):
    def test_shape_and_order(self):
        current = np.arange(2858, dtype=np.float32)
        history = [
            (np.ones(74, dtype=np.float32), LEFT),
            (np.full(74, 2.0, dtype=np.float32), RIGHT),
        ]
        values = build_probe_input(current, history)
        self.assertEqual(values.shape, (2858 + 2 * 79,))
        np.testing.assert_allclose(values[:2858], current)
        np.testing.assert_allclose(values[2858:2858 + 74], 1.0)
        np.testing.assert_allclose(
            values[2858 + 74:2858 + 78], [0, 1, 0, 0]
        )
        self.assertEqual(values[2858 + 78], 1.0)
        np.testing.assert_allclose(
            values[2858 + 79:2858 + 79 + 74], 2.0
        )
        np.testing.assert_allclose(
            values[2858 + 79 + 74:2858 + 79 + 78], [0, 0, 1, 0]
        )


class TestRecoveryPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = ProbeRecoveryPolicy(
            ProbeRecoveryConfig(
                threshold=0.5, recovery_steps=2, cooldown=3,
                side_window=10, side_margin=0.05,
            )
        )

    def test_trigger_uses_last_high_side_then_forward(self):
        # Step 0: high-confidence target on the left; no trigger.
        action, recovering, trigger, info = self.policy.act(
            0.0, FORWARD, 0, True, -0.4
        )
        self.assertEqual((action, recovering, trigger), (FORWARD, 0, 0))
        # Step 1: probe fires; correction turns LEFT.
        action, recovering, trigger, info = self.policy.act(
            0.9, FORWARD, 1, False, 0.0
        )
        self.assertEqual((action, recovering, trigger), (LEFT, 1, 1))
        self.assertEqual(self.policy.trigger_count, 1)
        # Step 2: remaining budget is one FORWARD; forward is allowed.
        action, recovering, trigger, info = self.policy.act(
            0.0, LEFT, 2, False, 0.0
        )
        self.assertEqual((action, recovering, trigger), (FORWARD, 1, 0))
        self.assertFalse(self.policy.recovering)
        # Events record the two actions and budget exit.
        self.assertEqual(len(self.policy.events), 1)
        self.assertEqual(self.policy.events[0]["actions"], [LEFT, FORWARD])
        self.assertEqual(self.policy.events[0]["exit_reason"], "budget")

    def test_cooldown_blocks_immediate_retrigger(self):
        self.policy.act(0.9, FORWARD, 0, False, 0.0)
        self.policy.act(0.0, FORWARD, 1, False, 0.0)
        # Cooldown is 3; repeated high scores must not trigger.
        for step in (2, 3, 4):
            _action, recovering, trigger, _info = self.policy.act(
                0.9, FORWARD, step, False, 0.0
            )
            self.assertEqual(trigger, 0)
        # After three cooldown decrements, the next step may trigger.
        _action, recovering, trigger, _info = self.policy.act(
            0.9, FORWARD, 5, False, 0.0
        )
        self.assertEqual(trigger, 1)
        self.assertEqual(self.policy.trigger_count, 2)

    def test_stop_is_not_overridden(self):
        action, recovering, trigger, _info = self.policy.act(
            0.99, STOP, 0, False, 0.0
        )
        self.assertEqual((action, recovering, trigger), (STOP, 0, 0))
        self.assertEqual(self.policy.trigger_count, 0)

    def test_high_confidence_exits_recovery(self):
        self.policy.act(0.9, FORWARD, 0, False, 0.0)
        action, recovering, trigger, _info = self.policy.act(
            0.0, LEFT, 1, True, -0.3
        )
        self.assertEqual((action, recovering, trigger), (LEFT, 0, 0))
        self.assertEqual(self.policy.events[0]["exit_reason"], "high_confidence")
        self.assertEqual(self.policy.events[0]["reacquired_step"], 1)

    def test_single_step_correction_without_side(self):
        action, recovering, trigger, _info = self.policy.act(
            0.9, FORWARD, 0, False, 0.0
        )
        self.assertEqual((action, recovering, trigger), (LEFT, 1, 1))
        # No last high side: centered x = 0 -> bounded LEFT correction.
        self.assertEqual(self.policy.events, [])


if __name__ == "__main__":
    unittest.main()
