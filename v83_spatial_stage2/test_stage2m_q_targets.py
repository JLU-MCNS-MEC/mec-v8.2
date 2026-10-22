"""Unit tests for the audited DQN target and Double DQN update.

Run with:
    python -m unittest -v v83_spatial_stage2.test_stage2m_q_targets
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT := PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_state import masked_argmax


def _constant_q_network(network, values):
    """Force a QNetwork to output ``values`` for every input."""
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.net[0].bias.fill_(1.0)
        network.net[2].bias.fill_(1.0)
        network.net[4].bias.copy_(torch.as_tensor(values))


def _fill_buffer(agent, next_mask, reward=0.0, done=0.0, count=128):
    state = np.zeros(agent.state_dim, dtype=np.float32)
    for _ in range(count):
        agent.buffer.add(
            state, 0, reward, state, done,
            np.asarray(next_mask, dtype=bool),
        )


class TargetMaskTest(unittest.TestCase):
    def test_legacy_zero_fill_is_reproduced(self):
        agent = DQNAgentV2("cpu", seed=0, state_dim=2, legacy_target=True)
        values = [-5.0] + [0.0] * 19
        _constant_q_network(agent.target, values)
        _constant_q_network(agent.online, values)
        next_mask = np.zeros(20, dtype=bool)
        next_mask[0] = True
        _fill_buffer(agent, next_mask)
        stats = agent.train_step()
        # Legacy fills unavailable entries with 0.0, so max = 0.
        self.assertAlmostEqual(stats["target_mean"], 0.0, places=5)

    def test_fixed_mask_uses_available_minimum(self):
        agent = DQNAgentV2("cpu", seed=0, state_dim=2)
        values = [-5.0] + [0.0] * 19
        _constant_q_network(agent.target, values)
        _constant_q_network(agent.online, values)
        next_mask = np.zeros(20, dtype=bool)
        next_mask[0] = True
        _fill_buffer(agent, next_mask)
        stats = agent.train_step()
        # Only action 0 is available and its Q is -5: no phantom zero.
        self.assertAlmostEqual(stats["target_mean"], -5.0, places=5)

    def test_double_dqn_evaluates_online_argmax(self):
        agent = DQNAgentV2(
            "cpu", seed=0, state_dim=2, double_dqn=True
        )
        target_values = [-5.0, -7.0] + [0.0] * 18
        online_values = [-6.0, -1.0] + [0.0] * 18
        _constant_q_network(agent.target, target_values)
        _constant_q_network(agent.online, online_values)
        next_mask = np.zeros(20, dtype=bool)
        next_mask[0] = True
        next_mask[1] = True
        _fill_buffer(agent, next_mask)
        stats = agent.train_step()
        # Online picks action 1; target evaluates Q(1) = -7, not max = -5.
        self.assertAlmostEqual(stats["target_mean"], -7.0, places=5)

    def test_terminal_transition_does_not_bootstrap(self):
        agent = DQNAgentV2("cpu", seed=0, state_dim=2)
        _constant_q_network(agent.target, [10.0] * 20)
        _constant_q_network(agent.online, [10.0] * 20)
        next_mask = np.zeros(20, dtype=bool)
        _fill_buffer(agent, next_mask, reward=0.25, done=1.0)
        stats = agent.train_step()
        self.assertAlmostEqual(stats["target_mean"], 0.25, places=5)

    def test_masked_argmax_never_selects_unavailable(self):
        q_values = np.asarray([100.0, -100.0] + [-50.0] * 18)
        mask = np.zeros(20, dtype=bool)
        mask[1] = True
        self.assertEqual(masked_argmax(q_values, mask), 1)


if __name__ == "__main__":
    unittest.main()
