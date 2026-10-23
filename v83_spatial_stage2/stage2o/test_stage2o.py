"""Stage-2o unit tests: reward identities, leakage and RL target handling.

Run:

    python v83_spatial_stage2/stage2o/test_stage2o.py

The tests are deliberately independent of the frozen task pools.  They cover

* scheme B / D / one per-step and cumulative reward identities,
* the 100 ms threshold, horizon and terminal-correction behaviour,
* absence of future-information leakage into non-terminal rewards,
* PPO GAE terminal resets and exact J targets for scheme one,
* DQN replay terminal handling with a fake Q network,
* the online EpisodeRewardController matching the pure implementation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_ppo_agent import RolloutBuffer
from v83_spatial_stage2.stage2n_common import latency_rewards
from v83_spatial_stage2.stage2o.rewards import (
    H_MAX,
    TAU_MS,
    EpisodeRewardController,
    episode_rewards,
    expected_return,
    scheme_b_increment,
    scheme_d_step_reward,
    scheme_d_terminal_correction,
    scheme_one_terminal_reward,
)


def _sequence(count=37, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(5.0, 180.0, size=count)


class RewardIdentityTests(unittest.TestCase):
    def test_scheme_b_matches_stage2n_times_four(self):
        delays = _sequence()
        rewards, diagnostics = episode_rewards("B", delays)
        stage2n, mean = latency_rewards(delays)
        np.testing.assert_allclose(rewards, 4.0 * stage2n, rtol=0, atol=1e-12)
        self.assertAlmostEqual(diagnostics["return"], -mean / TAU_MS, 12)
        self.assertAlmostEqual(
            diagnostics["return"],
            expected_return("B", float(np.mean(delays))),
            12,
        )

    def test_scheme_d_identity_and_correction(self):
        delays = _sequence(41, seed=11)
        rewards, diagnostics = episode_rewards("D", delays)
        steps = len(delays)
        mean = float(np.mean(delays))
        base = [scheme_d_step_reward(value) for value in delays]
        correction = scheme_d_terminal_correction(steps, mean)
        expected = 1.0 - mean / TAU_MS
        self.assertAlmostEqual(diagnostics["return"], expected, 12)
        self.assertAlmostEqual(sum(base) + correction, expected, 12)
        self.assertAlmostEqual(diagnostics["terminal_correction"],
                               correction, 12)
        self.assertAlmostEqual(rewards[-1], base[-1] + correction, 12)
        np.testing.assert_allclose(rewards[:-1], base[:-1], rtol=0, atol=0)
        self.assertAlmostEqual(diagnostics["base_sum"], sum(base), 12)

    def test_scheme_one_identity_and_sparsity(self):
        delays = _sequence(53, seed=13)
        rewards, diagnostics = episode_rewards("one", delays)
        mean = float(np.mean(delays))
        self.assertTrue(all(value == 0.0 for value in rewards[:-1]))
        self.assertAlmostEqual(
            rewards[-1], scheme_one_terminal_reward(mean), 12
        )
        self.assertAlmostEqual(
            diagnostics["return"], 1.0 - mean / TAU_MS, 12
        )
        self.assertEqual(diagnostics["base_sum"], 0.0)

    def test_d_and_one_share_return_b_is_shifted_by_one(self):
        delays = _sequence(29, seed=17)
        _, diag_b = episode_rewards("B", delays)
        _, diag_d = episode_rewards("D", delays)
        _, diag_one = episode_rewards("one", delays)
        self.assertAlmostEqual(diag_d["return"], diag_one["return"], 12)
        self.assertAlmostEqual(
            diag_d["return"] - diag_b["return"], 1.0, 12
        )
        self.assertAlmostEqual(diag_one["return"] - diag_b["return"], 1.0, 12)

    def test_tau_threshold_behaviour(self):
        for value, expected_b, expected_d in (
            (0.0, 0.0, 1.0),
            (50.0, -0.5, 0.5),
            (100.0, -1.0, 0.0),
            (200.0, -2.0, -1.0),
        ):
            delays = [value] * 23
            _, diag_b = episode_rewards("B", delays)
            _, diag_d = episode_rewards("D", delays)
            _, diag_one = episode_rewards("one", delays)
            self.assertAlmostEqual(diag_b["return"], expected_b, 12)
            self.assertAlmostEqual(diag_d["return"], expected_d, 12)
            self.assertAlmostEqual(diag_one["return"], expected_d, 12)

    def test_d_correction_vanishes_at_horizon(self):
        delays = [42.0] * H_MAX
        rewards, diagnostics = episode_rewards("D", delays)
        self.assertAlmostEqual(diagnostics["terminal_correction"], 0.0, 12)
        self.assertAlmostEqual(
            diagnostics["return"], 1.0 - np.mean(delays) / TAU_MS, 12
        )
        expected_last = scheme_d_step_reward(42.0)
        self.assertAlmostEqual(rewards[-1], expected_last, 12)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            episode_rewards("B", [])
        with self.assertRaises(ValueError):
            episode_rewards("B", [-1.0])
        with self.assertRaises(ValueError):
            episode_rewards("D", [10.0] * (H_MAX + 1))
        with self.assertRaises(ValueError):
            episode_rewards("nope", [10.0])
        with self.assertRaises(ValueError):
            scheme_b_increment(10.0, 0.0, 0)


class NoFutureLeakageTests(unittest.TestCase):
    def test_nonterminal_rewards_do_not_depend_on_future_e2e(self):
        base = _sequence(31, seed=19)
        for scheme in ("B", "D", "one"):
            rewards, _ = episode_rewards(scheme, base)
            for change_index in (5, 17, 30):
                changed = list(base)
                changed[change_index] += 40.0
                rewards_changed, _ = episode_rewards(scheme, changed)
                prefix = min(change_index, len(base) - 1)
                # For scheme one every non-terminal reward is zero; for B a
                # later e2e changes the running mean only from that step on.
                np.testing.assert_allclose(
                    rewards[:prefix], rewards_changed[:prefix],
                    rtol=0, atol=0,
                    err_msg=f"{scheme} changed reward before step "
                            f"{change_index}",
                )
                # The terminal transition is allowed to depend on the whole
                # episode; all rewards before it are not.

    def test_only_terminal_transition_carries_correction(self):
        delays = _sequence(25, seed=23)
        mean = float(np.mean(delays))
        rewards, diagnostics = episode_rewards("D", delays)
        self.assertNotEqual(diagnostics["terminal_correction"], 0.0)
        base = [scheme_d_step_reward(value) for value in delays]
        np.testing.assert_allclose(rewards[:-1], base[:-1], rtol=0, atol=0)
        self.assertAlmostEqual(rewards[-1], base[-1]
                               + scheme_d_terminal_correction(
                                   len(delays), mean), 12)
        one_rewards, _ = episode_rewards("one", delays)
        self.assertTrue(all(value == 0.0 for value in one_rewards[:-1]))


class ControllerTests(unittest.TestCase):
    def test_controller_matches_pure_implementation(self):
        delays = _sequence(33, seed=29)
        for scheme in ("B", "D", "one"):
            controller = EpisodeRewardController(scheme)
            online = []
            for value in delays:
                online.append(
                    controller.step_reward(value)
                )
            final_reward, diagnostics = controller.finalize(
                float(np.mean(delays)), len(delays)
            )
            online[-1] = final_reward
            pure, pure_diag = episode_rewards(scheme, delays)
            np.testing.assert_allclose(online, pure, rtol=0, atol=1e-12)
            self.assertAlmostEqual(
                diagnostics["return"], pure_diag["return"], 12
            )
            self.assertAlmostEqual(
                sum(online), expected_return(
                    scheme, float(np.mean(delays))
                ), 12
            )

    def test_controller_rejects_double_finalize(self):
        controller = EpisodeRewardController("D")
        controller.step_reward(50.0)
        controller.finalize(50.0, 1)
        with self.assertRaises(RuntimeError):
            controller.finalize(50.0, 1)


class FakeQNet(nn.Module):
    """Constant Q output; a dummy parameter keeps Adam/SGD happy."""

    def __init__(self, value, action_dim=20):
        super().__init__()
        self.value = float(value)
        self.dummy = nn.Parameter(torch.zeros(1))
        self.action_dim = int(action_dim)

    def forward(self, state):
        batch = state.shape[0]
        return torch.full(
            (batch, self.action_dim), self.value,
            dtype=state.dtype, device=state.device,
        ) + 0.0 * self.dummy


class DQNTargetTests(unittest.TestCase):
    def _agent(self, online_value, target_value):
        agent = DQNAgentV2(
            torch.device("cpu"), seed=0, gamma=1.0,
            state_dim=384, state_mode="pure_mec_v2",
        )
        agent.online = FakeQNet(online_value)
        agent.target = FakeQNet(target_value)
        agent.optimizer = torch.optim.SGD(agent.online.parameters(), lr=0.0)
        return agent

    def test_terminal_transition_target_equals_terminal_reward(self):
        agent = self._agent(0.0, 0.0)
        mean = 42.0
        terminal_reward = 1.0 - mean / TAU_MS
        state = np.zeros(384, dtype=np.float32)
        mask = np.ones(20, dtype=bool)
        # One full batch of identical terminal transitions (done=1).
        for _ in range(128):
            agent.buffer.add(
                state, 3, terminal_reward, None, True, None,
            )
        stats = agent.train_step()
        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats["target_mean"], terminal_reward, 5)
        self.assertAlmostEqual(stats["td_error_abs_mean"],
                               abs(terminal_reward), 5)

    def test_nonterminal_target_bootstraps_from_next_state(self):
        agent = self._agent(0.0, 0.25)
        state = np.zeros(384, dtype=np.float32)
        mask = np.ones(20, dtype=bool)
        next_state = np.ones(384, dtype=np.float32)
        for _ in range(128):
            agent.buffer.add(
                state, 3, 0.0, next_state, False, mask,
            )
        stats = agent.train_step()
        self.assertAlmostEqual(stats["target_mean"], 0.25, 5)

    def test_scheme_one_sparse_episode_stored_correctly(self):
        delays = _sequence(19, seed=31)
        rewards, diagnostics = episode_rewards("one", delays)
        agent = self._agent(0.0, 0.0)
        mean = float(np.mean(delays))
        for index, value in enumerate(delays):
            done = index == len(delays) - 1
            next_state = None if done else np.zeros(384, dtype=np.float32)
            next_mask = None if done else np.ones(20, dtype=bool)
            agent.buffer.add(
                np.zeros(384, dtype=np.float32), 0, rewards[index],
                next_state, done, next_mask,
            )
        self.assertEqual(int(agent.buffer.dones[:len(delays)].sum()), 1)
        self.assertTrue(
            not agent.buffer.next_masks[len(delays) - 1].any()
        )
        self.assertAlmostEqual(
            float(agent.buffer.rewards[:len(delays)].sum()),
            diagnostics["return"], 6,
        )
        self.assertAlmostEqual(diagnostics["return"],
                               1.0 - mean / TAU_MS, 12)


class PPOAdvantageTests(unittest.TestCase):
    def test_gae_terminal_reset_and_exact_episode_return(self):
        delays = [30.0] * 9
        rewards, diagnostics = episode_rewards("one", delays)
        J = diagnostics["return"]
        values = [0.0] * len(delays)
        rollout = RolloutBuffer()
        for index, reward in enumerate(rewards):
            rollout.add(
                np.zeros(384, dtype=np.float32),
                np.ones(20, dtype=bool),
                0, 0.0, values[index], reward,
                index == len(rewards) - 1,
            )
        advantages, returns = rollout.compute_gae(gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(returns, np.full(len(delays), J),
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(advantages, np.full(len(delays), J),
                                   rtol=0, atol=1e-6)
        # Standard PPO lambda=0.95 keeps the terminal target exactly J and
        # only attenuates the terminal signal for earlier steps.
        advantages_95, returns_95 = rollout.compute_gae(
            gamma=1.0, gae_lambda=0.95
        )
        self.assertAlmostEqual(float(returns_95[-1]), J, 6)
        self.assertTrue(np.all(np.asarray(returns_95) <= J + 1e-12))
        self.assertTrue(np.all(np.diff(returns_95) >= -1e-12))

    def test_gae_resets_between_two_episodes(self):
        J = 0.6
        rollout = RolloutBuffer()
        for episode in range(2):
            for index in range(5):
                done = index == 4
                rollout.add(
                    np.zeros(384, dtype=np.float32),
                    np.ones(20, dtype=bool),
                    0, 0.0, 0.0,
                    J if done else 0.0, done,
                )
        advantages, returns = rollout.compute_gae(gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(returns, np.full(10, J), rtol=0, atol=1e-6)
        np.testing.assert_allclose(advantages, np.full(10, J),
                                   rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
