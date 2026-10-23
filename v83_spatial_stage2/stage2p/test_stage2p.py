"""Stage-2p unit tests: baseline vs scheme D reward identities and PPO GAE.

Run:
    python v83_spatial_stage2/stage2p/test_stage2p.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_ppo_agent import RolloutBuffer
from v83_spatial_stage2.stage2n_common import (
    latency_reward_increment,
    latency_rewards,
)
from v83_spatial_stage2.stage2o.rewards import (
    scheme_d_step_reward,
    scheme_d_terminal_correction,
)
from v83_spatial_stage2.stage2p.rewards import (
    H_MAX,
    TAU_MS,
    EpisodeRewardController,
    aligned_baseline_return,
    baseline_increment,
    episode_rewards,
    expected_return,
)


def _sequence(count=37, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(5.0, 180.0, size=count)


class RewardIdentityTests(unittest.TestCase):
    def test_baseline_matches_stage2n_exactly(self):
        delays = _sequence()
        rewards, diagnostics = episode_rewards("baseline", delays)
        stage2n, mean = latency_rewards(delays)
        np.testing.assert_allclose(rewards, stage2n, rtol=0, atol=1e-15)
        self.assertAlmostEqual(
            diagnostics["return"], -0.25 * mean / TAU_MS, 12
        )
        self.assertAlmostEqual(
            diagnostics["return"],
            expected_return("baseline", float(np.mean(delays))),
            12,
        )

    def test_baseline_increment_matches_stage2n(self):
        mean_stage2p = 0.0
        mean_stage2n = 0.0
        for index, delay in enumerate(_sequence(41, seed=11), start=1):
            mean_stage2p, reward = baseline_increment(
                delay, mean_stage2p, index
            )
            mean_stage2n, reward_stage2n = latency_reward_increment(
                delay, mean_stage2n, index
            )
            self.assertAlmostEqual(mean_stage2p, mean_stage2n, 14)
            self.assertAlmostEqual(reward, reward_stage2n, 14)

    def test_scheme_d_identity(self):
        delays = _sequence(41, seed=13)
        rewards, diagnostics = episode_rewards("D", delays)
        mean = float(np.mean(delays))
        base = [scheme_d_step_reward(value) for value in delays]
        correction = scheme_d_terminal_correction(len(delays), mean)
        self.assertAlmostEqual(
            diagnostics["return"], 1.0 - mean / TAU_MS, 12
        )
        self.assertAlmostEqual(sum(base) + correction, sum(rewards), 12)
        self.assertAlmostEqual(correction,
                               diagnostics["terminal_correction"], 12)
        np.testing.assert_allclose(rewards[:-1], base[:-1], rtol=0, atol=0)

    def test_alignment_four_r_plus_one(self):
        delays = _sequence(29, seed=17)
        _, baseline = episode_rewards("baseline", delays)
        _, scheme_d = episode_rewards("D", delays)
        self.assertAlmostEqual(
            aligned_baseline_return(baseline["return"]),
            scheme_d["return"], 12,
        )
        for value in (-0.5, -0.25, 0.0, 0.4):
            self.assertAlmostEqual(
                aligned_baseline_return(value), 4.0 * value + 1.0, 14
            )

    def test_threshold_100ms(self):
        delays = [100.0] * 23
        _, baseline = episode_rewards("baseline", delays)
        _, scheme_d = episode_rewards("D", delays)
        self.assertAlmostEqual(baseline["return"], -0.25, 12)
        self.assertAlmostEqual(scheme_d["return"], 0.0, 12)
        self.assertAlmostEqual(
            aligned_baseline_return(baseline["return"]),
            scheme_d["return"], 12,
        )

    def test_d_correction_zero_at_horizon(self):
        delays = [42.0] * H_MAX
        rewards, diagnostics = episode_rewards("D", delays)
        self.assertAlmostEqual(diagnostics["terminal_correction"], 0.0, 12)
        self.assertAlmostEqual(
            rewards[-1], scheme_d_step_reward(42.0), 12
        )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            episode_rewards("baseline", [])
        with self.assertRaises(ValueError):
            episode_rewards("baseline", [-1.0])
        with self.assertRaises(ValueError):
            episode_rewards("D", [10.0] * (H_MAX + 1))
        with self.assertRaises(ValueError):
            episode_rewards("nope", [10.0])


class NoFutureLeakageTests(unittest.TestCase):
    def test_nonterminal_rewards_do_not_depend_on_future(self):
        base = _sequence(31, seed=19)
        for scheme in ("baseline", "D"):
            rewards, _ = episode_rewards(scheme, base)
            for change_index in (5, 17, 30):
                changed = list(base)
                changed[change_index] += 40.0
                changed_rewards, _ = episode_rewards(scheme, changed)
                prefix = min(change_index, len(base) - 1)
                np.testing.assert_allclose(
                    rewards[:prefix], changed_rewards[:prefix],
                    rtol=0, atol=0,
                    err_msg=f"{scheme} changed reward before step "
                            f"{change_index}",
                )


class ControllerTests(unittest.TestCase):
    def test_controller_matches_pure_implementation(self):
        delays = _sequence(33, seed=29)
        for scheme in ("baseline", "D"):
            controller = EpisodeRewardController(scheme)
            online = [controller.step_reward(value) for value in delays]
            final_reward, diagnostics = controller.finalize(
                float(np.mean(delays)), len(delays)
            )
            online[-1] = final_reward
            pure, pure_diag = episode_rewards(scheme, delays)
            np.testing.assert_allclose(online, pure, rtol=0, atol=1e-12)
            self.assertAlmostEqual(
                diagnostics["return"], pure_diag["return"], 12
            )

    def test_controller_rejects_double_finalize(self):
        controller = EpisodeRewardController("D")
        controller.step_reward(50.0)
        controller.finalize(50.0, 1)
        with self.assertRaises(RuntimeError):
            controller.finalize(50.0, 1)


class PPOAdvantageTests(unittest.TestCase):
    def test_gae_terminal_reset_with_d_rewards(self):
        delays = [30.0] * 9
        rewards, diagnostics = episode_rewards("D", delays)
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
        self.assertTrue(np.isfinite(advantages).all())
        self.assertTrue(np.isfinite(returns).all())
        # With gamma=1, lambda=1, V=0, returns_t = sum of rewards from t.
        np.testing.assert_allclose(
            returns, np.cumsum(rewards[::-1])[::-1], rtol=0, atol=1e-6
        )
        self.assertAlmostEqual(returns[-1], rewards[-1], 6)
        self.assertAlmostEqual(
            returns[0], diagnostics["return"], 6
        )

    def test_gae_resets_between_episodes(self):
        rollout = RolloutBuffer()
        for episode in range(2):
            for index in range(5):
                done = index == 4
                rollout.add(
                    np.zeros(384, dtype=np.float32),
                    np.ones(20, dtype=bool),
                    0, 0.0, 0.0,
                    0.2 if done else 0.0, done,
                )
        _, returns = rollout.compute_gae(gamma=1.0, gae_lambda=1.0)
        np.testing.assert_allclose(
            returns, np.full(10, 0.2), rtol=0, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
