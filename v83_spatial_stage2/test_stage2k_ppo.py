"""Tests for the Stage-2k PPO agent and its fairness protocol."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BANDS,
    SCENARIOS,
    load_split_specs,
)
from v83_spatial_stage2.stage2k_dqn import (
    SEED,
    dev_validation_specs,
)
from v83_spatial_stage2.stage2k_jointr_v2_order import (
    random_training_order,
    verify_validation_isolation,
)
from v83_spatial_stage2.stage2k_ppo_agent import (
    CLIP_EPS,
    GAE_LAMBDA,
    GAMMA,
    JOINTR_REWARD_DESCRIPTION,
    PPOAgent,
    PPOAnomalyError,
    RolloutBuffer,
    masked_logits,
)

DQN_ORDER_PATH = Path(
    "runs/v83_spatial_stage2/stage2k_pure_mec_v2_smoke/training_order.json"
)


class MaskedPolicyTest(unittest.TestCase):
    def setUp(self):
        self.agent = PPOAgent(
            torch.device("cpu"), seed=3, state_dim=8,
            rollout_steps=64, minibatch_size=16, update_epochs=2,
        )
        with torch.no_grad():
            self.agent.network.actor.weight.zero_()
            self.agent.network.actor.bias.zero_()
            self.agent.network.actor.bias[0] = 10.0
        self.state = np.zeros(8, dtype=np.float32)
        self.mask = np.zeros(20, dtype=bool)
        self.mask[[3, 5]] = True

    def test_greedy_only_selects_available_actions(self):
        for _ in range(20):
            action = self.agent.greedy_action(self.state, self.mask)
            self.assertIn(action, (3, 5))

    def test_sampling_only_selects_available_actions(self):
        actions = {
            self.agent.act(self.state, self.mask)["action"]
            for _ in range(300)
        }
        self.assertTrue(actions.issubset({3, 5}))
        self.assertTrue(len(actions) >= 1)

    def test_log_prob_and_entropy_finite(self):
        decision = self.agent.act(self.state, self.mask)
        self.assertTrue(np.isfinite(decision["log_prob"]))
        self.assertTrue(np.isfinite(decision["entropy"]))
        self.assertTrue(np.isfinite(decision["value"]))
        # Entropy cannot exceed log(number of available actions).
        self.assertLessEqual(
            decision["entropy"], np.log(2.0) + 1e-5
        )

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            masked_logits(
                torch.zeros(1, 20), torch.zeros(1, 20, dtype=torch.bool)
            )


class GAETest(unittest.TestCase):
    @staticmethod
    def reference_gae(rewards, values, dones, gamma, gae_lambda):
        advantages = np.zeros(len(rewards), dtype=np.float64)
        last = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            if dones[index]:
                delta = rewards[index] - values[index]
                last = delta
            else:
                delta = (
                    rewards[index] + gamma * values[index + 1]
                    - values[index]
                )
                last = delta + gamma * gae_lambda * last
            advantages[index] = last
        return advantages, advantages + values

    def test_two_episodes_match_reference(self):
        buffer = RolloutBuffer()
        rewards = [0.1, -0.2, 0.3, 0.4, -0.1]
        values = [0.05, 0.0, -0.05, 0.2, 0.1]
        dones = [False, False, True, False, True]
        for index, (reward, value, done) in enumerate(
            zip(rewards, values, dones)
        ):
            buffer.add(
                np.zeros(8), np.ones(20, dtype=bool), 0,
                0.0, value, reward, done,
            )
        advantages, returns = buffer.compute_gae(
            gamma=GAMMA, gae_lambda=GAE_LAMBDA
        )
        expected_adv, expected_ret = self.reference_gae(
            np.asarray(rewards), np.asarray(values), dones,
            GAMMA, GAE_LAMBDA,
        )
        # RolloutBuffer stores float32 values; compare at float32 precision.
        np.testing.assert_allclose(advantages, expected_adv, atol=1e-6)
        np.testing.assert_allclose(returns, expected_ret, atol=1e-6)

    def test_terminal_reward_is_not_modified(self):
        buffer = RolloutBuffer()
        buffer.add(
            np.zeros(8), np.ones(20, dtype=bool), 0, 0.0, 0.7, 0.123, True
        )
        advantages, returns = buffer.compute_gae(
            gamma=GAMMA, gae_lambda=GAE_LAMBDA
        )
        self.assertAlmostEqual(float(returns[0]), 0.123, places=6)
        self.assertAlmostEqual(
            float(advantages[0]), 0.123 - 0.7, places=6
        )

    def test_no_advantage_leakage_across_episodes(self):
        def build(second_reward):
            buffer = RolloutBuffer()
            for reward, done in (
                (0.1, False), (0.2, False), (0.3, True),
                (second_reward, False), (0.4, True),
            ):
                buffer.add(
                    np.zeros(8), np.ones(20, dtype=bool), 0,
                    0.0, 0.1, reward, done,
                )
            return buffer.compute_gae(GAMMA, GAE_LAMBDA)[0][:3]

        np.testing.assert_allclose(
            build(0.5), build(-0.9), atol=1e-6
        )


class PPOUpdateTest(unittest.TestCase):
    def _agent(self):
        return PPOAgent(
            torch.device("cpu"), seed=5, state_dim=8,
            rollout_steps=64, minibatch_size=16, update_epochs=2,
        )

    def _buffer(self, count=64, anomaly=False):
        rng = np.random.default_rng(11)
        buffer = RolloutBuffer()
        for index in range(count):
            state = rng.normal(size=8).astype(np.float32)
            if anomaly and index == 3:
                state[0] = np.inf
            mask = np.ones(20, dtype=bool)
            buffer.add(
                state, mask, int(index % 20), -3.0,
                float(rng.normal()), float(rng.normal()) * 0.1,
                bool((index + 1) % 16 == 0),
            )
        return buffer

    def test_update_is_finite_and_counts_minibatches(self):
        agent = self._agent()
        stats = agent.update(self._buffer())
        for key in (
            "policy_loss", "value_loss", "entropy", "loss", "approx_kl",
            "clip_fraction", "grad_norm", "explained_variance",
        ):
            self.assertTrue(np.isfinite(stats[key]), key)
        self.assertEqual(stats["minibatches"], 8)  # 4 per epoch x 2
        self.assertEqual(stats["epochs_run"], 2)
        self.assertEqual(agent.update_steps, 8)
        # The single-sample KL estimator can be slightly negative.
        self.assertLess(abs(stats["approx_kl"]), 0.1)
        self.assertGreater(stats["entropy"], 0.0)
        self.assertGreaterEqual(stats["clip_fraction"], 0.0)
        self.assertLessEqual(stats["clip_fraction"], 1.0)

    def test_non_finite_state_raises_diagnostics(self):
        agent = self._agent()
        with self.assertRaises(PPOAnomalyError):
            agent.update(self._buffer(anomaly=True))

    def test_resume_roundtrip(self):
        agent = self._agent()
        agent.update(self._buffer())
        agent.total_steps = 123
        path = Path("/tmp/stage2k_ppo_test_full.pt")
        extra = {"episode": 7, "best_joint": 0.12, "no_improve": 1}
        agent.save_full(path, extra=extra)
        restored = self._agent()
        payload = restored.load_full(path)
        self.assertEqual(restored.total_steps, 123)
        self.assertEqual(payload["episode"], 7)
        self.assertEqual(
            restored.update_steps, agent.update_steps
        )
        path.unlink(missing_ok=True)


class ProtocolTest(unittest.TestCase):
    def test_hyperparameters(self):
        self.assertEqual(GAMMA, 1.0)
        self.assertEqual(GAE_LAMBDA, 0.95)
        self.assertEqual(CLIP_EPS, 0.2)
        self.assertIn("running_mean_E2E", JOINTR_REWARD_DESCRIPTION)
        self.assertNotIn("success", JOINTR_REWARD_DESCRIPTION.lower())

    def test_order_matches_dqn_reference(self):
        payload = load_split_specs("train")
        order, targets = random_training_order(payload, total=3500)
        self.assertEqual(len(order), 3500)
        self.assertEqual(len(set(order)), 3500)
        lookup = {
            item["episode_uid"]: item
            for scenario in SCENARIOS
            for item in payload["scenarios"][scenario]["episodes"]
        }
        bands = Counter(lookup[uid]["band"] for uid in order)
        self.assertEqual(
            [bands[band] for band in BANDS], [1167, 1167, 1166]
        )
        if DQN_ORDER_PATH.is_file():
            import json
            dqn = json.loads(DQN_ORDER_PATH.read_text(encoding="utf-8"))
            self.assertEqual(order, dqn["order"])
            self.assertEqual(
                targets, {
                    tuple(key.split("/")): value
                    for key, value in dqn["target_counts"].items()
                },
            )

    def test_validation_isolation_and_size(self):
        checks, validation = verify_validation_isolation()
        self.assertTrue(checks["ok"])
        self.assertEqual(len(validation), 60)
        by_band = Counter(
            (item["scenario"], item["band"]) for item in validation
        )
        for scenario in SCENARIOS:
            self.assertEqual(by_band[(scenario, "short")], 6)
            self.assertEqual(by_band[(scenario, "medium")], 10)
            self.assertEqual(by_band[(scenario, "long")], 4)
        self.assertEqual(len(dev_validation_specs()), 60)

    def test_seed_matches_dqn(self):
        self.assertEqual(SEED, 20261031)


if __name__ == "__main__":
    unittest.main()
