"""Strict reward-consistency and data-isolation tests for Stage-2k v2.

Run from the repository root:

    python -m unittest -v v83_spatial_stage2.test_stage2k_jointr_v2
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BASELINE_DIR,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    VISUAL_PCA_PATH,
    load_split_specs,
    load_topology,
    read_json,
)
from v83_spatial_stage2.stage2k_dqn import ReplayBuffer, flat_quality
from v83_spatial_stage2.stage2k_jointr_reward import (
    aligned_delay_rewards,
    aligned_episode_audit,
    aligned_episode_jointr,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_jointr_v2_order import (
    stratified_training_order,
    order_statistics,
    verify_order,
    verify_validation_isolation,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import STATE_DIM, VisualProjector


class RewardFormulaTests(unittest.TestCase):
    def test_single_step(self):
        rewards, mean = aligned_delay_rewards([120.0])
        self.assertAlmostEqual(mean, 120.0)
        self.assertAlmostEqual(float(rewards.sum()), -0.30, places=12)
        self.assertAlmostEqual(
            aligned_episode_jointr(0.7, 120.0), 0.4, places=12
        )

    def test_constant_delay(self):
        rewards, mean = aligned_delay_rewards([100.0] * 5)
        self.assertAlmostEqual(mean, 100.0)
        self.assertAlmostEqual(float(rewards.sum()), -0.25, places=12)
        self.assertAlmostEqual(
            aligned_episode_jointr(0.8, 100.0), 0.55, places=12
        )

    def test_varying_delay(self):
        rewards, mean = aligned_delay_rewards([50.0, 150.0, 100.0])
        self.assertAlmostEqual(mean, 100.0)
        self.assertAlmostEqual(float(rewards.sum()), -0.25, places=12)
        audit = aligned_episode_audit([0.1, 0.2, 0.5], [50.0, 150.0, 100.0],
                                      0.8)
        self.assertLess(audit["abs_error"], 1e-12)
        self.assertLess(abs(audit["nav_minus_ndtw"]), 1e-12)

    def test_max_400_steps(self):
        rewards, mean = aligned_delay_rewards([200.0] * 400)
        self.assertAlmostEqual(mean, 200.0)
        self.assertAlmostEqual(float(rewards.sum()), -0.5, places=12)
        audit = aligned_episode_audit(
            [0.0] * 399 + [0.9], [200.0] * 400, 0.9
        )
        self.assertLess(audit["abs_error"], 1e-12)

    def test_baseline_900_recomputation(self):
        steps_path = BASELINE_DIR / "dev" / "steps.csv"
        episodes_path = BASELINE_DIR / "dev" / "per_episode.csv"
        self.assertTrue(steps_path.is_file())
        grouped = {}
        with steps_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["strategy"], row["episode_uid"])
                grouped.setdefault(key, []).append(row)
        episodes = {
            (row["strategy"], row["episode_uid"]): row
            for row in csv.DictReader(episodes_path.open(encoding="utf-8"))
        }
        self.assertEqual(len(grouped), 900)
        max_error = 0.0
        max_nav_error = 0.0
        max_delay_error = 0.0
        for key, rows in grouped.items():
            rows.sort(key=lambda row: int(row["step"]))
            nav = [float(row["navigation_reward"]) for row in rows]
            delays = [float(row["e2e_ms"]) for row in rows]
            episode = episodes[key]
            ndtw = float(episode["ndtw"])
            audit = aligned_episode_audit(nav, delays, ndtw)
            max_error = max(max_error, audit["abs_error"])
            max_nav_error = max(max_nav_error, abs(audit["nav_minus_ndtw"]))
            max_delay_error = max(
                max_delay_error,
                abs(audit["delay_sum"] + 0.25 * audit["mean_delay_ms"] / 100.0),
            )
            self.assertLess(
                abs(audit["episode_return"] - float(episode["joint_reward"])),
                1e-6,
                msg=f"{key} return/JointR mismatch",
            )
        self.assertLess(max_error, 1e-6)
        self.assertLess(max_nav_error, 1e-6)
        self.assertLess(max_delay_error, 1e-9)


class OrderAndIsolationTests(unittest.TestCase):
    def test_stratified_order_covers_all_windows(self):
        payload = load_split_specs("train")
        order = stratified_training_order(payload)
        lookup = verify_order(order, payload)
        stats = order_statistics(order, lookup)
        self.assertEqual(len(order), 4000)
        self.assertEqual(len(set(order)), 4000)
        self.assertTrue(stats["all_full_windows_cover_9_groups"])
        self.assertEqual(stats["min_groups_in_full_window"], 9)
        self.assertEqual(sum(stats["counts"].values()), 4000)
        for band in ("short", "medium", "long"):
            total = sum(
                count for key, count in stats["counts"].items()
                if key.endswith(f"/{band}")
            )
            self.assertGreater(total, 1300, msg=band)

    def test_validation_isolation(self):
        checks, validation = verify_validation_isolation()
        self.assertTrue(checks["ok"], msg=json.dumps(checks))
        self.assertEqual(len(validation), 60)
        self.assertTrue(all(item["split"] == "dev" for item in validation))


class ReplayBufferTests(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(7)
        buffer = ReplayBuffer(capacity=8, state_dim=5, action_dim=3, seed=11)
        for _ in range(6):
            buffer.add(
                rng.normal(size=5).astype(np.float32),
                int(rng.integers(0, 3)),
                float(rng.normal()),
                rng.normal(size=5).astype(np.float32),
                bool(rng.integers(0, 2)),
                rng.integers(0, 2, size=3).astype(bool),
            )
        payload = buffer.state_dict()
        restored = ReplayBuffer(capacity=8, state_dim=5, action_dim=3, seed=99)
        restored.load_state_dict(payload)
        np.testing.assert_array_equal(buffer.states, restored.states)
        np.testing.assert_array_equal(buffer.actions, restored.actions)
        np.testing.assert_array_equal(buffer.next_masks, restored.next_masks)
        self.assertEqual(buffer.size, restored.size)
        self.assertEqual(buffer.position, restored.position)
        first = buffer.rng.integers(0, buffer.size, size=20)
        second = restored.rng.integers(0, restored.size, size=20)
        np.testing.assert_array_equal(first, second)


class AgentCheckpointTests(unittest.TestCase):
    def test_full_roundtrip(self):
        agent = DQNAgentV2("cpu", seed=3)
        agent.buffer = ReplayBuffer(
            capacity=4, state_dim=STATE_DIM, action_dim=20, seed=3
        )
        for index in range(2):
            agent.buffer.add(
                np.zeros(STATE_DIM, dtype=np.float32), index,
                -0.1 * index, np.zeros(STATE_DIM, dtype=np.float32),
                index == 1, np.ones(20, dtype=bool),
            )
        agent.total_steps = 17
        agent.update_steps = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_full.pt"
            agent.save_full(path, extra={"episode": 2})
            expected_rng = agent.rng.integers(0, 1000, size=5)
            restored = DQNAgentV2("cpu", seed=999)
            restored.buffer = ReplayBuffer(
                capacity=4, state_dim=STATE_DIM, action_dim=20, seed=999
            )
            payload = restored.load_full(path)
        self.assertEqual(payload["episode"], 2)
        self.assertEqual(restored.total_steps, 17)
        self.assertEqual(restored.update_steps, 3)
        self.assertEqual(restored.buffer.size, 2)
        for key, value in agent.online.state_dict().items():
            torch.testing.assert_close(
                value, restored.online.state_dict()[key]
            )
        np.testing.assert_array_equal(
            expected_rng, restored.rng.integers(0, 1000, size=5)
        )


class AlignedRunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quality = flat_quality(read_json(QUALITY_PRIOR_PATH))
        cls.projector = VisualProjector.load(VISUAL_PCA_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.device, cls.encoder, cls.backends = load_runtime(device)
        cls.gab_config = load_gab_config()
        cls.specs = load_split_specs("dev")
        cls.topology = load_topology("balanced", "dev")

    def _runner(self):
        spec = self.specs["scenarios"]["balanced"]["episodes"][0]
        scene = build_scene(spec, self.encoder, self.device)
        return EpisodeRunner(
            spec=spec,
            topology=self.topology,
            encoder=self.encoder,
            backends=self.backends,
            projector=self.projector,
            quality_prior=self.quality,
            gab_config=self.gab_config,
            scene=scene,
            device=self.device,
            jointr_aligned=True,
        )

    def test_reset_and_one_episode_consistency(self):
        runner = self._runner()
        self.assertEqual(runner.running_mean_e2e_ms, 0.0)
        self.assertEqual(len(runner.step_data["state"]), STATE_DIM)
        self.assertAlmostEqual(
            float(runner.step_data["state"][128 + 5]), 0.0, places=12
        )
        while True:
            action = runner.select_node("min_latency")
            transition = runner.apply(action)
            if transition["done"]:
                break
        result = runner.finalize("jointr_v2")
        self.assertLess(result["jointr_abs_error"], 1e-6)
        nav = sum(
            row["navigation_reward"] for row in result["step_records"]
        )
        delay = sum(
            row["latency_penalty"] for row in result["step_records"]
        )
        self.assertLess(abs(nav - result["ndtw"]), 1e-6)
        self.assertLess(
            abs(delay + 0.25 * result["mean_episode_e2e_ms"] / 100.0),
            1e-9,
        )
        self.assertAlmostEqual(
            result["running_mean_e2e_ms"],
            result["mean_episode_e2e_ms"],
            places=9,
        )
        runner.reset()
        self.assertEqual(runner.running_mean_e2e_ms, 0.0)
        self.assertEqual(runner.return_sum, 0.0)
        self.assertEqual(len(runner.step_data["state"]), STATE_DIM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
