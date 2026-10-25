"""CPU tests for the 10-node Stage-2r contract.

Run:
    python -m unittest -v \
        v83_spatial_stage2.stage2r_10node_small_vs_tm.test_stage2r
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_dqn import QNetwork, ReplayBuffer
from v83_spatial_stage2.stage2k_latency import Stage2KLatency
from v83_spatial_stage2.stage2k_ppo_agent import (
    ActorCriticMLP,
    PPOAgent,
    PPOAnomalyError,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    CANDIDATE_DIM,
    build_pure_mec_v2_state,
    pure_mec_v2_candidate_features,
)
from v83_spatial_stage2.stage2o.rewards import EpisodeRewardController
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (
    ACTION_DIM,
    MEC_MODELS,
    NUM_NODES,
    STATE_DIM,
    TRAIN_BAND_COUNTS,
    build_deployment,
    canonical_sha256,
    load_bundle,
    select_a10_indices,
)


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployments = {
            model: build_deployment(model) for model in MEC_MODELS
        }

    def test_ten_unique_nodes_and_shared_geometry(self):
        positions = {}
        for model, payload in self.deployments.items():
            self.assertEqual(len(payload["nodes"]), NUM_NODES)
            self.assertTrue(payload["homogeneity_checks"]["ok"])
            self.assertEqual(
                {node["model_type"] for node in payload["nodes"]}, {model}
            )
            self.assertEqual(len({
                (node["x"], node["y"]) for node in payload["nodes"]
            }), NUM_NODES)
            positions[model] = [
                (node["x"], node["y"]) for node in payload["nodes"]
            ]
        self.assertEqual(positions["Small"], positions["Transformer-M"])

    def test_selected_indices_and_full_task_coverage(self):
        indices = select_a10_indices()
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices, sorted(indices))
        bundle = load_bundle("Transformer-M")
        nodes = np.asarray([
            [node["x"], node["y"]] for node in bundle["deployment"]["nodes"]
        ])
        starts = np.asarray([
            spec["start"] for spec in bundle["train_specs"] + bundle["dev_specs"]
        ])
        distances = np.linalg.norm(
            starts[:, None, :] - nodes[None, :, :], axis=2
        ).min(axis=1)
        self.assertEqual(int((distances > 500.0).sum()), 0)

    def test_deployment_hash_and_model_checkpoint(self):
        for model, payload in self.deployments.items():
            expected = canonical_sha256({
                "nodes": payload["nodes"],
                "mec": payload["mec"],
                "source_topology_sha256": payload[
                    "source_topology_sha256"
                ],
            })
            self.assertEqual(payload["deployment_sha256"], expected)
            self.assertEqual(payload["mec"]["model"], model)
            self.assertEqual(payload["mec"]["num_nodes"], 10)


class StateActionTests(unittest.TestCase):
    def test_candidate_and_state_dimension(self):
        deployment = build_deployment("Small")
        bundle = load_bundle("Small")
        candidate, mask, metrics, fallback = pure_mec_v2_candidate_features(
            deployment["nodes"],
            (750.0, 750.0),
            Stage2KLatency(),
            capability_prior=bundle["quality_prior"],
            num_nodes=10,
        )
        self.assertEqual(candidate.shape, (10, CANDIDATE_DIM))
        self.assertEqual(mask.shape, (10,))
        self.assertTrue(mask.any())
        self.assertFalse(fallback)
        state = build_pure_mec_v2_state(
            (750.0, 750.0), 0, 0.0, candidate
        )
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(STATE_DIM, 194)

    def test_networks_have_ten_outputs(self):
        q = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        output = q(torch.zeros(4, STATE_DIM))
        self.assertEqual(tuple(output.shape), (4, 10))
        actor_critic = ActorCriticMLP(
            state_dim=STATE_DIM, action_dim=ACTION_DIM
        )
        logits, value = actor_critic(torch.zeros(3, STATE_DIM))
        self.assertEqual(tuple(logits.shape), (3, 10))
        self.assertEqual(tuple(value.shape), (3,))

    def test_replay_buffer_action_dim(self):
        buffer = ReplayBuffer(
            capacity=8, state_dim=STATE_DIM, action_dim=10, seed=7
        )
        buffer.add(
            np.zeros(STATE_DIM), 3, 1.0, np.zeros(STATE_DIM), True,
            np.ones(10, dtype=bool),
        )
        batch = buffer.sample(2, torch.device("cpu"))
        self.assertEqual(tuple(batch["next_mask"].shape), (2, 10))
        self.assertEqual(buffer.action_dim, 10)

    def test_ppo_rejects_twenty_dim_mask(self):
        agent = PPOAgent(
            torch.device("cpu"), seed=11, state_dim=STATE_DIM,
            state_mode="pure_mec_v2", action_dim=10, gamma=1.0,
        )
        with self.assertRaises(PPOAnomalyError):
            agent.act(np.zeros(STATE_DIM, dtype=np.float32),
                      np.ones(20, dtype=bool))


class RewardAndProtocolTests(unittest.TestCase):
    def test_scheme_d_identity(self):
        rng = np.random.default_rng(11)
        delays = rng.uniform(15.0, 140.0, size=137)
        controller = EpisodeRewardController("D", tau=100.0, horizon=400)
        for delay in delays:
            controller.step_reward(float(delay))
        _, diagnostics = controller.finalize(float(delays.mean()), len(delays))
        expected = 1.0 - float(delays.mean()) / 100.0
        self.assertAlmostEqual(diagnostics["return"], expected, places=12)
        self.assertLess(diagnostics["abs_error"], 1e-9)

    def test_task_bundle_never_reads_test(self):
        bundle = load_bundle("Small")
        self.assertFalse(bundle["test_read"])
        self.assertEqual(len(bundle["train_specs"]), 4000)
        self.assertEqual(len(bundle["dev_specs"]), 60)
        self.assertEqual(
            {spec["band"] for spec in bundle["train_specs"]},
            {"short", "medium", "long"},
        )
        for band, count in TRAIN_BAND_COUNTS.items():
            self.assertEqual(
                sum(1 for spec in bundle["train_specs"]
                    if spec["band"] == band),
                count,
            )

    def test_source_does_not_request_test_pool(self):
        text = Path(__file__).read_text(encoding="utf-8")
        # The test itself may mention the string; the Stage-2r modules may
        # not call load_task_pool("test").
        stage_dir = Path(__file__).resolve().parent
        for path in stage_dir.glob("*.py"):
            if path.name == Path(__file__).name:
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn(
                'load_task_pool("test"', source,
                f"{path.name} must not load the sealed Test pool",
            )
            self.assertNotIn(
                "load_task_pool('test'", source,
                f"{path.name} must not load the sealed Test pool",
            )


if __name__ == "__main__":
    unittest.main()
