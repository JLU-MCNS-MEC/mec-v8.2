"""Tests for the 342-D Stage-2k no-visual DQN state."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    QUALITY_PRIOR_PATH,
    load_split_specs,
    load_topology,
    read_json,
)
from v83_spatial_stage2.stage2k_dqn import flat_quality
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_latency import Stage2KLatency
from v83_spatial_stage2.stage2k_no_visual_state import (
    CANDIDATE_DIM,
    CANDIDATE_FIELDS,
    INFERENCE_SCALE_MS,
    NUM_NODES,
    STATE_DIM_NO_VISUAL,
    UAV_POSITION_SCALE_M,
    build_no_visual_state,
    no_visual_candidate_features,
    state_layout_no_visual,
)
from v83_spatial_stage2.stage2k_no_visual_test import (
    build_freeze,
    paired_deltas,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)


class StateLayoutTests(unittest.TestCase):
    def test_dimension_and_layout(self):
        layout = state_layout_no_visual()
        self.assertEqual(STATE_DIM_NO_VISUAL, 342)
        self.assertEqual(layout["uav_position"], [0, 2])
        self.assertEqual(layout["candidates"], [2, 342])
        self.assertEqual(layout["total"], 342)
        self.assertEqual(len(layout["candidate_fields"]), CANDIDATE_DIM)
        forbidden = " ".join(layout["forbidden"]).lower()
        for word in ("visual", "goal", "gab", "action", "step", "success"):
            self.assertIn(word, forbidden)

    def test_candidate_fields_exclude_forbidden(self):
        forbidden = (
            "visual", "goal", "history", "heatmap", "gab", "map",
            "action", "step", "oracle", "success", "previous",
        )
        for field in CANDIDATE_FIELDS:
            for word in forbidden:
                self.assertNotIn(word, field.lower())

    def test_builder_signatures_have_no_visual_inputs(self):
        candidate = inspect.signature(no_visual_candidate_features)
        self.assertEqual(
            set(candidate.parameters),
            {"nodes", "uav_position", "latency_model", "quality_prior"},
        )
        builder = inspect.signature(build_no_visual_state)
        self.assertEqual(
            set(builder.parameters),
            {"uav_position", "candidate20x17"},
        )

    def test_build_state(self):
        candidate = np.zeros((NUM_NODES, CANDIDATE_DIM), dtype=np.float32)
        candidate[3, 0] = 1.0
        candidate[3, 9] = 1.44
        candidate[3, 10] = 0.55
        candidate[3, 11] = 0.20
        state = build_no_visual_state(
            [UAV_POSITION_SCALE_M / 2, UAV_POSITION_SCALE_M / 4], candidate
        )
        self.assertEqual(state.shape, (342,))
        self.assertAlmostEqual(float(state[0]), 0.5, places=6)
        self.assertAlmostEqual(float(state[1]), 0.25, places=6)
        self.assertAlmostEqual(float(state[2 + 3 * 17 + 9]), 1.44, places=5)
        self.assertAlmostEqual(float(state[2 + 3 * 17 + 10]), 0.55, places=5)
        self.assertAlmostEqual(float(state[2 + 3 * 17 + 11]), 0.20, places=5)


class CandidateFeatureTests(unittest.TestCase):
    def test_features_and_fields(self):
        quality_payload = read_json(QUALITY_PRIOR_PATH)
        quality = flat_quality(quality_payload)
        topology = load_topology("balanced", "dev")
        spec = load_split_specs("dev")["scenarios"]["balanced"]["episodes"][0]
        from env import ToyUAVEnv
        env = ToyUAVEnv()
        env.reset(
            seed=int(spec["seed"]),
            goal_distance_range_m=tuple(spec["distance_range"]),
            render_observation=False,
        )
        latency = Stage2KLatency()
        features, mask, metrics, fallback = no_visual_candidate_features(
            topology["nodes"], env.position, latency, quality
        )
        self.assertEqual(features.shape, (20, 17))
        self.assertEqual(mask.shape, (20,))
        self.assertTrue(mask.any())
        self.assertFalse(fallback)
        self.assertTrue(np.all(np.isfinite(features)))
        for index, node in enumerate(topology["nodes"]):
            available = bool(mask[index])
            self.assertEqual(
                available,
                float(features[index, 0]) == 1.0,
            )
            if not available:
                np.testing.assert_allclose(features[index, 6:10], 0.0)
                self.assertGreaterEqual(float(features[index, 10]), 0.0)
            else:
                metric = metrics[index]
                self.assertIsNotNone(metric)
                expected = [
                    metric["uplink_rate_bps"] / 200e6,
                    metric["communication_ms"] / 100.0,
                    min(metric["inference_ms"] / INFERENCE_SCALE_MS, 3.0),
                    min(metric["total_ms"] / 100.0, 3.0),
                    quality[node["model_type"]],
                    np.clip(metric["snr_db"] / 40.0, -1.0, 1.0),
                ]
                np.testing.assert_allclose(
                    features[index, 6:12], expected, rtol=1e-5, atol=1e-6
                )
                self.assertAlmostEqual(
                    float(features[index, 10]),
                    float(quality[node["model_type"]]),
                    places=6,
                )
            expected_one_hot = np.zeros(5, dtype=np.float32)
            expected_one_hot[int(node["model_index"])] = 1.0
            np.testing.assert_array_equal(features[index, 12:17],
                                          expected_one_hot)
            self.assertGreaterEqual(float(features[index, 11]), -1.0)
            self.assertLessEqual(float(features[index, 11]), 1.0)
        # SNR must actually vary across available nodes.
        available_snr = features[mask, 11]
        if len(available_snr) > 1:
            self.assertGreater(float(available_snr.std()), 0.0)

    def test_fallback_reenables_nearest_node(self):
        quality = flat_quality(read_json(QUALITY_PRIOR_PATH))
        topology = load_topology("balanced", "dev")
        features, mask, metrics, fallback = no_visual_candidate_features(
            topology["nodes"],
            np.asarray([1e9, 1e9], dtype=np.float64),
            Stage2KLatency(),
            quality,
        )
        self.assertTrue(fallback)
        self.assertEqual(int(mask.sum()), 1)
        index = int(np.argmax(mask))
        self.assertEqual(float(features[index, 0]), 1.0)
        self.assertIsNotNone(metrics[index])
        self.assertGreater(float(features[index, 9]), 0.0)


class NetworkTests(unittest.TestCase):
    def test_qnetwork_342(self):
        agent = DQNAgentV2(
            "cpu", state_dim=STATE_DIM_NO_VISUAL, state_mode="no_visual"
        )
        state = np.zeros(STATE_DIM_NO_VISUAL, dtype=np.float32)
        q = agent.q_values(state)
        self.assertEqual(q.shape, (20,))
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertEqual(agent.config()["state_dim"], 342)
        self.assertEqual(agent.config()["state_mode"], "no_visual")
        self.assertEqual(agent.online.net[0].in_features, 342)
        self.assertEqual(agent.buffer.state_dim, 342)


class RunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.quality = flat_quality(read_json(QUALITY_PRIOR_PATH))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.device, cls.encoder, cls.backends = load_runtime(device)
        cls.gab_config = load_gab_config()
        cls.specs = load_split_specs("dev")
        cls.topology = load_topology("balanced", "dev")

    def test_one_episode_state_and_reward(self):
        spec = self.specs["scenarios"]["balanced"]["episodes"][0]
        scene = build_scene(spec, self.encoder, self.device)
        runner = EpisodeRunner(
            spec=spec,
            topology=self.topology,
            encoder=self.encoder,
            backends=self.backends,
            projector=None,
            quality_prior=self.quality,
            gab_config=self.gab_config,
            scene=scene,
            device=self.device,
            jointr_aligned=True,
            no_visual_state=True,
        )
        self.assertEqual(len(runner.step_data["state"]), 342)
        self.assertEqual(
            runner.step_data["candidate_features"].shape, (20, 17)
        )
        uav = np.asarray(runner.step_data["pose"], dtype=np.float32)[:2]
        np.testing.assert_allclose(
            runner.step_data["state"][:2],
            uav / np.float32(UAV_POSITION_SCALE_M),
            rtol=1e-6, atol=1e-6,
        )
        while True:
            action = runner.select_node("min_latency")
            transition = runner.apply(action)
            self.assertEqual(transition["node_index"], action)
            if transition["done"]:
                break
        result = runner.finalize("jointr_v2")
        self.assertLess(result["jointr_abs_error"], 1e-6)
        self.assertEqual(len(runner.step_data["state"]), 342)


class FinalTestMathTests(unittest.TestCase):
    def test_paired_deltas(self):
        rows = [
            {"episode_uid": "e1", "strategy": "dqn", "success": 1,
             "ndtw": 0.8, "mean_e2e_ms": 100.0, "joint_reward": 0.55},
            {"episode_uid": "e1", "strategy": "heuristic", "success": 1,
             "ndtw": 0.7, "mean_e2e_ms": 120.0, "joint_reward": 0.40},
            {"episode_uid": "e2", "strategy": "dqn", "success": 0,
             "ndtw": 0.4, "mean_e2e_ms": 90.0, "joint_reward": 0.175},
            {"episode_uid": "e2", "strategy": "heuristic", "success": 1,
             "ndtw": 0.5, "mean_e2e_ms": 80.0, "joint_reward": 0.30},
        ]
        item = paired_deltas(rows, "heuristic")
        self.assertEqual(item["n"], 2)
        self.assertAlmostEqual(item["delta_SR"]["mean"], -0.5)
        self.assertAlmostEqual(item["delta_nDTW"]["mean"], 0.0)
        self.assertAlmostEqual(item["delta_E2E_ms"]["mean"], -5.0)
        self.assertAlmostEqual(item["delta_JointR"]["mean"], 0.0125)
        self.assertEqual((item["wins"], item["losses"], item["ties"]),
                         (1, 1, 0))

    def test_dev_freeze_is_skipped(self):
        self.assertIsNone(build_freeze("dev", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
