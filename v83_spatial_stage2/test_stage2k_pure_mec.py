"""Tests for the Stage-2k pure-MEC state and random training order."""

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
from v83_spatial_stage2.stage2k_jointr_v2_order import (
    order_statistics,
    random_training_order,
    stratified_training_order,
    verify_order,
)
from v83_spatial_stage2.stage2k_latency import Stage2KLatency
from v83_spatial_stage2.stage2k_pure_mec_state import (
    CANDIDATE_DIM,
    CANDIDATE_FIELDS,
    DECISION_DIM_PURE_MEC,
    MASK_DIM,
    NUM_NODES,
    STATE_DIM_PURE_MEC,
    UAV_POSITION_SCALE_M,
    build_pure_mec_state,
    pure_mec_candidate_features,
    state_layout_pure_mec,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)


class StateLayoutTests(unittest.TestCase):
    def test_dimension_and_layout(self):
        layout = state_layout_pure_mec()
        self.assertEqual(STATE_DIM_PURE_MEC, 322)
        self.assertEqual(CANDIDATE_DIM, 16)
        self.assertEqual(MASK_DIM, 20)
        self.assertEqual(DECISION_DIM_PURE_MEC, 342)
        self.assertEqual(layout["uav_position"], [0, 2])
        self.assertEqual(layout["candidates"], [2, 322])
        self.assertEqual(layout["total_mlp_input"], 322)
        self.assertEqual(layout["total_decision_state"], 342)
        self.assertEqual(layout["availability_mask"]["dim"], 20)
        self.assertTrue(
            layout["availability_mask"]["separate_from_mlp_input"]
        )
        self.assertEqual(len(CANDIDATE_FIELDS), 16)
        forbidden = " ".join(layout["forbidden"]).lower()
        for word in ("visual", "goal", "gab", "action", "step", "success"):
            self.assertIn(word, forbidden)

    def test_availability_is_mask_only(self):
        self.assertNotIn("availability", CANDIDATE_FIELDS)
        layout = state_layout_pure_mec()
        self.assertEqual(
            layout["availability_mask"]["dim"], MASK_DIM
        )

    def test_candidate_fields_exclude_forbidden(self):
        forbidden = (
            "visual", "goal", "history", "heatmap", "gab", "map",
            "action", "step", "oracle", "success", "previous",
        )
        for field in CANDIDATE_FIELDS:
            for word in forbidden:
                self.assertNotIn(word, field.lower())

    def test_builder_signatures_have_no_visual_inputs(self):
        candidate = inspect.signature(pure_mec_candidate_features)
        self.assertEqual(
            set(candidate.parameters),
            {"nodes", "uav_position", "latency_model", "quality_prior"},
        )
        builder = inspect.signature(build_pure_mec_state)
        self.assertEqual(
            set(builder.parameters),
            {"uav_position", "candidate20x16"},
        )

    def test_build_state(self):
        candidate = np.zeros((NUM_NODES, CANDIDATE_DIM), dtype=np.float32)
        candidate[3, 8] = 1.44
        candidate[3, 9] = 0.55
        candidate[3, 10] = 0.20
        state = build_pure_mec_state(
            [UAV_POSITION_SCALE_M / 2, UAV_POSITION_SCALE_M / 4],
            candidate,
        )
        self.assertEqual(state.shape, (322,))
        self.assertAlmostEqual(float(state[0]), 0.5, places=6)
        self.assertAlmostEqual(float(state[1]), 0.25, places=6)
        self.assertAlmostEqual(
            float(state[2 + 3 * 16 + 8]), 1.44, places=5
        )
        self.assertAlmostEqual(
            float(state[2 + 3 * 16 + 9]), 0.55, places=5
        )
        self.assertAlmostEqual(
            float(state[2 + 3 * 16 + 10]), 0.20, places=5
        )


class CandidateFeatureTests(unittest.TestCase):
    def test_features_and_mask(self):
        quality = flat_quality(read_json(QUALITY_PRIOR_PATH))
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
        features, mask, metrics, fallback = pure_mec_candidate_features(
            topology["nodes"], env.position, latency, quality
        )
        self.assertEqual(features.shape, (20, 16))
        self.assertEqual(mask.shape, (20,))
        self.assertTrue(mask.any())
        self.assertFalse(fallback)
        self.assertTrue(np.all(np.isfinite(features)))
        for index, node in enumerate(topology["nodes"]):
            available = bool(mask[index])
            if not available:
                np.testing.assert_allclose(features[index, 5:9], 0.0)
                self.assertGreaterEqual(float(features[index, 9]), 0.0)
            else:
                metric = metrics[index]
                self.assertIsNotNone(metric)
                expected = [
                    metric["uplink_rate_bps"] / 200e6,
                    metric["communication_ms"] / 100.0,
                    min(metric["inference_ms"] / 10.0, 3.0),
                    min(metric["total_ms"] / 100.0, 3.0),
                    quality[node["model_type"]],
                    np.clip(metric["snr_db"] / 40.0, -1.0, 1.0),
                ]
                np.testing.assert_allclose(
                    features[index, 5:11], expected,
                    rtol=1e-5, atol=1e-6,
                )
            expected_one_hot = np.zeros(5, dtype=np.float32)
            expected_one_hot[int(node["model_index"])] = 1.0
            np.testing.assert_array_equal(
                features[index, 11:16], expected_one_hot
            )
            self.assertGreaterEqual(float(features[index, 10]), -1.0)
            self.assertLessEqual(float(features[index, 10]), 1.0)
        available_snr = features[mask, 10]
        if len(available_snr) > 1:
            self.assertGreater(float(available_snr.std()), 0.0)

    def test_fallback_reenables_nearest_node(self):
        quality = flat_quality(read_json(QUALITY_PRIOR_PATH))
        topology = load_topology("balanced", "dev")
        features, mask, metrics, fallback = pure_mec_candidate_features(
            topology["nodes"],
            np.asarray([1e9, 1e9], dtype=np.float64),
            Stage2KLatency(),
            quality,
        )
        self.assertTrue(fallback)
        self.assertEqual(int(mask.sum()), 1)
        index = int(np.argmax(mask))
        self.assertIsNotNone(metrics[index])
        self.assertGreater(float(features[index, 8]), 0.0)


class NetworkTests(unittest.TestCase):
    def test_qnetwork_322(self):
        agent = DQNAgentV2(
            "cpu", state_dim=STATE_DIM_PURE_MEC, state_mode="pure_mec"
        )
        state = np.zeros(STATE_DIM_PURE_MEC, dtype=np.float32)
        q = agent.q_values(state)
        self.assertEqual(q.shape, (20,))
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertEqual(agent.config()["state_dim"], 322)
        self.assertEqual(agent.config()["state_mode"], "pure_mec")
        self.assertEqual(agent.online.net[0].in_features, 322)
        self.assertEqual(agent.buffer.state_dim, 322)


class RandomOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_split_specs("train")
        cls.order, cls.targets = random_training_order(
            cls.train, total=3500
        )
        cls.lookup = verify_order(
            cls.order, cls.train, require_full=False
        )
        cls.dev = load_split_specs("dev")
        cls.test = load_split_specs("test")

    def test_counts_and_subset(self):
        self.assertEqual(len(self.order), 3500)
        self.assertEqual(len(set(self.order)), 3500)
        from collections import Counter
        bands = Counter(
            self.lookup[uid]["band"] for uid in self.order
        )
        self.assertEqual(
            dict(bands), {"short": 1167, "medium": 1167, "long": 1166}
        )
        scenarios = Counter(
            self.lookup[uid]["scenario"] for uid in self.order
        )
        self.assertEqual(
            dict(scenarios),
            {"balanced": 1167, "quality_rich": 1167,
             "tradeoff_stress": 1166},
        )
        train_uids = {
            item["episode_uid"]
            for scenario in self.train["scenarios"].values()
            for item in scenario["episodes"]
        }
        dev_uids = {
            item["episode_uid"]
            for scenario in self.dev["scenarios"].values()
            for item in scenario["episodes"]
        }
        test_uids = {
            item["episode_uid"]
            for scenario in self.test["scenarios"].values()
            for item in scenario["episodes"]
        }
        self.assertTrue(set(self.order).issubset(train_uids))
        self.assertFalse(set(self.order) & dev_uids)
        self.assertFalse(set(self.order) & test_uids)

    def test_random_not_round_robin(self):
        deterministic = stratified_training_order(self.train)[:3500]
        self.assertNotEqual(self.order, deterministic)
        from collections import Counter
        first = Counter(
            self.lookup[uid]["band"] for uid in self.order[:200]
        )
        self.assertEqual(set(first), {"short", "medium", "long"})
        for name in ("short", "medium", "long"):
            self.assertGreater(first[name], 30)
            self.assertLess(first[name], 100)
        # The sequence must not be the fixed S,M,L,S,M,L,... cycle.
        first_nine = [
            self.lookup[uid]["band"] for uid in self.order[:9]
        ]
        self.assertNotEqual(
            first_nine,
            ["short", "medium", "long"] * 3,
        )

    def test_reproducible(self):
        order2, targets2 = random_training_order(
            self.train, total=3500
        )
        self.assertEqual(self.order, order2)
        self.assertEqual(
            {k: v for k, v in self.targets.items()},
            {k: v for k, v in targets2.items()},
        )

    def test_window_statistics(self):
        stats = order_statistics(self.order, self.lookup)
        self.assertEqual(stats["episodes"], 3500)
        self.assertEqual(len(stats["windows"]), 17)
        self.assertEqual(
            stats["counts"]["balanced/short"], 389
        )
        self.assertEqual(
            stats["counts"]["tradeoff_stress/long"], 388
        )
        for window in stats["windows"]:
            bands = {
                name: sum(
                    window["counts"][f"{scenario}/{name}"]
                    for scenario in (
                        "balanced", "quality_rich", "tradeoff_stress"
                    )
                )
                for name in ("short", "medium", "long")
            }
            self.assertTrue(all(value > 0 for value in bands.values()))


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
            pure_mec_state=True,
        )
        self.assertEqual(len(runner.step_data["state"]), 322)
        self.assertEqual(
            runner.step_data["candidate_features"].shape, (20, 16)
        )
        self.assertEqual(runner.step_data["mask"].shape, (20,))
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
        self.assertEqual(len(runner.step_data["state"]), 322)


if __name__ == "__main__":
    unittest.main(verbosity=2)
