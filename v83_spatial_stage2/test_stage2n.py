"""Stage-2n unit and integration tests.

Run after ``stage2n_build.py``:

    python v83_spatial_stage2/test_stage2n.py

The tests cover task generation/sealing, seed isolation, homogeneous
deployment, checkpoint provenance, the latency-only reward identity, the
384-D state and 20-D mask, the convergence rule, checkpoint selection, and a
real one-episode closed loop.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json
from v83_spatial_stage2.stage2n_common import (
    ACTION_DIM,
    AGENT_SEEDS,
    BAND_COUNTS_DEV,
    BAND_COUNTS_TEST,
    BAND_COUNTS_TRAIN,
    MEC_CHECKPOINT_SHA256,
    MEC_MODEL,
    PROTOCOL_PATH,
    STAGE2N_ROOT,
    STATE_DIM,
    STATE_MODE,
    build_homogeneous_deployment,
    canonical_sha256,
    convergence_status,
    deployment_topology,
    dev_specs,
    generate_split_tasks,
    isolation_audit,
    latency_return_from_mean,
    latency_rewards,
    latency_reward_increment,
    load_deployment,
    load_task_pool,
    load_training_order,
    prepare_runner_spec,
    select_best_validation,
    sha256_file,
    verify_mec_checkpoint,
    verify_task_hash,
)
from v83_spatial_stage2.stage2n_runtime import (
    HomogeneousBackendPool,
    LatencyEvaluator,
    LatencyOnlyEpisodeRunner,
)
from v83_spatial_stage2.stage2n_train import load_bundle, verify_frozen_protocol


class TaskPoolTests(unittest.TestCase):
    def test_counts_and_bands(self):
        pools = {
            split: load_task_pool(
                split, allow_test=(split == "test")
            ) for split in ("train", "dev", "test")
        }
        expected = {
            "train": BAND_COUNTS_TRAIN,
            "dev": BAND_COUNTS_DEV,
            "test": BAND_COUNTS_TEST,
        }
        for split, pool in pools.items():
            self.assertEqual(pool["num_tasks"], sum(expected[split].values()))
            self.assertEqual(pool["band_counts"], expected[split])
            for task in pool["tasks"]:
                self.assertTrue(verify_task_hash(task))
                self.assertGreaterEqual(task["oracle_steps"], 1)
                self.assertTrue(
                    task["distance_range"][0]
                    <= task["start_goal_distance_m"]
                    <= task["distance_range"][1]
                )

    def test_test_is_sealed_by_default(self):
        with self.assertRaises(PermissionError):
            load_task_pool("test")
        pool = load_task_pool("test", allow_test=True)
        self.assertTrue(pool["sealed"])

    def test_seed_and_uid_isolation(self):
        train = load_task_pool("train")["tasks"]
        dev = load_task_pool("dev")["tasks"]
        test = load_task_pool("test", allow_test=True)["tasks"]
        audit = isolation_audit(train, dev, test)
        self.assertTrue(audit["checks"]["ok"], audit["checks"])
        self.assertTrue(audit["checks"]["test_sealed"])

    def test_same_map_and_fingerprint(self):
        pools = [
            load_task_pool("train"),
            load_task_pool("dev"),
            load_task_pool("test", allow_test=True),
        ]
        fingerprints = {
            item["tasks"][0]["city_fingerprint"] for item in pools
        }
        self.assertEqual(len(fingerprints), 1)
        self.assertEqual(
            {item["map_seed"] for item in pools}, {pools[0]["map_seed"]}
        )

    def test_training_order_covers_pool_once(self):
        order = load_training_order()
        pool = load_task_pool("train")
        self.assertEqual(len(order["order"]), pool["num_tasks"])
        self.assertEqual(len(set(order["order"])), len(order["order"]))
        self.assertEqual(
            set(order["order"]),
            {task["episode_uid"] for task in pool["tasks"]},
        )
        self.assertEqual(order["task_pool_sha256"],
                         canonical_sha256(pool["tasks"]))

    def test_small_generation_uses_same_rules(self):
        specs = generate_split_tasks(
            "train", {"short": 2, "medium": 2, "long": 2}
        )
        self.assertEqual(len(specs), 6)
        self.assertEqual([item["band"] for item in specs],
                         ["short", "short", "medium", "medium",
                          "long", "long"])
        for task in specs:
            self.assertTrue(verify_task_hash(task))
            band = task["band"]
            self.assertGreaterEqual(task["oracle_steps"], 1)
            self.assertEqual(task["map_seed"],
                             load_task_pool("train")["map_seed"])

    def test_cross_stage_disjoint(self):
        path = Path(
            "runs/v83_spatial_stage2/stage2n_latency_only/specs/"
            "cross_stage_isolation.json"
        )
        payload = read_json(path)
        self.assertIsNotNone(payload)
        self.assertTrue(payload["ok"], payload)


class DeploymentTests(unittest.TestCase):
    def test_homogeneous_deployment(self):
        payload = load_deployment()
        checks = payload["homogeneity_checks"]
        self.assertTrue(checks["ok"], checks)
        self.assertEqual(len(payload["nodes"]), 20)
        self.assertEqual(
            {node["model_type"] for node in payload["nodes"]},
            {MEC_MODEL},
        )
        self.assertEqual(len({
            (node["x"], node["y"]) for node in payload["nodes"]
        }), 20)
        for key in (
            "compute_capacity_flops_per_s", "bandwidth_hz",
            "coverage_radius_m", "quality_prior",
        ):
            self.assertEqual(
                len({node[key] for node in payload["nodes"]}), 1
            )

    def test_positions_match_stage2l_a_fixed(self):
        deployment = load_deployment()
        source = read_json(Path(
            "runs/v83_spatial_stage2/stage2l_fixed_city/topologies/"
            "A_fixed.json"
        ))
        self.assertIsNotNone(source)
        self.assertEqual(
            [(round(node["x"], 6), round(node["y"], 6))
             for node in deployment["nodes"]],
            [(round(node["x"], 6), round(node["y"], 6))
             for node in source["nodes"]],
        )

    def test_deployment_hash_is_stable(self):
        deployment = load_deployment()
        expected = canonical_sha256({
            "nodes": deployment["nodes"],
            "mec": deployment["mec"],
            "source_topology_sha256": deployment[
                "source_topology_sha256"
            ],
        })
        self.assertEqual(deployment["deployment_sha256"], expected)
        rebuilt = build_homogeneous_deployment(verbose=False)
        self.assertEqual(
            rebuilt["deployment_sha256"], deployment["deployment_sha256"]
        )
        self.assertEqual(
            [n["node_id"] for n in rebuilt["nodes"]],
            [n["node_id"] for n in deployment["nodes"]],
        )

    def test_topology_view(self):
        topology = deployment_topology()
        self.assertEqual(len(topology["nodes"]), 20)
        self.assertEqual(
            topology["deployment_sha256"],
            load_deployment()["deployment_sha256"],
        )


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_hash_and_records(self):
        record = verify_mec_checkpoint()
        self.assertEqual(record["checkpoint_sha256"], MEC_CHECKPOINT_SHA256)
        self.assertEqual(record["model"], MEC_MODEL)
        self.assertAlmostEqual(record["dev_val_180"]["sr"], 0.5722222222,
                               places=9)
        self.assertAlmostEqual(
            record["final_test_300_reference"]["sr"], 0.5533333333,
            places=9,
        )
        self.assertFalse(record["final_test_read"])

    def test_same_observation_same_action(self):
        pool = HomogeneousBackendPool(torch.device("cpu"))
        inputs = {
            "current_tokens": torch.randn(1, 576, 384),
            "goal_tokens": torch.randn(1, 3, 384),
            "history_vis": torch.randn(1, 16, 768),
            "history_pose": torch.randn(1, 16, 10),
            "history_action": torch.randn(1, 16, 4),
            "history_mask": torch.ones(1, 16),
            "gab": torch.randn(1, 20),
            "map_feat": torch.randn(1, 22),
        }
        first = pool.get(MEC_MODEL, inputs)
        second = pool.get(MEC_MODEL, inputs)
        for key in ("action", "arrival_probability", "move_action"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(list(pool.load_calls), [MEC_MODEL])
        with self.assertRaises(ValueError):
            pool.get("Large", inputs)


class RewardTests(unittest.TestCase):
    def test_latency_reward_telescopes(self):
        rng = np.random.default_rng(7)
        delays = rng.uniform(20.0, 300.0, size=137)
        rewards, mean = latency_rewards(delays)
        closed = latency_return_from_mean(mean)
        self.assertAlmostEqual(float(rewards.sum()), closed, places=12)
        self.assertAlmostEqual(mean, float(delays.mean()), places=12)

    def test_first_step_reward(self):
        mean, reward = latency_reward_increment(200.0, 0.0, 1)
        self.assertAlmostEqual(mean, 200.0)
        self.assertAlmostEqual(reward, -0.5)

    def test_invalid_step(self):
        with self.assertRaises(ValueError):
            latency_reward_increment(100.0, 0.0, 0)
        with self.assertRaises(ValueError):
            latency_reward_increment(float("nan"), 0.0, 1)


class ConvergenceTests(unittest.TestCase):
    def test_stable_is_converged(self):
        rows = [
            {"episode": 200 * (index + 1), "E2E_ms": value}
            for index, value in enumerate([100.0, 99.0, 98.5, 98.6, 98.4])
        ]
        result = convergence_status(rows, reached_cap=True)
        self.assertTrue(result["converged"])
        self.assertEqual(result["status"], "converged")

    def test_unstable_is_not_converged(self):
        rows = [
            {"episode": 200 * (index + 1), "E2E_ms": value}
            for index, value in enumerate([100.0, 90.0, 110.0, 95.0, 105.0])
        ]
        result = convergence_status(rows, reached_cap=True)
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "not_converged")

    def test_checkpoint_selection(self):
        rows = [
            {"episode": 200, "E2E_ms": 100.0, "SR": 0.5},
            {"episode": 400, "E2E_ms": 98.0, "SR": 0.4},
            {"episode": 600, "E2E_ms": 98.0, "SR": 0.6},
        ]
        self.assertEqual(select_best_validation(rows)["episode"], 600)


class FrozenLaneEquivalenceTests(unittest.TestCase):
    """The cached MEC lane must reproduce the full closed loop exactly."""

    @classmethod
    def setUpClass(cls):
        from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
        from v83_spatial_stage2.stage2n_lanes import (
            LaneEvaluator,
            build_lanes,
        )
        from v83_spatial_stage2.stage2n_runtime import (
            LatencyEvaluator,
            load_stage2n_runtime,
            make_stage2n_scene_cache,
        )

        cls.bundle = load_bundle()
        cls.specs = [
            prepare_runner_spec(item) for item in cls.bundle["dev_specs"][:2]
        ]
        cls.device, encoder, backends = load_stage2n_runtime("auto")
        cls.scene_cache = make_stage2n_scene_cache(
            encoder, cls.device, capacity=8
        )
        cls.full = LatencyEvaluator(
            specs=cls.specs,
            topology=cls.bundle["topology"],
            encoder=encoder,
            backends=backends,
            quality_prior=cls.bundle["quality_prior"],
            gab_config=cls.bundle["gab_config"],
            device=cls.device,
            scene_cache=cls.scene_cache,
        )
        lanes = build_lanes(
            cls.specs,
            cls.bundle["topology"],
            encoder,
            backends,
            cls.bundle["quality_prior"],
            cls.bundle["gab_config"],
            cls.device,
            cls.scene_cache,
        )
        cls.lane = LaneEvaluator(lanes)
        cls.agent = DQNAgentV2(
            cls.device, seed=0, gamma=1.0,
            state_dim=STATE_DIM, state_mode=STATE_MODE,
        )

    @staticmethod
    def _compare(test, full_rows, lane_rows):
        for full, cached in zip(full_rows, lane_rows):
            for key in (
                "success", "osr", "steps", "action_sha256",
            ):
                test.assertEqual(full[key], cached[key], key)
            for key in (
                "spl", "ndtw", "mean_e2e_ms", "p95_e2e_ms",
                "delay_violation_rate", "latency_return",
            ):
                test.assertAlmostEqual(
                    float(full[key]), float(cached[key]), places=9,
                    msg=key,
                )

    def test_heuristics_match_full_closed_loop(self):
        for strategy in (
            "random", "nearest", "max_uplink_rate", "greedy_min_e2e"
        ):
            self._compare(
                self,
                self.full.evaluate_strategy(strategy),
                self.lane.evaluate_strategy(strategy),
            )

    def test_dqn_matches_full_closed_loop(self):
        self._compare(
            self,
            self.full.evaluate_dqn(self.agent),
            self.lane.evaluate_dqn(self.agent),
        )


class ProtocolAndStateTests(unittest.TestCase):
    def test_frozen_protocol_hashes(self):
        protocol = verify_frozen_protocol(verify_test_file=True)
        self.assertFalse(protocol["final_test_read"])
        self.assertTrue(protocol["test_sealed"])
        self.assertEqual(
            protocol["mec"]["checkpoint_sha256"], MEC_CHECKPOINT_SHA256
        )

    def test_trainer_bundle_has_no_test_pool(self):
        bundle = load_bundle()
        self.assertEqual(len(bundle["train_specs"]), 4000)
        self.assertEqual(len(bundle["dev_specs"]), 60)
        self.assertNotIn("test_specs", bundle)
        self.assertNotIn("test_tasks", bundle)
        for key in bundle["train_specs"][0]:
            self.assertNotIn("test", key.lower())

    def test_state_and_mask_shapes(self):
        bundle = load_bundle()
        spec = prepare_runner_spec(bundle["dev_specs"][0])
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        from v83_spatial_stage2.stage2n_runtime import (
            load_stage2n_runtime,
            make_stage2n_scene_cache,
        )
        device, encoder, backends = load_stage2n_runtime(device)
        cache = make_stage2n_scene_cache(encoder, device, capacity=4)
        runner = LatencyOnlyEpisodeRunner(
            spec=spec,
            topology=bundle["topology"],
            encoder=encoder,
            backends=backends,
            projector=None,
            quality_prior=bundle["quality_prior"],
            gab_config=bundle["gab_config"],
            scene_cache=cache,
            device=device,
        )
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(mask.shape, (ACTION_DIM,))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask.any())
        candidate = runner.step_data["candidate_features"]
        self.assertEqual(candidate.shape, (20, 19))
        unavailable = ~mask
        self.assertTrue(np.all(candidate[unavailable][:, 5:10] == 0.0))
        # The MEC still receives the full visual contract.
        inputs = runner.step_data["inputs"]
        self.assertEqual(tuple(inputs["current_tokens"].shape), (1, 576, 384))
        self.assertEqual(tuple(inputs["goal_tokens"].shape), (1, 3, 384))
        self.assertEqual(tuple(inputs["history_vis"].shape), (1, 16, 768))
        self.assertEqual(tuple(inputs["history_pose"].shape), (1, 16, 10))
        self.assertEqual(tuple(inputs["history_action"].shape), (1, 16, 4))
        self.assertEqual(tuple(inputs["history_mask"].shape), (1, 16))
        self.assertEqual(tuple(inputs["gab"].shape), (1, 20))
        self.assertEqual(tuple(inputs["map_feat"].shape), (1, 22))

        # One full closed-loop episode with a deterministic heuristic.
        evaluator = LatencyEvaluator(
            specs=[spec],
            topology=bundle["topology"],
            encoder=encoder,
            backends=backends,
            quality_prior=bundle["quality_prior"],
            gab_config=bundle["gab_config"],
            device=device,
            scene_cache=cache,
        )
        rows = evaluator.evaluate_strategy("greedy_min_e2e")
        row = rows[0]
        self.assertEqual(row["reward_mode"], "latency_only_v1")
        self.assertLess(row["latency_return_abs_error"], 1e-9)
        self.assertAlmostEqual(
            row["latency_return"],
            latency_return_from_mean(row["mean_e2e_ms"]),
            places=9,
        )
        self.assertGreater(len(row["actions"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
