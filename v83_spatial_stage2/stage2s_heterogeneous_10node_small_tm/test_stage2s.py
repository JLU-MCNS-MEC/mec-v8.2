"""CPU tests for the Stage-2s heterogeneous 10-node contract.

Run:
    python -m unittest -v \
        v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s

GPU closed-loop tests live in ``test_stage2s_gpu.py``.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import MODEL_NAMES  # noqa: E402
from v83_spatial_stage2.stage2k_latency import (  # noqa: E402
    Stage2KLatency,
    stage2j_model_flops,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (  # noqa: E402
    BLOCK_E2E,
    BLOCK_INFERENCE,
    BLOCK_MODEL_ONE_HOT,
    CANDIDATE_DIM,
    build_pure_mec_v2_state,
    pure_mec_v2_candidate_features,
)
from v83_spatial_stage2.stage2o.rewards import (  # noqa: E402
    EpisodeRewardController,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm import (  # noqa: E402
    common,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.runtime import (  # noqa: E402
    HeterogeneousLatency,
    latency_stats,
    scheme_d_return,
)


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = common.load_deployment()
        cls.nodes = cls.deployment["nodes"]

    def test_a10_geometry_matches_stage2r(self):
        self.assertEqual(
            tuple(common.select_a10_indices()),
            common.A10_SOURCE_INDICES,
        )
        source = common.read_json(common.DEPLOYMENT_SOURCE)
        for node in self.nodes:
            source_node = source["nodes"][node["source_index"]]
            self.assertAlmostEqual(node["x"], source_node["x"], places=9)
            self.assertAlmostEqual(node["y"], source_node["y"], places=9)

    def test_five_small_five_transformer_m(self):
        counts = {
            model: sum(
                1 for node in self.nodes if node["model_type"] == model
            )
            for model in common.MEC_MODELS
        }
        self.assertEqual(counts, {"Small": 5, "Transformer-M": 5})
        self.assertEqual(len({(node["x"], node["y"])
                              for node in self.nodes}), 10)
        self.assertTrue(self.deployment["checks"]["ok"])

    def test_nominal_and_effective_compute_are_separate(self):
        for node in self.nodes:
            self.assertEqual(
                node["compute_capacity_flops_per_s"],
                common.MEC_NOMINAL_COMPUTE_FLOPS_PER_S,
            )
            self.assertEqual(
                node["effective_inference_compute_flops_per_s"],
                common.MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S,
            )
        mec = self.deployment["mec"]
        self.assertEqual(
            mec["compute_capacity_flops_per_s"], 2.0e12
        )
        self.assertEqual(
            mec["effective_inference_compute_flops_per_s"], 0.2e12
        )

    def test_deployment_hash_and_spatial_balance(self):
        expected = common.canonical_sha256({
            "nodes": self.deployment["nodes"],
            "mec": self.deployment["mec"],
            "source_topology_sha256": self.deployment[
                "source_topology_sha256"
            ],
            "assignment_rule": common.ASSIGNMENT_RULE,
        })
        self.assertEqual(
            self.deployment["deployment_sha256"], expected
        )
        balance = self.deployment["assignment"]["balance"]
        self.assertLessEqual(balance["grid_coverage_gap"], 0.10)
        self.assertGreaterEqual(balance["grid_coverage_min"], 0.80)
        self.assertLessEqual(balance["start_coverage_gap"], 0.10)
        self.assertGreaterEqual(balance["start_coverage_min"], 0.80)
        self.assertLessEqual(balance["mean_nearest_gap_m"], 50.0)
        centre = balance["centre_nearest_node_index"]
        self.assertEqual(self.nodes[centre]["model_type"], "Small")

    def test_model_indices_match_global_order(self):
        for node in self.nodes:
            self.assertEqual(
                node["model_index"], MODEL_NAMES.index(node["model_type"])
            )


class LatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = common.load_deployment()
        cls.latency = HeterogeneousLatency()
        cls.node = {
            "x": 750.0,
            "y": 750.0,
            "bandwidth_hz": common.MEC_BANDWIDTH_HZ,
            "compute_capacity_flops_per_s": (
                common.MEC_NOMINAL_COMPUTE_FLOPS_PER_S
            ),
            "coverage_radius_m": common.MEC_COVERAGE_RADIUS_M,
        }

    def test_effective_throughput_gives_expected_inference(self):
        for model in common.MEC_MODELS:
            metric = self.latency.evaluate(
                self.node, (750.0, 750.0), model
            )
            expected = (
                1000.0 * stage2j_model_flops(model) / 0.2e12
            )
            self.assertAlmostEqual(
                metric["inference_ms"], expected, places=9
            )
            self.assertAlmostEqual(
                metric["total_ms"],
                metric["extraction_ms"]
                + metric["communication_ms"]
                + metric["inference_ms"],
                places=9,
            )
            self.assertEqual(
                metric["nominal_compute_capacity_flops_per_s"], 2.0e12
            )
            self.assertEqual(
                metric["effective_inference_compute_flops_per_s"], 0.2e12
            )

    def test_transformer_m_inference_far_slower_than_small(self):
        small = self.latency.evaluate(
            self.node, (750.0, 750.0), "Small"
        )
        large = self.latency.evaluate(
            self.node, (750.0, 750.0), "Transformer-M"
        )
        ratio = large["inference_ms"] / small["inference_ms"]
        self.assertGreater(ratio, 100.0)
        self.assertGreater(
            large["total_ms"] - small["total_ms"], 10.0
        )

    def test_nominal_stage2k_latency_is_not_used_for_inference(self):
        nominal = Stage2KLatency().evaluate(
            self.node, (750.0, 750.0), "Transformer-M"
        )
        effective = self.latency.evaluate(
            self.node, (750.0, 750.0), "Transformer-M"
        )
        self.assertAlmostEqual(
            nominal["inference_ms"], 1.407297824, places=6
        )
        self.assertAlmostEqual(
            effective["inference_ms"], 14.07297824, places=6
        )
        self.assertEqual(
            effective["compute_capacity_flops_per_s"],
            nominal["compute_capacity_flops_per_s"],
        )

    def test_state_features_include_effective_inference_and_e2e(self):
        deployment = common.load_deployment()
        prior, flat = common._load_prior()
        candidate, mask, metrics, fallback = (
            pure_mec_v2_candidate_features(
                deployment["nodes"], (750.0, 750.0), self.latency,
                capability_prior=flat, num_nodes=10,
            )
        )
        self.assertEqual(candidate.shape, (10, CANDIDATE_DIM))
        self.assertEqual(mask.shape, (10,))
        self.assertFalse(fallback)
        for index, metric in enumerate(metrics):
            if metric is None:
                continue
            self.assertAlmostEqual(
                float(candidate[index, BLOCK_INFERENCE]),
                min(metric["inference_ms"] / 10.0, 3.0),
                places=6,
            )
            self.assertAlmostEqual(
                float(candidate[index, BLOCK_E2E]),
                min(metric["total_ms"] / 100.0, 3.0),
                places=6,
            )
        state = build_pure_mec_v2_state(
            (750.0, 750.0), 0, 0.0, candidate
        )
        self.assertEqual(state.shape, (common.STATE_DIM,))
        self.assertEqual(common.STATE_DIM, 194)

    def test_model_one_hot_and_capability_prior(self):
        deployment = common.load_deployment()
        prior, flat = common._load_prior()
        candidate, _, metrics, _ = pure_mec_v2_candidate_features(
            deployment["nodes"], (750.0, 750.0), self.latency,
            capability_prior=flat, num_nodes=10,
        )
        for index, node in enumerate(deployment["nodes"]):
            one_hot = candidate[index, BLOCK_MODEL_ONE_HOT]
            self.assertEqual(int(np.argmax(one_hot)), node["model_index"])
            self.assertAlmostEqual(
                float(one_hot.sum()), 1.0, places=6
            )


class RewardTests(unittest.TestCase):
    def test_scheme_d_identity_synthetic(self):
        controller = EpisodeRewardController("D")
        delays = [41.0, 58.5, 73.25, 99.0]
        for delay in delays:
            controller.step_reward(delay)
        mean = float(np.mean(delays))
        terminal_reward, diagnostics = controller.finalize(
            mean, len(delays)
        )
        expected = 1.0 - mean / 100.0
        self.assertAlmostEqual(
            diagnostics["return"], expected, places=12
        )
        self.assertAlmostEqual(
            diagnostics["expected_return"], expected, places=12
        )
        self.assertAlmostEqual(diagnostics["abs_error"], 0.0, places=12)
        self.assertAlmostEqual(
            diagnostics["terminal_reward"], terminal_reward, places=12
        )
        self.assertAlmostEqual(scheme_d_return(mean), expected, places=12)

    def test_latency_stats_counts_both_models(self):
        records = [
            {
                "model_type": "Small", "inference_ms": 0.071,
                "communication_ms": 12.0, "extraction_ms": 22.7,
                "e2e_ms": 34.8,
            },
            {
                "model_type": "Transformer-M", "inference_ms": 14.07,
                "communication_ms": 20.0, "extraction_ms": 22.7,
                "e2e_ms": 56.8,
            },
        ]
        stats = latency_stats(records)
        self.assertEqual(stats["small_steps"], 1)
        self.assertEqual(stats["transformer_m_steps"], 1)
        self.assertAlmostEqual(stats["small_ratio"], 0.5)
        self.assertAlmostEqual(stats["transformer_m_ratio"], 0.5)
        self.assertAlmostEqual(stats["mean_inference_small_ms"], 0.071)
        self.assertAlmostEqual(
            stats["mean_inference_transformer_m_ms"], 14.07
        )

    def test_no_success_or_navigation_term_in_reward(self):
        source = inspect.getsource(EpisodeRewardController)
        self.assertNotIn("success", source)
        self.assertNotIn("ndtw", source.lower())
        self.assertNotIn("nav_reward", source)


class IsolationTests(unittest.TestCase):
    def test_bundle_never_reads_test(self):
        bundle = common.load_bundle()
        self.assertFalse(bundle["test_read"])
        self.assertEqual(len(bundle["train_specs"]), 4000)
        self.assertEqual(len(bundle["dev_specs"]), 60)
        self.assertNotIn("test_specs", bundle)
        self.assertNotIn("test_pool", bundle)
        self.assertFalse(common.load_deployment()["final_test_read"])

    def test_common_source_has_no_test_pool_call(self):
        source = inspect.getsource(common)
        self.assertNotIn('load_task_pool("test"', source)
        self.assertNotIn("load_task_pool('test'", source)

    def test_training_is_closed_loop_without_lane_cache(self):
        from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm import (
            train,
        )

        source = inspect.getsource(train)
        self.assertIn("full_closed_loop_heterogeneous", source)
        self.assertNotIn("LaneRunnerAdapter", source)
        self.assertNotIn("load_lane_cache", source)
        self.assertNotIn("lane_cache.py", source)

    def test_task_coverage_and_bands(self):
        bundle = common.load_bundle()
        nodes = np.asarray([
            [node["x"], node["y"]]
            for node in bundle["deployment"]["nodes"]
        ])
        starts = np.asarray([
            spec["start"]
            for spec in bundle["train_specs"] + bundle["dev_specs"]
        ])
        distances = np.linalg.norm(
            starts[:, None, :] - nodes[None, :, :], axis=2
        ).min(axis=1)
        self.assertEqual(int((distances > 500.0).sum()), 0)
        for band, count in common.TRAIN_BAND_COUNTS.items():
            self.assertEqual(
                sum(1 for spec in bundle["train_specs"]
                    if spec["band"] == band),
                count,
            )
        for band, count in common.DEV_BAND_COUNTS.items():
            self.assertEqual(
                sum(1 for spec in bundle["dev_specs"]
                    if spec["band"] == band),
                count,
            )


if __name__ == "__main__":
    unittest.main()
