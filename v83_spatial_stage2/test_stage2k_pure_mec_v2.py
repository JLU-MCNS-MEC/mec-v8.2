"""Tests for the Stage-2k v2 Pure MEC and visual-ablation state."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    VISUAL_PCA_PATH,
    load_topology,
)
from v83_spatial_stage2.stage2k_latency import Stage2KLatency
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    AUX_SUMMARY_DIM,
    CANDIDATE_DIM,
    CANDIDATE_FIELDS,
    GLOBAL_DIM,
    MASK_DIM,
    NUM_NODES,
    PCA_SUMMARY_DIM,
    STATE_DIM_PURE_MEC_V2,
    STATE_DIM_VISUAL_ABLATION_V2,
    VISUAL_DIM,
    VISUAL_SUMMARY_DIM,
    build_pure_mec_v2_state,
    build_visual_ablation_v2_state,
    pure_mec_v2_candidate_features,
    state_layout_pure_mec_v2,
    state_layout_visual_ablation_v2,
    visual_context_from_projector,
)
from v83_spatial_stage2.stage2k_state import (
    VisualProjector,
    visual_summary,
)
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_FIELDS,
    CAPABILITY_METRICS,
    CAPABILITY_PRIOR_PATH,
    load_capability_prior,
    validate_capability_prior,
)


class DummyLatency:
    def evaluate(self, node, uav_position, model_type):
        return {
            "uplink_rate_bps": 120e6,
            "communication_ms": 8.0,
            "inference_ms": 2.5,
            "total_ms": 20.0,
            "snr_db": 12.0,
        }


def dummy_nodes():
    nodes = []
    for index in range(NUM_NODES):
        name = MODEL_NAMES[index % len(MODEL_NAMES)]
        nodes.append({
            "node_id": f"n{index}",
            "x": float(index) * 40.0,
            "y": float((index % 4) * 30.0),
            "coverage_radius_m": 1000.0,
            "compute_capacity_flops_per_s": 1.5e12,
            "bandwidth_hz": 25e6,
            "model_type": name,
            "model_index": MODEL_NAMES.index(name),
        })
    return nodes


def fake_projector(seed=7):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 0.1, size=PCA_SUMMARY_DIM).astype(np.float32)
    std = np.ones(PCA_SUMMARY_DIM, dtype=np.float32)
    components = rng.normal(
        0.0, 1.0 / np.sqrt(PCA_SUMMARY_DIM),
        size=(VISUAL_DIM, PCA_SUMMARY_DIM),
    ).astype(np.float32)
    return VisualProjector(
        mean=mean, std=std, components=components, meta={}
    )


class StateDimensionTest(unittest.TestCase):
    def test_dimensions_computed_from_fields(self):
        self.assertEqual(len(CANDIDATE_FIELDS), CANDIDATE_DIM)
        self.assertEqual(CANDIDATE_DIM, 19)
        self.assertEqual(
            STATE_DIM_PURE_MEC_V2,
            GLOBAL_DIM + NUM_NODES * len(CANDIDATE_FIELDS),
        )
        self.assertEqual(STATE_DIM_PURE_MEC_V2, 384)
        self.assertEqual(
            STATE_DIM_VISUAL_ABLATION_V2,
            VISUAL_DIM + STATE_DIM_PURE_MEC_V2,
        )
        self.assertEqual(STATE_DIM_VISUAL_ABLATION_V2, 512)
        self.assertEqual(MASK_DIM, 20)
        layout_a = state_layout_pure_mec_v2()
        layout_b = state_layout_visual_ablation_v2()
        self.assertEqual(layout_a["total_mlp_input"], 384)
        self.assertEqual(layout_a["total_decision_state"], 404)
        self.assertEqual(layout_b["total_mlp_input"], 512)
        self.assertEqual(layout_b["total_decision_state"], 532)

    def test_capability_fields_replace_single_quality(self):
        self.assertEqual(
            tuple(CAPABILITY_FIELDS), ("dev_sr", "dev_osr", "dev_spl", "dev_ndtw")
        )
        self.assertEqual(tuple(CAPABILITY_METRICS), ("sr", "osr", "spl", "ndtw"))
        self.assertNotIn("dev_val_quality_prior", CANDIDATE_FIELDS)
        self.assertNotIn("quality_prior", CANDIDATE_FIELDS)


class CandidateFeatureTest(unittest.TestCase):
    def setUp(self):
        self.prior, self.flat = load_capability_prior()
        self.nodes = dummy_nodes()
        self.latency = DummyLatency()

    def test_shape_mask_and_capability_values(self):
        features, mask, metrics, fallback = pure_mec_v2_candidate_features(
            self.nodes, (0.0, 0.0), self.latency, self.prior
        )
        self.assertEqual(features.shape, (NUM_NODES, CANDIDATE_DIM))
        self.assertEqual(mask.shape, (NUM_NODES,))
        self.assertTrue(mask.all())
        self.assertFalse(fallback)
        for index, node in enumerate(self.nodes):
            expected = np.asarray([
                self.flat[node["model_type"]][metric]
                for metric in CAPABILITY_METRICS
            ], dtype=np.float32)
            np.testing.assert_allclose(
                features[index, 10:14], expected, atol=1e-7
            )
            np.testing.assert_allclose(
                features[index, 14:19],
                np.eye(len(MODEL_NAMES), dtype=np.float32)[
                    node["model_index"]
                ],
                atol=1e-7,
            )
            self.assertIsNotNone(metrics[index])

    def test_unavailable_node_keeps_static_and_capability_but_zeroes_channel(self):
        nodes = copy.deepcopy(self.nodes)
        nodes[0]["coverage_radius_m"] = 1.0
        nodes[0]["x"] = 10_000.0
        features, mask, metrics, fallback = pure_mec_v2_candidate_features(
            nodes, (0.0, 0.0), self.latency, self.prior
        )
        self.assertFalse(mask[0])
        self.assertIsNone(metrics[0])
        np.testing.assert_allclose(features[0, 5:10], 0.0)
        self.assertGreater(features[0, 3], 0.0)  # compute
        self.assertGreater(features[0, 4], 0.0)  # bandwidth
        np.testing.assert_allclose(
            features[0, 10:14],
            [
                self.flat[nodes[0]["model_type"]][metric]
                for metric in CAPABILITY_METRICS
            ],
            atol=1e-7,
        )
        self.assertFalse(fallback)

    def test_no_coverage_fallback_enables_nearest_node(self):
        nodes = copy.deepcopy(self.nodes)
        for node in nodes:
            node["coverage_radius_m"] = 1.0
            node["x"] += 10_000.0
            node["y"] += 10_000.0
        features, mask, metrics, fallback = pure_mec_v2_candidate_features(
            nodes, (0.0, 0.0), self.latency, self.prior
        )
        self.assertTrue(fallback)
        self.assertEqual(int(mask.sum()), 1)
        nearest = int(mask.argmax())
        self.assertIsNotNone(metrics[nearest])
        self.assertGreater(features[nearest, 5], 0.0)

    def test_real_train_topology(self):
        topology = load_topology("balanced", "train")
        features, mask, metrics, fallback = pure_mec_v2_candidate_features(
            topology["nodes"],
            (750.0, 400.0),
            Stage2KLatency(),
            self.prior,
        )
        self.assertEqual(features.shape, (20, 19))
        self.assertTrue(mask.any() or fallback)
        state = build_pure_mec_v2_state(
            (750.0, 400.0), 12, 143.2, features
        )
        self.assertEqual(state.shape, (STATE_DIM_PURE_MEC_V2,))
        self.assertTrue(np.isfinite(state).all())


class StateBuilderTest(unittest.TestCase):
    def test_global_features_and_normalization(self):
        candidate = np.zeros((NUM_NODES, CANDIDATE_DIM), dtype=np.float32)
        state = build_pure_mec_v2_state(
            (750.0, 300.0), 100, 250.0, candidate
        )
        np.testing.assert_allclose(
            state[:4], [0.5, 0.2, 0.25, 2.5], atol=1e-7
        )
        state_hi = build_pure_mec_v2_state(
            (1500.0, 1500.0), 400, 350.0, candidate
        )
        np.testing.assert_allclose(
            state_hi[:4], [1.0, 1.0, 1.0, 3.5], atol=1e-7
        )

    def test_step_and_running_mean_reject_invalid_values(self):
        candidate = np.zeros((NUM_NODES, CANDIDATE_DIM), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_pure_mec_v2_state((0.0, 0.0), -1, 0.0, candidate)
        with self.assertRaises(ValueError):
            build_pure_mec_v2_state((0.0, 0.0), 401, 0.0, candidate)
        with self.assertRaises(ValueError):
            build_pure_mec_v2_state((0.0, 0.0), 0, -1.0, candidate)

    def test_visual_ablation_is_exact_concatenation(self):
        rng = np.random.default_rng(3)
        candidate = rng.normal(size=(NUM_NODES, CANDIDATE_DIM)).astype(
            np.float32
        )
        pure = build_pure_mec_v2_state((100.0, 200.0), 7, 90.0, candidate)
        visual = rng.normal(size=VISUAL_DIM).astype(np.float32)
        combined = build_visual_ablation_v2_state(visual, pure)
        self.assertEqual(combined.shape, (STATE_DIM_VISUAL_ABLATION_V2,))
        np.testing.assert_array_equal(combined[:VISUAL_DIM], visual)
        np.testing.assert_array_equal(combined[VISUAL_DIM:], pure)

    def test_layouts_document_forbidden_inputs(self):
        layout_a = state_layout_pure_mec_v2()
        self.assertIn("visual/PCA", layout_a["forbidden"])
        self.assertIn("goal tokens", layout_a["forbidden"])
        self.assertIn("GAB", layout_a["forbidden"])
        self.assertEqual(layout_a["episode_step"]["normalization"], "t/400")
        self.assertEqual(
            layout_a["running_mean_e2e"]["normalization"],
            "running_mean_E2E_ms/100",
        )
        layout_b = state_layout_visual_ablation_v2()
        self.assertEqual(layout_b["visual_context"], [0, VISUAL_DIM])
        self.assertEqual(
            layout_b["pure_mec_state"],
            [VISUAL_DIM, VISUAL_DIM + STATE_DIM_PURE_MEC_V2],
        )


class VisualContextTest(unittest.TestCase):
    def test_aux_blocks_are_neutralized_and_visual_blocks_are_not(self):
        projector = fake_projector()
        rng = np.random.default_rng(11)
        summary = rng.normal(size=PCA_SUMMARY_DIM).astype(np.float32)
        summary_aux_changed = summary.copy()
        summary_aux_changed[VISUAL_SUMMARY_DIM:] += 100.0
        summary_visual_changed = summary.copy()
        summary_visual_changed[0] += 1.0
        mean_before = projector.mean.copy()
        out = visual_context_from_projector(projector, summary)
        out_aux = visual_context_from_projector(
            projector, summary_aux_changed
        )
        out_visual = visual_context_from_projector(
            projector, summary_visual_changed
        )
        self.assertEqual(out.shape, (VISUAL_DIM,))
        np.testing.assert_allclose(out, out_aux, atol=1e-6)
        self.assertGreater(float(np.abs(out - out_visual).max()), 1e-5)
        np.testing.assert_array_equal(projector.mean, mean_before)

    def test_real_frozen_projector_compatibility(self):
        if not Path(VISUAL_PCA_PATH).is_file():
            self.skipTest("visual PCA is not present")
        projector = VisualProjector.load(VISUAL_PCA_PATH)
        rng = np.random.default_rng(13)
        current = rng.normal(size=(5, 384)).astype(np.float32)
        goal = rng.normal(size=(5, 384)).astype(np.float32)
        history = rng.normal(size=(16, 768)).astype(np.float32)
        mask = np.ones(16, dtype=np.float32)
        summary_a = visual_summary(
            current, goal, history, mask,
            np.zeros(20, dtype=np.float32),
            np.zeros(22, dtype=np.float32),
        )
        summary_b = visual_summary(
            current, goal, history, mask,
            np.full(20, 7.0, dtype=np.float32),
            np.full(22, -3.0, dtype=np.float32),
        )
        self.assertEqual(summary_a.shape, (PCA_SUMMARY_DIM,))
        visual_a = visual_context_from_projector(projector, summary_a)
        visual_b = visual_context_from_projector(projector, summary_b)
        self.assertEqual(visual_a.shape, (VISUAL_DIM,))
        self.assertTrue(np.isfinite(visual_a).all())
        np.testing.assert_allclose(visual_a, visual_b, atol=1e-6)
        self.assertEqual(
            VISUAL_SUMMARY_DIM + AUX_SUMMARY_DIM, PCA_SUMMARY_DIM
        )


class CapabilityPriorTest(unittest.TestCase):
    def test_real_prior_has_four_metrics_and_no_final_test(self):
        self.assertTrue(Path(CAPABILITY_PRIOR_PATH).is_file())
        payload, flat = load_capability_prior()
        self.assertEqual(payload["split"], "dev_val_180")
        self.assertFalse(payload["final_test_read"])
        self.assertEqual(payload["episodes"], 180)
        self.assertEqual(tuple(payload["metrics"]), CAPABILITY_METRICS)
        self.assertEqual(set(flat), set(MODEL_NAMES))
        for name in MODEL_NAMES:
            self.assertEqual(set(flat[name]), set(CAPABILITY_METRICS))
            for value in flat[name].values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_rejects_final_test_prior(self):
        payload, _ = load_capability_prior()
        bad = copy.deepcopy(payload)
        bad["final_test_read"] = True
        with self.assertRaises(ValueError):
            validate_capability_prior(bad, verify_hashes=False)

    def test_rejects_wrong_split_and_missing_metric(self):
        payload, _ = load_capability_prior()
        bad = copy.deepcopy(payload)
        bad["split"] = "final_test_300"
        with self.assertRaises(ValueError):
            validate_capability_prior(bad, verify_hashes=False)
        bad = copy.deepcopy(payload)
        del bad["models"]["Small"]["spl"]
        with self.assertRaises(ValueError):
            validate_capability_prior(bad, verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
