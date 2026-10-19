"""CPU tests for the Stage-2i input ablations and observable feature builders."""

from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from config import ENCODER
from v83_spatial_stage2 import stage2i_data, stage2i_model
from v83_spatial_stage2.stage2i_data import (
    GAB_FEATURE_DIM,
    HISTORY_STEPS,
    MAP_FEATURE_DIM,
    POSE_FEATURE_DIM,
    VISUAL_DESCRIPTOR_DIM,
    build_anonymous_landmark_nodes,
    build_gab_for_episode,
    gab_config_from_threshold,
    history_arrays,
    map_retrieval_features,
    online_history_arrays,
    pose_action_feature,
    retrieval_diagnostics,
)
from v83_spatial_stage2.stage2i_model import (
    SHARED_INIT_SEED,
    Stage2ITransformer,
    VARIANTS,
    parameter_count,
)


def random_inputs(batch=2):
    return {
        "current": torch.randn(batch, ENCODER.num_tokens, ENCODER.token_dim),
        "goal": torch.randn(batch, 3, ENCODER.token_dim),
        "heat": torch.randn(batch, 3, 8, 8),
        "network": torch.randn(batch, 5),
    }


def variant_kwargs(variant, batch=2):
    flags = VARIANTS[variant]
    kwargs = {}
    if flags["use_heatmap"]:
        kwargs["heatmap"] = torch.randn(batch, 3, 8, 8)
    if flags["use_network"]:
        kwargs["network_state"] = torch.randn(batch, 5)
    if flags["use_history"]:
        kwargs.update({
            "history_vis": torch.randn(batch, HISTORY_STEPS,
                                       VISUAL_DESCRIPTOR_DIM),
            "history_pose": torch.randn(batch, HISTORY_STEPS,
                                        POSE_FEATURE_DIM),
            "history_action": torch.zeros(batch, HISTORY_STEPS, 4),
            "history_mask": torch.ones(batch, HISTORY_STEPS),
        })
    if flags["use_gab"]:
        kwargs["gab"] = torch.randn(batch, GAB_FEATURE_DIM)
    if flags["use_map"]:
        kwargs["map_feat"] = torch.randn(batch, MAP_FEATURE_DIM)
    return kwargs


class ModelStructureTests(unittest.TestCase):
    def test_t0_matches_original_architecture_and_values(self):
        from gbs_model_v3 import build_gbs_model

        torch.manual_seed(SHARED_INIT_SEED)
        original = build_gbs_model("Transformer-M")
        t0 = Stage2ITransformer("T0")
        self.assertEqual(parameter_count(original), 2722164)
        self.assertEqual(parameter_count(t0), parameter_count(original))
        original_state = original.state_dict()
        t0_state = t0.state_dict()
        self.assertEqual(set(original_state), set(t0_state))
        for key in original_state:
            self.assertEqual(original_state[key].shape,
                             t0_state[key].shape, key)
            self.assertTrue(torch.equal(original_state[key], t0_state[key]),
                            key)

        inputs = random_inputs()
        original.eval()
        t0.eval()
        with torch.no_grad():
            expected = original.forward_components(
                inputs["current"], inputs["goal"], inputs["heat"],
                inputs["network"],
            )
            actual = t0.forward_components(
                inputs["current"], inputs["goal"], heatmap=inputs["heat"],
                network_state=inputs["network"],
            )
        for key in ("move_logits", "arrival_logit", "arrival_features"):
            self.assertTrue(torch.allclose(expected[key], actual[key], atol=1e-6),
                            key)

    def test_true_branch_removal(self):
        models = {v: Stage2ITransformer(v) for v in VARIANTS}
        self.assertTrue(hasattr(models["T0"].context_encoder, "heatmap_net"))
        self.assertTrue(hasattr(models["T0"].context_encoder, "network_net"))
        self.assertFalse(hasattr(models["T1"].context_encoder, "heatmap_net"))
        self.assertTrue(hasattr(models["T1"].context_encoder, "network_net"))
        self.assertTrue(hasattr(models["T2"].context_encoder, "heatmap_net"))
        self.assertFalse(hasattr(models["T2"].context_encoder, "network_net"))
        for variant in ("T3", "T4", "T5"):
            self.assertFalse(hasattr(models[variant].context_encoder,
                                      "heatmap_net"))
            self.assertFalse(hasattr(models[variant].context_encoder,
                                      "network_net"))
        self.assertFalse(hasattr(models["T3"], "history_encoder"))
        self.assertTrue(hasattr(models["T4"], "history_encoder"))
        self.assertTrue(hasattr(models["T5"], "history_encoder"))
        self.assertFalse(hasattr(models["T4"], "map_proj"))
        self.assertTrue(hasattr(models["T5"], "map_proj"))
        # Removal must reduce the parameter count.
        self.assertLess(parameter_count(models["T3"]),
                        parameter_count(models["T1"]))
        self.assertLess(parameter_count(models["T1"]),
                        parameter_count(models["T2"]))
        self.assertLess(parameter_count(models["T2"]),
                        parameter_count(models["T0"]))
        self.assertLess(parameter_count(models["T4"]),
                        parameter_count(models["T5"]))

    def test_removed_modalities_are_rejected(self):
        inputs = random_inputs()
        t3 = Stage2ITransformer("T3").eval()
        with self.assertRaises(ValueError):
            t3.forward_components(inputs["current"], inputs["goal"],
                                  heatmap=inputs["heat"])
        with self.assertRaises(ValueError):
            t3.forward_components(inputs["current"], inputs["goal"],
                                  network_state=inputs["network"])
        t1 = Stage2ITransformer("T1").eval()
        with self.assertRaises(ValueError):
            t1.forward_components(inputs["current"], inputs["goal"],
                                  heatmap=inputs["heat"],
                                  network_state=inputs["network"])
        t4 = Stage2ITransformer("T4").eval()
        with self.assertRaises(ValueError):
            t4.forward_components(inputs["current"], inputs["goal"])
        t5 = Stage2ITransformer("T5").eval()
        kwargs = variant_kwargs("T5")
        kwargs.pop("map_feat")
        with self.assertRaises(ValueError):
            t5.forward_components(inputs["current"], inputs["goal"], **kwargs)

    def test_shared_initialisation(self):
        models = {v: Stage2ITransformer(v) for v in VARIANTS}
        excluded = ("arrival_head", "history_encoder", "hist_proj",
                    "gab_proj", "map_proj")
        common = None
        for model in models.values():
            keys = {
                key for key in model.state_dict()
                if not key.startswith(excluded)
            }
            common = keys if common is None else common & keys
        self.assertGreater(len(common), 30)
        for key in common:
            reference = models["T0"].state_dict()[key]
            for variant, model in models.items():
                self.assertTrue(
                    torch.equal(reference, model.state_dict()[key]),
                    f"{key} differs for {variant}",
                )
        self.assertTrue(torch.equal(
            models["T0"].state_dict()["context_encoder.heatmap_net.0.weight"],
            models["T2"].state_dict()["context_encoder.heatmap_net.0.weight"],
        ))
        self.assertTrue(torch.equal(
            models["T0"].state_dict()["context_encoder.network_net.0.weight"],
            models["T1"].state_dict()["context_encoder.network_net.0.weight"],
        ))
        self.assertTrue(torch.equal(
            models["T4"].state_dict()["history_encoder.visual_proj.weight"],
            models["T5"].state_dict()["history_encoder.visual_proj.weight"],
        ))

    def test_forward_backward_all_variants(self):
        inputs = random_inputs()
        for variant in VARIANTS:
            model = Stage2ITransformer(variant)
            model.train()
            components = model.forward_components(
                inputs["current"], inputs["goal"], **variant_kwargs(variant)
            )
            loss = (components["move_logits"].sum()
                    + components["arrival_logit"].sum())
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item(), variant)
            named = dict(model.named_parameters())
            checks = {
                "T0": ["context_encoder.heatmap_net.0.weight",
                       "context_encoder.network_net.0.weight"],
                "T1": ["context_encoder.network_net.0.weight"],
                "T2": ["context_encoder.heatmap_net.0.weight"],
                "T3": ["visual_proj.weight"],
                "T4": ["history_encoder.visual_proj.weight",
                       "gab_proj.weight"],
                "T5": ["history_encoder.visual_proj.weight",
                       "gab_proj.weight", "map_proj.weight"],
            }[variant]
            for name in checks:
                self.assertIsNotNone(named[name].grad, f"{variant}/{name}")
                self.assertGreater(float(named[name].grad.norm()), 0.0,
                                   f"{variant}/{name}")


class HistoryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        steps = 20
        self.poses = np.zeros((steps, 3), dtype=np.float32)
        yaw = 0.0
        x = y = 0.0
        self.actions = rng.integers(0, 3, size=steps - 1)
        for index in range(steps):
            self.poses[index] = (x, y, yaw)
            action = int(self.actions[index]) if index < steps - 1 else 3
            if action == 0:
                x += 5.0 * math.cos(yaw)
                y += 5.0 * math.sin(yaw)
            elif action == 1:
                yaw += math.radians(15.0)
            elif action == 2:
                yaw -= math.radians(15.0)
        self.visuals = [
            rng.normal(size=VISUAL_DESCRIPTOR_DIM).astype(np.float16)
            for _ in range(steps)
        ]

    def test_history_uses_only_past_states(self):
        index = 19
        vis, pose, action, mask = history_arrays(
            self.poses, self.actions, index, self.visuals
        )
        self.assertEqual(int(mask.sum()), HISTORY_STEPS)
        np.testing.assert_array_equal(vis[0], self.visuals[index - 1])
        np.testing.assert_array_equal(vis[1], self.visuals[index - 2])
        np.testing.assert_array_equal(
            action[0], np.eye(4)[int(self.actions[index - 1])]
        )
        self.assertAlmostEqual(float(pose[0][0]), 1.0, places=5)
        # Slot 15 must refer to index-16 and never to a future state.
        np.testing.assert_array_equal(vis[15], self.visuals[index - 16])
        partial = history_arrays(self.poses, self.actions, 12, self.visuals)
        self.assertEqual(int(partial[3].sum()), 12)
        self.assertTrue(np.all(partial[3][12:] == 0))

    def test_online_history_matches_training_history(self):
        index = 11
        past = []
        for position in range(index):
            past.append({
                "pose": self.poses[position],
                "next_pose": self.poses[position + 1],
                "visual": self.visuals[position],
                "action": int(self.actions[position]),
            })
        expected = history_arrays(self.poses, self.actions, index, self.visuals)
        actual = online_history_arrays(past)
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left, right, atol=1e-6)

    def test_short_history_is_masked(self):
        vis, pose, action, mask = history_arrays(
            self.poses, self.actions, 3, self.visuals
        )
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(np.all(mask[3:] == 0))
        self.assertTrue(np.all(vis[3:] == 0))
        self.assertTrue(np.all(pose[3:] == 0))

    def test_pose_feature_never_reads_goal(self):
        feature = pose_action_feature(
            (10.0, 20.0, 0.0), (15.0, 20.0, 0.0), (15.0, 20.0, 0.0), 0,
            reference_pose=(10.0, 20.0, 0.0),
        )
        self.assertEqual(feature.shape, (POSE_FEATURE_DIM,))
        self.assertAlmostEqual(float(feature[0]), 1.0, places=5)


class GABTests(unittest.TestCase):
    def test_causal_prefix_and_episode_reset(self):
        rng = np.random.default_rng(3)
        heatmaps = rng.random((12, 3, 8, 8)).astype(np.float16)
        actions = rng.integers(0, 4, size=12)
        config = gab_config_from_threshold(0.0)
        full = build_gab_for_episode(heatmaps, actions, config)
        prefix = build_gab_for_episode(heatmaps[:7], actions[:7], config)
        np.testing.assert_allclose(full[:7], prefix, atol=1e-6)
        # A concatenation of two episodes with a step break resets the belief.
        reset = build_gab_for_episode(
            np.concatenate([heatmaps[:5], heatmaps[5:]]),
            np.concatenate([actions[:5], actions[5:]]),
            config,
            step_ids=np.concatenate([np.arange(5), np.arange(7)]),
        )
        second = build_gab_for_episode(heatmaps[5:], actions[5:], config)
        np.testing.assert_allclose(reset[5:], second, atol=1e-6)


class AnonymousMapTests(unittest.TestCase):
    class Node:
        def __init__(self, index):
            self.index = index
            self.position = np.asarray([index * 10.0, index * 5.0],
                                       dtype=np.float32)
            self.shape = "square"
            self.base_color = np.asarray([50.0 + index, 60.0, 70.0])
            self.marker_type = "dot"
            self.marker_color = np.asarray([200.0, 200.0, 200.0])

    class Env:
        def __init__(self):
            self.landmarks = [AnonymousMapTests.Node(i) for i in range(10)]
            self.goal_landmark = AnonymousMapTests.Node(99)

    def test_goal_identity_flags_are_not_read(self):
        import ast

        tree = ast.parse(inspect.getsource(stage2i_data))
        forbidden_attributes = {"is_goal", "is_distractor"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, forbidden_attributes)
            # The map code may use env.goal_landmark (anonymous node list) but
            # must never read env.goal (the true target coordinate).
            if (isinstance(node, ast.Attribute) and node.attr == "goal"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "env"):
                self.fail("stage2i_data must not read env.goal")

    def test_retrieval_is_cosine_topk(self):
        rng = np.random.default_rng(11)
        embeddings = rng.normal(size=(20, 16))
        goal = embeddings[7] + 0.01 * rng.normal(size=16)
        diagnostics = retrieval_diagnostics(7, embeddings, goal)
        self.assertEqual(diagnostics["goal_rank"], 1)
        self.assertEqual(diagnostics["top1"], 1)
        self.assertEqual(diagnostics["top5"], 1)

    def test_map_feature_uses_relative_geometry(self):
        rng = np.random.default_rng(5)
        embeddings = rng.normal(size=(6, 8))
        embeddings[2] = np.asarray([1.0] + [0.0] * 7)
        goal = np.asarray([1.0] + [0.0] * 7)
        positions = np.asarray([[100.0, 0.0]] * 6, dtype=np.float32)
        feature = map_retrieval_features(
            (0.0, 0.0, 0.0), positions, embeddings, goal
        )
        self.assertEqual(feature.shape, (MAP_FEATURE_DIM,))
        self.assertAlmostEqual(float(feature[0]), 1.0, places=4)
        # forward = 100 / 500 = 0.2 for the retrieved node.
        self.assertAlmostEqual(float(feature[1]), 0.2, places=4)

    def test_shuffle_is_deterministic_and_goal_index_is_diagnostic_only(self):
        env = self.Env()
        first, index_first = build_anonymous_landmark_nodes(env, 123)
        second, index_second = build_anonymous_landmark_nodes(env, 123)
        self.assertEqual(index_first, index_second)
        self.assertEqual([n.index for n in first], [n.index for n in second])
        self.assertEqual(first[index_first].index, 99)
        self.assertEqual(len(first), 11)


class EvalDisciplineTests(unittest.TestCase):
    def test_stop_threshold_is_not_short_circuited(self):
        source = Path("v83_spatial_stage2/stage2i_eval.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("STOP_THRESHOLD = 0.7", source)
        self.assertIn(
            "action = STOP if arrival_probability >= STOP_THRESHOLD else move_action",
            source,
        )
        # No diagnostic branch may appear between the probability and decision.
        start = source.index("arrival_probability = float(")
        end = source.index("action = STOP if arrival_probability")
        self.assertNotIn("if ", source[start:end].replace(
            "arrival_probability = float(", ""))


if __name__ == "__main__":
    unittest.main()
