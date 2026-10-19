"""Unit tests for V8.3 Stage-2d calibration / Stop-decision code."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2d_common import (
    FEATURE_DIM,
    SPLIT_SEED,
    STAGE2C_ROLLOUT,
    StopRule,
    bernoulli_kl,
    build_calibrator,
    build_training_rows,
    feature_normalization,
    load_rollout_steps,
    mine_far_negatives,
    parameter_count,
    per_decision_hazard,
    selection_score,
    split_episodes,
)
from v83_spatial_stage2.stage2d_calibration import _variant_configs


class EpisodeSplitTest(unittest.TestCase):
    def test_split_is_disjoint_and_covers_all_episodes(self):
        episode_index = np.repeat(np.arange(600), 3)
        train, validation = split_episodes(episode_index, seed=SPLIT_SEED,
                                           n_train=480)
        self.assertEqual(len(train), 480)
        self.assertEqual(len(validation), 120)
        self.assertEqual(len(set(train.tolist()) & set(validation.tolist())), 0)
        self.assertEqual(
            set(train.tolist()) | set(validation.tolist()),
            set(range(600)),
        )

    def test_split_is_deterministic(self):
        episode_index = np.repeat(np.arange(600), 2)
        first = split_episodes(episode_index, seed=7, n_train=480)
        second = split_episodes(episode_index, seed=7, n_train=480)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))


class MiningTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.episode = np.repeat(np.arange(10), 400)
        distance = np.concatenate([
            rng.uniform(0.0, 40.0, 1200),
            rng.uniform(40.0, 500.0, 2800),
        ])
        probability = np.where(
            distance > 40.0,
            rng.uniform(0.0, 0.2, len(distance)),
            rng.uniform(0.0, 1.0, len(distance)),
        )
        hard = rng.choice(np.flatnonzero(distance > 40.0), 60, replace=False)
        probability[hard] = rng.uniform(0.3, 0.9, len(hard))
        self.distance = distance
        self.label = (distance < 20.0).astype(np.float32)
        self.probability = probability
        self.train_episodes = np.arange(8)

    def test_hard_and_random_sets_are_equal_size_and_disjoint(self):
        hard, random = mine_far_negatives(
            self.distance, self.label, self.probability,
            np.isin(self.episode, self.train_episodes), seed=1,
        )
        self.assertEqual(len(hard), len(random))
        self.assertGreater(len(hard), 0)
        self.assertEqual(len(set(hard.tolist()) & set(random.tolist())), 0)
        self.assertTrue(np.all(self.distance[hard] > 40.0))
        self.assertTrue(np.all(self.distance[random] > 40.0))
        self.assertTrue(np.all(self.probability[hard] >= 0.3))
        self.assertTrue(np.all(self.probability[random] < 0.3))
        self.assertTrue(np.all(
            np.isin(self.episode[hard], self.train_episodes)
        ))
        self.assertTrue(np.all(
            np.isin(self.episode[random], self.train_episodes)
        ))

    def test_r2_and_r3_have_identical_row_budget(self):
        hard_rows = build_training_rows(
            self.distance, self.label, self.probability, self.episode,
            self.train_episodes, kind="hard", seed=1,
        )
        random_rows = build_training_rows(
            self.distance, self.label, self.probability, self.episode,
            self.train_episodes, kind="random", seed=1,
        )
        self.assertEqual(
            hard_rows["stats"]["total_rows"],
            random_rows["stats"]["total_rows"],
        )
        self.assertEqual(
            hard_rows["stats"]["added_far_rows"],
            random_rows["stats"]["added_far_rows"],
        )
        self.assertEqual(
            hard_rows["stats"]["base_rows"],
            random_rows["stats"]["base_rows"],
        )
        self.assertTrue(np.array_equal(
            hard_rows["row_indices"][:hard_rows["stats"]["base_rows"]],
            random_rows["row_indices"][:random_rows["stats"]["base_rows"]],
        ))
        self.assertEqual(
            int(hard_rows["anchor_mask"].sum()),
            hard_rows["stats"]["base_rows"],
        )


class CalibrationInitializationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.features = torch.randn(64, FEATURE_DIM)
        self.base_logit = torch.randn(64) * 3.0
        self.mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.std = np.ones(FEATURE_DIM, dtype=np.float32)

    def test_every_variant_exactly_reproduces_b0_at_initialization(self):
        for kind in ("R1", "R2", "R3"):
            calibrator = build_calibrator(kind, self.mean, self.std)
            output = calibrator(self.features, self.base_logit)
            self.assertTrue(
                torch.equal(output, self.base_logit),
                msg=f"{kind} initial output differs from B0",
            )

    def test_r2_and_r3_share_the_same_initialization(self):
        torch.manual_seed(77)
        r2 = build_calibrator("R2", self.mean, self.std)
        torch.manual_seed(77)
        r3 = build_calibrator("R3", self.mean, self.std)
        state2 = r2.state_dict()
        state3 = r3.state_dict()
        self.assertEqual(set(state2), set(state3))
        for key in state2:
            self.assertTrue(torch.equal(state2[key], state3[key]), msg=key)

    def test_parameter_counts(self):
        self.assertEqual(parameter_count(build_calibrator("R1")), 2)
        self.assertEqual(
            parameter_count(build_calibrator("R2", self.mean, self.std)),
            104 * 32 + 32 + 32 + 1,
        )
        self.assertEqual(
            parameter_count(build_calibrator("R3", self.mean, self.std)),
            parameter_count(build_calibrator("R2", self.mean, self.std)),
        )

    def test_calibrator_interface_has_no_ground_truth_input(self):
        for module in ("R1", "R2", "R3"):
            calibrator = build_calibrator(module, self.mean, self.std)
            signature = inspect.signature(calibrator.forward)
            self.assertEqual(
                list(signature.parameters), ["features", "base_logit"],
                msg=f"{module} interface leaked a non-observable input",
            )

    def test_anchor_kl_is_zero_at_initialization(self):
        calibrator = build_calibrator("R2", self.mean, self.std)
        output = calibrator(self.features, self.base_logit)
        self.assertLess(
            float(bernoulli_kl(self.base_logit, output).detach()), 1e-12
        )

    def test_bernoulli_kl_is_positive_when_distributions_differ(self):
        base = torch.zeros(32)
        other = torch.full((32,), 3.0)
        self.assertGreater(float(bernoulli_kl(base, other)), 0.1)


class StopRuleTest(unittest.TestCase):
    def test_threshold_rule(self):
        rule = StopRule("t0.7", 0.7)
        state = rule.new_state()
        self.assertFalse(state.update(0.69))
        state = rule.new_state()
        self.assertTrue(state.update(0.70))

    def test_two_of_two_requires_two_consecutive_decisions(self):
        rule = StopRule("2of2", 0.5, 2, 2)
        state = rule.new_state()
        self.assertFalse(state.update(0.6))
        self.assertTrue(state.update(0.6))
        state.reset()
        self.assertFalse(state.update(0.6))
        self.assertFalse(state.update(0.4))
        self.assertFalse(state.update(0.6))
        self.assertTrue(state.update(0.6))

    def test_three_of_three_requires_three_consecutive_decisions(self):
        rule = StopRule("3of3", 0.5, 3, 3)
        state = rule.new_state()
        self.assertFalse(state.update(0.6))
        self.assertFalse(state.update(0.6))
        self.assertTrue(state.update(0.6))

    def test_decision_state_is_episode_local(self):
        rule = StopRule("2of2", 0.5, 2, 2)
        first = rule.new_state()
        self.assertFalse(first.update(0.6))
        second = rule.new_state()
        self.assertFalse(
            second.update(0.6),
            msg="a new episode must not inherit the previous confirmation state",
        )
        self.assertTrue(second.update(0.6))


class HazardTest(unittest.TestCase):
    def test_hazard_can_use_the_actual_persistence_decision(self):
        probabilities = np.array([0.9, 0.9, 0.9, 0.9])
        distance = np.array([100.0, 100.0, 100.0, 100.0])
        episode = np.array([0, 0, 1, 1])
        actual_stop = np.array([False, True, False, True])
        hazard = per_decision_hazard(
            probabilities, distance, episode, threshold=0.5,
            stopped=actual_stop,
        )
        self.assertEqual(hazard["far40_stops"], 2)
        self.assertEqual(hazard["far40_stop_rate"], 0.5)
        self.assertEqual(
            hazard["expected_false_stop_decisions_per_episode"], 1.0
        )
        self.assertTrue(hazard["stop_decisions_from_actual_rule"])

    def test_hazard_falls_back_to_threshold(self):
        probabilities = np.array([0.9, 0.9, 0.9, 0.9])
        distance = np.array([100.0, 100.0, 100.0, 100.0])
        episode = np.array([0, 0, 1, 1])
        hazard = per_decision_hazard(
            probabilities, distance, episode, threshold=0.5
        )
        self.assertEqual(hazard["far40_stops"], 4)
        self.assertFalse(hazard["stop_decisions_from_actual_rule"])


class SelectionTest(unittest.TestCase):
    @staticmethod
    def _hazard(far_stops, predicted, near_recall):
        return {
            "expected_false_stop_decisions_per_episode": far_stops,
            "predicted_false_stop_decisions_per_episode": predicted,
            "near20_stop_rate": near_recall,
        }

    def test_lower_far_risk_is_preferred(self):
        low = self._hazard(0.1, 1.0, 0.60)
        high = self._hazard(0.5, 2.0, 0.60)
        self.assertLess(
            selection_score(low, b0_near_recall=0.60),
            selection_score(high, b0_near_recall=0.60),
        )

    def test_near_recall_collapse_is_penalized(self):
        safe = self._hazard(0.1, 1.0, 0.60)
        collapsed = self._hazard(0.1, 1.0, 0.50)
        self.assertGreater(
            selection_score(collapsed, b0_near_recall=0.60),
            selection_score(safe, b0_near_recall=0.60),
        )


class SourceDisciplineTest(unittest.TestCase):
    def test_no_positive_weighting_in_training_source(self):
        source = (
            PROJECT_ROOT / "v83_spatial_stage2/stage2d_calibration.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("pos_weight=", source)
        self.assertNotIn("BCEWithLogitsLoss(pos_weight", source)
        self.assertIn("F.binary_cross_entropy_with_logits", source)

    def test_optimizer_only_contains_calibrator_parameters(self):
        source = (
            PROJECT_ROOT / "v83_spatial_stage2/stage2d_calibration.py"
        ).read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"torch\.optim\.AdamW\(\s*calibrator\.parameters\(\)",
        )
        self.assertNotRegex(source, r"AdamW\([^)]*model\.parameters\(\)")

    def test_no_final_test_dataset_path_in_stage2d_scripts(self):
        for name in (
            "stage2d_common.py",
            "stage2d_calibration.py",
            "stage2d_r4_decision_rules.py",
        ):
            source = (
                PROJECT_ROOT / "v83_spatial_stage2" / name
            ).read_text(encoding="utf-8")
            self.assertNotIn("datasets/final_test", source)
            self.assertNotIn("final_test_300", source)

    def test_r2_and_r3_use_the_same_config_grid(self):
        args = SimpleNamespace(
            r1_lrs=[1e-2], r1_anchors=[1.0],
            r23_lrs=[1e-3, 3e-3],
            r23_anchors=[1.0, 10.0],
            r23_residuals=[1e-3],
        )
        self.assertEqual(
            _variant_configs("R2", args), _variant_configs("R3", args)
        )

    def test_confirmation_state_is_created_inside_the_episode(self):
        source = (
            PROJECT_ROOT / "v83_spatial_stage2/stage2d_common.py"
        ).read_text(encoding="utf-8")
        self.assertIn("confirmation = rule.new_state()", source)
        self.assertNotIn("global confirmation", source)


class CachedDataTest(unittest.TestCase):
    @unittest.skipUnless(STAGE2C_ROLLOUT.exists(), "rollout cache not available")
    def test_rollout_cache_is_reused_with_104d_features(self):
        data = load_rollout_steps()
        self.assertEqual(data["features"].shape[0], 74001)
        self.assertEqual(data["features"].shape[1], FEATURE_DIM)
        self.assertEqual(len(np.unique(data["episode_index"])), 600)
        self.assertLessEqual(data["probability_logit_max_abs_diff"], 1e-6)

    def test_feature_normalization_is_finite(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(100, FEATURE_DIM)).astype(np.float32)
        mask = np.ones(100, dtype=bool)
        mean, std = feature_normalization(features, mask)
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(std)))
        self.assertTrue(np.all(std > 0.0))


if __name__ == "__main__":
    unittest.main()
