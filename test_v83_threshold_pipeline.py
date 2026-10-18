"""CPU unit tests for validation isolation and Stop policy; no GPU/models."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import prepare_closedloop_validation_v83 as preparation
from eval_stop_threshold_v83 import parse_threshold, attach_stop_policy


class MockModel:
    def predict_action(self, *args, **kwargs):
        return {
            "action": torch.tensor([3]),
            "move_probabilities": torch.tensor([[0.1, 0.8, 0.1]]),
            "arrival_probability": torch.tensor([0.9]),
            "used_threshold": kwargs["stop_threshold"]
        }


class ThresholdPolicyTest(unittest.TestCase):
    def test_threshold_range(self):
        self.assertIsNone(parse_threshold("never"))
        self.assertEqual(parse_threshold("0.7"), 0.7)
        for value in ("0", "1", "-0.1", "1.1"):
            with self.assertRaises(ValueError):
                parse_threshold(value)

    def test_policy_override(self):
        model = MockModel()
        attach_stop_policy(model, None)
        decision = model.predict_action()
        self.assertEqual(decision["action"].tolist(), [1])
        self.assertEqual(decision["used_threshold"], 0.5)
        model = MockModel()
        attach_stop_policy(model, 0.8)
        decision = model.predict_action()
        self.assertEqual(decision["action"].tolist(), [3])
        self.assertEqual(decision["used_threshold"], 0.8)

    def test_heldout_specs_exclude_old_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in preparation.EXCLUDE_SPLITS:
                d = root / split
                d.mkdir()
                (d / "dataset_stats.json").write_text(
                    '{"trajectories": [{"seed": 900000}]}', encoding="utf-8"
                )
            forbidden, sources = preparation.excluded_trajectory_seeds(root)
            self.assertEqual(forbidden, {900000})
            self.assertEqual(len(sources), 4)

        accepted = lambda env, seed, band: {
            "accepted": True, "reason": "accepted", "band": band[0],
            "seed": seed, "steps": 30, "start_goal_distance_m": 200.0
        }
        with mock.patch.object(preparation, "_inspect_oracle_candidate", side_effect=accepted), mock.patch.object(
            preparation, "ToyUAVEnv", return_value=SimpleNamespace()
        ):
            records, _, _ = preparation.generate_specs(5, 900000, {900000})
        self.assertEqual(len(records), 5)
        self.assertNotIn(900000, [r["seed"] for r in records])
        self.assertEqual(len(set(r["seed"] for r in records)), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
