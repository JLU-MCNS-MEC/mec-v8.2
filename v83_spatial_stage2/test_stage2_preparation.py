"""CPU tests for Stage-2 equal splits, Dev-Val spec and M1 direction audit."""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import token_data
from config import TRAIN
from v83_spatial_stage2.audit_stage2 import direction_audit, write_dev_spec
from v83_spatial_stage2.generate_stage2_datasets import EQUAL_MIX


class Stage2PreparationTest(unittest.TestCase):
    def test_equal_band_counts(self):
        original = token_data.TRAIN
        try:
            token_data.TRAIN = dataclasses.replace(
                original, trajectory_mix=EQUAL_MIX
            )
            for target in (600, 180, 300):
                _, counts = token_data._trajectory_band_schedule(target, 1_000_000)
                self.assertEqual(counts, [target // 3] * 3)
        finally:
            token_data.TRAIN = original

    def test_direction_audit(self):
        audit = direction_audit()
        self.assertTrue(audit["current_left_right_invariant"])
        self.assertTrue(audit["proposed_direction_sensitive"])
        self.assertEqual(audit["proposed_param_delta"], 144)
        self.assertEqual(audit["m1_total_params_proposed"], 2724983)

    def test_dev_spec_builder(self):
        stats = {
            "version": "v8.2-input-redesign",
            "trajectories": [
                {
                    "seed": 1000001, "band": "short", "steps": 40,
                    "start_goal_distance_m": 200.0,
                    "network_layout_seed": 17820001,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats_path = root / "dataset_stats.json"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")
            out = write_dev_spec(stats_path, root / "dev_spec.json")
            doc = json.loads(out.read_text())
            self.assertEqual(doc["purpose"], "stage2-dev-val")
            self.assertFalse(doc["split_is_final_test"])
            self.assertEqual(doc["trajectories"][0]["seed"], 1000001)
            self.assertEqual(doc["trajectories"][0]["network_layout_seed"], 17820001)
            with self.assertRaises(SystemExit):
                write_dev_spec(stats_path, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
