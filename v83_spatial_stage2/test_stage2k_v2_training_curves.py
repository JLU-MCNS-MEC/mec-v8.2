"""Data-integrity tests for the Stage-2k V2 training-curve artifacts."""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_v2_training_curves import (
    A_RUN,
    B_RUN,
    OUT,
    load_training,
)


class TrainingCurveDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {
            "A": load_training(A_RUN),
            "B": load_training(B_RUN),
        }
        cls.summary = json.loads(
            (OUT / "stage2k_v2_training_curves_summary.json").read_text(
                encoding="utf-8"
            )
        )

    def test_3500_real_episodes_and_ma100_window(self):
        for key, run in self.runs.items():
            self.assertEqual(len(run["rows"]), 3500, key)
            np.testing.assert_array_equal(
                run["episode"], np.arange(1, 3501, dtype=np.float64)
            )
            self.assertTrue(np.isnan(run["ma100"][:99]).all(), key)
            self.assertTrue(np.isfinite(run["ma100"][99:]).all(), key)
            self.assertTrue(np.isfinite(run["returns"]).all(), key)

    def test_summary_peak_matches_history(self):
        for key in ("A", "B"):
            item = self.summary["training_curves"][key]
            run = self.runs[key]
            self.assertAlmostEqual(
                item["ma100_peak"], float(np.nanmax(run["ma100"])),
                places=12,
            )
            self.assertEqual(
                item["ma100_peak_episode"], run["peak_episode"]
            )
            self.assertAlmostEqual(
                item["ma100_final"], float(run["ma100"][-1]), places=12
            )

    def test_episode_csv_has_real_rows_and_blank_prefix(self):
        for key in ("A", "B"):
            path = OUT / f"stage2k_v2_train_{key}_episodes.csv"
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3500)
            self.assertEqual(rows[0]["episode"], "1")
            self.assertEqual(rows[-1]["episode"], "3500")
            self.assertEqual(rows[0]["ma100"], "")
            self.assertEqual(rows[98]["ma100"], "")
            self.assertNotEqual(rows[99]["ma100"], "")
            self.assertEqual(rows[0]["ma100_defined"], "0")
            self.assertEqual(rows[99]["ma100_defined"], "1")

    def test_no_final_test_read_and_no_interpolation(self):
        self.assertFalse(self.summary["final_test_read"])
        self.assertFalse(self.summary["interpolation_used"])
        self.assertEqual(self.summary["episodes"], {"A": 3500, "B": 3500})


if __name__ == "__main__":
    unittest.main()
