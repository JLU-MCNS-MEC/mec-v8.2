"""Tests for the eval-only M1 Stop-threshold migration.

The replay/aggregate tests require the M1 run directory. They are skipped on
a fresh checkout where the run artifacts are not present.
"""

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

from v83_spatial_stage2.stage2d_m1_threshold_eval import (
    M1_CKPT,
    verify_m1_replay,
)


RUN_DIR = (
    PROJECT_ROOT
    / "runs/v83_spatial_stage2/stage2d/m1_threshold_20261019T013737Z"
)
HAS_RUN = RUN_DIR.exists()


def read_rows(name):
    with (RUN_DIR / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class M1CheckpointTest(unittest.TestCase):
    def test_m1_checkpoint_exists(self):
        self.assertTrue(M1_CKPT.is_file(), msg=str(M1_CKPT))


@unittest.skipUnless(HAS_RUN, "M1 threshold run artifacts not present")
class M1ReplayTest(unittest.TestCase):
    def test_m1_0_5_reproduces_stage2a_aggregates(self):
        rows = read_rows("M1@0.5_per_episode.csv")
        self.assertEqual(len(rows), 180)
        self.assertEqual(
            sum(int(row["success"]) for row in rows), 50
        )
        self.assertAlmostEqual(
            float(np.mean([int(row["success"]) for row in rows])),
            0.2778, places=4,
        )
        self.assertAlmostEqual(
            float(np.mean([int(row["osr"]) for row in rows])),
            0.3056, places=4,
        )
        self.assertAlmostEqual(
            float(np.mean([float(row["ndtw"]) for row in rows])),
            0.3648, places=4,
        )

    def test_m1_0_5_outcome_replay_is_exact(self):
        replay = verify_m1_replay(read_rows("M1@0.5_per_episode.csv"))
        self.assertEqual(replay["outcome_mismatch_count"], 0)
        self.assertEqual(replay["episodes"], 180)

    def test_m1_0_7_improves_success_and_preserves_ndtw(self):
        metrics = json.loads(
            (RUN_DIR / "metrics.json").read_text(encoding="utf-8")
        )
        base = metrics["summaries"]["M1@0.5"]["all"]
        tuned = metrics["summaries"]["M1@0.7"]["all"]
        self.assertGreater(tuned["sr"], base["sr"])
        self.assertGreater(tuned["osr"], base["osr"])
        self.assertLess(tuned["premature_stop"], base["premature_stop"])
        self.assertGreaterEqual(tuned["ndtw"], base["ndtw"] - 0.002)

    def test_m0_0_7_reference_is_the_existing_stage2d_run(self):
        metrics = json.loads(
            (RUN_DIR / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertAlmostEqual(
            metrics["m0_0.7_reference"]["all"]["sr"], 0.3444, places=4
        )
        self.assertAlmostEqual(
            metrics["m0_0.7_reference"]["all"]["osr"], 0.3833, places=4
        )
        self.assertAlmostEqual(
            metrics["m0_0.7_reference"]["all"]["ndtw"], 0.3573, places=4
        )


if __name__ == "__main__":
    unittest.main()
