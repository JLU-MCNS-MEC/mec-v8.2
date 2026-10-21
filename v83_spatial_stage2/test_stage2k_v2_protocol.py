"""Protocol-level tests for the Stage-2k v2 3500-episode ablations."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BANDS,
    SCENARIOS,
    load_split_specs,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import epsilon_at
from v83_spatial_stage2.stage2k_jointr_v2_order import (
    random_training_order,
    verify_order,
)
from v83_spatial_stage2.stage2k_jointr_v2_train import (
    classify_status,
    early_stop_decision,
)
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_PRIOR_PATH,
    load_capability_prior,
)


class SharedTrainingOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_split_specs("train")
        cls.order_a, cls.targets_a = random_training_order(
            cls.payload, total=3500
        )
        cls.order_b, cls.targets_b = random_training_order(
            cls.payload, total=3500
        )
        cls.lookup = {
            item["episode_uid"]: item
            for scenario in SCENARIOS
            for item in cls.payload["scenarios"][scenario]["episodes"]
        }

    def test_two_variants_share_the_same_order(self):
        self.assertEqual(self.order_a, self.order_b)
        self.assertEqual(self.targets_a, self.targets_b)

    def test_order_is_3500_unique_train_uids(self):
        self.assertEqual(len(self.order_a), 3500)
        self.assertEqual(len(set(self.order_a)), 3500)
        verify_order(self.order_a, self.payload, require_full=False)
        self.assertTrue(set(self.order_a).issubset(self.lookup))

    def test_band_and_scenario_coverage(self):
        bands = Counter(self.lookup[uid]["band"] for uid in self.order_a)
        scenarios = Counter(
            self.lookup[uid]["scenario"] for uid in self.order_a
        )
        self.assertEqual(
            [bands[band] for band in BANDS], [1167, 1167, 1166]
        )
        self.assertEqual(
            [scenarios[scenario] for scenario in SCENARIOS],
            [1167, 1167, 1166],
        )


class EpsilonScheduleTest(unittest.TestCase):
    def test_500k_schedule(self):
        self.assertAlmostEqual(epsilon_at(0, 500_000), 1.0, places=12)
        self.assertAlmostEqual(
            epsilon_at(250_000, 500_000), 1.0 - 0.95 * 0.5, places=12
        )
        self.assertAlmostEqual(
            epsilon_at(500_000, 500_000), 0.05, places=12
        )
        self.assertAlmostEqual(
            epsilon_at(900_000, 500_000), 0.05, places=12
        )

    def test_invalid_decay(self):
        with self.assertRaises(ValueError):
            epsilon_at(0, 0)


class StopRuleTest(unittest.TestCase):
    def test_disabled_runs_never_early_stop_on_reward(self):
        stop, reason = early_stop_decision(
            1600, 10, [0.16, 0.10, 0.05], True
        )
        self.assertFalse(stop)
        self.assertIsNone(reason)
        stop, reason = early_stop_decision(
            1600, 10, [0.16, 0.10, 0.05], False
        )
        self.assertTrue(stop)
        self.assertIn("no new best", reason)

    def test_min_episode_guard_still_applies_when_enabled(self):
        stop, _ = early_stop_decision(200, 10, [0.1, 0.05], False)
        self.assertFalse(stop)


class ConvergenceRuleTest(unittest.TestCase):
    def test_stable_platform_is_converged(self):
        payload = classify_status(
            [0.10, 0.15, 0.20, 0.205, 0.203], False, True
        )
        self.assertTrue(payload["stable_platform"])
        self.assertFalse(payload["improving"])
        self.assertTrue(payload["converged"])
        self.assertEqual(payload["status"], "converged")

    def test_rising_curve_at_cap_is_not_converged(self):
        payload = classify_status(
            [0.05, 0.10, 0.15, 0.20, 0.25], False, True
        )
        self.assertFalse(payload["converged"])
        self.assertEqual(payload["status"], "not_converged")

    def test_early_stop_alone_is_not_convergence(self):
        payload = classify_status(
            [0.20, 0.15, 0.10, 0.08, 0.06], True, False
        )
        self.assertFalse(payload["converged"])
        self.assertEqual(payload["status"], "early_stopped_no_improvement")


class CapabilityPriorProtocolTest(unittest.TestCase):
    def test_prior_is_dev_only_and_has_four_metrics(self):
        self.assertTrue(Path(CAPABILITY_PRIOR_PATH).is_file())
        payload, flat = load_capability_prior()
        self.assertFalse(payload["final_test_read"])
        self.assertEqual(payload["split"], "dev_val_180")
        self.assertEqual(payload["episodes"], 180)
        self.assertIn("2,000,000", payload["source"])
        self.assertEqual(payload["metrics"], ["sr", "osr", "spl", "ndtw"])
        for values in flat.values():
            self.assertEqual(
                set(values), {"sr", "osr", "spl", "ndtw"}
            )


if __name__ == "__main__":
    unittest.main()
