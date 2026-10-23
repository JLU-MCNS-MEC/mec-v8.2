"""Unit/integration tests for the Stage-2n latency calibration.

Run from the repository root:

    python v83_spatial_stage2/test_stage2n_latency_calibrate.py

The tests are CPU-only except for one short CUDA check that is skipped when
CUDA is unavailable.  They do not read the sealed Final Test set, do not
train, and do not modify the simulation latency model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2n_latency_calibrate import (
    ANALYTICAL_TFLOPS,
    EXPECTED_PARAMS,
    EXPECTED_SHA256,
    HISTORICAL_GREEDY_E2E_MS,
    REFERENCE_DISTANCE_M,
    TOKEN_SAMPLE,
    AUX_SAMPLE,
    analytical_record,
    load_model,
    load_real_inputs,
    measure_forward_per_call,
    measure_request_to_result,
    _cv_round_means,
    _stats,
    verify_checkpoint,
)


class StatisticsTests(unittest.TestCase):
    def test_stats_units_and_percentiles(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _stats(values)
        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["mean_ms"], 3.0)
        self.assertAlmostEqual(stats["p50_ms"], 3.0)
        self.assertGreaterEqual(stats["p95_ms"], stats["p90_ms"])
        self.assertGreaterEqual(stats["p99_ms"], stats["p95_ms"])
        self.assertLessEqual(stats["min_ms"], stats["max_ms"])

    def test_stats_reject_empty_and_nonfinite(self):
        with self.assertRaises(ValueError):
            _stats([])
        with self.assertRaises(ValueError):
            _stats([1.0, float("nan")])

    def test_round_cv(self):
        self.assertAlmostEqual(
            _cv_round_means([1.0, 1.0, 1.0]), 0.0
        )
        self.assertGreater(_cv_round_means([1.0, 1.1, 1.2]), 0.0)


class AnalyticalTests(unittest.TestCase):
    def test_analytical_matches_flops(self):
        for name in ("Small", "Transformer-M"):
            record = analytical_record(name)
            self.assertAlmostEqual(
                record["analytical_inference_ms"],
                1000.0 * record["flops"] / (ANALYTICAL_TFLOPS * 1e12),
                places=12,
            )
            self.assertAlmostEqual(record["extraction_ms"], 22.7149777099,
                                   places=6)
            distance = record["per_distance"][
                f"{REFERENCE_DISTANCE_M:.0f}"
            ]
            self.assertGreater(distance["communication_ms"], 0.0)
            self.assertGreater(distance["total_ms"], 0.0)
            self.assertEqual(record["latency_model"]["modified"], False)

    def test_historical_reference_is_fixed(self):
        self.assertEqual(HISTORICAL_GREEDY_E2E_MS, 49.11)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_hashes_and_parameters(self):
        for name in ("Small", "Transformer-M"):
            record = verify_checkpoint(name)
            self.assertEqual(
                record["checkpoint_sha256"], EXPECTED_SHA256[name]
            )
            self.assertEqual(record["parameters"], EXPECTED_PARAMS[name])
            self.assertFalse(record["final_test_read"])


class InputTests(unittest.TestCase):
    def test_real_sample_shapes_and_dtype(self):
        inputs, provenance = load_real_inputs(0, torch.device("cpu"))
        expected = {
            "current_tokens": (1, 576, 384),
            "goal_tokens": (1, 3, 384),
            "history_vis": (1, 16, 768),
            "history_pose": (1, 16, 10),
            "history_action": (1, 16, 4),
            "history_mask": (1, 16),
            "gab": (1, 20),
            "map_feat": (1, 22),
        }
        self.assertEqual(
            {key: tuple(value.shape) for key, value in inputs.items()},
            expected,
        )
        for value in inputs.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(provenance["deployed_dtype"], "float32")
        self.assertFalse(provenance["final_test_read"])
        self.assertNotIn("final_test", str(TOKEN_SAMPLE).lower())
        self.assertNotIn("final_test", str(AUX_SAMPLE).lower())


class CudaMeasurementTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_short_real_measurement(self):
        device = torch.device("cuda")
        inputs, _ = load_real_inputs(0, device)
        model, _ = load_model("Small", device)
        with torch.inference_mode():
            for _ in range(20):
                model.forward_components(**inputs)
        torch.cuda.synchronize()
        forward = measure_forward_per_call(model, inputs, 20)
        request = measure_request_to_result(model, inputs, 20)
        for values in (forward, request):
            array = np.asarray(values, dtype=np.float64)
            self.assertEqual(array.shape, (20,))
            self.assertTrue(np.isfinite(array).all())
            self.assertTrue((array > 0.0).all())
            self.assertTrue((array < 100.0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
