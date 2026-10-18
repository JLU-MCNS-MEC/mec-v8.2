"""CPU-only tests for MEC relative geometry (no datasets, no checkpoints)."""
import unittest
from types import SimpleNamespace

import numpy as np
from mec_geometry_v83 import mec_relative_geometry


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            SimpleNamespace(x=100.0, y=0.0, bandwidth_hz=30e6),
            SimpleNamespace(x=0.0, y=100.0, bandwidth_hz=5e6),
        ]
        self.state = np.array([0.8, 0.4, 0.6, 1.0, 0.3], dtype=np.float32)

    def test_static_geometry_and_no_absolute_location_leakage(self):
        a = mec_relative_geometry(self.nodes, (0, 0), 0, self.state)
        shifted = [SimpleNamespace(x=n.x+500, y=n.y+400,
                                  bandwidth_hz=n.bandwidth_hz) for n in self.nodes]
        b = mec_relative_geometry(shifted, (500, 400), 0, self.state)
        self.assertEqual(a.heatmaps.shape, (3, 8, 8))
        self.assertEqual(a.network_delta.shape, (5,))
        np.testing.assert_allclose(a.heatmaps, b.heatmaps, atol=1e-6)
        np.testing.assert_allclose(a.network_delta, np.zeros(5))
        self.assertTrue(np.isfinite(a.heatmaps).all())

    def test_dynamic_trend_and_network_delta(self):
        a = mec_relative_geometry(self.nodes, (5, 0), 0, self.state,
                                  previous_xy=(0, 0),
                                  previous_network_state=np.zeros(5))
        self.assertTrue(a.heatmaps[2].max() > 0)
        np.testing.assert_allclose(a.network_delta, self.state)

    def test_only_observable_fields_required(self):
        # No goal, capability, stop label, history image, or distance_to_goal.
        b = mec_relative_geometry(self.nodes, (0, 0), 1.0, self.state)
        self.assertEqual(b.visible_nodes, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
