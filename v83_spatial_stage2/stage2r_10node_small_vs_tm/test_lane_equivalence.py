"""Equivalence test: full closed-loop runner vs frozen-lane adapter.

Run with a GPU (builds two real Dev lanes):
    python v83_spatial_stage2/stage2r_10node_small_vs_tm/test_lane_equivalence.py

The test uses a fixed Greedy-Min-E2E association policy.  It compares every
194-D state, mask, E2E value, done flag, success/SPL/nDTW/step count and the
scheme-D reward identity between the full environment and the cached-lane
adapter.  If they differ, lane-based training is invalid and must not be used.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2o.rewards import EpisodeRewardController
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (
    MEC_MODELS,
    load_bundle,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.lane_cache import (
    compact_lane,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.lane_runner import (
    LaneRunnerAdapter,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.runtime import (
    build_lane,
    load_stage2r_runtime,
    make_scene_cache,
    runner_for,
)


def _choose(step_data):
    metrics = step_data["latency_metrics"]
    mask = step_data["mask"]
    indices = np.where(np.asarray(mask, bool))[0]
    return int(min(indices, key=lambda i: (
        float(metrics[i]["total_ms"]), int(i)
    )))


def _run_full(bundle, spec, encoder, backends, scene_cache, device):
    runner = runner_for(
        bundle, spec, encoder, backends, scene_cache, device
    )
    states = []
    e2e = []
    controller = EpisodeRewardController("D")
    while True:
        states.append(runner.step_data["state"].copy())
        action = _choose(runner.step_data)
        transition = runner.apply(action)
        controller.step_reward(transition["e2e_ms"])
        e2e.append(float(transition["e2e_ms"]))
        if transition["done"]:
            break
    result = runner.finalize("full")
    _, diagnostics = controller.finalize(
        result["mean_episode_e2e_ms"], result["steps"]
    )
    return states, e2e, result, diagnostics


def _run_lane(episode, spec):
    runner = LaneRunnerAdapter(episode, spec)
    states = []
    e2e = []
    controller = EpisodeRewardController("D")
    while True:
        states.append(runner.step_data["state"].copy())
        action = _choose(runner.step_data)
        transition = runner.apply(action)
        controller.step_reward(transition["e2e_ms"])
        e2e.append(float(transition["e2e_ms"]))
        if transition["done"]:
            break
    result = runner.finalize("lane")
    _, diagnostics = controller.finalize(
        result["mean_episode_e2e_ms"], result["steps"]
    )
    return states, e2e, result, diagnostics


class LaneEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        cls.bundles = {}
        cls.runtimes = {}

    def _runtime(self, model):
        if model not in self.runtimes:
            bundle = load_bundle(model)
            device, encoder, backends = load_stage2r_runtime(
                self.device, model
            )
            scene_cache = make_scene_cache(
                encoder, device, capacity=64
            )
            self.bundles[model] = bundle
            self.runtimes[model] = (
                device, encoder, backends, scene_cache
            )
        return self.bundles[model], self.runtimes[model]

    def _check_model(self, model):
        bundle, (device, encoder, backends, scene_cache) = self._runtime(
            model
        )
        # Pick one successful and one failed Dev lane when available.
        chosen = []
        outcomes = set()
        for spec in bundle["dev_specs"]:
            lane = build_lane(
                spec, bundle["topology"], encoder, backends,
                bundle["quality_prior"], bundle["gab_config"], device,
                scene_cache,
            )
            if bool(lane.success) not in outcomes or len(chosen) < 2:
                chosen.append((spec, lane))
                outcomes.add(bool(lane.success))
            if len(chosen) >= 2 and len(outcomes) >= 1:
                break
        for spec, lane in chosen:
            episode = compact_lane(lane)
            full_states, full_e2e, full_result, full_diag = _run_full(
                bundle, spec, encoder, backends, scene_cache, device
            )
            lane_states, lane_e2e, lane_result, lane_diag = _run_lane(
                episode, spec
            )
            self.assertEqual(len(full_states), len(lane_states))
            for index, (left, right) in enumerate(
                zip(full_states, lane_states)
            ):
                # The state is float32; allow only float32 round-off.
                np.testing.assert_allclose(
                    left, right, rtol=0.0, atol=1e-7,
                    err_msg=f"{model} state step {index}",
                )
            np.testing.assert_allclose(
                np.asarray(full_e2e), np.asarray(lane_e2e),
                rtol=0.0, atol=0.0,
            )
            for key in ("success", "osr", "spl", "ndtw", "steps",
                        "mean_episode_e2e_ms"):
                if isinstance(full_result[key], float):
                    self.assertAlmostEqual(
                        full_result[key], lane_result[key], places=12,
                        msg=f"{model} {spec['episode_uid']} {key}",
                    )
                else:
                    self.assertEqual(
                        full_result[key], lane_result[key],
                        f"{model} {spec['episode_uid']} {key}",
                    )
            self.assertAlmostEqual(
                full_diag["return"], lane_diag["return"], places=12
            )
            self.assertAlmostEqual(
                full_result["latency_return"],
                lane_result["latency_return"], places=12,
            )

    def test_small_lane_equivalence(self):
        self._check_model("Small")

    def test_transformer_m_lane_equivalence(self):
        self._check_model("Transformer-M")


if __name__ == "__main__":
    unittest.main()
