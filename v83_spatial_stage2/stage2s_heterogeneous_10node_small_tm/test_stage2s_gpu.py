"""GPU closed-loop tests for Stage-2s heterogeneous dispatch.

These tests load both frozen checkpoints and run the real environment.  They
verify that the selected node's model actually produces the navigation action,
that the action changes the UAV pose and next 194-D state, that the 0.2 TFLOP/s
inference term enters E2E/state/reward, and that scheme D telescopes exactly.

Run:
    python -m unittest -v \
        v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s_gpu
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP  # noqa: E402
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (  # noqa: E402
    BLOCK_E2E,
    BLOCK_INFERENCE,
    CANDIDATE_DIM,
)
from v83_spatial_stage2.stage2o.rewards import (  # noqa: E402
    EpisodeRewardController,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm import (  # noqa: E402
    common,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.runtime import (  # noqa: E402
    HeterogeneousEvaluator,
    dual_model_probe,
    latency_stats,
    load_stage2s_runtime,
    make_scene_cache,
    runner_for,
)


def _available_by_model(runner):
    mask = np.asarray(runner.step_data["mask"], bool)
    return {
        model: [
            index for index in np.where(mask)[0]
            if runner.nodes[int(index)]["model_type"] == model
        ]
        for model in common.MEC_MODELS
    }


class HeterogeneousClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for closed-loop tests")
        cls.bundle = common.load_bundle()
        cls.device, cls.encoder, cls.backends, cls.latency = (
            load_stage2s_runtime("cuda")
        )
        cls.cache = make_scene_cache(cls.encoder, cls.device, capacity=8)

    def _runner(self, spec_index=0):
        return runner_for(
            self.bundle, self.bundle["dev_specs"][spec_index],
            self.encoder, self.backends, self.cache, self.device,
            self.latency,
        )

    def test_both_models_loaded_and_frozen(self):
        self.assertEqual(set(self.backends.loaded_models), set(common.MEC_MODELS))
        for model in common.MEC_MODELS:
            self.assertTrue(
                self.backends.models[model].training is False
            )
            self.assertTrue(all(
                not parameter.requires_grad
                for parameter in self.backends.models[model].parameters()
            ))
        self.assertEqual(
            self.backends.checkpoint_sha256,
            dict(common.MEC_CHECKPOINT_SHA256),
        )

    def test_dispatch_follows_selected_node_model(self):
        runner = self._runner()
        used = set()
        steps = 0
        while not used == set(common.MEC_MODELS) and steps < 120:
            available = _available_by_model(runner)
            choices = []
            for model in common.MEC_MODELS:
                if model not in used and available[model]:
                    choices.append((model, available[model][0]))
            if not choices:
                choices = [
                    (runner.nodes[int(index)]["model_type"], int(index))
                    for index in np.where(runner.step_data["mask"])[0]
                ]
            model, node_index = choices[0]
            before = dict(self.backends.dispatch_calls)
            transition = runner.apply(node_index)
            self.assertGreater(
                self.backends.dispatch_calls[model],
                before.get(model, 0),
            )
            self.assertEqual(
                runner.step_records[-1]["model_type"], model
            )
            used.add(model)
            steps += 1
            if transition["done"]:
                break
        self.assertEqual(used, set(common.MEC_MODELS))

    @staticmethod
    def _find_divergent_step(bundle, encoder, backends, cache, device,
                             latency, max_tasks=20, max_steps=80):
        for task_index in range(max_tasks):
            runner = runner_for(
                bundle, bundle["dev_specs"][task_index], encoder,
                backends, cache, device, latency,
            )
            prefix = []
            for step in range(max_steps):
                available = _available_by_model(runner)
                if available["Small"] and available["Transformer-M"]:
                    probe = dual_model_probe(runner)
                    if (
                        probe["actions_differ"]
                        and probe["actions"]["Small"] != STOP
                        and probe["actions"]["Transformer-M"] != STOP
                    ):
                        return task_index, list(prefix), probe
                mask = np.where(runner.step_data["mask"])[0]
                if len(mask) == 0:
                    break
                action = int(mask[0])
                prefix.append(action)
                transition = runner.apply(action)
                if transition["done"]:
                    break
        return None, None, None

    def test_model_choice_changes_navigation_action_and_state(self):
        task_index, prefix, probe = self._find_divergent_step(
            self.bundle, self.encoder, self.backends, self.cache,
            self.device, self.latency,
        )
        self.assertIsNotNone(
            probe, "no observation with different Small/TM actions found"
        )
        runners = []
        for _ in range(2):
            runner = runner_for(
                self.bundle, self.bundle["dev_specs"][task_index],
                self.encoder, self.backends, self.cache, self.device,
                self.latency,
            )
            for action in prefix:
                runner.apply(int(action))
            runners.append(runner)
        available = _available_by_model(runners[0])
        small_index = int(available["Small"][0])
        transformer_index = int(available["Transformer-M"][0])
        state_before = runners[0].step_data["state"].copy()
        pose_before = runners[0].step_data["pose"].copy()
        small = runners[0].apply(small_index)
        transformer = runners[1].apply(transformer_index)
        self.assertEqual(
            int(runners[0].step_records[-1]["action"]),
            int(probe["actions"]["Small"]),
        )
        self.assertEqual(
            int(runners[1].step_records[-1]["action"]),
            int(probe["actions"]["Transformer-M"]),
        )
        self.assertFalse(np.allclose(
            runners[0].env.position, runners[1].env.position
        ))
        if small["next_state"] is not None and transformer[
            "next_state"
        ] is not None:
            self.assertFalse(np.allclose(
                small["next_state"], transformer["next_state"]
            ))
        self.assertEqual(int(state_before.shape[0]), 194)
        self.assertFalse(np.allclose(
            pose_before, runners[0].env.pose
        ))

    def test_effective_inference_enters_e2e_state_and_reward(self):
        runner = self._runner()
        available = _available_by_model(runner)
        node_index = int(available["Transformer-M"][0])
        node = runner.nodes[node_index]
        metric = runner.step_data["latency_metrics"][node_index]
        candidate = runner.step_data["candidate_features"][node_index]
        expected_norm = min(
            float(metric["inference_ms"]) / 10.0, 3.0
        )
        self.assertAlmostEqual(
            float(candidate[BLOCK_INFERENCE]), expected_norm, places=6
        )
        self.assertAlmostEqual(
            float(candidate[BLOCK_E2E]),
            min(float(metric["total_ms"]) / 100.0, 3.0),
            places=6,
        )
        self.assertAlmostEqual(
            float(metric["total_ms"]),
            float(metric["extraction_ms"])
            + float(metric["communication_ms"])
            + float(metric["inference_ms"]),
            places=9,
        )
        controller = EpisodeRewardController("D")
        transition = runner.apply(node_index)
        self.assertAlmostEqual(
            float(transition["e2e_ms"]),
            float(metric["total_ms"]),
            places=9,
        )
        controller.step_reward(transition["e2e_ms"])
        self.assertAlmostEqual(
            controller.rewards[-1],
            (1.0 - float(metric["total_ms"]) / 100.0) / 400.0,
            places=12,
        )
        if transition["next_state"] is not None:
            self.assertAlmostEqual(
                float(transition["next_state"][3]),
                float(metric["total_ms"]) / 100.0,
                places=6,
            )
        self.assertEqual(node["model_type"], "Transformer-M")

    def test_full_episode_scheme_d_and_model_usage(self):
        runner = self._runner()
        controller = EpisodeRewardController("D")
        steps = 0
        while True:
            mask = np.where(runner.step_data["mask"])[0]
            self.assertGreater(len(mask), 0)
            node_index = int(mask[steps % len(mask)])
            transition = runner.apply(node_index)
            controller.step_reward(transition["e2e_ms"])
            steps += 1
            if transition["done"]:
                break
            self.assertLess(steps, 400)
        result = runner.finalize("test")
        _, diagnostics = controller.finalize(
            result["mean_episode_e2e_ms"], result["steps"]
        )
        expected = 1.0 - float(result["mean_episode_e2e_ms"]) / 100.0
        self.assertAlmostEqual(
            diagnostics["return"], expected, places=12
        )
        self.assertAlmostEqual(
            result["reward_D_reconstruction_error"], 0.0, places=12
        )
        stats = latency_stats(result["step_records"])
        self.assertEqual(
            stats["small_steps"] + stats["transformer_m_steps"],
            result["steps"],
        )
        self.assertEqual(
            result["model_usage"]["Small"] + result[
                "model_usage"
            ]["Transformer-M"],
            result["steps"],
        )

    def test_cuda_safe_full_checkpoint_resume(self):
        import tempfile

        from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
        from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.train import (
            _load_full_cuda_safe,
        )

        agent = DQNAgentV2(
            self.device, seed=7, gamma=1.0, state_dim=common.STATE_DIM,
            state_mode=common.STATE_MODE, action_dim=common.ACTION_DIM,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_full.pt"
            agent.save_full(path, extra={
                "stage": common.STAGE,
                "algorithm": "dqn",
                "seed": 7,
                "deployment_sha256": self.bundle["deployment"][
                    "deployment_sha256"
                ],
            })
            restored = DQNAgentV2(
                self.device, seed=7, gamma=1.0,
                state_dim=common.STATE_DIM,
                state_mode=common.STATE_MODE,
                action_dim=common.ACTION_DIM,
            )
            payload = _load_full_cuda_safe(restored, path)
        self.assertEqual(payload.get("stage"), common.STAGE)
        self.assertEqual(restored.total_steps, agent.total_steps)
        # A forward pass after RNG restore must remain finite.
        state = np.zeros(common.STATE_DIM, dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(restored.q_values(state))))

    def test_greedy_min_e2e_uses_current_total_ms_only(self):
        source = inspect.getsource(HeterogeneousEvaluator.select_strategy)
        self.assertNotIn("oracle", source.lower())
        runner = self._runner()
        metrics = runner.step_data["latency_metrics"]
        mask = runner.step_data["mask"]
        indices = np.where(mask)[0]
        chosen = HeterogeneousEvaluator.select_strategy(
            runner, "greedy_min_e2e", np.random.default_rng(1)
        )
        best = min(
            indices, key=lambda i: (float(metrics[i]["total_ms"]), int(i))
        )
        self.assertEqual(int(chosen), int(best))
        node = runner.nodes[int(chosen)]
        self.assertIn(node["model_type"], common.MEC_MODELS)


if __name__ == "__main__":
    unittest.main()
