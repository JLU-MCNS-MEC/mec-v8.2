"""Stage-2n closed-loop runtime: homogeneous MEC + latency-only reward.

The MEC side is unchanged from Stage-2j/2l: the frozen Transformer-M still
receives RGB tokens, goal-reference tokens and visual history and emits the
navigation action.  What changes is:

* only **one** MEC checkpoint is loaded (all 20 nodes are identical);
* the association reward contains **only** the per-step E2E-delay term, with
  no navigation-progress / nDTW reward.

The latency-only reward is applied by subtracting the inherited navigation
potential from the transition after ``FixedCityEpisodeRunner.apply`` has run.
The next state is built from the running mean E2E, not from the reward, so
this post-adjustment cannot change the environment dynamics or the MEC input.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP
from v83_spatial_stage2.stage2j_eval import load_checkpoint
from v83_spatial_stage2.stage2j_model import build_stage2j_model
from v83_spatial_stage2.stage2k_common import STOP_THRESHOLD, sha256_file
from v83_spatial_stage2.stage2k_runtime import load_encoder
from v83_spatial_stage2.stage2k_state import masked_argmax
from v83_spatial_stage2.stage2l_runtime import (
    FixedCityEpisodeRunner,
    FixedCitySceneCache,
)
from v83_spatial_stage2.stage2n_common import (
    MEC_CHECKPOINT,
    MEC_CHECKPOINT_SHA256,
    MEC_MODEL,
    latency_return_from_mean,
    prepare_runner_spec,
)


class HomogeneousBackendPool:
    """One frozen MEC model shared by every node of a homogeneous deployment.

    ``model_name`` defaults to the Stage-2n Transformer-M so every existing
    Stage-2n/2o/2p caller is byte-for-byte unchanged.  Stage-2r passes
    ``Small`` or ``Transformer-M`` explicitly; only that checkpoint is loaded.
    """

    def __init__(self, device, checkpoint=MEC_CHECKPOINT,
                 expected_sha256=MEC_CHECKPOINT_SHA256,
                 model_name=MEC_MODEL):
        self.device = torch.device(device)
        self.model_name = str(model_name)
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual = sha256_file(checkpoint)
        if actual != expected_sha256:
            raise ValueError(
                f"{self.model_name} checkpoint SHA-256 {actual} != "
                f"{expected_sha256}"
            )
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = actual
        self.model = build_stage2j_model(self.model_name).to(self.device)
        load_checkpoint(self.model, checkpoint, self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.load_calls = {}

    @torch.inference_mode()
    def get(self, model_name, inputs, cache=None):
        if str(model_name) != self.model_name:
            raise ValueError(
                f"This homogeneous pool only loads {self.model_name}, got "
                f"{model_name}"
            )
        if cache is not None and self.model_name in cache:
            return cache[self.model_name]
        components = self.model.forward_components(
            inputs["current_tokens"],
            inputs["goal_tokens"],
            heatmap=None,
            network_state=None,
            history_vis=inputs["history_vis"],
            history_pose=inputs["history_pose"],
            history_action=inputs["history_action"],
            history_mask=inputs["history_mask"],
            gab=inputs["gab"],
            map_feat=inputs["map_feat"],
        )
        move_logits = components["move_logits"][0].float()
        arrival_logit = components["arrival_logit"][0].float()
        arrival_probability = float(torch.sigmoid(arrival_logit).item())
        move_probabilities = torch.softmax(move_logits, dim=-1)
        move_action = int(torch.argmax(move_probabilities).item())
        action = STOP if arrival_probability >= STOP_THRESHOLD else move_action
        joint = np.concatenate([
            (1.0 - arrival_probability)
            * move_probabilities.detach().cpu().numpy(),
            np.asarray([arrival_probability], dtype=np.float32),
        ]).astype(np.float32)
        output = {
            "model_type": self.model_name,
            "move_logits": move_logits.detach().cpu().numpy(),
            "move_probabilities": move_probabilities.detach().cpu().numpy(),
            "arrival_probability": arrival_probability,
            "joint_probabilities": joint,
            "move_action": move_action,
            "action": int(action),
        }
        self.load_calls[self.model_name] = (
            self.load_calls.get(self.model_name, 0) + 1
        )
        if cache is not None:
            cache[self.model_name] = output
        return output


def load_stage2n_runtime(device=None):
    if device is None or str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    encoder = load_encoder(device)
    backends = HomogeneousBackendPool(device)
    return device, encoder, backends


class LatencyOnlyEpisodeRunner(FixedCityEpisodeRunner):
    """Fixed-city runner whose agent reward has no navigation term."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("jointr_aligned", None)
        kwargs.pop("pure_mec_v2_state", None)
        kwargs.pop("visual_ablation_v2_state", None)
        super().__init__(
            *args,
            jointr_aligned=True,
            pure_mec_v2_state=True,
            visual_ablation_v2_state=False,
            **kwargs,
        )

    def apply(self, node_index):
        transition = super().apply(node_index)
        navigation_reward = float(transition["nav_reward"])
        # The parent added nav_reward + latency_penalty to return_sum and to
        # the step record.  Remove the navigation term; keep the latency term.
        self.return_sum -= navigation_reward
        row = self.step_records[-1]
        row["navigation_reward"] = 0.0
        row["navigation_reward_removed"] = navigation_reward
        row["reward"] = float(transition["latency_penalty"])
        transition["reward"] = float(transition["latency_penalty"])
        transition["nav_reward"] = 0.0
        transition["navigation_reward_removed"] = navigation_reward
        transition["reward_mode"] = "latency_only_v1"
        return transition

    def finalize(self, strategy):
        output = super().finalize(strategy)
        mean_e2e = float(output["mean_episode_e2e_ms"])
        latency_return = float(self.return_sum)
        closed_form = latency_return_from_mean(mean_e2e)
        output["strategy"] = str(strategy)
        output["reward_mode"] = "latency_only_v1"
        output["latency_return"] = latency_return
        output["latency_return_closed_form"] = closed_form
        output["latency_return_abs_error"] = abs(latency_return - closed_form)
        output["episode_return"] = latency_return
        output["navigation_reward"] = 0.0
        output["mean_latency_reward"] = float(np.mean([
            row["latency_penalty"] for row in self.step_records
        ])) if self.step_records else 0.0
        output["step_e2e_ms"] = [
            float(row["e2e_ms"]) for row in self.step_records
        ]
        # JointR is intentionally absent from Stage-2n results.
        for key in (
            "episode_jointR", "jointr_abs_error", "joint_reward",
            "jointr_delay_weight", "mean_navigation_reward",
            "mean_latency_penalty",
        ):
            output.pop(key, None)
        return output


def make_stage2n_scene_cache(encoder, device, capacity=64):
    return FixedCitySceneCache(encoder, device, max_entries=int(capacity))


class LatencyEvaluator:
    """Shared Dev/Test evaluator for learned policies and heuristics."""

    STRATEGIES = ("random", "nearest", "max_uplink_rate", "greedy_min_e2e")
    LEARNED = ("dqn", "ppo")

    def __init__(
        self,
        specs,
        topology,
        encoder,
        backends,
        quality_prior,
        gab_config,
        device,
        scene_cache=None,
    ):
        self.specs = [prepare_runner_spec(item) for item in specs]
        self.topology = topology
        self.encoder = encoder
        self.backends = backends
        self.quality_prior = quality_prior
        self.gab_config = gab_config
        self.device = device
        self.scene_cache = scene_cache or make_stage2n_scene_cache(
            encoder, device, capacity=max(64, len(self.specs))
        )

    def _runner(self, spec):
        return LatencyOnlyEpisodeRunner(
            spec=spec,
            topology=self.topology,
            encoder=self.encoder,
            backends=self.backends,
            projector=None,
            quality_prior=self.quality_prior,
            gab_config=self.gab_config,
            scene_cache=self.scene_cache,
            device=self.device,
        )

    @staticmethod
    def _row(result, strategy, actions=None):
        actions = list(actions or [])
        action_text = ",".join(str(int(action)) for action in actions)
        return {
            "strategy": str(strategy),
            "action_sha256": hashlib.sha256(
                action_text.encode("ascii")
            ).hexdigest(),
            "actions": actions,
            "episode_uid": result["episode_uid"],
            "split": result["split"],
            "episode_index": int(result["episode_index"]),
            "task_seed": int(result["seed"]),
            "band": result["band"],
            "band_id": int(result["band_id"]),
            "scenario": result["scenario"],
            "success": int(result["success"]),
            "osr": int(result["osr"]),
            "spl": float(result["spl"]),
            "ndtw": float(result["ndtw"]),
            "steps": int(result["steps"]),
            "mean_e2e_ms": float(result["mean_e2e_ms"]),
            "p95_e2e_ms": float(result["p95_e2e_ms"]),
            "delay_violation_rate": float(result["delay_violation_rate"]),
            "latency_return": float(result["latency_return"]),
            "latency_return_closed_form": float(
                result["latency_return_closed_form"]
            ),
            "latency_return_abs_error": float(
                result["latency_return_abs_error"]
            ),
            "step_e2e_ms": list(result["step_e2e_ms"]),
            "reward_mode": "latency_only_v1",
        }

    def evaluate_dqn(self, agent):
        rows = []
        for spec in self.specs:
            runner = self._runner(spec)
            while True:
                action = int(masked_argmax(
                    agent.q_values(runner.step_data["state"]),
                    runner.step_data["mask"],
                ))
                transition = runner.apply(action)
                if transition["done"]:
                    break
            actions = [item["action"] for item in runner.step_records]
            rows.append(self._row(
                runner.finalize("dqn_greedy"), "dqn", actions
            ))
        return rows

    def evaluate_ppo(self, agent):
        rows = []
        for spec in self.specs:
            runner = self._runner(spec)
            while True:
                action = int(agent.greedy_action(
                    runner.step_data["state"], runner.step_data["mask"]
                ))
                transition = runner.apply(action)
                if transition["done"]:
                    break
            actions = [item["action"] for item in runner.step_records]
            rows.append(self._row(
                runner.finalize("ppo_greedy"), "ppo", actions
            ))
        return rows

    @staticmethod
    def select_strategy(runner, strategy, rng):
        indices = np.where(np.asarray(runner.step_data["mask"], bool))[0]
        if len(indices) == 0:
            raise RuntimeError("no available MEC")
        metrics = runner.step_data["latency_metrics"]
        if strategy == "random":
            return int(rng.choice(indices))
        if strategy == "nearest":
            return int(min(
                indices,
                key=lambda i: (float(metrics[i]["distance_m"]), int(i)),
            ))
        if strategy == "max_uplink_rate":
            return int(max(
                indices,
                key=lambda i: (
                    float(metrics[i]["uplink_rate_bps"]), -int(i)
                ),
            ))
        if strategy == "greedy_min_e2e":
            return int(min(
                indices,
                key=lambda i: (float(metrics[i]["total_ms"]), int(i)),
            ))
        raise ValueError(f"unknown strategy {strategy}")

    def evaluate_strategy(self, strategy, rng_seed=1_000_003):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unknown strategy {strategy}")
        rows = []
        for index, spec in enumerate(self.specs):
            runner = self._runner(spec)
            rng = np.random.default_rng(int(rng_seed) + index)
            while True:
                node_index = self.select_strategy(runner, strategy, rng)
                transition = runner.apply(node_index)
                if transition["done"]:
                    break
            actions = [item["action"] for item in runner.step_records]
            rows.append(self._row(
                runner.finalize(strategy), strategy, actions
            ))
        return rows


def consistency_probe(inputs, device):
    """Run the same observation twice through the single frozen MEC.

    Node identity is not part of the MEC input, so different nodes must
    produce byte-identical navigation outputs for the same observation.
    """
    pool = HomogeneousBackendPool(device)
    first = pool.get(MEC_MODEL, inputs)
    second = pool.get(MEC_MODEL, inputs)
    fields = ("action", "arrival_probability", "move_action")
    equal = all(first[key] == second[key] for key in fields)
    return {
        "model": MEC_MODEL,
        "checkpoint_sha256": pool.checkpoint_sha256,
        "action": int(first["action"]),
        "arrival_probability": float(first["arrival_probability"]),
        "move_action": int(first["move_action"]),
        "identical": bool(equal),
        "fields": list(fields),
        "loaded_models": list(pool.load_calls),
        "all_nodes_one_model": list(pool.load_calls) == [MEC_MODEL],
    }


__all__ = [
    "HomogeneousBackendPool",
    "LatencyOnlyEpisodeRunner",
    "LatencyEvaluator",
    "load_stage2n_runtime",
    "make_stage2n_scene_cache",
    "consistency_probe",
]
