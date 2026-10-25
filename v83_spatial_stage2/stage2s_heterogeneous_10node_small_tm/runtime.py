"""Heterogeneous closed-loop runtime for Stage-2s.

Two frozen Stage-2j checkpoints are loaded at the same time.  The association
policy chooses a node; the node's ``model_type`` determines which checkpoint
produces the navigation action for that step.  This is a real closed loop:

    node choice -> model -> navigation action -> UAV pose
                -> next 194-D state / availability mask -> next node choice

Inference latency is computed from the model's analytic FLOPs and the
**effective** inference throughput 0.2 TFLOP/s.  The node's nominal
2.0 TFLOP/s capacity is kept unchanged and is still what the state feature
reports; the two numbers are deliberately separate and both are logged.

No Stage-2r frozen lane cache is used anywhere in this module.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP  # noqa: E402
from v83_spatial_stage2.stage2j_eval import load_checkpoint  # noqa: E402
from v83_spatial_stage2.stage2j_model import build_stage2j_model  # noqa: E402
from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    STOP_THRESHOLD,
    latency_penalty,
    sha256_file,
)
from v83_spatial_stage2.stage2k_latency import Stage2KLatency  # noqa: E402
from v83_spatial_stage2.stage2k_runtime import load_encoder  # noqa: E402
from v83_spatial_stage2.stage2k_state import masked_argmax  # noqa: E402
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    prepare_runner_spec,
    summarize_episode_rows,
)
from v83_spatial_stage2.stage2n_runtime import (  # noqa: E402
    LatencyOnlyEpisodeRunner,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S,
    MEC_MODELS,
    MEC_NOMINAL_COMPUTE_FLOPS_PER_S,
    TAU_MS,
)
from v83_spatial_stage2.stage2l_runtime import (  # noqa: E402
    FixedCitySceneCache,
)


class HeterogeneousLatency:
    """E2E latency with explicit effective inference throughput.

    ``Stage2KLatency`` already gives extraction, channel/communication and
    model FLOPs.  This wrapper replaces only the inference term:

        inference_ms = 1000 * model_flops / effective_compute_flops_per_s

    The nominal ``compute_capacity_flops_per_s`` reported by the base model
    is preserved so the 194-D state keeps its original compute feature.
    No fixed per-model latency is written by hand.
    """

    def __init__(
        self,
        effective_compute_flops_per_s=MEC_EFFECTIVE_INFERENCE_FLOPS_PER_S,
        base: Stage2KLatency | None = None,
    ):
        self.base = base or Stage2KLatency()
        self.effective_compute_flops_per_s = float(
            effective_compute_flops_per_s
        )
        if self.effective_compute_flops_per_s <= 0.0:
            raise ValueError("effective inference throughput must be positive")

    @property
    def extraction_ms(self) -> float:
        return float(self.base.extraction_ms)

    @property
    def deadline_ms(self) -> float:
        return float(self.base.deadline_ms)

    def evaluate(self, node, uav_position, model_name=None):
        nominal = self.base.evaluate(node, uav_position, model_name)
        flops = float(nominal["inference_flops"])
        inference_ms = (
            1000.0 * flops / self.effective_compute_flops_per_s
        )
        total_ms = (
            float(nominal["extraction_ms"])
            + float(nominal["communication_ms"])
            + inference_ms
        )
        output = dict(nominal)
        output.update({
            "inference_ms": float(inference_ms),
            "total_ms": float(total_ms),
            "normalized_e2e": float(total_ms / self.deadline_ms),
            "deadline_violation": bool(total_ms > self.deadline_ms),
            "latency_penalty": float(latency_penalty(total_ms)),
            "nominal_compute_capacity_flops_per_s": float(
                nominal["compute_capacity_flops_per_s"]
            ),
            "effective_inference_compute_flops_per_s": float(
                self.effective_compute_flops_per_s
            ),
            "inference_compute_flops_per_s": float(
                self.effective_compute_flops_per_s
            ),
        })
        return output

    def describe(self):
        payload = dict(self.base.describe())
        payload.update({
            "effective_inference_compute_flops_per_s": float(
                self.effective_compute_flops_per_s
            ),
            "effective_inference_compute_tflops": float(
                self.effective_compute_flops_per_s / 1e12
            ),
            "nominal_compute_capacity_flops_per_s": float(
                MEC_NOMINAL_COMPUTE_FLOPS_PER_S
            ),
            "semantics": (
                "nominal capacity is reported in topology/state; inference "
                "latency uses effective_inference_compute_flops_per_s only"
            ),
        })
        return payload


class HeterogeneousBackendPool:
    """Both frozen checkpoints, dispatched by the selected node's model."""

    def __init__(self, device, checkpoints, expected_sha256):
        self.device = torch.device(device)
        self.models = {}
        self.checkpoint_paths = {}
        self.checkpoint_sha256 = {}
        for model_name in MEC_MODELS:
            path = Path(checkpoints[model_name])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256_file(path)
            if actual != expected_sha256[model_name]:
                raise ValueError(
                    f"{model_name} checkpoint SHA-256 {actual} != "
                    f"{expected_sha256[model_name]}"
                )
            model = build_stage2j_model(model_name).to(self.device)
            load_checkpoint(model, path, self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            self.models[model_name] = model
            self.checkpoint_paths[model_name] = path
            self.checkpoint_sha256[model_name] = actual
        self.forward_calls = Counter()
        self.reset_counts()

    def reset_counts(self):
        self.forward_calls = {model: 0 for model in MEC_MODELS}
        self.dispatch_calls = {model: 0 for model in MEC_MODELS}

    @property
    def loaded_models(self):
        return tuple(sorted(self.models))

    @torch.inference_mode()
    def get(self, model_name, inputs, cache=None):
        model_name = str(model_name)
        if model_name not in self.models:
            raise ValueError(
                f"heterogeneous pool has no model {model_name!r}; "
                f"loaded={self.loaded_models}"
            )
        self.dispatch_calls[model_name] = (
            self.dispatch_calls.get(model_name, 0) + 1
        )
        if cache is not None and model_name in cache:
            return cache[model_name]
        components = self.models[model_name].forward_components(
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
            "model_type": model_name,
            "move_logits": move_logits.detach().cpu().numpy(),
            "move_probabilities": move_probabilities.detach().cpu().numpy(),
            "arrival_probability": arrival_probability,
            "joint_probabilities": joint,
            "move_action": move_action,
            "action": int(action),
        }
        self.forward_calls[model_name] = (
            self.forward_calls.get(model_name, 0) + 1
        )
        if cache is not None:
            cache[model_name] = output
        return output

    def provenance(self):
        return {
            "loaded_models": list(self.loaded_models),
            "checkpoints": {
                model: str(self.checkpoint_paths[model])
                for model in MEC_MODELS
            },
            "checkpoint_sha256": dict(self.checkpoint_sha256),
            "forward_calls": dict(self.forward_calls),
            "dispatch_calls": dict(self.dispatch_calls),
            "fully_frozen": all(
                not parameter.requires_grad
                for model in self.models.values()
                for parameter in model.parameters()
            ),
        }


class Stage2sEpisodeRunner(LatencyOnlyEpisodeRunner):
    """LatencyOnly runner with heterogeneous bookkeeping and scheme D.

    The inherited ``apply`` already dispatches to
    ``backends.get(node["model_type"], ...)`` and then calls the simulator,
    so the navigation action is produced by the model of the selected node.
    This subclass only adds provenance fields and the closed-form scheme-D
    episode return; it never changes the dynamics or the policy input.
    """

    def apply(self, node_index):
        step_data = self.step_data
        node_index = int(node_index)
        metric = None
        metrics = step_data.get("latency_metrics") or []
        if 0 <= node_index < len(metrics):
            metric = metrics[node_index]
        transition = super().apply(node_index)
        record = self.step_records[-1]
        node = self.nodes[node_index]
        record["model_type"] = str(node["model_type"])
        record["model_index"] = int(node["model_index"])
        record["node_id"] = str(node["node_id"])
        if metric is not None:
            record["extraction_ms"] = float(metric["extraction_ms"])
            record["communication_ms"] = float(
                metric["communication_ms"]
            )
            record["inference_ms"] = float(metric["inference_ms"])
            record["e2e_ms"] = float(metric["total_ms"])
            record["nominal_compute_capacity_flops_per_s"] = float(
                metric["nominal_compute_capacity_flops_per_s"]
            )
            record["effective_inference_compute_flops_per_s"] = float(
                metric["effective_inference_compute_flops_per_s"]
            )
            transition["e2e_ms"] = float(metric["total_ms"])
        transition["model_type"] = str(node["model_type"])
        transition["node_id"] = str(node["node_id"])
        return transition

    def finalize(self, strategy):
        output = super().finalize(strategy)
        records = list(self.step_records)
        mean_e2e = float(output["mean_episode_e2e_ms"])
        reward_d = float(1.0 - mean_e2e / TAU_MS)
        steps = int(output["steps"])
        horizon = 400
        closed = (
            float(np.mean([
                1.0 - float(row["e2e_ms"]) / TAU_MS for row in records
            ])) if records else 0.0
        )
        # The scheme-D identity is exact: per-step utility plus the terminal
        # correction telescopes to 1 - mean_E2E/tau.
        reconstructed = float(
            sum(
                (1.0 - float(row["e2e_ms"]) / TAU_MS) / horizon
                for row in records
            )
        ) if records else 0.0
        if steps:
            reconstructed += (
                1.0 - steps / horizon
            ) * reward_d
        stats = latency_stats(records)
        output.update({
            "reward_D": reward_d,
            "reward_D_closed_form": float(1.0 - mean_e2e / TAU_MS),
            "reward_D_abs_error": abs(reward_d - closed),
            "reward_D_reconstructed": reconstructed,
            "reward_D_reconstruction_error": abs(reconstructed - reward_d),
            "latency_return_stage2n": float(
                output.get("latency_return", 0.0)
            ),
            **stats,
        })
        output["model_usage"] = {
            model: int(stats["steps_per_model"].get(model, 0))
            for model in MEC_MODELS
        }
        output["step_node_indices"] = [
            int(row["node_index"]) for row in records
        ]
        output["step_model_types"] = [
            str(row["model_type"]) for row in records
        ]
        output["step_inference_ms"] = [
            float(row.get("inference_ms", 0.0)) for row in records
        ]
        output["step_communication_ms"] = [
            float(row.get("communication_ms", 0.0)) for row in records
        ]
        output["step_extraction_ms"] = [
            float(row.get("extraction_ms", 0.0)) for row in records
        ]
        return output


def make_scene_cache(encoder, device, capacity=64):
    return FixedCitySceneCache(encoder, device, max_entries=int(capacity))


def load_stage2s_runtime(device="auto"):
    if device is None or str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    encoder = load_encoder(device)
    backends = HeterogeneousBackendPool(
        device,
        checkpoints={
            model: str(path)
            for model, path in _checkpoint_paths().items()
        },
        expected_sha256=_checkpoint_hashes(),
    )
    latency = HeterogeneousLatency()
    return device, encoder, backends, latency


def _checkpoint_paths():
    from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (
        MEC_CHECKPOINTS,
    )

    return dict(MEC_CHECKPOINTS)


def _checkpoint_hashes():
    from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (
        MEC_CHECKPOINT_SHA256,
    )

    return dict(MEC_CHECKPOINT_SHA256)


def runner_for(bundle, spec, encoder, backends, scene_cache, device,
               latency_model):
    return Stage2sEpisodeRunner(
        spec=spec,
        topology=bundle["topology"],
        encoder=encoder,
        backends=backends,
        projector=None,
        quality_prior=bundle["quality_prior"],
        gab_config=bundle["gab_config"],
        scene_cache=scene_cache,
        device=device,
        latency_model=latency_model,
    )


def scheme_d_return(mean_e2e_ms):
    """Scheme-D episode return, exactly ``1 - mean_E2E/tau``."""
    return float(1.0 - float(mean_e2e_ms) / TAU_MS)


def add_scheme_d(rows):
    for row in rows:
        mean_e2e = float(row["mean_e2e_ms"])
        row["reward_D"] = scheme_d_return(mean_e2e)
        row["reward_D_closed_form"] = float(1.0 - mean_e2e / TAU_MS)
        row["reward_D_abs_error"] = abs(
            row["reward_D"] - row["reward_D_closed_form"]
        )
        row["scheme"] = "D"
    return rows


def latency_stats(records: Sequence[Mapping]) -> dict:
    """Counts and latency decomposition over one or more step records."""
    records = list(records)
    steps = len(records)
    per_model_steps = {model: 0 for model in MEC_MODELS}
    per_model = {
        model: {"inference": [], "communication": [], "e2e": []}
        for model in MEC_MODELS
    }
    for row in records:
        model = str(row.get("model_type", ""))
        if model not in per_model:
            continue
        per_model_steps[model] += 1
        per_model[model]["inference"].append(
            float(row.get("inference_ms", 0.0))
        )
        per_model[model]["communication"].append(
            float(row.get("communication_ms", 0.0))
        )
        per_model[model]["e2e"].append(float(row.get("e2e_ms", 0.0)))

    def _mean(values):
        return float(np.mean(values)) if values else float("nan")

    output = {
        "steps": int(steps),
        "steps_per_model": per_model_steps,
        "small_steps": int(per_model_steps["Small"]),
        "transformer_m_steps": int(
            per_model_steps["Transformer-M"]
        ),
        "small_ratio": (
            float(per_model_steps["Small"]) / steps if steps else 0.0
        ),
        "transformer_m_ratio": (
            float(per_model_steps["Transformer-M"]) / steps if steps else 0.0
        ),
        "mean_inference_ms": _mean([
            float(row.get("inference_ms", 0.0)) for row in records
        ]),
        "mean_communication_ms": _mean([
            float(row.get("communication_ms", 0.0)) for row in records
        ]),
        "mean_extraction_ms": _mean([
            float(row.get("extraction_ms", 0.0)) for row in records
        ]),
        "mean_step_e2e_ms": _mean([
            float(row.get("e2e_ms", 0.0)) for row in records
        ]),
    }
    for model, slug in (
        ("Small", "small"), ("Transformer-M", "transformer_m")
    ):
        output[f"mean_inference_{slug}_ms"] = _mean(
            per_model[model]["inference"]
        )
        output[f"mean_communication_{slug}_ms"] = _mean(
            per_model[model]["communication"]
        )
        output[f"mean_e2e_{slug}_ms"] = _mean(per_model[model]["e2e"])
    return output


def dual_model_probe(runner) -> dict:
    """Run both frozen models on the same observation (no state change)."""
    inputs = runner.step_data["inputs"]
    cache = {}
    outputs = {
        model: runner.backends.get(model, inputs, cache=cache)
        for model in MEC_MODELS
    }
    latencies = {}
    for index, node in enumerate(runner.nodes):
        metric = runner.step_data["latency_metrics"][index]
        if metric is None:
            continue
        latencies.setdefault(node["model_type"], {
            "inference_ms": float(metric["inference_ms"]),
            "communication_ms": float(metric["communication_ms"]),
            "total_ms": float(metric["total_ms"]),
            "node_id": node["node_id"],
            "node_index": index,
        })
    return {
        "step": int(runner.step_data["step"]),
        "actions": {
            model: int(outputs[model]["action"]) for model in MEC_MODELS
        },
        "arrival_probability": {
            model: float(outputs[model]["arrival_probability"])
            for model in MEC_MODELS
        },
        "move_action": {
            model: int(outputs[model]["move_action"])
            for model in MEC_MODELS
        },
        "latencies": latencies,
        "actions_differ": (
            int(outputs["Small"]["action"])
            != int(outputs["Transformer-M"]["action"])
        ),
    }


class HeterogeneousEvaluator:
    """Closed-loop Dev evaluation for learned policies and heuristics."""

    STRATEGIES = ("random", "nearest", "max_uplink_rate", "greedy_min_e2e")
    LEARNED = ("dqn", "ppo")

    def __init__(self, bundle, encoder, backends, scene_cache, device,
                 latency_model):
        self.bundle = bundle
        self.specs = [prepare_runner_spec(item) for item in bundle["dev_specs"]]
        self.encoder = encoder
        self.backends = backends
        self.scene_cache = scene_cache
        self.device = device
        self.latency_model = latency_model

    def _runner(self, spec):
        return runner_for(
            self.bundle, spec, self.encoder, self.backends,
            self.scene_cache, self.device, self.latency_model,
        )

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

    def _row(self, result, strategy, actions, nodes, models):
        e2e = np.asarray(
            [float(row["e2e_ms"]) for row in result["step_records"]],
            dtype=np.float64,
        )
        stats = latency_stats(result["step_records"])
        row = {
            "strategy": str(strategy),
            "action_sha256": _sha256_ints(actions),
            "actions": [int(value) for value in actions],
            "node_choices": [int(value) for value in nodes],
            "model_choices": [str(value) for value in models],
            "episode_uid": result["episode_uid"],
            "split": result["split"],
            "episode_index": int(result["episode_index"]),
            "task_seed": int(result["seed"]),
            "band": result["band"],
            "band_id": int(result["band_id"]),
            "scenario": result["scenario"],
            "success": int(bool(result["success"])),
            "osr": int(bool(result["osr"])),
            "spl": float(result["spl"]),
            "ndtw": float(result["ndtw"]),
            "steps": int(result["steps"]),
            "mean_e2e_ms": float(result["mean_episode_e2e_ms"]),
            "p95_e2e_ms": float(result["p95_e2e_ms"]),
            "max_e2e_ms": float(result["max_e2e_ms"]),
            "delay_violation_rate": float(result["delay_violation_rate"]),
            "latency_return": float(result["latency_return"]),
            "latency_return_closed_form": float(
                result["latency_return_closed_form"]
            ),
            "latency_return_abs_error": float(
                result["latency_return_abs_error"]
            ),
            "reward_D": float(result["reward_D"]),
            "reward_D_abs_error": float(result["reward_D_abs_error"]),
            "reward_mode": "stage2s_scheme_D_heterogeneous",
            "step_e2e_ms": [float(value) for value in e2e],
            **stats,
        }
        return row

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
            result = runner.finalize("dqn_greedy")
            rows.append(self._row(
                result, "dqn",
                [row["action"] for row in runner.step_records],
                [row["node_index"] for row in runner.step_records],
                [row["model_type"] for row in runner.step_records],
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
            result = runner.finalize("ppo_greedy")
            rows.append(self._row(
                result, "ppo",
                [row["action"] for row in runner.step_records],
                [row["node_index"] for row in runner.step_records],
                [row["model_type"] for row in runner.step_records],
            ))
        return rows

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
            result = runner.finalize(strategy)
            rows.append(self._row(
                result, strategy,
                [row["action"] for row in runner.step_records],
                [row["node_index"] for row in runner.step_records],
                [row["model_type"] for row in runner.step_records],
            ))
        return rows

    def evaluate_forced_model(self, agent, algorithm, forced_model,
                              rng_seed=20261025):
        """Diagnostic: force every available step to one model type.

        If no node of the forced type is inside coverage, the step falls back
        to the unrestricted greedy action and is counted explicitly.  This is
        a diagnostic only, never part of the pre-registered comparison.
        """
        forced_model = str(forced_model)
        if forced_model not in MEC_MODELS:
            raise ValueError(forced_model)
        model_mask = np.asarray([
            node["model_type"] == forced_model for node in self.bundle[
                "topology"
            ]["nodes"]
        ], dtype=bool)
        rows = []
        for index, spec in enumerate(self.specs):
            runner = self._runner(spec)
            fallbacks = 0
            while True:
                mask = np.asarray(runner.step_data["mask"], bool)
                forced = mask & model_mask
                if not forced.any():
                    forced = mask
                    fallbacks += 1
                if algorithm == "dqn":
                    action = int(masked_argmax(
                        agent.q_values(runner.step_data["state"]), forced
                    ))
                else:
                    action = int(agent.greedy_action(
                        runner.step_data["state"], forced
                    ))
                transition = runner.apply(action)
                if transition["done"]:
                    break
            result = runner.finalize(
                f"{algorithm}_forced_{forced_model.lower()}"
            )
            row = self._row(
                result, f"forced_{forced_model.lower()}",
                [record["action"] for record in runner.step_records],
                [record["node_index"] for record in runner.step_records],
                [record["model_type"] for record in runner.step_records],
            )
            row["forced_model"] = forced_model
            row["forced_fallback_steps"] = int(fallbacks)
            rows.append(row)
        return rows


def _sha256_ints(values):
    text = ",".join(str(int(value)) for value in values)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def summarize_rows(rows):
    overall = dict(summarize_episode_rows(rows).get("overall", {}))
    overall["RewardD"] = float(1.0 - float(overall["E2E_ms"]) / TAU_MS)
    return overall


__all__ = [
    "HeterogeneousLatency", "HeterogeneousBackendPool",
    "Stage2sEpisodeRunner", "HeterogeneousEvaluator",
    "load_stage2s_runtime", "make_scene_cache", "runner_for",
    "scheme_d_return", "add_scheme_d", "latency_stats",
    "dual_model_probe", "summarize_rows",
]
