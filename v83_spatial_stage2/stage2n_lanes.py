"""Frozen MEC lanes for fast, exact Stage-2n evaluation.

All 20 nodes share one Transformer-M and the MEC input contains no node
identity, so the navigation action sequence and the trajectory of a task are
independent of the association policy.  We can therefore render/run the MEC
once per task ("lane") and replay any association policy over the cached
per-step observations and latency features.

For a policy with its own running mean E2E, the 384-D state is rebuilt from
the cached pose, step index and 20x19 candidate features; the MEC action and
the environment outcome come from the lane.  This is exact, not an
approximation.  ``test_stage2n.py`` checks the lane evaluator against the full
closed-loop evaluator on real tasks.

The lane builder still runs the full MEC (RGB, goal tokens, history) and is
used only for Dev validation and the final evaluation.  Training always uses
the full environment.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    build_pure_mec_v2_state,
)
from v83_spatial_stage2.stage2k_state import masked_argmax
from v83_spatial_stage2.stage2n_common import (
    latency_return_from_mean,
    latency_reward_increment,
    prepare_runner_spec,
)
from v83_spatial_stage2.stage2n_runtime import LatencyOnlyEpisodeRunner


@dataclass
class LaneStep:
    step_index: int
    pose: np.ndarray
    candidate_features: np.ndarray
    mask: np.ndarray
    latency_metrics: tuple
    action: int


@dataclass
class Lane:
    episode_uid: str
    split: str
    episode_index: int
    task_seed: int
    band: str
    band_id: int
    scenario: str
    steps: tuple
    success: bool
    osr: bool
    spl: float
    ndtw: float
    episode_steps: int
    action_sha256: str


def _choose_greedy(metrics, mask):
    indices = np.where(np.asarray(mask, bool))[0]
    return int(min(indices, key=lambda i: (float(metrics[i]["total_ms"]), i)))


def build_lane(
    spec,
    topology,
    encoder,
    backends,
    quality_prior,
    gab_config,
    device,
    scene_cache,
):
    spec = prepare_runner_spec(spec)
    runner = LatencyOnlyEpisodeRunner(
        spec=spec,
        topology=topology,
        encoder=encoder,
        backends=backends,
        projector=None,
        quality_prior=quality_prior,
        gab_config=gab_config,
        scene_cache=scene_cache,
        device=device,
    )
    steps = []
    actions = []
    while True:
        step = runner.step_data
        metrics = tuple(
            None if item is None else dict(item)
            for item in step["latency_metrics"]
        )
        node = _choose_greedy(metrics, step["mask"])
        step_index = int(step["step"])
        pose = np.asarray(step["pose"], dtype=np.float32).copy()
        candidate_features = np.asarray(
            step["candidate_features"], dtype=np.float32
        ).copy()
        mask = np.asarray(step["mask"], dtype=bool).copy()
        transition = runner.apply(node)
        steps.append(LaneStep(
            step_index=step_index,
            pose=pose,
            candidate_features=candidate_features,
            mask=mask,
            latency_metrics=metrics,
            action=int(transition["action"]),
        ))
        actions.append(int(transition["action"]))
        if transition["done"]:
            break
    result = runner.finalize("lane")
    return Lane(
        episode_uid=str(result["episode_uid"]),
        split=str(result["split"]),
        episode_index=int(result["episode_index"]),
        task_seed=int(result["seed"]),
        band=str(result["band"]),
        band_id=int(result["band_id"]),
        scenario=str(result["scenario"]),
        steps=tuple(steps),
        success=bool(result["success"]),
        osr=bool(result["osr"]),
        spl=float(result["spl"]),
        ndtw=float(result["ndtw"]),
        episode_steps=int(result["steps"]),
        action_sha256=hashlib.sha256(
            ",".join(str(action) for action in actions).encode("ascii")
        ).hexdigest(),
    )


def build_lanes(
    specs,
    topology,
    encoder,
    backends,
    quality_prior,
    gab_config,
    device,
    scene_cache,
    verbose=False,
):
    lanes = []
    for index, spec in enumerate(specs):
        if verbose and (index + 1) % 10 == 0:
            print(
                f"[lanes] built {index + 1}/{len(specs)} frozen MEC lanes",
                flush=True,
            )
        lanes.append(build_lane(
            spec, topology, encoder, backends, quality_prior, gab_config,
            device, scene_cache,
        ))
    return lanes


class LaneEvaluator:
    """Evaluate association policies on cached frozen MEC lanes."""

    STRATEGIES = ("random", "nearest", "max_uplink_rate", "greedy_min_e2e")

    def __init__(self, lanes):
        self.lanes = list(lanes)
        if not self.lanes:
            raise ValueError("empty lane set")
        self.by_uid = {lane.episode_uid: lane for lane in self.lanes}

    def _runner_row(self, lane, strategy, node_choices, mean_e2e, e2e_values,
                    latency_return):
        closed = latency_return_from_mean(mean_e2e)
        return {
            "strategy": strategy,
            "action_sha256": lane.action_sha256,
            "actions": [lane.steps[index].action
                        for index in range(len(lane.steps))],
            "episode_uid": lane.episode_uid,
            "split": lane.split,
            "episode_index": lane.episode_index,
            "task_seed": lane.task_seed,
            "band": lane.band,
            "band_id": lane.band_id,
            "scenario": lane.scenario,
            "success": int(lane.success),
            "osr": int(lane.osr),
            "spl": float(lane.spl),
            "ndtw": float(lane.ndtw),
            "steps": int(lane.episode_steps),
            "mean_e2e_ms": float(mean_e2e),
            "p95_e2e_ms": float(np.percentile(e2e_values, 95)),
            "delay_violation_rate": float(np.mean(
                np.asarray(e2e_values) > 100.0
            )),
            "latency_return": float(latency_return),
            "latency_return_closed_form": float(closed),
            "latency_return_abs_error": float(abs(latency_return - closed)),
            "step_e2e_ms": [float(value) for value in e2e_values],
            "node_choices": [int(value) for value in node_choices],
            "reward_mode": "latency_only_v1",
        }

    def _replay(self, lane, chooser, strategy):
        mean = 0.0
        latency_return = 0.0
        e2e_values = []
        node_choices = []
        for step in lane.steps:
            state = build_pure_mec_v2_state(
                step.pose[:2],
                step.step_index,
                mean,
                step.candidate_features,
            )
            node = int(chooser(state, step.mask, step.latency_metrics))
            if not bool(step.mask[node]):
                raise RuntimeError("policy selected an unavailable MEC")
            metric = step.latency_metrics[node]
            if metric is None:
                raise RuntimeError("available MEC has no latency metric")
            mean, reward = latency_reward_increment(
                metric["total_ms"], mean, step.step_index + 1
            )
            latency_return += reward
            e2e_values.append(float(metric["total_ms"]))
            node_choices.append(node)
        return self._runner_row(
            lane, strategy, node_choices, mean, e2e_values, latency_return
        )

    def evaluate_dqn(self, agent):
        def chooser(state, mask, metrics):
            return int(masked_argmax(agent.q_values(state), mask))
        return [
            self._replay(lane, chooser, "dqn")
            for lane in self.lanes
        ]

    def evaluate_ppo(self, agent):
        def chooser(state, mask, metrics):
            return int(agent.greedy_action(state, mask))
        return [
            self._replay(lane, chooser, "ppo")
            for lane in self.lanes
        ]

    def evaluate_strategy(self, strategy, rng_seed=1_000_003):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"unknown strategy {strategy}")
        rows = []
        for index, lane in enumerate(self.lanes):
            rng = np.random.default_rng(int(rng_seed) + index)

            def chooser(state, mask, metrics, rng=rng):
                indices = np.where(np.asarray(mask, bool))[0]
                if strategy == "random":
                    return int(rng.choice(indices))
                if strategy == "nearest":
                    return int(min(
                        indices,
                        key=lambda i: (float(metrics[i]["distance_m"]), i),
                    ))
                if strategy == "max_uplink_rate":
                    return int(max(
                        indices,
                        key=lambda i: (
                            float(metrics[i]["uplink_rate_bps"]), -i
                        ),
                    ))
                return int(min(
                    indices,
                    key=lambda i: (float(metrics[i]["total_ms"]), i),
                ))

            rows.append(self._replay(lane, chooser, strategy))
        return rows


__all__ = ["Lane", "LaneStep", "LaneEvaluator", "build_lane", "build_lanes"]
