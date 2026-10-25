"""Exact frozen-lane episode runner for Stage-2r training.

The adapter deliberately exposes the same small interface used by the
Stage-2r DQN/PPO loops (``step_data``, ``apply``, ``finalize``) as
``LatencyOnlyEpisodeRunner``.  Every step rebuilds the 194-D Pure-MEC v2
state from the cached pose, candidate features and the policy's own running
mean E2E.  Navigation, success, SPL and nDTW come from the real closed-loop
lane and do not depend on the association action.  Only node choice, E2E and
the running mean change.

This is the same exactness argument as ``stage2n_lanes.LaneEvaluator``; the
Stage-2r equivalence test compares this adapter with the full runner on real
Dev tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import ACTION_NAMES  # noqa: E402
from v83_spatial_stage2.stage2k_common import latency_penalty  # noqa: E402
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (  # noqa: E402
    build_pure_mec_v2_state,
)
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    latency_return_from_mean,
)


class LaneRunnerAdapter:
    def __init__(self, episode, spec):
        self.episode = episode
        self.spec = dict(spec)
        self.episode_uid = str(episode["episode_uid"])
        self.steps_total = int(episode["steps"])
        self.pose = np.asarray(episode["pose"], dtype=np.float64)
        self.candidate = np.asarray(
            episode["candidate"], dtype=np.float32
        )
        self.mask_all = np.asarray(episode["mask"], dtype=bool)
        self.total_ms = np.asarray(
            episode["total_ms"], dtype=np.float64
        )
        self.distance_m = np.asarray(
            episode["distance_m"], dtype=np.float64
        )
        self.actions = np.asarray(episode["actions"], dtype=np.int64)
        self.step_index_all = np.asarray(
            episode["step_index"], dtype=np.int64
        )
        if not (
            len(self.pose) == len(self.candidate) == len(self.mask_all)
            == len(self.total_ms) == len(self.actions) == self.steps_total
        ):
            raise ValueError(f"inconsistent lane arrays for {self.episode_uid}")
        self.step_index = 0
        self.running_mean_e2e_ms = 0.0
        self.step_records = []
        self.return_sum = 0.0
        self.finalized = False
        self.step_data = None
        self._build_step_data()

    def _metrics(self, index):
        mask = self.mask_all[index]
        metrics = []
        for node in range(self.mask_all.shape[1]):
            if not bool(mask[node]):
                metrics.append(None)
                continue
            metrics.append({
                "total_ms": float(self.total_ms[index, node]),
                "distance_m": float(self.distance_m[index, node]),
                "communication_ms": 0.0,
                "inference_ms": 0.0,
                "uplink_rate_mbps": 0.0,
                "deadline_violation": bool(
                    self.total_ms[index, node] > 100.0
                ),
            })
        return metrics

    def _build_step_data(self):
        index = self.step_index
        if index >= self.steps_total:
            raise RuntimeError(f"lane episode {self.episode_uid} exhausted")
        state = build_pure_mec_v2_state(
            self.pose[index, :2],
            int(self.step_index_all[index]),
            self.running_mean_e2e_ms,
            self.candidate[index],
        )
        self.step_data = {
            "step": int(self.step_index_all[index]),
            "pose": self.pose[index].copy(),
            "state": state,
            "mask": self.mask_all[index].copy(),
            "candidate_features": self.candidate[index].copy(),
            "latency_metrics": self._metrics(index),
            "fallback": False,
        }
        return self.step_data

    def apply(self, node_index):
        if self.finalized:
            raise RuntimeError("lane episode already finalized")
        node_index = int(node_index)
        index = self.step_index
        state = self.step_data["state"]
        mask = self.step_data["mask"]
        if not bool(mask[node_index]):
            raise ValueError(f"node {node_index} is masked")
        e2e_ms = float(self.total_ms[index, node_index])
        step_index = index + 1
        self.running_mean_e2e_ms += (
            e2e_ms - self.running_mean_e2e_ms
        ) / float(step_index)
        nav_action = int(self.actions[index])
        done = step_index >= self.steps_total
        success = bool(self.episode["success"]) if done else False
        self.step_records.append({
            "step": int(self.step_index_all[index]),
            "node_index": node_index,
            "node_id": f"M{node_index + 1:02d}",
            "action": nav_action,
            "action_name": ACTION_NAMES[nav_action],
            "e2e_ms": e2e_ms,
            "distance_m": float(self.distance_m[index, node_index]),
            "running_mean_e2e_ms": float(self.running_mean_e2e_ms),
            "success": bool(success),
            "done": bool(done),
            "navigation_reward": 0.0,
            "latency_penalty": float(latency_penalty(e2e_ms)),
            "distance_to_goal_m": float("nan"),
        })
        self.step_index = step_index
        next_state = None
        next_mask = None
        if not done:
            self._build_step_data()
            next_state = self.step_data["state"]
            next_mask = self.step_data["mask"]
        return {
            "state": state,
            "mask": mask,
            "node_index": node_index,
            "action": nav_action,
            "reward": 0.0,
            "nav_reward": 0.0,
            "latency_penalty": float(latency_penalty(e2e_ms)),
            "e2e_ms": e2e_ms,
            "done": bool(done),
            "success": bool(success),
            "next_state": next_state,
            "next_mask": next_mask,
        }

    def finalize(self, strategy="lane"):
        if self.finalized:
            raise RuntimeError("lane episode already finalized")
        if self.step_index != self.steps_total:
            raise RuntimeError(
                f"finalize after {self.step_index}/{self.steps_total} steps"
            )
        self.finalized = True
        e2e = np.asarray(
            [row["e2e_ms"] for row in self.step_records], dtype=np.float64
        )
        mean_e2e = float(e2e.mean()) if len(e2e) else 0.0
        latency_return = float(latency_return_from_mean(mean_e2e))
        closed = float(latency_return_from_mean(mean_e2e))
        episode = self.episode
        return {
            "strategy": str(strategy),
            "episode_uid": self.episode_uid,
            "split": str(episode["split"]),
            "scenario": str(episode["scenario"]),
            "episode_index": int(episode["episode_index"]),
            "seed": int(self.spec["task_seed"]),
            "band": str(episode["band"]),
            "band_id": int(episode["band_id"]),
            "success": bool(episode["success"]),
            "osr": bool(episode["osr"]),
            "ndtw": float(episode["ndtw"]),
            "spl": float(episode["spl"]),
            "steps": int(self.steps_total),
            "path_length_m": float(episode.get("path_length_m", 0.0)),
            "start_goal_distance_m": float(
                self.spec.get("start_goal_distance_m", 0.0)
            ),
            "mean_e2e_ms": mean_e2e,
            "p95_e2e_ms": float(np.percentile(e2e, 95)),
            "max_e2e_ms": float(e2e.max()) if len(e2e) else 0.0,
            "delay_violation_rate": float(np.mean(e2e > 100.0)),
            "mean_navigation_reward": 0.0,
            "mean_latency_penalty": float(np.mean([
                row["latency_penalty"] for row in self.step_records
            ])) if self.step_records else 0.0,
            "model_usage": {self.spec.get("model_type", ""): self.steps_total},
            "fallback_steps": 0,
            "mean_episode_e2e_ms": mean_e2e,
            "final_ndtw": float(episode["ndtw"]),
            "episode_return": latency_return,
            "latency_return": latency_return,
            "latency_return_closed_form": closed,
            "latency_return_abs_error": abs(latency_return - closed),
            "running_mean_e2e_ms": mean_e2e,
            "trajectory": self.pose.copy(),
            "step_records": self.step_records,
            "step_e2e_ms": [float(value) for value in e2e],
            "reward_mode": "stage2r_frozen_lane",
        }


__all__ = ["LaneRunnerAdapter"]
