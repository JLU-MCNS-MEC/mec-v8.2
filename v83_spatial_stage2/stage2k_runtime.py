"""Stage-2k closed-loop runtime: frozen T5 MECs + association policies.

The five Stage-2j MEC checkpoints are loaded once, kept in eval mode with
``requires_grad=False``, and never optimized.  The DQN only chooses a MEC
node; the selected frozen MEC produces the navigation action.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, NETWORK
from env import STOP, ToyUAVEnv
from gbs_model_v3 import goal_current_similarity
from navigation_reward import NavigationRewardTracker
from token_data import (
    encode_multiscale_goal_tokens,
    goal_conditioned_spatial_heatmap,
)
from utils import image_to_tensor
from v83_spatial_stage2.stage2h_gab import GABConfig, GABState
from v83_spatial_stage2.stage2i_data import (
    build_landmark_map,
    map_retrieval_features,
    online_history_arrays,
    visual_descriptor,
)
from v83_spatial_stage2.stage2j_eval import load_checkpoint
from v83_spatial_stage2.stage2j_model import build_stage2j_model
from v83_spatial_stage2.stage2k_common import (
    ACTION_NAMES,
    MAX_STEPS,
    MODEL_NAMES,
    STOP_THRESHOLD,
    build_oracle_reference,
    joint_reward,
    latency_penalty,
)
from v83_spatial_stage2.stage2k_jointr_reward import (
    aligned_episode_jointr,
    delay_reward_increment,
)
from v83_spatial_stage2.stage2k_latency import Stage2KLatency
from v83_spatial_stage2.stage2k_state import (
    build_state,
    candidate_features,
    global_features,
    visual_summary,
)
from v83_spatial_training.eval_closedloop_spatial import load_encoder


GAB_MANIFEST = Path("runs/v83_spatial_stage2/stage2i/aux/manifest.json")
DEFAULT_CHECKPOINT_DIR = Path("runs/v83_spatial_stage2/stage2j")


def load_gab_config():
    payload = json.loads(GAB_MANIFEST.read_text(encoding="utf-8"))
    return GABConfig(
        reliability_threshold=float(payload["gab_reliability_threshold"])
    )


def checkpoint_path(model_name):
    return DEFAULT_CHECKPOINT_DIR / model_name / "best.pt"


class BackendPool:
    """The five frozen Stage-2j MECs."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.models = {}
        for name in MODEL_NAMES:
            model = build_stage2j_model(name).to(self.device)
            load_checkpoint(model, checkpoint_path(name), self.device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            self.models[name] = model

    @torch.inference_mode()
    def get(self, model_name, inputs, cache=None):
        if cache is not None and model_name in cache:
            return cache[model_name]
        model = self.models[model_name]
        components = model.forward_components(
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
        if cache is not None:
            cache[model_name] = output
        return output


def build_scene(spec, encoder, device):
    """Goal tokens + anonymous map, cached per episode."""
    env = ToyUAVEnv()
    env.reset(
        seed=int(spec["seed"]),
        goal_distance_range_m=tuple(spec["distance_range"]),
        render_observation=False,
    )
    goal_tokens = encode_multiscale_goal_tokens(
        env.get_goal_reference_multiscale_rgbs(), encoder, device
    ).unsqueeze(0).to(device=device, dtype=torch.float32)
    landmark_map = build_landmark_map(
        env, int(spec["seed"]), encoder, device
    )
    return {
        "goal_tokens": goal_tokens,
        "goal_token_np": goal_tokens[0, 2].detach().cpu().float().numpy(),
        "landmark_map": landmark_map,
    }


class SceneCache:
    def __init__(self, encoder, device, max_entries=None):
        self.encoder = encoder
        self.device = device
        self.max_entries = max_entries
        self.cache = {}
        self.order = []

    def get(self, spec):
        key = (int(spec["seed"]), tuple(spec["distance_range"]))
        if key in self.cache:
            return self.cache[key]
        scene = build_scene(spec, self.encoder, self.device)
        if self.max_entries is not None:
            if len(self.order) >= self.max_entries:
                oldest = self.order.pop(0)
                self.cache.pop(oldest, None)
            self.order.append(key)
        self.cache[key] = scene
        return scene


class EpisodeRunner:
    """One closed-loop episode with the frozen MEC input contract."""

    def __init__(
        self,
        spec,
        topology,
        encoder,
        backends,
        projector,
        quality_prior,
        gab_config,
        latency_model=None,
        scene=None,
        scene_cache=None,
        device=None,
        jointr_aligned=False,
    ):
        self.jointr_aligned = bool(jointr_aligned)
        self.spec = spec
        self.topology = topology
        self.nodes = topology["nodes"]
        self.encoder = encoder
        self.backends = backends
        self.projector = projector
        self.quality_prior = quality_prior
        self.gab_config = gab_config
        self.latency_model = latency_model or Stage2KLatency()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scene = scene or (
            scene_cache.get(spec) if scene_cache is not None
            else build_scene(spec, encoder, self.device)
        )
        self.reference = build_oracle_reference(spec)
        self.reset()

    def reset(self):
        self.env = ToyUAVEnv()
        self.observation = self.env.reset(
            seed=int(self.spec["seed"]),
            goal_distance_range_m=tuple(self.spec["distance_range"]),
            render_observation=True,
        )
        reproduced = float(np.linalg.norm(self.env.position - self.env.goal))
        expected = float(self.spec["start_goal_distance_m"])
        if abs(reproduced - expected) > 1e-3:
            raise RuntimeError(
                f"Episode {self.spec['episode_uid']} start distance "
                f"{reproduced} != {expected}"
            )
        self.goal_tokens = self.scene["goal_tokens"]
        self.goal_token_np = self.scene["goal_token_np"]
        self.landmark_map = self.scene["landmark_map"]
        self.gab_state = GABState(self.gab_config)
        self.past = []
        self.previous_action = -1
        self.previous_selected = -1
        self.previous_e2e_ms = 0.0
        self.running_mean_e2e_ms = 0.0
        self.running_mean_steps = 0
        self.return_sum = 0.0
        self.step_index = 0
        self.trajectory = [self.env.position.copy()]
        self.success = False
        self.osr = bool(self.env.distance_to_goal < self.env.stop_radius)
        self.ended_by_stop = False
        self.model_usage = Counter()
        self.step_records = []
        self.fallback_steps = 0
        self.reward_tracker = NavigationRewardTracker(
            reference_trajectory=self.reference["reference"],
            start_position=self.observed_start,
            distance_threshold=self.env.stop_radius,
            max_forward_progress=None,
        )
        self._build_step_data(self.observation)

    @property
    def observed_start(self):
        return self.reference["start"]

    def _build_step_data(self, observation):
        with torch.no_grad():
            current_tokens = self.encoder(
                image_to_tensor(observation, self.device)
            )
            heatmap = goal_conditioned_spatial_heatmap(
                current_tokens, self.goal_tokens
            )
            similarity = goal_current_similarity(
                current_tokens, self.goal_tokens
            )[0]
        history = online_history_arrays(self.past)
        self.gab_state.update(
            heatmap[0].detach().cpu().float().numpy(),
            self.previous_action,
        )
        gab = self.gab_state.features()
        pose = self.env.pose.copy()
        map_feat = map_retrieval_features(
            pose,
            self.landmark_map["positions"],
            self.landmark_map["embeddings"],
            self.goal_token_np,
        )
        current_goal_similarity = float(similarity.max().item())
        visual = visual_descriptor(current_tokens)[0].cpu().float().numpy()
        summary = visual_summary(
            current_tokens[0].detach().cpu().float().numpy(),
            self.goal_tokens[0].detach().cpu().float().numpy(),
            history[0],
            history[3],
            gab,
            map_feat,
        )
        visual128 = self.projector.project(summary)
        candidate, mask, metrics, fallback = candidate_features(
            self.nodes,
            pose,
            self.latency_model,
            previous_selected=self.previous_selected,
            quality_prior=self.quality_prior,
        )
        state_e2e_ms = (
            self.running_mean_e2e_ms
            if self.jointr_aligned else self.previous_e2e_ms
        )
        global10 = global_features(
            self.step_index,
            MAX_STEPS,
            history[3],
            current_goal_similarity,
            gab,
            state_e2e_ms,
            self.previous_action,
            clip_e2e=not self.jointr_aligned,
        )
        state = build_state(visual128, global10, candidate)
        inputs = {
            "current_tokens": current_tokens,
            "goal_tokens": self.goal_tokens,
            "history_vis": torch.from_numpy(history[0]).unsqueeze(0).to(
                self.device
            ),
            "history_pose": torch.from_numpy(history[1]).unsqueeze(0).to(
                self.device
            ),
            "history_action": torch.from_numpy(history[2]).unsqueeze(0).to(
                self.device
            ),
            "history_mask": torch.from_numpy(history[3]).unsqueeze(0).to(
                self.device
            ),
            "gab": torch.from_numpy(gab).unsqueeze(0).to(self.device),
            "map_feat": torch.from_numpy(map_feat).unsqueeze(0).to(
                self.device
            ),
        }
        self.step_data = {
            "step": int(self.step_index),
            "pose": pose,
            "current_tokens": current_tokens,
            "visual": visual,
            "history_mask": history[3],
            "gab": gab,
            "map_feat": map_feat,
            "current_goal_similarity": current_goal_similarity,
            "state": state,
            "mask": mask,
            "candidate_features": candidate,
            "latency_metrics": metrics,
            "fallback": bool(fallback),
            "inputs": inputs,
        }
        if fallback:
            self.fallback_steps += 1

    def available_indices(self):
        return np.where(self.step_data["mask"])[0]

    def select_node(self, strategy, rng=None):
        indices = self.available_indices()
        metrics = self.step_data["latency_metrics"]
        if strategy == "random":
            if rng is None:
                raise ValueError("random requires rng")
            return int(rng.choice(indices))
        if strategy == "min_latency":
            return int(min(
                indices,
                key=lambda i: (metrics[i]["total_ms"], i),
            ))
        if strategy == "max_quality":
            return int(max(
                indices,
                key=lambda i: (
                    self.nodes[i]["quality_prior"],
                    -metrics[i]["total_ms"],
                    -i,
                ),
            ))
        if strategy == "heuristic":
            def score(index):
                quality = float(self.nodes[index]["quality_prior"])
                delay = float(metrics[index]["total_ms"]) / 100.0
                return (
                    0.5 * quality + 0.5 * (1.0 - min(max(delay, 0.0), 1.0)),
                    -metrics[index]["total_ms"],
                    -index,
                )
            return int(max(indices, key=score))
        if strategy == "oracle":
            return self._select_oracle(indices)
        raise ValueError(f"Unknown strategy {strategy}")

    def _select_oracle(self, indices):
        """Privileged upper bound; never used by DQN state/reward."""
        oracle_action = int(self.env.oracle_action())
        cache = {}
        outputs = {}
        for index in indices:
            model_name = self.nodes[index]["model_type"]
            outputs[index] = self.backends.get(
                model_name, self.step_data["inputs"], cache=cache
            )
        exact = [
            index for index in indices
            if outputs[index]["action"] == oracle_action
        ]
        if exact:
            return int(min(
                exact,
                key=lambda i: (
                    self.step_data["latency_metrics"][i]["total_ms"], i
                ),
            ))
        def probability(index):
            return float(
                outputs[index]["joint_probabilities"][oracle_action]
            )
        return int(max(
            indices,
            key=lambda i: (
                probability(i),
                -self.step_data["latency_metrics"][i]["total_ms"],
                -i,
            ),
        ))

    def apply(self, node_index):
        """Execute one association action and return its transition."""
        node_index = int(node_index)
        if not self.step_data["mask"][node_index]:
            raise ValueError(f"node {node_index} is masked")
        step_data = self.step_data
        node = self.nodes[node_index]
        metric = step_data["latency_metrics"][node_index]
        if metric is None:
            metric = self.latency_model.evaluate(
                node, self.env.position, node["model_type"]
            )
        output = self.backends.get(
            node["model_type"], step_data["inputs"]
        )
        action = int(output["action"])
        arrival_probability = float(output["arrival_probability"])
        e2e_ms = float(metric["total_ms"])
        pose_before = step_data["pose"].copy()
        self.observation, success, done = self.env.step(
            action, render_observation=(action != STOP)
        )
        self.success = bool(success)
        self.osr = bool(
            self.osr or self.env.distance_to_goal < self.env.stop_radius
        )
        self.ended_by_stop = bool(action == STOP)
        self.trajectory.append(self.env.position.copy())
        nav_reward, _ = self.reward_tracker.advance(
            new_position=self.env.position.copy(), done=bool(done)
        )
        if self.jointr_aligned:
            step_index = self.step_index + 1
            self.running_mean_e2e_ms, penalty = delay_reward_increment(
                e2e_ms, self.running_mean_e2e_ms, step_index
            )
            self.running_mean_steps = step_index
        else:
            penalty = float(latency_penalty(e2e_ms))
        reward = float(nav_reward + penalty)
        self.return_sum += reward
        self.model_usage[node["model_type"]] += 1
        self.step_records.append({
            "step": int(step_data["step"]),
            "node_index": node_index,
            "node_id": node["node_id"],
            "model_type": node["model_type"],
            "action": action,
            "action_name": ACTION_NAMES[action],
            "arrival_probability": arrival_probability,
            "e2e_ms": e2e_ms,
            "communication_ms": float(metric["communication_ms"]),
            "inference_ms": float(metric["inference_ms"]),
            "uplink_rate_mbps": float(metric["uplink_rate_mbps"]),
            "distance_m": float(metric["distance_m"]),
            "navigation_reward": float(nav_reward),
            "latency_penalty": penalty,
            "reward": reward,
            "running_mean_e2e_ms": float(self.running_mean_e2e_ms),
            "success": bool(success),
            "done": bool(done),
            "fallback": bool(step_data["fallback"]),
            "distance_to_goal_m": float(self.env.distance_to_goal),
        })
        self.past.append({
            "pose": pose_before,
            "next_pose": self.env.pose.copy(),
            "visual": step_data["visual"],
            "action": action,
        })
        self.previous_action = action
        self.previous_selected = node_index
        self.previous_e2e_ms = e2e_ms
        self.step_index += 1
        next_state = None
        next_mask = None
        if not done:
            self._build_step_data(self.observation)
            next_state = self.step_data["state"]
            next_mask = self.step_data["mask"]
        return {
            "state": step_data["state"],
            "mask": step_data["mask"],
            "node_index": node_index,
            "action": action,
            "reward": reward,
            "nav_reward": float(nav_reward),
            "latency_penalty": penalty,
            "e2e_ms": e2e_ms,
            "done": bool(done),
            "success": bool(success),
            "next_state": next_state,
            "next_mask": next_mask,
        }

    def run(self, strategy, rng=None):
        while True:
            node_index = self.select_node(strategy, rng=rng)
            transition = self.apply(node_index)
            if transition["done"]:
                break
        return self.finalize(strategy)

    def finalize(self, strategy):
        summary = self.reward_tracker.summary()
        ndtw = float(summary["final_ndtw"])
        trajectory = np.asarray(self.trajectory, dtype=np.float64)
        path_length = 0.0
        if len(trajectory) > 1:
            path_length = float(np.linalg.norm(
                np.diff(trajectory[:, :2], axis=0), axis=1
            ).sum())
        shortest = float(np.linalg.norm(
            self.reference["start"] - self.reference["goal"]
        ))
        spl = (
            shortest / max(shortest, path_length, 1e-8)
            if self.success else 0.0
        )
        e2e = np.asarray(
            [row["e2e_ms"] for row in self.step_records], dtype=np.float64
        )
        violations = np.asarray(
            [row["e2e_ms"] > 100.0 for row in self.step_records],
            dtype=np.float64,
        )
        mean_e2e = float(e2e.mean()) if len(e2e) else 0.0
        episode_return = float(self.return_sum)
        episode_joint = (
            aligned_episode_jointr(ndtw, mean_e2e)
            if self.jointr_aligned else joint_reward(ndtw, mean_e2e)
        )
        return {
            "strategy": strategy,
            "episode_uid": self.spec["episode_uid"],
            "split": self.spec["split"],
            "scenario": self.spec["scenario"],
            "episode_index": int(self.spec["episode_index"]),
            "seed": int(self.spec["seed"]),
            "band": self.spec["band"],
            "band_id": int(self.spec["band_id"]),
            "success": bool(self.success),
            "osr": bool(self.osr),
            "ndtw": ndtw,
            "spl": float(spl),
            "steps": len(self.step_records),
            "path_length_m": path_length,
            "start_goal_distance_m": shortest,
            "mean_e2e_ms": float(e2e.mean()) if len(e2e) else 0.0,
            "p95_e2e_ms": float(np.percentile(e2e, 95)) if len(e2e) else 0.0,
            "max_e2e_ms": float(e2e.max()) if len(e2e) else 0.0,
            "delay_violation_rate": (
                float(violations.mean()) if len(violations) else 0.0
            ),
            "mean_navigation_reward": float(np.mean([
                row["navigation_reward"] for row in self.step_records
            ])) if self.step_records else 0.0,
            "mean_latency_penalty": float(np.mean([
                row["latency_penalty"] for row in self.step_records
            ])) if self.step_records else 0.0,
            "model_usage": dict(self.model_usage),
            "fallback_steps": int(self.fallback_steps),
            "mean_episode_e2e_ms": mean_e2e,
            "final_ndtw": ndtw,
            "episode_return": episode_return,
            "episode_jointR": float(episode_joint),
            "jointr_abs_error": abs(episode_return - episode_joint),
            "jointr_delay_weight": 0.25,
            "running_mean_e2e_ms": float(self.running_mean_e2e_ms),
            "joint_reward": float(episode_joint),
            "trajectory": trajectory.astype(np.float32).tolist(),
            "step_records": self.step_records,
        }


def make_runner_factory(
    encoder,
    backends,
    projector,
    quality_prior,
    gab_config,
    latency_model=None,
    scene_cache=None,
    device=None,
):
    def factory(spec, topology, scene=None):
        return EpisodeRunner(
            spec=spec,
            topology=topology,
            encoder=encoder,
            backends=backends,
            projector=projector,
            quality_prior=quality_prior,
            gab_config=gab_config,
            latency_model=latency_model,
            scene=scene,
            scene_cache=scene_cache,
            device=device,
        )
    return factory


def load_runtime(device=None):
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    encoder = load_encoder(device)
    backends = BackendPool(device)
    return device, encoder, backends
