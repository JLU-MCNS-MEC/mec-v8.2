"""Stage-2l closed-loop runtime built on the fixed-city environment.

The frozen five Stage-2j T5 MECs, the existing encoder, the Pure-MEC v2 state
and the JointR-aligned reward are reused exactly as in Stage-2k v2.  Only the
episode scene construction changes: every task now replays a fixed city from
``stage2l_fixed_city.py`` instead of regenerating a city in ``reset``.

The runner is a thin subclass of the Stage-2k ``EpisodeRunner``.  All
decision, reward and metric code is inherited, which keeps the DQN/PPO
comparison against the previous stage honest.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP, ToyUAVEnv
from gbs_model_v3 import goal_current_similarity
from navigation_reward import NavigationRewardTracker
from token_data import (
    encode_multiscale_goal_tokens,
    goal_conditioned_spatial_heatmap,
)
from utils import image_to_tensor
from v83_spatial_stage2.stage2h_gab import GABState
from v83_spatial_stage2.stage2i_data import (
    encode_landmark_canvases,
    online_history_arrays,
    render_landmark_canvas,
)
from v83_spatial_stage2.stage2k_runtime import EpisodeRunner, load_runtime
from v83_spatial_stage2.stage2l_fixed_city import (
    FixedCityEnv,
    MAP_SEED_PHASE1,
)


def _shuffled_city_nodes(env: FixedCityEnv):
    """Return (nodes, city_index_to_node_index) for the fixed city map.

    The permutation depends only on ``map_seed``; every task in the city
    therefore sees exactly the same anonymous map in exactly the same order.
    """
    nodes = list(env.city_landmarks)
    order = np.random.default_rng(
        int(env.map_seed) ^ 0x5EED1234
    ).permutation(len(nodes))
    shuffled = [nodes[int(index)] for index in order]
    city_to_node = {
        int(city_index): int(node_index)
        for node_index, city_index in enumerate(order.tolist())
    }
    return shuffled, city_to_node


class FixedCitySceneCache:
    """LRU cache for goal tokens + the city map, keyed by map/task goal."""

    def __init__(self, encoder, device, max_entries=256):
        self.encoder = encoder
        self.device = device
        self.max_entries = max_entries
        self.cache = {}
        self.order = []
        self.city_cache = {}

    def _city_entry(self, map_seed):
        map_seed = int(map_seed)
        if map_seed in self.city_cache:
            return self.city_cache[map_seed]
        env = FixedCityEnv(map_seed)
        nodes, city_to_node = _shuffled_city_nodes(env)
        canvases = [render_landmark_canvas(env, node) for node in nodes]
        embeddings = encode_landmark_canvases(canvases, self.encoder, self.device)
        entry = {
            "nodes": nodes,
            "city_to_node": city_to_node,
            "positions": np.asarray(
                [node.position for node in nodes], dtype=np.float32
            ),
            "embeddings": embeddings.float().numpy(),
        }
        self.city_cache[map_seed] = entry
        return entry

    def get(self, spec):
        map_seed = int(spec["map_seed"])
        goal_index = int(spec["goal_index"])
        key = (map_seed, goal_index)
        if key in self.cache:
            return self.cache[key]

        city = self._city_entry(map_seed)
        env = FixedCityEnv(map_seed)
        env.reset(
            task_seed=int(spec["task_seed"]),
            goal_index=goal_index,
            start_xy=spec["start"],
            start_yaw=float(spec["start_yaw"]),
            goal_distance_range_m=tuple(spec["distance_range"]),
            render_observation=False,
        )
        goal_tokens = encode_multiscale_goal_tokens(
            env.get_goal_reference_multiscale_rgbs(), self.encoder, self.device
        ).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        landmark_map = {
            "positions": city["positions"],
            "embeddings": city["embeddings"],
            "goal_index": int(city["city_to_node"][goal_index]),
            "num_nodes": len(city["nodes"]),
        }
        scene = {
            "goal_tokens": goal_tokens,
            "goal_token_np": goal_tokens[0, 2].detach().cpu().float().numpy(),
            "landmark_map": landmark_map,
        }
        if self.max_entries is not None:
            if len(self.order) >= self.max_entries:
                oldest = self.order.pop(0)
                self.cache.pop(oldest, None)
            self.order.append(key)
        self.cache[key] = scene
        return scene


def build_fixed_scene(spec, encoder, device, cache=None):
    if cache is not None:
        return cache.get(spec)
    cache = FixedCitySceneCache(encoder, device, max_entries=1)
    return cache.get(spec)


_REFERENCE_CACHE: dict[str, dict] = {}


def build_fixed_oracle_reference(spec) -> dict:
    """Oracle reference trajectory for the reward tracker and nDTW/SPL."""
    uid = str(spec["episode_uid"])
    if uid in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[uid]
    env = FixedCityEnv(int(spec["map_seed"]))
    env.reset(
        task_seed=int(spec["task_seed"]),
        goal_index=int(spec["goal_index"]),
        start_xy=spec["start"],
        start_yaw=float(spec["start_yaw"]),
        goal_distance_range_m=tuple(spec["distance_range"]),
        render_observation=False,
    )
    start = env.position.copy()
    goal = env.goal.copy()
    reference = [start.copy()]
    success = False
    steps = 0
    for _ in range(int(env.max_steps)):
        action = int(env.oracle_action())
        steps += 1
        _, success, done = env.step(action, render_observation=False)
        if action != STOP and np.linalg.norm(env.position - reference[-1]) > 1e-9:
            reference.append(env.position.copy())
        if done:
            break
    reference = np.asarray(reference, dtype=np.float32)
    if np.linalg.norm(goal - reference[-1]) > 1e-9:
        reference = np.concatenate([reference, goal.reshape(1, 2)], axis=0)
    payload = {
        "start": start,
        "goal": goal,
        "reference": reference.astype(np.float32),
        "oracle_success": bool(success),
        "oracle_steps": int(steps),
        "shortest_distance_m": float(np.linalg.norm(start - goal)),
    }
    _REFERENCE_CACHE[uid] = payload
    return payload


class FixedCityEpisodeRunner(EpisodeRunner):
    """EpisodeRunner with fixed-city scene construction."""

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
        pure_mec_v2_state=False,
        visual_ablation_v2_state=False,
    ):
        if visual_ablation_v2_state:
            raise ValueError("Stage-2l only uses the Pure-MEC v2 state")
        if not pure_mec_v2_state:
            raise ValueError("Stage-2l requires pure_mec_v2_state=True")
        self.jointr_aligned = bool(jointr_aligned)
        self.no_visual_state = False
        self.pure_mec_state = False
        self.pure_mec_v2_state = True
        self.visual_ablation_v2_state = False
        self.spec = spec
        self.topology = topology
        self.nodes = topology["nodes"]
        self.encoder = encoder
        self.backends = backends
        self.projector = projector
        self.quality_prior = quality_prior
        self.gab_config = gab_config
        from v83_spatial_stage2.stage2k_latency import Stage2KLatency

        self.latency_model = latency_model or Stage2KLatency()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scene = scene or (
            scene_cache.get(spec) if scene_cache is not None
            else build_fixed_scene(spec, encoder, self.device)
        )
        self.reference = build_fixed_oracle_reference(spec)
        self.reset()

    def reset(self):
        self.env = FixedCityEnv(int(self.spec["map_seed"]))
        self.observation = self.env.reset(
            task_seed=int(self.spec["task_seed"]),
            goal_index=int(self.spec["goal_index"]),
            start_xy=self.spec["start"],
            start_yaw=float(self.spec["start_yaw"]),
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
        if int(self.env.task_info["goal_index"]) != int(self.spec["goal_index"]):
            raise RuntimeError(
                f"Episode {self.spec['episode_uid']} goal index mismatch"
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


def make_fixed_runner_factory(
    encoder,
    backends,
    quality_prior,
    gab_config,
    latency_model=None,
    scene_cache=None,
    device=None,
):
    def factory(spec, topology, scene=None):
        return FixedCityEpisodeRunner(
            spec=spec,
            topology=topology,
            encoder=encoder,
            backends=backends,
            projector=None,
            quality_prior=quality_prior,
            gab_config=gab_config,
            latency_model=latency_model,
            scene=scene,
            scene_cache=scene_cache,
            device=device,
            jointr_aligned=True,
            pure_mec_v2_state=True,
        )

    return factory


def clear_reference_cache() -> None:
    _REFERENCE_CACHE.clear()
