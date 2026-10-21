"""Stage-2l fixed-city navigation environment and task generator.

This module splits the old ``ToyUAVEnv.reset(seed)`` into two independent
seeds:

``map_seed``
    Builds the whole city once: landmark positions, count, physical sizes,
    base colours, shapes and internal markers, plus the fixed ground
    background / road grid.  Every episode that uses the same ``map_seed``
    sees exactly the same city.  Nothing in the city is regenerated when a
    different navigation task is selected.

``task_seed``
    Samples the UAV start position, heading and a target chosen **from the
    landmarks that already exist in the fixed city**.  The goal reference
    image is rendered from that landmark's existing appearance.  No new
    building is created for a task, and selecting a different goal never
    mutates another landmark.

The RGB observation still depends on the UAV pose: the same city is rendered
from a different position / heading, so frames are not fixed images.  The
frame-level sensor noise is deterministic given
``(map_seed, task_seed, step)`` to make closed-loop smoke tests reproducible.

The five frozen Stage-2j T5 MECs and the existing encoder are reused
unchanged.  This module only changes how the navigation scene is generated.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import ENV, NETWORK
from env import Landmark, ToyUAVEnv, wrap_angle
from gbs_network import build_random_gbs_layout
from v83_spatial_stage2.stage2k_common import (
    BAND_ACTION_RANGES,
    BAND_ID,
    BAND_RANGES,
    BANDS,
    MODEL_INDEX,
    MODEL_NAMES,
    SCENARIOS,
    atomic_write_json,
    read_json,
    utc_now,
)
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior


# ---------------------------------------------------------------------------
# Stage paths and protocol constants
# ---------------------------------------------------------------------------

STAGE2L_ROOT = Path("runs/v83_spatial_stage2/stage2l_fixed_city")
SPECS_DIR = STAGE2L_ROOT / "specs"
TOPOLOGY_DIR = STAGE2L_ROOT / "topologies"
CONFIG_DIR = STAGE2L_ROOT / "configs"
SMOKE_DIR = STAGE2L_ROOT / "smoke"
EVAL_DIR = STAGE2L_ROOT / "evaluation"
FIGURE_DIR = STAGE2L_ROOT / "figures"
LOG_DIR = STAGE2L_ROOT / "logs"

# One shared map in phase 1.  Train/Dev/Test task seeds and task IDs are
# disjoint; the map itself may be shared (explicitly allowed by the protocol).
MAP_SEED_PHASE1 = 10_000_001
TASK_SEED_BASE = {
    "train": 11_000_000,
    "dev": 12_000_000,
    "test": 13_000_000,
}
TASK_ID_PREFIX = "stage2l"
TASK_POOL_VERSION = "stage2l-fixed-city-v1"

TRAIN_EPISODES = 3500
DEV_EPISODES = 60
TEST_EPISODES = 60

# 1:1:1 band generation.  The remainder goes to the earlier bands so the
# three counts differ by at most one.
BAND_COUNTS_3500 = {"short": 1167, "medium": 1167, "long": 1166}
BAND_COUNTS_60 = {"short": 20, "medium": 20, "long": 20}

ORDER_SEED = 20261021
CONFIG_SEED = 20261021

# Independent topology seed block (Stage-2k used 8,000,000+; Stage-2j used
# 17.8M / 34.8M / 48.8M).  Nothing here overlaps those ranges.
TOPOLOGY_SEED_BASE = 9_000_000
TOPOLOGY_SEEDS = {
    "A_fixed": TOPOLOGY_SEED_BASE + 0,
    "B_dev": TOPOLOGY_SEED_BASE + 50,
    "B_test": TOPOLOGY_SEED_BASE + 60,
    "B_train": [TOPOLOGY_SEED_BASE + 100 + i for i in range(10)],
    "C_balanced": TOPOLOGY_SEED_BASE + 0,
    "C_quality_rich": TOPOLOGY_SEED_BASE + 200,
    "C_tradeoff_stress": TOPOLOGY_SEED_BASE + 300,
}
B_TRAIN_POOL_SIZE = len(TOPOLOGY_SEEDS["B_train"])

CONFIG_DESCRIPTIONS = {
    "A": "Balanced 场景，单张固定拓扑（训练与验证共用该固定拓扑）",
    "B": "Balanced 场景，10 张独立 Train 拓扑随机分配给训练任务；验证使用独立固定 Dev 拓扑",
    "C": "Balanced / Quality-rich / Tradeoff-stress 三种部署场景，每种各一张固定拓扑",
}


# ---------------------------------------------------------------------------
# Small IO / hashing helpers
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(payload) -> str:
    return sha256_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Fixed city
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CityMap:
    """Immutable description of one fixed city."""

    map_seed: int
    world_size_m: float
    background_value: float
    road_spacing_m: float
    road_half_width_m: float
    road_alpha: float
    block_tint: float
    landmarks: tuple
    fingerprint: str

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)


_CITY_CACHE: dict[int, CityMap] = {}


def _city_entry(landmark: Landmark) -> dict:
    return {
        "x": float(landmark.x),
        "y": float(landmark.y),
        "size": float(landmark.physical_size_m),
        "shape": str(landmark.shape),
        "base_color": [float(v) for v in landmark.base_color],
        "marker_type": str(landmark.marker_type),
        "marker_color": [float(v) for v in landmark.marker_color],
        "signature": list(landmark.signature),
        "is_distractor": bool(landmark.is_distractor),
    }


def city_fingerprint(city: CityMap) -> str:
    payload = {
        "map_seed": int(city.map_seed),
        "world_size_m": float(city.world_size_m),
        "background_value": float(city.background_value),
        "road_spacing_m": float(city.road_spacing_m),
        "road_half_width_m": float(city.road_half_width_m),
        "road_alpha": float(city.road_alpha),
        "block_tint": float(city.block_tint),
        "landmarks": [_city_entry(item) for item in city.landmarks],
    }
    return canonical_sha256(payload)


def build_city(map_seed: int, config=ENV) -> CityMap:
    """Build (or return the cached) fixed city for ``map_seed``.

    All appearance sampling uses a dedicated ``map_seed`` RNG.  No task seed
    participates, so two runs that share ``map_seed`` share the exact city.
    """
    map_seed = int(map_seed)
    if map_seed in _CITY_CACHE:
        return _CITY_CACHE[map_seed]

    template = ToyUAVEnv(config)
    rng = np.random.default_rng(map_seed)
    background_value = float(rng.uniform(24.0, 42.0))
    road_spacing_m = float(rng.uniform(220.0, 280.0))
    road_half_width_m = float(rng.uniform(6.0, 10.0))
    road_alpha = float(rng.uniform(0.16, 0.22))
    block_tint = float(rng.uniform(1.5, 2.8))

    margin = float(config.boundary_margin_m)
    world = float(config.world_size_m)
    low = margin * 0.5
    high = world - margin * 0.5

    used_signatures: set[tuple] = set()
    landmarks: list[Landmark] = []
    for _ in range(int(config.num_landmarks)):
        for _attempt in range(1000):
            signature = (
                int(rng.integers(0, len(template.shapes))),
                int(rng.integers(0, len(template.base_palette))),
                int(rng.integers(0, len(template.marker_types))),
                int(rng.integers(0, len(template.marker_palette))),
            )
            if signature not in used_signatures:
                break
        else:  # pragma: no cover - the palette has >1500 combinations
            raise RuntimeError("Unable to sample a unique landmark signature")
        used_signatures.add(signature)
        shape_id, base_id, marker_id, marker_color_id = signature
        base = template.base_palette[base_id].copy() + rng.normal(0.0, 5.0, size=3)
        marker = (
            template.marker_palette[marker_color_id].copy()
            + rng.normal(0.0, 3.0, size=3)
        )
        landmarks.append(
            Landmark(
                x=float(rng.uniform(low, high)),
                y=float(rng.uniform(low, high)),
                physical_size_m=float(
                    rng.uniform(config.landmark_size_min_m, config.landmark_size_max_m)
                ),
                shape=str(template.shapes[shape_id]),
                base_color=np.clip(base, 0, 255).astype(np.float32),
                marker_type=str(template.marker_types[marker_id]),
                marker_color=np.clip(marker, 0, 255).astype(np.float32),
                signature=signature,
                is_distractor=bool(rng.random() < config.distractor_ratio),
            )
        )

    city = CityMap(
        map_seed=map_seed,
        world_size_m=world,
        background_value=background_value,
        road_spacing_m=road_spacing_m,
        road_half_width_m=road_half_width_m,
        road_alpha=road_alpha,
        block_tint=block_tint,
        landmarks=tuple(landmarks),
        fingerprint="",  # filled below
    )
    city = CityMap(
        map_seed=city.map_seed,
        world_size_m=city.world_size_m,
        background_value=city.background_value,
        road_spacing_m=city.road_spacing_m,
        road_half_width_m=city.road_half_width_m,
        road_alpha=city.road_alpha,
        block_tint=city.block_tint,
        landmarks=city.landmarks,
        fingerprint=city_fingerprint(city),
    )
    _CITY_CACHE[map_seed] = city
    return city


def clear_city_cache() -> None:
    """Test helper; production code never clears the cache."""
    _CITY_CACHE.clear()


# ---------------------------------------------------------------------------
# Fixed-city environment
# ---------------------------------------------------------------------------


class FixedCityEnv(ToyUAVEnv):
    """ToyUAVEnv with a map_seed / task_seed split.

    The inherited navigation dynamics, projection, depth sensor, goal
    reference renderer and reward-relevant API are unchanged.  Only
    ``reset`` and ``render`` are overridden.
    """

    def __init__(self, map_seed: int = MAP_SEED_PHASE1, config=ENV):
        super().__init__(config)
        self.map_seed = int(map_seed)
        self.city = build_city(self.map_seed, config)
        self.city_landmarks = list(self.city.landmarks)
        self._task_seed = None
        self._task_info = None
        self._render_rng = None
        self._render_step = 0
        self._precompute_projection_grid()

    # -- projection helpers -------------------------------------------------

    def _precompute_projection_grid(self):
        height = width = int(self.img_size)
        focal_x = (width / 2.0) / math.tan(self.hfov_rad / 2.0)
        focal_y = (height / 2.0) / math.tan(self.vfov_rad / 2.0)
        px = np.arange(width, dtype=np.float32)[None, :]
        py = np.arange(height, dtype=np.float32)[:, None]
        self._focal_x = float(focal_x)
        self._focal_y = float(focal_y)
        self._rel_yaw_grid = np.arctan((width / 2.0 - px) / focal_x).astype(
            np.float32
        )
        self._rel_pitch_grid = np.arctan((height / 2.0 - py) / focal_y).astype(
            np.float32
        )

    # -- task sampling ------------------------------------------------------

    def reset(
        self,
        seed=None,
        task_seed=None,
        goal_index=None,
        start_xy=None,
        start_yaw=None,
        goal_distance_range_m=None,
        band=None,
        render_observation=True,
    ):
        """Reset to a task sampled from the fixed city.

        ``task_seed`` alone is sufficient to reproduce a task.  The stored
        ``goal_index`` / ``start_xy`` / ``start_yaw`` fields in a task spec
        are accepted as an explicit replay path so a training manifest can
        never silently drift from its audit record.
        """
        if task_seed is None:
            task_seed = seed
        if task_seed is None:
            raise ValueError("FixedCityEnv.reset requires task_seed")
        task_seed = int(task_seed)
        self._task_seed = task_seed
        task_rng = np.random.default_rng(task_seed)

        if band is not None:
            if band not in BAND_RANGES:
                raise ValueError(f"Unknown band {band}")
            distance_range = BAND_RANGES[band]
        elif goal_distance_range_m is not None:
            distance_range = tuple(float(v) for v in goal_distance_range_m)
        else:
            distance_range = (
                self.min_start_goal_distance,
                self.max_start_goal_distance,
            )

        n_landmarks = len(self.city_landmarks)
        if goal_index is None:
            goal_index = int(task_rng.integers(0, n_landmarks))
        goal_index = int(goal_index)
        if not 0 <= goal_index < n_landmarks:
            raise IndexError(f"goal_index {goal_index} outside city")
        goal_landmark = self.city_landmarks[goal_index]
        goal_position = goal_landmark.position.astype(np.float32)

        if start_xy is None:
            distance_min, distance_max = distance_range
            for _attempt in range(2000):
                distance = float(task_rng.uniform(distance_min, distance_max))
                bearing = float(task_rng.uniform(-np.pi, np.pi))
                candidate = goal_position - distance * np.asarray(
                    [math.cos(bearing), math.sin(bearing)], dtype=np.float32
                )
                if (
                    self.margin <= candidate[0] <= self.world_size - self.margin
                    and self.margin <= candidate[1] <= self.world_size - self.margin
                ):
                    start_xy = candidate
                    break
            else:  # pragma: no cover - with 160 landmarks this is rare
                raise RuntimeError(
                    "Unable to sample a start position inside the city for "
                    f"task_seed={task_seed}"
                )
        start = np.asarray(start_xy, dtype=np.float32)
        if start.shape != (2,):
            raise ValueError("start_xy must be a 2-D point")
        if not (
            self.margin - 1e-6 <= start[0] <= self.world_size - self.margin + 1e-6
            and self.margin - 1e-6 <= start[1] <= self.world_size - self.margin + 1e-6
        ):
            raise ValueError(f"start {start} outside city bounds")

        if start_yaw is None:
            goal_bearing = math.atan2(
                float(goal_position[1] - start[1]),
                float(goal_position[0] - start[0]),
            )
            yaw_error = np.deg2rad(
                task_rng.uniform(-self.cfg.initial_yaw_error_deg,
                                 self.cfg.initial_yaw_error_deg)
            )
            start_yaw = wrap_angle(goal_bearing + float(yaw_error))
        start_yaw = float(start_yaw)

        self.x = float(start[0])
        self.y = float(start[1])
        self.yaw = start_yaw
        self.start = start.copy()
        self.goal = goal_position.copy()
        self.goal_landmark = goal_landmark
        # The goal is one of the city landmarks; expose all the others as
        # ordinary landmarks exactly as the old environment did.
        self.landmarks = [
            item for index, item in enumerate(self.city_landmarks)
            if index != goal_index
        ]
        self.step_count = 0
        self._render_step = 0
        self._goal_reference_rgb = self._render_goal_reference_rgb()
        self._start_reference_mask = None
        self._task_info = {
            "map_seed": int(self.map_seed),
            "task_seed": int(task_seed),
            "goal_index": int(goal_index),
            "band": band,
            "distance_range": [float(v) for v in distance_range],
            "start": [float(v) for v in self.start],
            "goal": [float(v) for v in self.goal],
            "start_yaw": float(self.yaw),
            "start_goal_distance_m": float(
                np.linalg.norm(self.goal - self.start)
            ),
            "city_fingerprint": self.city.fingerprint,
        }
        # The inherited renderer used self.rng for city sampling.  Here the
        # city is already fixed; keep a separate RNG for any legacy helper
        # that still asks for one.
        self.rng = np.random.default_rng(task_seed ^ 0x5A5A5A5A)
        if render_observation:
            return self.render()
        return None

    @property
    def task_info(self) -> dict:
        if self._task_info is None:
            raise RuntimeError("Call reset() before task_info")
        return json.loads(json.dumps(self._task_info))

    # -- rendering ----------------------------------------------------------

    def _frame_seed(self) -> int:
        mask = (1 << 63) - 1
        return int(
            (
                int(self.map_seed) * 1_000_003
                + int(self._task_seed or 0) * 97
                + int(self._render_step) * 7919
                + int(self.step_count) * 104_729
            )
            & mask
        )

    def _ground_texture(self):
        """Return (road_alpha, block_tint) for the current pose.

        Both arrays are **world-space fixed**: they depend on the UAV pose
        only through the camera projection, never on a task seed.  The roads
        and block tints are therefore identical for every task in the city.
        """
        elevation = self.camera_pitch_rad + self._rel_pitch_grid
        valid = elevation < -0.02
        safe = np.where(valid, elevation, -0.02).astype(np.float32)
        distance = -float(self.uav_altitude_m) / np.tan(safe)
        distance = np.clip(distance, 0.0, 1500.0).astype(np.float32)
        distance = np.where(valid, distance, 0.0).astype(np.float32)

        angle = (float(self.yaw) + self._rel_yaw_grid).astype(np.float32)
        world_x = float(self.x) + distance * np.cos(angle)
        world_y = float(self.y) + distance * np.sin(angle)

        spacing = float(self.city.road_spacing_m)
        half_width = float(self.city.road_half_width_m)
        dx = np.abs(((world_x + spacing / 2.0) % spacing) - spacing / 2.0)
        dy = np.abs(((world_y + spacing / 2.0) % spacing) - spacing / 2.0)
        road = (dx < half_width) | (dy < half_width)
        fade = np.clip(1.0 - np.maximum(distance - 150.0, 0.0) / 800.0, 0.0, 1.0)
        road_alpha = (
            road.astype(np.float32) * fade * float(self.city.road_alpha)
        )
        road_alpha = np.where(valid, road_alpha, 0.0).astype(np.float32)

        block_x = np.floor(world_x / spacing).astype(np.int64)
        block_y = np.floor(world_y / spacing).astype(np.int64)
        hashed = (
            block_x * 73856093
            ^ block_y * 19349663
            ^ int(self.map_seed)
        ) & 0xFFFF
        tint = (
            hashed.astype(np.float32) / 65535.0 - 0.5
        ) * 2.0 * float(self.city.block_tint)
        tint *= np.clip(1.0 - distance / 900.0, 0.0, 1.0)
        tint = np.where(valid, tint, 0.0).astype(np.float32)
        return road_alpha, tint

    def render(self):
        height = width = int(self.img_size)
        self._render_rng = np.random.default_rng(self._frame_seed())
        self._render_step += 1
        rng = self._render_rng

        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        base_background = float(self.city.background_value)
        background = (
            base_background
            + 8.0 * (1.0 - yy)
            + 3.0 * np.sin(2.0 * np.pi * xx)
        )
        road_alpha, block_tint = self._ground_texture()
        image_float = np.repeat(background[:, :, None], 3, axis=2)
        road_color = np.asarray(
            [base_background - 6.0, base_background - 6.0,
             base_background - 5.0],
            dtype=np.float32,
        )
        image_float = (
            image_float * (1.0 - road_alpha[:, :, None])
            + road_color[None, None, :] * road_alpha[:, :, None]
            + block_tint[:, :, None]
        )
        image = Image.fromarray(
            np.clip(image_float, 0, 255).astype(np.uint8), mode="RGB"
        )
        draw = ImageDraw.Draw(image)

        visible = []
        for landmark in self.landmarks:
            projection = self._project(
                landmark.position, landmark.physical_size_m
            )
            if projection is not None:
                visible.append((projection[2], landmark, projection))
        goal_projection = self._project(
            self.goal_landmark.position, self.goal_landmark.physical_size_m
        )
        if goal_projection is not None:
            visible.append(
                (goal_projection[2], self.goal_landmark, goal_projection)
            )

        visible.sort(key=lambda item: item[0], reverse=True)
        for distance, obj, projection in visible:
            px, py, _, half_size = projection
            self._draw_landmark(
                draw, obj, px, py, half_size, distance, base_background
            )

        if rng.random() < 0.30:
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=float(rng.uniform(0.0, 0.45))
                )
            )

        array = np.asarray(image).astype(np.float32)
        array *= float(rng.uniform(self.cfg.brightness_min,
                                   self.cfg.brightness_max))
        array += rng.normal(0.0, self.cfg.sensor_noise_std, size=array.shape)
        return np.clip(array, 0, 255).astype(np.uint8)

    def _draw_landmark(
        self, draw, obj, px, py, half_size, distance, background_value
    ):
        rng = self._render_rng
        if rng is None:
            rng = np.random.default_rng(0)
        base = obj.base_color + rng.normal(0.0, 4.0, size=3)
        marker = obj.marker_color + rng.normal(0.0, 2.0, size=3)
        base = self._haze_mix(base, distance, background_value)
        marker = self._haze_mix(marker, distance, background_value)
        self._draw_shape(draw, obj.shape, px, py, half_size, base)
        self._draw_marker(
            draw, obj.marker_type, px, py, half_size, marker, base
        )


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def band_counts(total: int) -> dict:
    base = int(total) // len(BANDS)
    counts = {band: base for band in BANDS}
    counts[BANDS[0]] += int(total) - base * len(BANDS)
    return counts


def _inspect_task(env: FixedCityEnv, task_seed: int, band: str) -> dict:
    """Sample one candidate and run the oracle expert without rendering."""
    env.reset(
        task_seed=int(task_seed),
        band=band,
        render_observation=False,
    )
    task = env.task_info
    start_distance = float(env.distance_to_goal)
    action_range = BAND_ACTION_RANGES[band]
    actions = Counter()
    success = False
    done = False
    steps = 0
    while not done and steps < int(env.max_steps):
        action = int(env.oracle_action())
        actions[action] += 1
        steps += 1
        _, success, done = env.step(action, render_observation=False)
    accepted = bool(success and action_range[0] <= steps <= action_range[1])
    return {
        "accepted": accepted,
        "steps": int(steps),
        "success": bool(success),
        "start_goal_distance_m": start_distance,
        "action_counts": dict(actions),
        "task": task,
    }


def generate_split_tasks(
    split: str,
    map_seed: int,
    counts: Mapping[str, int],
    seed_start: int,
    max_candidates: int = 200_000,
) -> list[dict]:
    """Generate an accepted, band-balanced task list for one split."""
    if split not in TASK_SEED_BASE:
        raise ValueError(split)
    env = FixedCityEnv(map_seed)
    specs: list[dict] = []
    candidate_seed = int(seed_start)
    attempts = 0
    for band in BANDS:
        target = int(counts[band])
        found = 0
        while found < target:
            if attempts >= int(max_candidates):
                raise RuntimeError(
                    f"Unable to generate {target} {band} tasks for {split}"
                )
            info = _inspect_task(env, candidate_seed, band)
            candidate_seed += 1
            attempts += 1
            if not info["accepted"]:
                continue
            task = info["task"]
            index = len(specs)
            specs.append({
                "episode_uid": f"{TASK_ID_PREFIX}-{split}-{index:05d}",
                "split": split,
                "episode_index": int(index),
                "scenario": "balanced",
                "map_seed": int(map_seed),
                "task_seed": int(task["task_seed"]),
                "goal_index": int(task["goal_index"]),
                "band": band,
                "band_id": int(BAND_ID[band]),
                "distance_range": list(task["distance_range"]),
                "oracle_steps": int(info["steps"]),
                "oracle_action_counts": info["action_counts"],
                "start_goal_distance_m": float(task["start_goal_distance_m"]),
                "start": list(task["start"]),
                "goal": list(task["goal"]),
                "start_yaw": float(task["start_yaw"]),
                "city_fingerprint": str(task["city_fingerprint"]),
            })
            found += 1
    if len(specs) != sum(int(counts[band]) for band in BANDS):
        raise RuntimeError("Task count mismatch")
    return specs


def task_pool_payload(split: str, specs: Sequence[dict], sealed: bool = False) -> dict:
    return {
        "version": TASK_POOL_VERSION,
        "split": split,
        "created_utc": utc_now(),
        "sealed": bool(sealed),
        "final_test_read": False,
        "map_seed": int(specs[0]["map_seed"]) if specs else None,
        "seed_base": int(TASK_SEED_BASE[split]),
        "band_counts": {
            band: sum(1 for item in specs if item["band"] == band)
            for band in BANDS
        },
        "tasks": list(specs),
    }


def save_task_pool(split: str, specs: Sequence[dict], sealed: bool = False) -> Path:
    payload = task_pool_payload(split, specs, sealed=sealed)
    path = SPECS_DIR / f"{split}_tasks.json"
    atomic_write_json(path, payload)
    return path


def load_task_pool(split: str) -> dict:
    path = SPECS_DIR / f"{split}_tasks.json"
    payload = read_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    if payload.get("split") != split:
        raise ValueError(f"{path} is not a {split} task pool")
    if split == "test" and payload.get("sealed") is not True:
        raise ValueError("test task pool is not sealed")
    return payload


def build_training_order(train_specs: Sequence[dict], seed: int = ORDER_SEED) -> dict:
    order = [item["episode_uid"] for item in train_specs]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(order)
    payload = {
        "version": "stage2l-training-order-v1",
        "created_utc": utc_now(),
        "seed": int(seed),
        "map_seed": int(train_specs[0]["map_seed"]) if train_specs else None,
        "task_pool_sha256": canonical_sha256(list(train_specs)),
        "num_tasks": len(order),
        "band_counts": {
            band: sum(
                1 for item in train_specs if item["band"] == band
            )
            for band in BANDS
        },
        "order": order,
    }
    atomic_write_json(SPECS_DIR / "training_order.json", payload)
    return payload


def load_training_order() -> dict:
    payload = read_json(SPECS_DIR / "training_order.json")
    if payload is None:
        raise FileNotFoundError(SPECS_DIR / "training_order.json")
    return payload


# ---------------------------------------------------------------------------
# Deployment topologies
# ---------------------------------------------------------------------------


def _quality_rank(prior_payload: Mapping) -> list[str]:
    return sorted(
        MODEL_NAMES,
        key=lambda name: (
            -float(prior_payload["models"][name]["ndtw"]),
            MODEL_NAMES.index(name),
        ),
    )


def _scenario_model_counts(prior_payload: Mapping) -> dict:
    rank = _quality_rank(prior_payload)
    top, low = rank[:3], rank[3:]
    counts = {name: 0 for name in MODEL_NAMES}
    for name in top:
        counts[name] = 5
    counts[low[0]] = 3
    counts[low[1]] = 2
    return counts


def build_stage2l_topology(
    scenario: str,
    seed: int,
    prior_payload: Mapping,
    role: str,
    verbose: bool = False,
) -> dict:
    """Build one deterministic 20-node deployment for Stage-2l."""
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    base = build_random_gbs_layout(seed=int(seed), verbose=verbose)
    rng = np.random.default_rng(int(seed) ^ 0x2A5F2A5F)

    if scenario == "balanced":
        model_types = [node.model_type for node in base]
        computes = [node.compute_capacity_flops_per_s for node in base]
        bandwidths = [node.bandwidth_hz for node in base]
    else:
        counts = _scenario_model_counts(prior_payload)
        model_types = []
        for name in MODEL_NAMES:
            model_types.extend([name] * counts[name])
        rng.shuffle(model_types)
        rank = _quality_rank(prior_payload)
        top = set(rank[:3])
        if scenario == "quality_rich":
            top_compute, low_compute = 2.0e12, 1.0e12
            top_bandwidth, low_bandwidth = 30.0e6, 10.0e6
        elif scenario == "tradeoff_stress":
            top_compute, low_compute = 0.5e12, 2.0e12
            top_bandwidth, low_bandwidth = 5.0e6, 30.0e6
        else:
            raise ValueError(scenario)
        computes = [
            top_compute if name in top else low_compute for name in model_types
        ]
        bandwidths = [
            top_bandwidth if name in top else low_bandwidth
            for name in model_types
        ]

    nodes = []
    for index, (x, y) in enumerate(
        [(float(node.x), float(node.y)) for node in base]
    ):
        name = model_types[index]
        nodes.append({
            "node_id": f"G{index + 1}",
            "x": float(x),
            "y": float(y),
            "model_type": name,
            "model_index": int(MODEL_INDEX[name]),
            "quality_prior": float(prior_payload["models"][name]["ndtw"]),
            "compute_capacity_flops_per_s": float(computes[index]),
            "bandwidth_hz": float(bandwidths[index]),
            "coverage_radius_m": float(NETWORK.coverage_radius_m),
        })
    if len(nodes) != int(NETWORK.num_gbs):
        raise RuntimeError(f"Expected {NETWORK.num_gbs} nodes, got {len(nodes)}")
    return {
        "version": "stage2l-topology-v1",
        "created_utc": utc_now(),
        "topology_id": f"stage2l-{role}",
        "role": str(role),
        "scenario": scenario,
        "topology_seed": int(seed),
        "nodes": nodes,
        "source_layout": "build_random_gbs_layout + scenario override",
        "quality_prior_source": "Stage-2j Dev Val 180 (capability_prior_v2)",
        "final_test_read": False,
    }


def topology_path(role: str) -> Path:
    return TOPOLOGY_DIR / f"{role}.json"


def save_topology(payload: Mapping) -> Path:
    path = topology_path(str(payload["role"]))
    atomic_write_json(path, dict(payload))
    return path


def load_topology(role: str) -> dict:
    payload = read_json(topology_path(role))
    if payload is None:
        raise FileNotFoundError(topology_path(role))
    return payload


def topology_role_for(config: str, split: str, spec: Mapping) -> str:
    """Return the topology role used by one config for one task."""
    if config == "A":
        return "A_fixed"
    if config == "B":
        if split == "train":
            index = int(spec.get("topology_index", -1))
            if not 0 <= index < B_TRAIN_POOL_SIZE:
                raise ValueError(
                    f"B train task {spec.get('episode_uid')} has invalid "
                    f"topology_index={index}"
                )
            return f"B_train_{index:02d}"
        if split == "test":
            return "B_test"
        return "B_dev"
    if config == "C":
        scenario = str(spec.get("scenario"))
        if scenario not in SCENARIOS:
            raise ValueError(f"C task scenario {scenario} is invalid")
        return f"C_{scenario}"
    raise ValueError(f"Unknown config {config}")


# ---------------------------------------------------------------------------
# Config manifests
# ---------------------------------------------------------------------------


def _scenario_assignment(
    specs: Sequence[dict], seed: int
) -> dict[str, str]:
    """Assign the three deployment scenarios nearly 1:1:1 per band.

    The starting scenario rotates by band so that both the band counts and
    the scenario totals stay balanced (e.g. 60 Dev tasks -> 20 per scenario
    and 20 per band).
    """
    rng = np.random.default_rng(int(seed))
    mapping: dict[str, str] = {}
    for band_index, band in enumerate(BANDS):
        rows = [item for item in specs if item["band"] == band]
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            scenario = SCENARIOS[(position + band_index) % len(SCENARIOS)]
            mapping[rows[index]["episode_uid"]] = scenario
    return mapping


def _topology_assignment(
    specs: Sequence[dict], seed: int
) -> dict[str, int]:
    rng = np.random.default_rng(int(seed))
    return {
        item["episode_uid"]: int(rng.integers(0, B_TRAIN_POOL_SIZE))
        for item in specs
    }


def build_config_manifest(
    config: str,
    train_specs: Sequence[dict],
    dev_specs: Sequence[dict],
    test_specs: Sequence[dict],
    order_payload: Mapping,
) -> dict:
    if config not in CONFIG_DESCRIPTIONS:
        raise ValueError(config)
    payload = {
        "version": "stage2l-config-v1",
        "config": config,
        "description": CONFIG_DESCRIPTIONS[config],
        "created_utc": utc_now(),
        "map_seed": int(MAP_SEED_PHASE1),
        "task_pool_sha256": canonical_sha256(list(train_specs)),
        "dev_pool_sha256": canonical_sha256(list(dev_specs)),
        "test_pool_sha256": canonical_sha256(list(test_specs)),
        "training_order_sha256": canonical_sha256(order_payload["order"]),
        "train_tasks": {},
        "dev_tasks": {},
        "test_tasks": {},
    }
    if config in ("A", "B"):
        for item in train_specs:
            payload["train_tasks"][item["episode_uid"]] = {
                "scenario": "balanced",
            }
        for item in dev_specs:
            payload["dev_tasks"][item["episode_uid"]] = {
                "scenario": "balanced",
            }
        for item in test_specs:
            payload["test_tasks"][item["episode_uid"]] = {
                "scenario": "balanced",
            }
    else:
        train_scenarios = _scenario_assignment(train_specs, CONFIG_SEED + 1)
        dev_scenarios = _scenario_assignment(dev_specs, CONFIG_SEED + 2)
        test_scenarios = _scenario_assignment(test_specs, CONFIG_SEED + 4)
        for item in train_specs:
            payload["train_tasks"][item["episode_uid"]] = {
                "scenario": train_scenarios[item["episode_uid"]],
            }
        for item in dev_specs:
            payload["dev_tasks"][item["episode_uid"]] = {
                "scenario": dev_scenarios[item["episode_uid"]],
            }
        for item in test_specs:
            payload["test_tasks"][item["episode_uid"]] = {
                "scenario": test_scenarios[item["episode_uid"]],
            }
    if config == "B":
        assignment = _topology_assignment(train_specs, CONFIG_SEED + 3)
        for item in train_specs:
            payload["train_tasks"][item["episode_uid"]]["topology_index"] = int(
                assignment[item["episode_uid"]]
            )
    payload["topology_roles"] = {
        "train": sorted({
            topology_role_for(config, "train", {
                **item,
                **payload["train_tasks"][item["episode_uid"]],
            })
            for item in train_specs
        }),
        "dev": sorted({
            topology_role_for(config, "dev", {
                **item,
                **payload["dev_tasks"][item["episode_uid"]],
            })
            for item in dev_specs
        }),
        "test": sorted({
            topology_role_for(config, "test", {
                **item,
                **payload["test_tasks"][item["episode_uid"]],
            })
            for item in test_specs
        }),
        "reference": (
            "A_fixed" if config in ("A", "B") else "C_balanced"
        ),
    }
    atomic_write_json(CONFIG_DIR / f"{config}.json", payload)
    return payload


def load_config_manifest(config: str) -> dict:
    payload = read_json(CONFIG_DIR / f"{config}.json")
    if payload is None:
        raise FileNotFoundError(CONFIG_DIR / f"{config}.json")
    if payload.get("config") != config:
        raise ValueError(f"Config manifest is not {config}")
    return payload


def prepare_config_spec(
    base_spec: Mapping, config_manifest: Mapping, split: str
) -> dict:
    """Attach the config-specific deployment fields to a base nav task."""
    uid = str(base_spec["episode_uid"])
    key = {
        "train": "train_tasks",
        "dev": "dev_tasks",
        "test": "test_tasks",
    }[split]
    if uid not in config_manifest[key]:
        raise KeyError(f"{uid} missing from {key}")
    extra = config_manifest[key][uid]
    spec = dict(base_spec)
    # The inherited Stage-2k metric/finalize code uses the historical key
    # ``seed``.  Stage-2l keeps ``task_seed`` as the authoritative field and
    # exposes the same integer under the legacy alias.
    spec["seed"] = int(spec["task_seed"])
    spec["config"] = str(config_manifest["config"])
    spec["scenario"] = str(extra["scenario"])
    if "topology_index" in extra:
        spec["topology_index"] = int(extra["topology_index"])
    else:
        spec.pop("topology_index", None)
    return spec


def config_specs(
    config: str, split: str
) -> tuple[list[dict], dict, dict]:
    """Return (ordered train/dev/test specs, config manifest, order).

    Training code only calls ``train``/``dev``.  ``test`` exists for the
    post-convergence evaluation entry point; the test pool stays sealed and
    is never touched by a trainer.
    """
    order_payload = load_training_order()
    manifest = load_config_manifest(config)
    if split == "train":
        train_payload = load_task_pool("train")
        by_uid = {item["episode_uid"]: item for item in train_payload["tasks"]}
        if set(by_uid) != set(order_payload["order"]):
            raise ValueError("Training order and task pool differ")
        specs = [
            prepare_config_spec(by_uid[uid], manifest, "train")
            for uid in order_payload["order"]
        ]
    elif split == "dev":
        dev_payload = load_task_pool("dev")
        specs = [
            prepare_config_spec(item, manifest, "dev")
            for item in dev_payload["tasks"]
        ]
    elif split == "test":
        test_payload = load_task_pool("test")
        if not test_payload.get("sealed"):
            raise ValueError("Test pool is not sealed")
        specs = [
            prepare_config_spec(item, manifest, "test")
            for item in test_payload["tasks"]
        ]
    else:
        raise ValueError(f"Unknown split {split}")
    return specs, manifest, order_payload


# ---------------------------------------------------------------------------
# Isolation and distribution audits
# ---------------------------------------------------------------------------


def seed_isolation_audit(
    train_specs: Sequence[dict],
    dev_specs: Sequence[dict],
    test_specs: Sequence[dict],
) -> dict:
    def seeds(specs):
        return {int(item["task_seed"]) for item in specs}

    def uids(specs):
        return {str(item["episode_uid"]) for item in specs}

    train_seeds, dev_seeds, test_seeds = (
        seeds(train_specs), seeds(dev_specs), seeds(test_specs)
    )
    train_uids, dev_uids, test_uids = (
        uids(train_specs), uids(dev_specs), uids(test_specs)
    )
    checks = {
        "train_dev_seed_disjoint": not (train_seeds & dev_seeds),
        "train_test_seed_disjoint": not (train_seeds & test_seeds),
        "dev_test_seed_disjoint": not (dev_seeds & test_seeds),
        "train_dev_uid_disjoint": not (train_uids & dev_uids),
        "train_test_uid_disjoint": not (train_uids & test_uids),
        "dev_test_uid_disjoint": not (dev_uids & test_uids),
        "all_share_phase1_map": (
            {int(item["map_seed"]) for item in train_specs}
            | {int(item["map_seed"]) for item in dev_specs}
            | {int(item["map_seed"]) for item in test_specs}
        ) == {int(MAP_SEED_PHASE1)},
        "test_sealed": True,
    }
    checks["ok"] = all(
        value for key, value in checks.items() if key != "test_sealed"
    )
    return {
        "version": "stage2l-isolation-v1",
        "created_utc": utc_now(),
        "checks": checks,
        "counts": {
            "train": len(train_specs),
            "dev": len(dev_specs),
            "test": len(test_specs),
        },
        "task_seed_ranges": {
            "train": [min(train_seeds), max(train_seeds)],
            "dev": [min(dev_seeds), max(dev_seeds)],
            "test": [min(test_seeds), max(test_seeds)],
        },
    }


def distribution_report(
    train_specs: Sequence[dict],
    dev_specs: Sequence[dict],
    test_specs: Sequence[dict],
) -> dict:
    def describe(specs):
        by_band = {
            band: sum(1 for item in specs if item["band"] == band)
            for band in BANDS
        }
        steps = {
            band: [
                int(item["oracle_steps"])
                for item in specs
                if item["band"] == band
            ]
            for band in BANDS
        }
        distances = {
            band: [
                float(item["start_goal_distance_m"])
                for item in specs
                if item["band"] == band
            ]
            for band in BANDS
        }
        return {
            "count": len(specs),
            "bands": by_band,
            "oracle_steps": {
                band: {
                    "min": int(min(values)),
                    "mean": float(np.mean(values)),
                    "max": int(max(values)),
                }
                for band, values in steps.items()
            },
            "start_goal_distance_m": {
                band: {
                    "min": float(min(values)),
                    "mean": float(np.mean(values)),
                    "max": float(max(values)),
                }
                for band, values in distances.items()
            },
        }

    return {
        "version": "stage2l-distribution-v1",
        "created_utc": utc_now(),
        "map_seed": int(MAP_SEED_PHASE1),
        "train": describe(train_specs),
        "dev": describe(dev_specs),
        "test": describe(test_specs),
        "band_ranges_m": {
            band: [float(v) for v in BAND_RANGES[band]] for band in BANDS
        },
        "band_action_ranges": {
            band: [int(v) for v in BAND_ACTION_RANGES[band]] for band in BANDS
        },
    }


# ---------------------------------------------------------------------------
# Build entry point (called by stage2l_build.py)
# ---------------------------------------------------------------------------


def build_all(verbose: bool = True) -> dict:
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    TOPOLOGY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[stage2l] generating fixed city + task pools", flush=True)
    train_specs = generate_split_tasks(
        "train", MAP_SEED_PHASE1, BAND_COUNTS_3500, TASK_SEED_BASE["train"]
    )
    dev_specs = generate_split_tasks(
        "dev", MAP_SEED_PHASE1, BAND_COUNTS_60, TASK_SEED_BASE["dev"]
    )
    test_specs = generate_split_tasks(
        "test", MAP_SEED_PHASE1, BAND_COUNTS_60, TASK_SEED_BASE["test"]
    )
    save_task_pool("train", train_specs)
    save_task_pool("dev", dev_specs)
    save_task_pool("test", test_specs, sealed=True)
    order_payload = build_training_order(train_specs)

    isolation = seed_isolation_audit(train_specs, dev_specs, test_specs)
    atomic_write_json(SPECS_DIR / "isolation.json", isolation)
    if not isolation["checks"]["ok"]:
        raise RuntimeError(f"Seed/ID isolation failed: {isolation['checks']}")
    distribution = distribution_report(train_specs, dev_specs, test_specs)
    atomic_write_json(SPECS_DIR / "distribution.json", distribution)

    prior_payload, _flat = load_capability_prior()
    topologies = {}
    topologies["A_fixed"] = save_topology(build_stage2l_topology(
        "balanced", TOPOLOGY_SEEDS["A_fixed"], prior_payload, "A_fixed"
    ))
    topologies["B_dev"] = save_topology(build_stage2l_topology(
        "balanced", TOPOLOGY_SEEDS["B_dev"], prior_payload, "B_dev"
    ))
    topologies["B_test"] = save_topology(build_stage2l_topology(
        "balanced", TOPOLOGY_SEEDS["B_test"], prior_payload, "B_test"
    ))
    for index, seed in enumerate(TOPOLOGY_SEEDS["B_train"]):
        role = f"B_train_{index:02d}"
        topologies[role] = save_topology(build_stage2l_topology(
            "balanced", seed, prior_payload, role
        ))
    for scenario, seed in (
        ("balanced", TOPOLOGY_SEEDS["C_balanced"]),
        ("quality_rich", TOPOLOGY_SEEDS["C_quality_rich"]),
        ("tradeoff_stress", TOPOLOGY_SEEDS["C_tradeoff_stress"]),
    ):
        role = f"C_{scenario}"
        topologies[role] = save_topology(build_stage2l_topology(
            scenario, seed, prior_payload, role
        ))

    configs = {}
    for config in CONFIG_DESCRIPTIONS:
        configs[config] = build_config_manifest(
            config, train_specs, dev_specs, test_specs, order_payload
        )

    protocol = {
        "version": "stage2l-protocol-freeze-v1",
        "created_utc": utc_now(),
        "stage": "stage2l_fixed_city",
        "map_seed_phase1": int(MAP_SEED_PHASE1),
        "task_seed_base": TASK_SEED_BASE,
        "train_episodes": TRAIN_EPISODES,
        "dev_episodes": DEV_EPISODES,
        "test_episodes": TEST_EPISODES,
        "band_counts_train": BAND_COUNTS_3500,
        "band_counts_dev_test": BAND_COUNTS_60,
        "training_order_seed": ORDER_SEED,
        "configs": CONFIG_DESCRIPTIONS,
        "topology_seeds": {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in TOPOLOGY_SEEDS.items()
        },
        "reward": (
            "r_nav_t + (-0.25/100)*(running_mean_E2E_t - "
            "running_mean_E2E_{t-1})"
        ),
        "state_mode": "pure_mec_v2",
        "state_dim": 384,
        "action_mask_dim": 20,
        "max_episodes": TRAIN_EPISODES,
        "validation_every": 200,
        "validation_tasks": DEV_EPISODES,
        "stop_radius_m": float(ENV.stop_radius_m),
        "max_steps": int(ENV.max_steps),
        "frozen_mecs": list(MODEL_NAMES),
        "final_test_read": False,
        "test_sealed": True,
        "artifacts": {
            "specs": str(SPECS_DIR),
            "topologies": str(TOPOLOGY_DIR),
            "configs": str(CONFIG_DIR),
            "isolation_sha256": sha256_file(SPECS_DIR / "isolation.json"),
            "distribution_sha256": sha256_file(SPECS_DIR / "distribution.json"),
            "training_order_sha256": sha256_file(
                SPECS_DIR / "training_order.json"
            ),
        },
    }
    atomic_write_json(STAGE2L_ROOT / "protocol_freeze.json", protocol)
    if verbose:
        print(json.dumps({
            "train": distribution["train"]["bands"],
            "dev": distribution["dev"]["bands"],
            "test": distribution["test"]["bands"],
            "map_seed": MAP_SEED_PHASE1,
            "city_fingerprint": build_city(MAP_SEED_PHASE1).fingerprint,
            "train_order_sha256": canonical_sha256(
                order_payload["order"]
            ),
        }, indent=2, ensure_ascii=False), flush=True)
    return {
        "train_specs": train_specs,
        "dev_specs": dev_specs,
        "test_specs": test_specs,
        "order": order_payload,
        "isolation": isolation,
        "distribution": distribution,
        "configs": configs,
        "topologies": topologies,
        "protocol": protocol,
    }
