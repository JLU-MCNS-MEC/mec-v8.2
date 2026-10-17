"""
V8.2 token dataset generation.

MEC input modalities:
  Current RGB tokens: 576x384
  Goal RGB tokens: 3x384
  Spatial relation heatmap: 3x8x8
  Global network state: 5D

Removed from MEC input:
  start mask, visual history, depth, absolute UAV position,
  MEC model capability/quality, success reward.

The heatmap is generated entirely from frozen Current/Goal RGB token similarity:
one channel for each Goal scale (14/28/56 px), using the 24x24 current-token grid
and downsampling to 8x8. No simulator goal mask or goal coordinate is used.

The network state is generated from a fixed random 20-MEC deployment per episode.
Only five global observable network quantities are retained; model type and compute
capacity are explicitly ignored.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import AUX, ENCODER, NETWORK, NUM_ACTIONS, TRAIN
from encoder import VisualTokenEncoder
from env import ACTION_NAMES, STOP, ToyUAVEnv, wrap_angle
from gbs_model_v3 import goal_current_similarity
from gbs_network import build_random_gbs_layout, global_network_state
from utils import image_to_tensor


SAMPLE_NATURAL = 0
SAMPLE_HARD_POSITIVE = 1
SAMPLE_BOUNDARY_NEGATIVE = 2
SAMPLE_DISTRACTOR_NEGATIVE = 3

SAMPLE_TYPE_NAMES = {
    SAMPLE_NATURAL: "natural",
    SAMPLE_HARD_POSITIVE: "hard_positive",
    SAMPLE_BOUNDARY_NEGATIVE: "boundary_negative",
    SAMPLE_DISTRACTOR_NEGATIVE: "distractor_negative",
}


def load_frozen_encoder(device):
    encoder = VisualTokenEncoder().to(device)
    encoder.load_state_dict(
        torch.load("shared_encoder.pt", map_location=device, weights_only=True)
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def encode_multiscale_goal_tokens(goal_rgbs, encoder, device):
    """Each Goal reference scale becomes one 384-D foreground-weighted token."""
    arr = np.asarray(goal_rgbs, dtype=np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"goal_rgbs must be [S,H,W,3], got {arr.shape}")

    x = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    if x.shape[-2:] != (ENCODER.image_size, ENCODER.image_size):
        x = F.interpolate(
            x,
            size=(ENCODER.image_size, ENCODER.image_size),
            mode="bilinear",
            align_corners=False,
        )
    x = x.to(device)
    full_tokens = encoder(x)

    s = int(full_tokens.shape[0])
    if s != AUX.goal_token_count:
        raise ValueError(
            f"Goal scales={s}, config expects {AUX.goal_token_count}"
        )

    raw = torch.from_numpy(arr).float() / 255.0
    bg = raw[:, 0:1, 0:1, :]
    diff = (raw - bg).abs().amax(dim=-1)
    fg = (diff > (8.0 / 255.0)).float().unsqueeze(1)
    fg = F.interpolate(
        fg,
        size=(ENCODER.grid_size, ENCODER.grid_size),
        mode="area",
    ).squeeze(1)
    weights = fg.reshape(s, -1).to(device)

    denom = weights.sum(dim=1, keepdim=True)
    empty = denom.squeeze(1) < 1e-6
    if bool(empty.any()):
        weights[empty] = 0.0
        center = (
            (ENCODER.grid_size // 2) * ENCODER.grid_size
            + ENCODER.grid_size // 2
        )
        weights[empty, center] = 1.0
        denom = weights.sum(dim=1, keepdim=True)

    goal_tokens = (full_tokens * weights.unsqueeze(-1)).sum(dim=1) / denom
    return goal_tokens.detach().cpu()


@torch.no_grad()
def goal_conditioned_spatial_heatmap(current_tokens, goal_tokens):
    """
    Build [3,8,8] scale-specific Goal-Current similarity heatmaps.

    For each Goal reference scale:
      H_s = cosine(CurrentPatch_i, Goal_s)
    on the 24x24 patch grid, then adaptive-average-pool to 8x8.

    The heatmap is an observable RGB-derived representation; it does not use
    env.goal, env.distance_to_goal, or env.goal_token_soft_mask.
    """
    squeeze = False
    if current_tokens.ndim == 2:
        current_tokens = current_tokens.unsqueeze(0)
        squeeze = True
    if goal_tokens.ndim == 2:
        goal_tokens = goal_tokens.unsqueeze(0)

    if current_tokens.ndim != 3 or goal_tokens.ndim != 3:
        raise ValueError("current_tokens/goal_tokens must be rank-3")

    if current_tokens.shape[1:] != (ENCODER.num_tokens, ENCODER.token_dim):
        raise ValueError(f"Unexpected current token shape {tuple(current_tokens.shape)}")
    if goal_tokens.shape[1:] != (AUX.goal_token_count, ENCODER.token_dim):
        raise ValueError(f"Unexpected goal token shape {tuple(goal_tokens.shape)}")

    sim = goal_current_similarity(current_tokens, goal_tokens)
    b, s, n = sim.shape
    if s != AUX.spatial_heatmap_channels:
        raise ValueError(
            f"Goal scales {s} != heatmap channels {AUX.spatial_heatmap_channels}"
        )

    maps = sim.reshape(
        b,
        s,
        ENCODER.grid_size,
        ENCODER.grid_size,
    )
    maps = maps.reshape(
        b * s,
        1,
        ENCODER.grid_size,
        ENCODER.grid_size,
    )
    pooled = F.adaptive_avg_pool2d(
        maps,
        (AUX.spatial_heatmap_grid, AUX.spatial_heatmap_grid),
    )
    heatmap = pooled.reshape(
        b,
        s,
        AUX.spatial_heatmap_grid,
        AUX.spatial_heatmap_grid,
    )
    return heatmap.squeeze(0) if squeeze else heatmap


def _save_shard(output_dir, shard_index, buf):
    if not buf["tokens"]:
        return
    path = Path(output_dir) / f"shard_{shard_index:04d}.pt"
    torch.save(
        {
            "tokens": torch.stack(buf["tokens"]).to(torch.float16),
            "goal_tokens": torch.stack(buf["goal_tokens"]).to(torch.float16),
            "spatial_heatmaps": torch.stack(buf["spatial_heatmaps"]).to(torch.float16),
            "network_states": torch.stack(buf["network_states"]).to(torch.float16),
            "labels": torch.tensor(buf["labels"], dtype=torch.long),
            "arrival_labels": torch.tensor(buf["arrival_labels"], dtype=torch.float32),
            "move_valid": torch.tensor(buf["move_valid"], dtype=torch.uint8),
            "arrival_train_mask": torch.tensor(buf["arrival_train_mask"], dtype=torch.uint8),
            "distance_to_goal_m": torch.tensor(
                buf["distance_to_goal_m"], dtype=torch.float32
            ),
            "sample_type": torch.tensor(buf["sample_type"], dtype=torch.uint8),
            "goal_visible": torch.tensor(buf["goal_visible"], dtype=torch.uint8),
            "episode_ids": torch.tensor(buf["episode_ids"], dtype=torch.long),
            "step_ids": torch.tensor(buf["step_ids"], dtype=torch.long),
            "episode_seeds": torch.tensor(buf["episode_seeds"], dtype=torch.long),
            "network_layout_seeds": torch.tensor(
                buf["network_layout_seeds"], dtype=torch.long
            ),
            "trajectory_band_ids": torch.tensor(
                buf["trajectory_band_ids"], dtype=torch.long
            ),
        },
        path,
    )
    print(f"Saved {path} ({len(buf['tokens'])} samples)")


def _new_buffer():
    return {
        "tokens": [],
        "goal_tokens": [],
        "spatial_heatmaps": [],
        "network_states": [],
        "labels": [],
        "arrival_labels": [],
        "move_valid": [],
        "arrival_train_mask": [],
        "distance_to_goal_m": [],
        "sample_type": [],
        "goal_visible": [],
        "episode_ids": [],
        "step_ids": [],
        "episode_seeds": [],
        "network_layout_seeds": [],
        "trajectory_band_ids": [],
    }


def _append_sample(
    buf,
    current_tokens,
    goal_token,
    spatial_heatmap,
    network_state,
    label,
    arrival_label,
    move_valid,
    arrival_train_mask,
    distance_to_goal_m,
    sample_type,
    goal_visible,
    episode_id,
    step_id,
    episode_seed,
    network_layout_seed,
    band_id,
):
    buf["tokens"].append(current_tokens)
    buf["goal_tokens"].append(goal_token.clone())
    buf["spatial_heatmaps"].append(spatial_heatmap.clone())
    buf["network_states"].append(network_state.clone())
    buf["labels"].append(int(label))
    buf["arrival_labels"].append(float(arrival_label))
    buf["move_valid"].append(int(move_valid))
    buf["arrival_train_mask"].append(int(arrival_train_mask))
    buf["distance_to_goal_m"].append(float(distance_to_goal_m))
    buf["sample_type"].append(int(sample_type))
    buf["goal_visible"].append(int(goal_visible))
    buf["episode_ids"].append(int(episode_id))
    buf["step_ids"].append(int(step_id))
    buf["episode_seeds"].append(int(episode_seed))
    buf["network_layout_seeds"].append(int(network_layout_seed))
    buf["trajectory_band_ids"].append(int(band_id))


def _trajectory_band_schedule(target_episodes, seed_start):
    bands = list(TRAIN.trajectory_mix)
    weights = np.asarray([float(x[1]) for x in bands], dtype=np.float64)
    weights /= weights.sum()
    raw_counts = weights * int(target_episodes)
    counts = np.floor(raw_counts).astype(int)
    remaining = int(target_episodes) - int(counts.sum())
    if remaining > 0:
        order = np.argsort(-(raw_counts - counts))
        for i in range(remaining):
            counts[int(order[i % len(order)])] += 1

    schedule = []
    for band_id, (band, count) in enumerate(zip(bands, counts.tolist())):
        schedule.extend([(band_id, band)] * int(count))
    rng = np.random.default_rng(int(seed_start) + 2026)
    rng.shuffle(schedule)
    return schedule, counts.tolist()


def _inspect_oracle_candidate(env, seed, band):
    band_name, _weight, distance_min, distance_max, action_min, action_max = band
    env.reset(
        seed=seed,
        goal_distance_range_m=(distance_min, distance_max),
        render_observation=False,
    )
    start_goal_distance = float(env.distance_to_goal)
    action_counter = Counter()
    success = False
    done = False
    steps = 0
    while not done and steps < env.max_steps:
        action = int(env.oracle_action())
        action_counter[action] += 1
        steps += 1
        _, success, done = env.step(action, render_observation=False)

    if not success:
        reason, accepted = "timeout_or_no_stop", False
    elif not (int(action_min) <= steps <= int(action_max)):
        reason, accepted = "action_count_out_of_range", False
    else:
        reason, accepted = "accepted", True

    return {
        "accepted": accepted,
        "reason": reason,
        "band": str(band_name),
        "seed": int(seed),
        "steps": int(steps),
        "start_goal_distance_m": start_goal_distance,
        "success": bool(success),
        "forward": int(action_counter[0]),
        "left": int(action_counter[1]),
        "right": int(action_counter[2]),
        "stop": int(action_counter[3]),
    }


def _set_pose(env, position, yaw):
    env.x = float(position[0])
    env.y = float(position[1])
    env.yaw = wrap_angle(float(yaw))


def _sample_pose_around_target(
    env,
    target_xy,
    distance_range,
    rng,
    yaw_jitter_deg,
    require_goal_visible=False,
):
    lo, hi = map(float, distance_range)
    for _ in range(200):
        d = float(rng.uniform(lo, hi))
        bearing = float(rng.uniform(-math.pi, math.pi))
        unit = np.asarray([math.cos(bearing), math.sin(bearing)], dtype=np.float32)
        pos = np.asarray(target_xy, dtype=np.float32) - d * unit
        if not (
            1.0 <= pos[0] <= env.world_size - 1.0
            and 1.0 <= pos[1] <= env.world_size - 1.0
        ):
            continue
        yaw = bearing + math.radians(
            float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        )
        _set_pose(env, pos, yaw)
        if require_goal_visible and env.goal_projection() is None:
            continue
        return pos, yaw, d, bearing
    return None


def _collect_hard_state(
    env,
    sample_type,
    rng,
    encoder,
    device,
    goal_token,
    network_nodes,
    episode_id,
    episode_seed,
    network_layout_seed,
    band_id,
    hard_index,
):
    if sample_type == SAMPLE_HARD_POSITIVE:
        sampled = _sample_pose_around_target(
            env,
            env.goal,
            TRAIN.arrival_positive_distance_range_m,
            rng,
            TRAIN.arrival_pose_yaw_jitter_deg,
            require_goal_visible=True,
        )
        focal = env.goal
    elif sample_type == SAMPLE_BOUNDARY_NEGATIVE:
        sampled = _sample_pose_around_target(
            env,
            env.goal,
            TRAIN.arrival_boundary_negative_range_m,
            rng,
            TRAIN.arrival_pose_yaw_jitter_deg,
            require_goal_visible=True,
        )
        focal = env.goal
    elif sample_type == SAMPLE_DISTRACTOR_NEGATIVE:
        candidates = [
            lm for lm in env.landmarks
            if 100.0 <= lm.x <= env.world_size - 100.0
            and 100.0 <= lm.y <= env.world_size - 100.0
        ]
        rng.shuffle(candidates)
        sampled = None
        focal = None
        for lm in candidates[:80]:
            trial = _sample_pose_around_target(
                env,
                lm.position,
                TRAIN.arrival_distractor_near_range_m,
                rng,
                TRAIN.arrival_pose_yaw_jitter_deg,
                require_goal_visible=False,
            )
            if trial is None:
                continue
            if float(np.linalg.norm(env.position - env.goal)) < float(
                TRAIN.arrival_distractor_goal_min_distance_m
            ):
                continue
            if env._project(lm.position, lm.physical_size_m) is None:
                continue
            sampled = trial
            focal = lm.position.copy()
            break
    else:
        raise ValueError(sample_type)

    if sampled is None:
        return None

    current_pos, current_yaw, _focal_distance, _bearing = sampled
    _set_pose(env, current_pos, current_yaw)

    obs = env.render()
    current_tokens = (
        encoder(image_to_tensor(obs, device)).squeeze(0).detach().cpu()
    )
    spatial_heatmap = goal_conditioned_spatial_heatmap(
        current_tokens, goal_token
    ).cpu()
    network_state = torch.from_numpy(
        global_network_state(network_nodes, env.position)
    ).float()

    distance_to_goal = float(env.distance_to_goal)
    arrival = float(distance_to_goal < env.stop_radius)
    label = STOP if arrival > 0.5 else int(env.oracle_action())

    return {
        "current_tokens": current_tokens,
        "goal_token": goal_token,
        "spatial_heatmap": spatial_heatmap,
        "network_state": network_state,
        "label": label,
        "arrival_label": arrival,
        "move_valid": 0,
        "arrival_train_mask": 1,
        "distance_to_goal_m": distance_to_goal,
        "sample_type": sample_type,
        "goal_visible": int(env.goal_projection() is not None),
        "episode_id": episode_id,
        "step_id": -(hard_index + 1),
        "episode_seed": episode_seed,
        "network_layout_seed": network_layout_seed,
        "band_id": band_id,
    }


def _print_and_save_dataset_stats(
    output_dir,
    trajectory_records,
    total_samples,
    sample_type_counter,
    arrival_counter,
    arrival_train_counter,
    planned_counts,
    rejected_reasons,
    candidate_attempts,
    include_hard_states,
):
    steps = np.asarray(
        [x["steps"] for x in trajectory_records], dtype=np.float64
    )
    distances = np.asarray(
        [x["start_goal_distance_m"] for x in trajectory_records],
        dtype=np.float64,
    )
    action_counter = Counter()
    band_counter = Counter()
    for record in trajectory_records:
        for i, key in enumerate(["forward", "left", "right", "stop"]):
            action_counter[i] += int(record[key])
        band_counter[str(record["band"])] += 1

    natural_samples = int(sum(int(x["steps"]) for x in trajectory_records))
    hard_samples = int(total_samples - natural_samples)
    stop_count = int(action_counter[STOP])
    num_episodes = len(trajectory_records)

    stats = {
        "version": "v8.2-input-redesign",
        "num_episodes": int(num_episodes),
        "total_samples": int(total_samples),
        "natural_samples": natural_samples,
        "hard_arrival_samples": hard_samples,
        "include_hard_states": bool(include_hard_states),
        "mean_episode_actions": float(steps.mean()),
        "median_episode_actions": float(np.median(steps)),
        "p90_episode_actions": float(np.percentile(steps, 90)),
        "min_episode_actions": int(steps.min()),
        "max_episode_actions": int(steps.max()),
        "mean_start_goal_distance_m": float(distances.mean()),
        "median_start_goal_distance_m": float(np.median(distances)),
        "p90_start_goal_distance_m": float(np.percentile(distances, 90)),
        "stop_samples_natural": stop_count,
        "stop_ratio_natural": float(stop_count / max(natural_samples, 1)),
        "action_counts_natural": {
            ACTION_NAMES[i]: int(action_counter[i]) for i in range(NUM_ACTIONS)
        },
        "sample_type_counts": {
            SAMPLE_TYPE_NAMES[int(k)]: int(v)
            for k, v in sample_type_counter.items()
        },
        "arrival_label_counts": {
            "negative": int(arrival_counter[0]),
            "positive": int(arrival_counter[1]),
        },
        "arrival_train_mask_counts": {
            "excluded": int(arrival_train_counter[0]),
            "included": int(arrival_train_counter[1]),
        },
        "trajectory_band_counts": dict(band_counter),
        "planned_band_counts": [int(x) for x in planned_counts],
        "candidate_attempts": int(candidate_attempts),
        "rejected_candidates": {
            str(k): int(v) for k, v in rejected_reasons.items()
        },
        "input_modalities": {
            "current_rgb_tokens": [ENCODER.num_tokens, ENCODER.token_dim],
            "goal_rgb_tokens": [AUX.goal_token_count, ENCODER.token_dim],
            "spatial_heatmap": [
                AUX.spatial_heatmap_channels,
                AUX.spatial_heatmap_grid,
                AUX.spatial_heatmap_grid,
            ],
            "network_state": [
                AUX.network_state_dim
            ],
        },
        "removed_modalities": [
            "uav_absolute_position",
            "start_mask",
            "visual_history",
            "depth",
            "mec_model_capability",
            "success_reward_hint",
        ],
        "heatmap_definition": {
            "channels": list(AUX.goal_reference_object_pixels),
            "source": "Current-Goal cosine similarity",
            "source_grid": ENCODER.grid_size,
            "output_grid": AUX.spatial_heatmap_grid,
        },
        "network_state_names": [
            "availability_ratio",
            "nearest_relative_distance",
            "best_link_snr",
            "best_link_bandwidth",
            "best_link_visual_tcom",
        ],
        "trajectories": trajectory_records,
    }

    stats_path = Path(output_dir) / "dataset_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 88)
    print("V8.2 dataset statistics")
    print("=" * 88)
    print(f"Episodes             : {num_episodes}")
    print(f"Natural samples      : {natural_samples}")
    print(f"Arrival hard samples : {hard_samples}")
    print(f"Total samples        : {total_samples}")
    print(
        f"Episode actions      : mean={steps.mean():.2f}, "
        f"median={np.median(steps):.1f}, "
        f"P90={np.percentile(steps,90):.1f}, "
        f"range=[{int(steps.min())},{int(steps.max())}]"
    )
    print(
        f"Start-goal distance  : mean={distances.mean():.2f}m, "
        f"median={np.median(distances):.2f}m, "
        f"P90={np.percentile(distances,90):.2f}m"
    )
    print(f"Trajectory bands     : {dict(band_counter)}")
    print("Natural action distribution:")
    for action_id in range(NUM_ACTIONS):
        n = int(action_counter[action_id])
        print(
            f"  {ACTION_NAMES[action_id]:>8s}: "
            f"{n:6d} ({100*n/max(natural_samples,1):6.2f}%)"
        )
    print(
        "Arrival labels       : "
        f"positive={arrival_counter[1]}, negative={arrival_counter[0]}"
    )
    print(
        "Input                : Current 576x384 | Goal 3x384 | "
        "Heatmap 3x8x8 | Network 5D"
    )
    print(f"Saved statistics     : {stats_path}")


def generate_token_dataset(
    output_dir,
    target_episodes,
    seed_start,
    include_hard_states=False,
):
    """
    Generate one split.

    Train keeps the V8.1 natural expert trajectories + six hard arrival states
    per accepted episode. Val/Test contain natural trajectories only.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        f"Current={ENCODER.num_tokens}x{ENCODER.token_dim}; "
        f"Goal={AUX.goal_token_count}x{ENCODER.token_dim}; "
        f"Heatmap={AUX.spatial_heatmap_channels}x"
        f"{AUX.spatial_heatmap_grid}x{AUX.spatial_heatmap_grid}; "
        f"Network={AUX.network_state_dim}D"
    )
    print(f"Target episodes: {target_episodes}")
    print(f"Arrival hard states: {bool(include_hard_states)}")

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = load_frozen_encoder(device)
    env = ToyUAVEnv()
    schedule, planned_counts = _trajectory_band_schedule(
        target_episodes, seed_start
    )

    buf = _new_buffer()
    total_samples = 0
    shard_index = 0
    candidate_seed = int(seed_start)
    candidate_attempts = 0
    rejected_reasons = Counter()
    trajectory_records = []
    sample_type_counter = Counter()
    arrival_counter = Counter()
    arrival_train_counter = Counter()
    arrival_rng = np.random.default_rng(int(seed_start) + 99173)

    def append_and_maybe_save(sample):
        nonlocal buf, total_samples, shard_index
        _append_sample(
            buf,
            sample["current_tokens"],
            sample["goal_token"],
            sample["spatial_heatmap"],
            sample["network_state"],
            sample["label"],
            sample["arrival_label"],
            sample["move_valid"],
            sample["arrival_train_mask"],
            sample["distance_to_goal_m"],
            sample["sample_type"],
            sample["goal_visible"],
            sample["episode_id"],
            sample["step_id"],
            sample["episode_seed"],
            sample["network_layout_seed"],
            sample["band_id"],
        )
        total_samples += 1
        sample_type_counter[int(sample["sample_type"])] += 1
        arrival_counter[int(float(sample["arrival_label"]) >= 0.5)] += 1
        arrival_train_counter[int(sample["arrival_train_mask"])] += 1
        if len(buf["tokens"]) >= int(TRAIN.token_shard_size):
            _save_shard(output_dir, shard_index, buf)
            buf = _new_buffer()
            shard_index += 1

    with torch.no_grad():
        for episode_id, (band_id, band) in enumerate(schedule):
            accepted_info = None
            for _ in range(int(TRAIN.trajectory_max_candidate_attempts)):
                seed = candidate_seed
                candidate_seed += 1
                candidate_attempts += 1
                info = _inspect_oracle_candidate(env, seed, band)
                if info["accepted"]:
                    accepted_info = info
                    break
                rejected_reasons[info["reason"]] += 1
            if accepted_info is None:
                raise RuntimeError(f"Unable to find accepted {band[0]} trajectory")

            # Keep network layout independent of environment RNG.
            network_layout_seed = (
                int(NETWORK.layout_seed_base)
                + int(seed_start) * 17
                + int(episode_id)
            )
            network_nodes = build_random_gbs_layout(seed=network_layout_seed, verbose=False)

            obs = env.reset(
                seed=accepted_info["seed"],
                goal_distance_range_m=(float(band[2]), float(band[3])),
                render_observation=True,
            )
            goal_token = encode_multiscale_goal_tokens(
                env.get_goal_reference_multiscale_rgbs(), encoder, device
            )

            done = False
            success = False
            step_id = 0

            while not done:
                distance = float(env.distance_to_goal)
                oracle_action = int(env.oracle_action())
                current_tokens = (
                    encoder(image_to_tensor(obs, device))
                    .squeeze(0)
                    .detach()
                    .cpu()
                )
                spatial_heatmap = goal_conditioned_spatial_heatmap(
                    current_tokens, goal_token
                ).cpu()
                network_state = torch.from_numpy(
                    global_network_state(network_nodes, env.position)
                ).float()

                arrival_label = float(distance < env.stop_radius)
                move_valid = int(oracle_action != STOP)

                if (
                    arrival_label > 0.5
                    or distance <= float(TRAIN.arrival_natural_near_radius_m)
                ):
                    arrival_train_mask = 1
                else:
                    arrival_train_mask = int(
                        arrival_rng.random()
                        < float(TRAIN.arrival_far_negative_keep_prob)
                    )

                append_and_maybe_save(
                    {
                        "current_tokens": current_tokens,
                        "goal_token": goal_token,
                        "spatial_heatmap": spatial_heatmap,
                        "network_state": network_state,
                        "label": oracle_action,
                        "arrival_label": arrival_label,
                        "move_valid": move_valid,
                        "arrival_train_mask": arrival_train_mask,
                        "distance_to_goal_m": distance,
                        "sample_type": SAMPLE_NATURAL,
                        "goal_visible": int(env.goal_projection() is not None),
                        "episode_id": episode_id,
                        "step_id": step_id,
                        "episode_seed": int(accepted_info["seed"]),
                        "network_layout_seed": int(network_layout_seed),
                        "band_id": band_id,
                    }
                )

                step_id += 1
                obs, success, done = env.step(
                    oracle_action,
                    render_observation=(oracle_action != STOP),
                )

            if not success:
                raise RuntimeError("Accepted trajectory did not end with natural Stop")
            if step_id != int(accepted_info["steps"]):
                raise RuntimeError("Oracle trajectory length changed during collection")

            if include_hard_states:
                plan = (
                    [SAMPLE_HARD_POSITIVE]
                    * int(TRAIN.arrival_hard_positive_per_episode)
                    + [SAMPLE_BOUNDARY_NEGATIVE]
                    * int(TRAIN.arrival_hard_boundary_negative_per_episode)
                    + [SAMPLE_DISTRACTOR_NEGATIVE]
                    * int(TRAIN.arrival_hard_distractor_negative_per_episode)
                )
                for hard_idx, sample_type in enumerate(plan):
                    hard = _collect_hard_state(
                        env=env,
                        sample_type=sample_type,
                        rng=arrival_rng,
                        encoder=encoder,
                        device=device,
                        goal_token=goal_token,
                        network_nodes=network_nodes,
                        episode_id=episode_id,
                        episode_seed=int(accepted_info["seed"]),
                        network_layout_seed=int(network_layout_seed),
                        band_id=band_id,
                        hard_index=hard_idx,
                    )
                    if hard is None:
                        print(
                            f"Warning: unable to generate hard state "
                            f"episode={episode_id}, "
                            f"type={SAMPLE_TYPE_NAMES[sample_type]}"
                        )
                        continue
                    append_and_maybe_save(hard)

            trajectory_records.append(
                {
                    **accepted_info,
                    "network_layout_seed": int(network_layout_seed),
                }
            )
            if (episode_id + 1) % 10 == 0 or episode_id + 1 == int(
                target_episodes
            ):
                print(
                    f"Accepted episode {episode_id+1}/{target_episodes} | "
                    f"band={accepted_info['band']} | "
                    f"distance={accepted_info['start_goal_distance_m']:.1f}m | "
                    f"actions={accepted_info['steps']} | "
                    f"layout={network_layout_seed} | "
                    f"total samples={total_samples}"
                )

    if buf["tokens"]:
        _save_shard(output_dir, shard_index, buf)

    _print_and_save_dataset_stats(
        output_dir=output_dir,
        trajectory_records=trajectory_records,
        total_samples=total_samples,
        sample_type_counter=sample_type_counter,
        arrival_counter=arrival_counter,
        arrival_train_counter=arrival_train_counter,
        planned_counts=planned_counts,
        rejected_reasons=rejected_reasons,
        candidate_attempts=candidate_attempts,
        include_hard_states=include_hard_states,
    )
    print(
        f"\nGenerated {total_samples} V8.2 samples from "
        f"{len(trajectory_records)} accepted episodes in {output_dir}"
    )
