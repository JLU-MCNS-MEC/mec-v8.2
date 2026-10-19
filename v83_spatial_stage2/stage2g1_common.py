"""Shared Stage-2g.1 near-target supplement and history utilities.

Only observable quantities are used in model inputs. True distance, target
visibility, projection size and goal coordinates are labels / analysis fields.

History convention: frame 0 is the most recent previous state (t-1), frame 1
is t-2, frame 2 is t-3. Actions are the actions executed at those previous
states (i.e. the transitions into t-2, t-1, t).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NEAR_DISTANCE_M = 100.0
HISTORY_FRAMES = 3
ACTION_DIM = 4

# 74 compact observable features used for the short history. They are a subset
# of the 2,858 Stage-2g current features.
COMPACT_FEATURE_NAMES = (
    [
        f"{stat}_s{scale}"
        for scale in range(3)
        for stat in ("sim_max", "sim_top5", "sim_mean", "sim_std")
    ]
    + [
        "sim_max_all_top5", "sim_max_all_mean", "sim_max_all_std",
        "sim_argmax_x", "sim_argmax_y",
    ]
    + [
        f"heat_{channel}_{stat}"
        for channel in range(3)
        for stat in ("ent", "top1", "top5_mass", "cx", "cy")
    ]
    + [
        f"{source}_{stat}"
        for source in ("base", "coh")
        for stat in ("max", "mean", "std", "cx", "cy")
    ]
    + [
        f"{source}_pool_{index}"
        for source in ("base", "coh")
        for index in range(0, 64, 4)
    ]
)
COMPACT_DIM = len(COMPACT_FEATURE_NAMES)

GOAL_FEATURE_NAMES = (
    [f"goal_mean_{i}" for i in range(384)]
    + [f"goal_std_{i}" for i in range(384)]
)
GOAL_DIM = len(GOAL_FEATURE_NAMES)


def compact_indices(feature_names):
    index = {str(name): i for i, name in enumerate(feature_names)}
    missing = [name for name in COMPACT_FEATURE_NAMES if name not in index]
    if missing:
        raise ValueError(f"Missing compact features: {missing[:10]}")
    return np.asarray(
        [index[name] for name in COMPACT_FEATURE_NAMES], dtype=np.int64
    )


def goal_indices(feature_names):
    index = {str(name): i for i, name in enumerate(feature_names)}
    missing = [name for name in GOAL_FEATURE_NAMES if name not in index]
    if missing:
        raise ValueError(f"Missing goal features: {missing[:10]}")
    return np.asarray(
        [index[name] for name in GOAL_FEATURE_NAMES], dtype=np.int64
    )


def near_invisible_label(distance_m, target_visible):
    return int(
        float(distance_m) < NEAR_DISTANCE_M
        and int(target_visible) == 0
    )


def distance_band(distance_m):
    distance = float(distance_m)
    if distance < 20.0:
        return "d_0_20"
    if distance < 50.0:
        return "d_20_50"
    if distance < 100.0:
        return "d_50_100"
    if distance < 200.0:
        return "d_100_200"
    if distance < 400.0:
        return "d_200_400"
    return "d_400_plus"


def one_hot_actions(actions):
    actions = np.asarray(actions, dtype=np.int64)
    one_hot = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    valid = (actions >= 0) & (actions < ACTION_DIM)
    one_hot[np.arange(len(actions))[valid], actions[valid]] = 1.0
    return one_hot


def load_action_map(data_dir, max_shards=None):
    """Read (episode_id, step_id) -> archived action from token shards."""
    import torch

    action_map = {}
    paths = sorted(Path(data_dir).glob("shard_*.pt"))
    if max_shards is not None:
        paths = paths[: int(max_shards)]
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        episodes = data["episode_ids"].long().tolist()
        steps = data["step_ids"].long().tolist()
        actions = data["labels"].long().tolist()
        for episode, step, action in zip(episodes, steps, actions):
            action_map[(int(episode), int(step))] = int(action)
    return action_map


def build_far_history(feature_data, action_map, compact_idx):
    """History for archived far states using previous steps of the same episode."""
    episodes = feature_data["episode_id"].astype(np.int64)
    steps = feature_data["step_id"].astype(np.int64)
    sample_type = feature_data.get(
        "sample_type", np.zeros(len(episodes), dtype=np.int64)
    )
    features = feature_data["features"]
    key_to_index = {
        (int(episode), int(step)): i
        for i, (episode, step) in enumerate(zip(episodes, steps))
    }
    n = len(episodes)
    history_features = np.zeros(
        (n, HISTORY_FRAMES, COMPACT_DIM), dtype=np.float32
    )
    history_actions = np.full((n, HISTORY_FRAMES), -1, dtype=np.int64)
    history_dt = np.zeros((n, HISTORY_FRAMES), dtype=np.float32)
    history_valid = np.zeros(n, dtype=np.int8)
    history_count = np.zeros(n, dtype=np.int8)
    for i in range(n):
        episode = int(episodes[i])
        step = int(steps[i])
        if int(sample_type[i]) != 0 or step < HISTORY_FRAMES:
            continue
        keys = [
            (episode, step - 1 - frame) for frame in range(HISTORY_FRAMES)
        ]
        if any(key not in key_to_index for key in keys):
            continue
        if any(key not in action_map for key in keys):
            continue
        for frame, key in enumerate(keys):
            index = key_to_index[key]
            history_features[i, frame] = features[index][compact_idx]
            history_actions[i, frame] = action_map[key]
            history_dt[i, frame] = 1.0
        history_valid[i] = 1
        history_count[i] = HISTORY_FRAMES
    return {
        "history_features": history_features,
        "history_actions": history_actions,
        "history_dt": history_dt,
        "history_valid": history_valid,
        "history_count": history_count,
    }
