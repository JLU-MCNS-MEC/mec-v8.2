"""Stage-2i data/feature utilities.

Everything here is derived from observable inputs:
  * visual history: mean/max-pooled frozen Current-RGB tokens;
  * UAV pose history: relative displacement / heading deltas / accumulated
    relative pose, plus executed action one-hot (never goal coordinates,
    true distance, true bearing, oracle action or success);
  * GAB: causal 8-bin target-bearing belief from the observable 3x8x8
    Goal-Current similarity heatmap (same rule as Stage-2h);
  * T5 map: an anonymous global landmark database.  Every node goes through
    the same render + frozen-encoder pipeline.  Retrieval is by cosine
    similarity to the Goal RGB tokens; node identity flags are never read.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from config import AUX, ENCODER, ENV, TRAIN
from env import ToyUAVEnv, wrap_angle
from v83_spatial_stage2.stage2h_gab import (
    GABConfig,
    GABState,
    current_evidence,
    normalized_entropy,
)

HISTORY_STEPS = 16
VISUAL_DESCRIPTOR_DIM = 2 * ENCODER.token_dim
POSE_FEATURE_DIM = 10
GAB_FEATURE_DIM = 20
MAP_FEATURE_DIM = 22
MAP_TOP_K = 5
MAP_RENDER_SIZE = ENCODER.image_size
MAP_OBJECT_PIXELS = int(AUX.goal_reference_object_pixels[-1])
MAP_POSITION_SCALE_M = 500.0
TURN_STEP_RAD = math.radians(float(ENV.turn_step_deg))
FORWARD_STEP_M = float(ENV.forward_step_m)


# ---------------------------------------------------------------------------
# Visual descriptor (identical in training and closed loop)
# ---------------------------------------------------------------------------


@torch.no_grad()
def visual_descriptor(tokens):
    """Mean + max pooled frozen Current-RGB tokens.

    Tokens are round-tripped through float16 to match the archived shard
    precision exactly (training features were stored as float16).
    """
    if tokens.ndim != 3 or tokens.shape[-1] != ENCODER.token_dim:
        raise ValueError(f"tokens must be [B,{ENCODER.num_tokens},"
                         f"{ENCODER.token_dim}], got {tuple(tokens.shape)}")
    x = tokens.detach().to(torch.float16).float()
    return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)


# ---------------------------------------------------------------------------
# Pose / history construction
# ---------------------------------------------------------------------------


def pose_action_feature(previous_pose, current_pose, next_pose, action,
                        reference_pose=None):
    """One history slot: relative motion + pose relative to the query state.

    previous_pose/next_pose bracket the executed action; reference_pose is the
    query state (the newest state).  The action is the one actually executed
    after ``previous_pose``.
    """
    if reference_pose is None:
        reference_pose = current_pose
    px, py, pyaw = map(float, previous_pose)
    nx, ny, _ = map(float, next_pose)
    rx, ry, ryaw = map(float, reference_pose)

    dx = nx - px
    dy = ny - py
    motion_forward = (dx * math.cos(pyaw) + dy * math.sin(pyaw)) / FORWARD_STEP_M
    motion_right = (-dx * math.sin(pyaw) + dy * math.cos(pyaw)) / FORWARD_STEP_M
    motion_yaw = wrap_angle(float(next_pose[2]) - pyaw) / TURN_STEP_RAD

    gx = px - rx
    gy = py - ry
    relative_forward = (gx * math.cos(ryaw) + gy * math.sin(ryaw)) / 100.0
    relative_right = (-gx * math.sin(ryaw) + gy * math.cos(ryaw)) / 100.0
    relative_yaw = wrap_angle(pyaw - ryaw) / math.pi

    one_hot = np.zeros(4, dtype=np.float32)
    if 0 <= int(action) < 4:
        one_hot[int(action)] = 1.0
    return np.concatenate([
        np.asarray([motion_forward, motion_right, motion_yaw,
                    relative_forward, relative_right, relative_yaw],
                   dtype=np.float32),
        one_hot,
    ]).astype(np.float32)


def history_arrays(poses, actions, index, visuals, steps=HISTORY_STEPS):
    """Causal history for the state at ``index`` in an episode sequence.

    Slot 0 is the most recent past state (index-1); slot 15 is index-16.
    Missing slots are zero and masked out.  No future state is touched.
    """
    vis = np.zeros((steps, VISUAL_DESCRIPTOR_DIM), dtype=np.float16)
    pose = np.zeros((steps, POSE_FEATURE_DIM), dtype=np.float32)
    action = np.zeros((steps, 4), dtype=np.float32)
    mask = np.zeros((steps,), dtype=np.uint8)
    current_pose = poses[index]
    for offset in range(1, steps + 1):
        source = index - offset
        if source < 0:
            continue
        vis[offset - 1] = np.asarray(visuals[source], dtype=np.float16)
        pose[offset - 1] = pose_action_feature(
            poses[source], poses[source + 1], poses[source + 1], actions[source],
            reference_pose=current_pose,
        )
        action[offset - 1] = np.eye(4, dtype=np.float32)[int(actions[source])]
        mask[offset - 1] = 1
    return vis, pose, action, mask


def online_history_arrays(past, steps=HISTORY_STEPS):
    """Closed-loop history from a list of past (pose, visual, action) tuples.

    ``past`` is ordered oldest -> newest and holds the *previous* states.
    """
    vis = np.zeros((steps, VISUAL_DESCRIPTOR_DIM), dtype=np.float16)
    pose = np.zeros((steps, POSE_FEATURE_DIM), dtype=np.float32)
    action = np.zeros((steps, 4), dtype=np.float32)
    mask = np.zeros((steps,), dtype=np.uint8)
    if not past:
        return vis, pose, action, mask
    current_pose = past[-1]["next_pose"]
    for slot, offset in enumerate(range(1, steps + 1)):
        if len(past) < offset:
            break
        item = past[-offset]
        vis[slot] = np.asarray(item["visual"], dtype=np.float16)
        pose[slot] = pose_action_feature(
            item["pose"], item["next_pose"], item["next_pose"],
            item["action"], reference_pose=current_pose,
        )
        action[slot] = np.eye(4, dtype=np.float32)[int(item["action"])]
        mask[slot] = 1
    return vis, pose, action, mask


# ---------------------------------------------------------------------------
# GAB (native 3x8x8 heatmap; identical update rule to Stage-2h)
# ---------------------------------------------------------------------------


def fit_gab_threshold(heatmaps, percentile=60.0):
    """Fit the reliability threshold on Train observable heatmaps only."""
    confidences = np.asarray([
        current_evidence(np.asarray(h, dtype=np.float64))["confidence"]
        for h in heatmaps
    ], dtype=np.float64)
    return float(np.percentile(confidences, percentile))


def build_gab_for_episode(heatmaps, actions, config, step_ids=None):
    """Causal GAB features for one ordered episode.

    ``heatmaps`` and ``actions`` are ordered by step; the previous action is
    the action executed after the previous state.  The state is reset at the
    episode boundary.
    """
    state = GABState(config)
    features = np.zeros((len(heatmaps), GAB_FEATURE_DIM), dtype=np.float32)
    previous_action = -1
    previous_step = None
    for index in range(len(heatmaps)):
        if step_ids is not None:
            step = int(step_ids[index])
            if previous_step is not None and step != previous_step + 1:
                state = GABState(config)
                previous_action = -1
            previous_step = step
        state.update(np.asarray(heatmaps[index], dtype=np.float64), previous_action)
        features[index] = state.features()
        previous_action = int(actions[index])
    return features


def gab_config_from_threshold(threshold):
    return GABConfig(reliability_threshold=float(threshold))


# ---------------------------------------------------------------------------
# Anonymous global landmark map (T5)
# ---------------------------------------------------------------------------


def build_anonymous_landmark_nodes(env, seed):
    """All environment landmarks as anonymous map nodes.

    The list is the 160 ordinary landmarks plus the goal object, then a
    seed-driven permutation.  No is_goal/is_distractor/Goal-ID field is read;
    every node is handled by the same renderer and encoder.  The goal node's
    index is returned **for diagnostics only** and is never passed to the
    feature builder.
    """
    nodes = list(env.landmarks)
    nodes.append(env.goal_landmark)
    order = np.random.default_rng(int(seed) ^ 0x5EED1234).permutation(len(nodes))
    shuffled = [nodes[int(i)] for i in order]
    goal_index = int(np.where(order == len(nodes) - 1)[0][0])
    return shuffled, goal_index


def render_landmark_canvas(env, landmark, canvas_size=MAP_RENDER_SIZE,
                           object_width_px=MAP_OBJECT_PIXELS):
    """Canonical 336x336 survey image for one anonymous landmark.

    Uses the same shape/marker drawing primitives and the same canvas size as
    the Goal reference scales.  No noise, goal identity or visibility flag is
    read.
    """
    background = 34
    image = Image.new("RGB", (canvas_size, canvas_size),
                      color=(background,) * 3)
    draw = ImageDraw.Draw(image)
    cx = cy = (canvas_size - 1) / 2.0
    half = max(float(object_width_px) / 2.0, 1.0)
    env._draw_shape(draw, landmark.shape, cx, cy, half, landmark.base_color)
    env._draw_marker(draw, landmark.marker_type, cx, cy, half,
                     landmark.marker_color, landmark.base_color)
    return np.asarray(image, dtype=np.uint8)


@torch.no_grad()
def encode_landmark_canvases(canvases, encoder, device):
    """Frozen-encoder embeddings for a batch of canvases.

    Mirrors ``encode_multiscale_goal_tokens``: resize to the encoder input,
    then foreground-weighted token pooling.  Identical for every node.
    """
    arr = np.asarray(canvases, dtype=np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"canvases must be [N,H,W,3], got {arr.shape}")
    raw = torch.from_numpy(arr).float() / 255.0
    x = raw.permute(0, 3, 1, 2)
    if x.shape[-2:] != (ENCODER.image_size, ENCODER.image_size):
        x = F.interpolate(
            x, size=(ENCODER.image_size, ENCODER.image_size),
            mode="bilinear", align_corners=False,
        )
    tokens = encoder(x.to(device))

    background = raw[:, 0:1, 0:1, :]
    diff = (raw - background).abs().amax(dim=-1)
    fg = (diff > (8.0 / 255.0)).float().unsqueeze(1)
    fg = F.interpolate(
        fg, size=(ENCODER.grid_size, ENCODER.grid_size), mode="area"
    ).squeeze(1)
    weights = fg.reshape(arr.shape[0], -1).to(device)
    denom = weights.sum(dim=1, keepdim=True)
    empty = denom.squeeze(1) < 1e-6
    if bool(empty.any()):
        weights[empty] = 0.0
        center = (ENCODER.grid_size // 2) * ENCODER.grid_size + \
            ENCODER.grid_size // 2
        weights[empty, center] = 1.0
        denom = weights.sum(dim=1, keepdim=True)
    embeddings = (tokens * weights.unsqueeze(-1)).sum(dim=1) / denom
    return embeddings.detach().cpu()


def build_landmark_map(env, seed, encoder, device):
    """Build the anonymous map and frozen-encoder node embeddings."""
    nodes, goal_index = build_anonymous_landmark_nodes(env, seed)
    canvases = [render_landmark_canvas(env, node) for node in nodes]
    embeddings = encode_landmark_canvases(canvases, encoder, device)
    positions = np.asarray([node.position for node in nodes], dtype=np.float32)
    return {
        "positions": positions,
        "embeddings": embeddings.float().numpy(),
        "goal_index": int(goal_index),  # diagnostics only
        "num_nodes": len(nodes),
    }


def retrieval_similarities(node_embeddings, goal_embedding):
    node = np.asarray(node_embeddings, dtype=np.float64)
    goal = np.asarray(goal_embedding, dtype=np.float64).reshape(-1)
    node_norm = node / np.maximum(np.linalg.norm(node, axis=1, keepdims=True), 1e-12)
    goal_norm = goal / max(np.linalg.norm(goal), 1e-12)
    return node_norm @ goal_norm


def map_retrieval_features(pose, node_positions, node_embeddings, goal_embedding,
                           top_k=MAP_TOP_K, scale=MAP_POSITION_SCALE_M):
    """Top-k anonymous map retrieval features.

    Only visual similarity to the Goal RGB reference selects nodes.  No node
    ID, is_goal, is_distractor or env.goal is read.
    """
    sims = retrieval_similarities(node_embeddings, goal_embedding)
    k = min(int(top_k), len(sims))
    order = np.argsort(-sims)[:k]
    x, y, yaw = map(float, pose)
    features = []
    for index in order:
        nx, ny = float(node_positions[index, 0]), float(node_positions[index, 1])
        dx = nx - x
        dy = ny - y
        forward = (dx * math.cos(yaw) + dy * math.sin(yaw)) / float(scale)
        right = (-dx * math.sin(yaw) + dy * math.cos(yaw)) / float(scale)
        distance = math.hypot(dx, dy) / float(scale)
        features.extend([float(sims[index]), forward, right, distance])
    top1 = float(sims[order[0]])
    margin = float(sims[order[0]] - sims[order[1]]) if k > 1 else 0.0
    features.extend([top1, margin])
    return np.asarray(features, dtype=np.float32)


def retrieval_diagnostics(goal_index, node_embeddings, goal_embedding):
    """Post-hoc Top-1/Top-5 diagnostics (goal identity used only here)."""
    sims = retrieval_similarities(node_embeddings, goal_embedding)
    order = np.argsort(-sims)
    rank = int(np.where(order == int(goal_index))[0][0]) + 1
    return {
        "top1": int(order[0] == int(goal_index)),
        "top5": int(rank <= 5),
        "goal_rank": rank,
        "top1_index": int(order[0]),
        "top1_similarity": float(sims[order[0]]),
        "goal_similarity": float(sims[int(goal_index)]),
        "margin": float(sims[order[0]] - sims[order[1]]),
    }


# ---------------------------------------------------------------------------
# Shard / auxiliary loading
# ---------------------------------------------------------------------------


ORIGINAL_FIELDS = (
    "tokens", "goal_tokens", "spatial_heatmaps", "network_states",
    "labels", "arrival_labels", "move_valid", "arrival_train_mask",
    "distance_to_goal_m", "sample_type",
)


def sorted_shards(directory):
    paths = sorted(Path(directory).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt shards in {directory}")
    return paths


def load_shard_bundle(path, aux_path=None, aux_fields=()):
    """Load one shard and (optionally) its aligned Stage-2i auxiliary arrays."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    missing = [key for key in ORIGINAL_FIELDS if key not in data]
    if missing:
        raise KeyError(f"{path} missing fields {missing}")
    bundle = {key: data[key] for key in ORIGINAL_FIELDS}
    bundle["num_samples"] = int(bundle["labels"].shape[0])
    for key in ("episode_ids", "step_ids", "episode_seeds",
                "trajectory_band_ids"):
        if key in data:
            bundle[key] = data[key]
    if aux_fields:
        if aux_path is None or not Path(aux_path).is_file():
            raise FileNotFoundError(f"Missing auxiliary features {aux_path}")
        aux = np.load(aux_path)
        for key in aux_fields:
            if key not in aux:
                raise KeyError(f"{aux_path} missing auxiliary field {key}")
            bundle[key] = torch.from_numpy(np.asarray(aux[key]))
    return bundle


def soft_class_weights(counts, gamma=0.5):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    freq = counts / counts.sum().clamp_min(1.0)
    weights = (freq.clamp_min(1e-6)).pow(-float(gamma))
    return (weights / weights.mean().clamp_min(1e-6)).float()


def movement_class_counts(shard_paths):
    counts = torch.zeros(3, dtype=torch.float64)
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        labels = data["labels"].long()
        valid = data["move_valid"].bool()
        for index in range(3):
            counts[index] += ((labels == index) & valid).sum().double()
    return counts
