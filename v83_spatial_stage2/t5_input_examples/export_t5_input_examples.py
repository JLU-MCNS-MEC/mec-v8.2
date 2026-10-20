"""Export three real Stage-2j T5 input examples from Train 600.

Scope
-----
Read-only export of existing archived inputs.  No training, no closed-loop
evaluation, no change to the MEC, checkpoints or feature caches.  Only
``runs/v83_spatial_stage2/datasets/train_600`` and the aligned Stage-2i
``aux/train`` sidecars are read; Final Test 300 is rejected.

Selection
---------
The three cases are selected by fixed, pre-declared rules over **observable**
features only (current Goal-Current match confidence, history validity, GAB
validity/confidence, anonymous map Top-5 similarity spread).  Navigation
success, true distance, goal coordinates and oracle actions are never used to
choose a sample.  See ``SELECTION_RULES`` below and the emitted
``selection_rules.json``.

Outputs
-------
One directory per case with the eight model-input arrays as a compressed NPZ,
metadata, input summary, offline-only diagnostics, and six visualisations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

from config import ENCODER, TRAIN  # noqa: E402
from env import STOP, ToyUAVEnv  # noqa: E402
from gbs_model_v3 import goal_current_similarity  # noqa: E402
from utils import image_to_tensor  # noqa: E402
from v83_spatial_stage2.stage2h_gab import (  # noqa: E402
    current_evidence,
    normalized_entropy,
)
from v83_spatial_stage2.stage2i_data import (  # noqa: E402
    GAB_FEATURE_DIM,
    HISTORY_STEPS,
    MAP_FEATURE_DIM,
    MAP_POSITION_SCALE_M,
    MAP_TOP_K,
    POSE_FEATURE_DIM,
    VISUAL_DESCRIPTOR_DIM,
    build_gab_for_episode,
    build_landmark_map,
    gab_config_from_threshold,
    history_arrays,
    map_retrieval_features,
    retrieval_diagnostics,
    retrieval_similarities,
    sorted_shards,
    visual_descriptor,
)
from v83_spatial_training.eval_closedloop_spatial import load_encoder  # noqa: E402


DEFAULT_TRAIN_DIR = Path("runs/v83_spatial_stage2/datasets/train_600")
DEFAULT_AUX_DIR = Path("runs/v83_spatial_stage2/stage2i/aux/train")
DEFAULT_AUX_MANIFEST = Path("runs/v83_spatial_stage2/stage2i/aux/manifest.json")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

CASE_DEFINITIONS = {
    "case_01_clear_match": {
        "title": "High current Goal-Current match confidence",
        "description": (
            "Current local match confidence is high; a full 16-step history "
            "is available."
        ),
    },
    "case_02_weak_match_history": {
        "title": "Weak current match with valid history and GAB belief",
        "description": (
            "Current local match confidence is low, but the 16-step visual "
            "history is valid and the GAB has valid, reliable-branch evidence."
        ),
    },
    "case_03_ambiguous_map": {
        "title": "Ambiguous anonymous map Top-5 candidates",
        "description": (
            "The five retrieved anonymous landmark candidates have nearly "
            "identical Goal-RGB cosine similarity."
        ),
    },
}

CASE_ORDER = (
    "case_01_clear_match",
    "case_02_weak_match_history",
    "case_03_ambiguous_map",
)

# The single Stage-2j input contract.  Keep in the exact order used by the
# model and training sidecars.
EXPECTED_ARRAYS = {
    "current_tokens": {
        "shape": (ENCODER.num_tokens, ENCODER.token_dim),
        "dtype": np.float16,
        "source": "archived Train-600 token shard",
        "definition": (
            "Frozen shared-encoder tokens of the current 336x336 RGB "
            "observation, 24x24 patch grid, row-major."
        ),
    },
    "goal_tokens": {
        "shape": (3, ENCODER.token_dim),
        "dtype": np.float16,
        "source": "archived Train-600 token shard",
        "definition": (
            "Foreground-weighted frozen-encoder tokens for the 14/28/56-px "
            "Goal reference canvases."
        ),
    },
    "history_vis": {
        "shape": (HISTORY_STEPS, VISUAL_DESCRIPTOR_DIM),
        "dtype": np.float16,
        "source": "Stage-2i aux/train sidecar",
        "definition": (
            "Causal past visual descriptors: mean+max pooled Current tokens, "
            "slot 0 = t-1."
        ),
    },
    "history_pose": {
        "shape": (HISTORY_STEPS, POSE_FEATURE_DIM),
        "dtype": np.float32,
        "source": "Stage-2i aux/train sidecar",
        "definition": (
            "Causal relative motion/pose history (6 dims) plus the executed "
            "action one-hot (4 dims)."
        ),
    },
    "history_action": {
        "shape": (HISTORY_STEPS, 4),
        "dtype": np.float16,
        "source": "Stage-2i aux/train sidecar",
        "definition": "Executed-action one-hot history; slot 0 = t-1.",
    },
    "history_mask": {
        "shape": (HISTORY_STEPS,),
        "dtype": np.uint8,
        "source": "Stage-2i aux/train sidecar",
        "definition": "Causal validity mask: 1 for real past slots, 0 for padding.",
    },
    "gab": {
        "shape": (GAB_FEATURE_DIM,),
        "dtype": np.float32,
        "source": "Stage-2i aux/train sidecar",
        "definition": (
            "20-D Goal-Anchor Belief: 8 current bearing bins, 8 last-reliable "
            "bins, confidence, entropy, age/max_age, valid."
        ),
    },
    "map_feat": {
        "shape": (MAP_FEATURE_DIM,),
        "dtype": np.float32,
        "source": "Stage-2i aux/train sidecar",
        "definition": (
            "Anonymous landmark Top-5 retrieval: 5 x (similarity, relative "
            "forward, relative right, normalized distance), then top-1 "
            "similarity and top-1 minus top-2 margin."
        ),
    },
}

SELECTION_RULES = {
    "pool": (
        "All natural (sample_type=0) samples of Train 600; hard-positive, "
        "boundary-negative and distractor-negative samples are excluded."
    ),
    "common_eligibility": "history_valid_slots >= 8",
    "case_01_clear_match": (
        "Maximize the current Goal-Current match confidence "
        "(current_evidence confidence of the archived 3x8x8 heatmap); "
        "ties by smallest episode_id then step_id."
    ),
    "case_02_weak_match_history": (
        "Distinct episode; GAB valid=1 and GAB confidence >= the Train-fitted "
        "reliability threshold; minimize current match confidence; ties by "
        "smallest episode_id then step_id."
    ),
    "case_03_ambiguous_map": (
        "Distinct episode; top-1 retrieval similarity >= the pool median; "
        "minimize round(top-1 minus top-5 similarity, 6 decimals); ties by "
        "larger top-1, then smallest episode_id then step_id."
    ),
}

# Static plotting tokens (light surface).  The two-series palette was validated
# (blue/orange: all checks pass); the heatmap uses one sequential blue hue.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#9b9a95"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "mec_blue",
    ["#f7fbff", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
)

plt.rcParams.update({
    "font.size": 9.5,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.6,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_project(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def reject_final_test(*paths):
    for path in paths:
        if "final_test" in str(path):
            raise ValueError(f"Final Test 300 is sealed; refusing to read {path}")


def current_confidence_from_heatmaps(heatmaps):
    """Vectorised ``current_evidence`` confidence for [N,3,8,8] heatmaps."""
    heat = np.asarray(heatmaps, dtype=np.float64)
    if heat.ndim != 4 or heat.shape[1] != 3:
        raise ValueError(f"heatmaps must be [N,3,8,8], got {heat.shape}")
    flat = heat[:, :3].max(axis=1).reshape(len(heat), -1)
    logits = flat / 0.10
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    return probabilities.max(axis=1)


def one_hot_check(action_rows):
    rows = np.asarray(action_rows, dtype=np.float32)
    return bool(np.all(rows.sum(axis=1) == 1.0)) and bool(
        np.all((rows == 0.0) | (rows == 1.0))
    )


# ---------------------------------------------------------------------------
# Train scan and deterministic case selection
# ---------------------------------------------------------------------------


def scan_train(train_dir, aux_dir):
    """Read Train-600 metadata and the aligned Stage-2i sidecars.

    Only natural samples enter the selection pool.  Returns the pool index and
    an ``episode -> [(shard, row, step)]`` map for exact episode replay.
    """
    train_dir = Path(train_dir)
    aux_dir = Path(aux_dir)
    reject_final_test(train_dir, aux_dir)
    shard_paths = sorted_shards(train_dir)
    aux_paths = sorted(aux_dir.glob("aux_shard_*.npz"))
    if len(shard_paths) != len(aux_paths):
        raise RuntimeError(
            f"shard/sidecar mismatch: {len(shard_paths)} vs {len(aux_paths)}"
        )

    rows = {key: [] for key in (
        "shard", "row", "episode", "step", "band", "seed",
        "current_confidence", "history_valid_slots", "gab_confidence",
        "gab_valid", "gab_entropy", "top1", "top2", "top5",
    )}
    episode_rows = {}
    sample_type_counts = {}
    started = time.time()

    for shard_index, (shard_path, aux_path) in enumerate(
        zip(shard_paths, aux_paths)
    ):
        data = torch.load(shard_path, map_location="cpu", weights_only=True)
        aux = np.load(aux_path)
        n = int(data["labels"].shape[0])
        if aux["history_mask"].shape[0] != n:
            raise RuntimeError(f"{aux_path} is not aligned with {shard_path}")

        sample_type = data["sample_type"].cpu().numpy()
        for value in np.unique(sample_type):
            sample_type_counts[int(value)] = sample_type_counts.get(
                int(value), 0
            ) + int((sample_type == value).sum())

        confidence = current_confidence_from_heatmaps(
            data["spatial_heatmaps"].cpu().numpy()
        )
        episode = data["episode_ids"].cpu().numpy()
        step = data["step_ids"].cpu().numpy()
        band = data["trajectory_band_ids"].cpu().numpy()
        seed = data["episode_seeds"].cpu().numpy()
        history_mask = aux["history_mask"].sum(axis=1).astype(np.int32)
        gab = aux["gab"]
        map_feat = aux["map_feat"]
        top1 = map_feat[:, 0].astype(np.float64)
        top2 = map_feat[:, 4].astype(np.float64)
        top5 = map_feat[:, 16].astype(np.float64)

        natural = np.where(sample_type == 0)[0]
        for row in natural:
            episode_rows.setdefault(int(episode[row]), []).append(
                (int(shard_index), int(row), int(step[row]))
            )

        rows["shard"].append(np.full(len(natural), shard_index, np.int32))
        rows["row"].append(natural.astype(np.int32))
        rows["episode"].append(episode[natural].astype(np.int32))
        rows["step"].append(step[natural].astype(np.int32))
        rows["band"].append(band[natural].astype(np.int8))
        rows["seed"].append(seed[natural].astype(np.int64))
        rows["current_confidence"].append(confidence[natural])
        rows["history_valid_slots"].append(history_mask[natural])
        rows["gab_confidence"].append(gab[natural, 16].astype(np.float64))
        rows["gab_valid"].append(gab[natural, 19].astype(np.int8))
        rows["gab_entropy"].append(gab[natural, 17].astype(np.float64))
        rows["top1"].append(top1[natural])
        rows["top2"].append(top2[natural])
        rows["top5"].append(top5[natural])

    index = {key: np.concatenate(value) for key, value in rows.items()}
    index["top5_spread"] = index["top1"] - index["top5"]
    index["top5_margin"] = index["top1"] - index["top2"]

    # Every episode must be a contiguous 0..T-1 natural sequence with one
    # consistent seed and band.  This is what makes replay exact.
    for episode, items in episode_rows.items():
        items.sort(key=lambda item: item[2])
        steps = [item[2] for item in items]
        if steps != list(range(len(steps))):
            raise RuntimeError(
                f"Episode {episode} natural steps are not 0..T-1: {steps[:8]}..."
            )
        positions = np.where(index["episode"] == episode)[0]
        seeds = np.unique(index["seed"][positions])
        bands = np.unique(index["band"][positions])
        if len(seeds) != 1 or len(bands) != 1:
            raise RuntimeError(f"Episode {episode} has mixed seed/band")

    elapsed = time.time() - started
    print(
        f"[scan] {len(index['episode'])} natural samples, "
        f"{len(episode_rows)} episodes, {elapsed:.1f}s",
        flush=True,
    )
    return {
        "index": index,
        "episode_rows": episode_rows,
        "shard_paths": shard_paths,
        "aux_paths": aux_paths,
        "sample_type_counts": sample_type_counts,
        "elapsed_s": elapsed,
    }


def _lexsort_index(mask, keys):
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        raise RuntimeError("Empty selection candidate pool")
    order = np.lexsort(tuple(key[candidates] for key in keys))
    return int(candidates[order[0]]), int(len(candidates))


def select_cases(index, gab_threshold):
    """Apply the pre-declared observable rules exactly once."""
    history_ok = index["history_valid_slots"] >= 8

    i1, pool1 = _lexsort_index(
        history_ok,
        (index["step"], index["episode"], -index["current_confidence"]),
    )
    episode1 = int(index["episode"][i1])

    mask2 = (
        history_ok
        & (index["gab_valid"] == 1)
        & (index["gab_confidence"] >= float(gab_threshold))
        & (index["episode"] != episode1)
    )
    i2, pool2 = _lexsort_index(
        mask2,
        (index["step"], index["episode"], index["current_confidence"]),
    )
    episode2 = int(index["episode"][i2])

    top1_median = float(np.median(index["top1"]))
    spread_rounded = np.round(index["top5_spread"], 6)
    mask3 = (
        history_ok
        & (index["top1"] >= top1_median)
        & (index["episode"] != episode1)
        & (index["episode"] != episode2)
    )
    i3, pool3 = _lexsort_index(
        mask3,
        (index["step"], index["episode"], -index["top1"], spread_rounded),
    )

    selected = [
        ("case_01_clear_match", i1, pool1),
        ("case_02_weak_match_history", i2, pool2),
        ("case_03_ambiguous_map", i3, pool3),
    ]
    episodes = [int(index["episode"][item[1]]) for item in selected]
    if len(set(episodes)) != len(episodes):
        raise RuntimeError("Selection rules did not yield distinct episodes")

    records = []
    confidence = index["current_confidence"]
    spread = index["top5_spread"]
    for case, position, pool in selected:
        records.append({
            "case": case,
            "row_index": int(position),
            "candidate_pool": int(pool),
            "episode_id": int(index["episode"][position]),
            "step_id": int(index["step"][position]),
            "band_id": int(index["band"][position]),
            "episode_seed": int(index["seed"][position]),
            "shard": int(index["shard"][position]),
            "shard_row": int(index["row"][position]),
            "current_confidence": float(confidence[position]),
            "current_confidence_percentile": float(
                100.0 * np.mean(confidence <= confidence[position])
            ),
            "history_valid_slots": int(index["history_valid_slots"][position]),
            "gab_confidence": float(index["gab_confidence"][position]),
            "gab_entropy": float(index["gab_entropy"][position]),
            "gab_valid": int(index["gab_valid"][position]),
            "map_top1": float(index["top1"][position]),
            "map_top2": float(index["top2"][position]),
            "map_top5": float(index["top5"][position]),
            "map_top5_spread": float(spread[position]),
            "map_top5_spread_percentile": float(
                100.0 * np.mean(spread <= spread[position])
            ),
            "map_top5_margin": float(index["top5_margin"][position]),
        })
    return {
        "selected": records,
        "top1_median": top1_median,
        "gab_reliability_threshold": float(gab_threshold),
    }


# ---------------------------------------------------------------------------
# Episode reconstruction and exact feature reproduction
# ---------------------------------------------------------------------------


def load_episode_sequence(episode, episode_rows, shard_paths, shard_cache):
    """Ordered actions, heatmaps and source rows for one episode."""
    items = sorted(episode_rows[episode], key=lambda item: item[2])
    actions = []
    heatmaps = []
    for shard_index, row, _step in items:
        if shard_index not in shard_cache:
            shard_cache[shard_index] = torch.load(
                shard_paths[shard_index], map_location="cpu", weights_only=True
            )
        data = shard_cache[shard_index]
        actions.append(int(data["labels"][row]))
        heatmaps.append(data["spatial_heatmaps"][row].cpu().numpy())
    return {
        "items": items,
        "steps": np.asarray([item[2] for item in items], dtype=np.int64),
        "actions": np.asarray(actions, dtype=np.int64),
        "heatmaps": np.stack(heatmaps).astype(np.float16),
    }


def replay_episode(seed, band_id, actions, step):
    """Replay the archived expert actions to the selected state.

    Rendering is enabled exactly once per state, matching dataset generation,
    so the reconstructed observation is the same scene as the archived one.
    Oracle agreement is recorded as an offline replay check only.
    """
    band = TRAIN.trajectory_mix[int(band_id)]
    env = ToyUAVEnv()
    observation = env.reset(
        seed=int(seed),
        goal_distance_range_m=(float(band[2]), float(band[3])),
        render_observation=True,
    )
    mismatches = 0
    for index in range(int(step)):
        oracle = int(env.oracle_action())
        stored = int(actions[index])
        if oracle != stored:
            mismatches += 1
            raise RuntimeError(
                f"Replay oracle mismatch at step {index}: {oracle} != {stored}"
            )
        observation, _, _ = env.step(
            stored, render_observation=(stored != STOP)
        )
    if env.step_count != int(step):
        raise RuntimeError(
            f"Replay length {env.step_count} != requested step {step}"
        )
    return env, observation, mismatches


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def save_current_rgb(observation, path):
    image = Image.fromarray(np.asarray(observation, dtype=np.uint8), mode="RGB")
    image.save(path)


def save_goal_reference_scales(goal_rgbs, path, case_title):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), constrained_layout=True)
    pixels = (14, 28, 56)
    for axis, image, pixels_ in zip(axes, goal_rgbs, pixels):
        axis.imshow(image)
        axis.set_title(f"{pixels_} px object width", color=INK, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f"Goal RGB reference scales — {case_title}",
        color=INK, fontsize=11,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_local_similarity(
    current_tokens, goal_tokens, path, case_title, gab_confidence
):
    current = torch.from_numpy(np.asarray(current_tokens, dtype=np.float32)).unsqueeze(0)
    goal = torch.from_numpy(np.asarray(goal_tokens, dtype=np.float32)).unsqueeze(0)
    similarity = goal_current_similarity(current, goal)[0].cpu().numpy()
    grid_size = ENCODER.grid_size
    grids = similarity.reshape(3, grid_size, grid_size)

    figure, axes = plt.subplots(
        1, 3, figsize=(10.4, 3.6), constrained_layout=True
    )
    # Per-case range keeps the within-case structure visible; the colorbar
    # always reports the absolute cosine values.
    low = float(math.floor(grids.min() * 20.0) / 20.0)
    high = float(math.ceil(grids.max() * 20.0) / 20.0)
    if high - low < 1e-6:
        high = low + 1e-6
    image = None
    for axis, grid, pixels_ in zip(axes, grids, (14, 28, 56)):
        image = axis.imshow(
            grid, cmap=SEQ_CMAP, vmin=low, vmax=high, origin="upper"
        )
        peak_row, peak_col = np.unravel_index(int(np.argmax(grid)), grid.shape)
        axis.scatter(
            [peak_col], [peak_row], s=34, facecolors="none",
            edgecolors=INK, linewidths=1.0,
        )
        axis.set_title(
            f"Goal scale {pixels_} px\nmax cosine {grid.max():.3f} at "
            f"(row {peak_row}, col {peak_col})",
            color=INK, fontsize=9,
        )
        ticks = [0, 6, 12, 18, 24]
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.tick_params(labelsize=8)
        axis.set_xlabel("current patch column", fontsize=8)
        axis.set_ylabel("current patch row", fontsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("cosine similarity", color=MUTED)
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        f"Current 576 tokens vs 3 Goal tokens — local cosine similarity "
        f"(24×24) — {case_title}",
        color=INK, fontsize=11,
    )
    figure.text(
        0.5, -0.02,
        "Offline visualisation of an observable intermediate. T5 receives the "
        "20-D GAB, not this 3×24×24 map.  "
        f"GAB confidence (softmax peakiness) = {gab_confidence:.4f}.",
        ha="center", color=MUTED, fontsize=8,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return grids


def save_gab_belief(features, evidence, path, case_title):
    belief = np.asarray(features[:8], dtype=np.float64)
    last_reliable = np.asarray(features[8:16], dtype=np.float64)
    confidence = float(features[16])
    entropy = float(features[17])
    age = float(features[18])
    valid = float(features[19])

    figure, axis = plt.subplots(figsize=(8.8, 4.1), constrained_layout=True)
    positions = np.arange(8)
    width = 0.38
    axis.bar(
        positions - width / 2, belief, width=width, color=BLUE,
        label="current belief", edgecolor="none",
    )
    axis.bar(
        positions + width / 2, last_reliable, width=width, color=ORANGE,
        label="last reliable belief", edgecolor="none",
    )
    axis.axhline(
        1.0 / 8.0, color=GRAY, linewidth=0.8, label="uniform 0.125"
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(index) for index in positions])
    axis.set_xlabel("egocentric bearing bin (left → right)")
    axis.set_ylabel("probability")
    axis.set_ylim(0.0, max(0.55, float(max(belief.max(), last_reliable.max())) * 1.25))
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.legend(loc="upper right", fontsize=8.5)

    peak = int(np.argmax(belief))
    last_peak = int(np.argmax(last_reliable))
    summary = (
        f"confidence      {confidence:.4f}\n"
        f"entropy         {entropy:.4f}\n"
        f"last-seen age   {age * 16:.1f} / 16 steps\n"
        f"valid           {int(valid)}\n"
        f"current peak    bin {peak} ({belief[peak]:.3f})\n"
        f"last reliable   bin {last_peak} ({last_reliable[last_peak]:.3f})"
    )
    axis.text(
        0.015, 0.975, summary, transform=axis.transAxes, va="top", ha="left",
        fontsize=8.5, family="monospace", color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=SURFACE,
                  edgecolor=GRID, linewidth=0.8),
    )
    axis.set_title(
        f"GAB 20-D belief input — {case_title}",
        color=INK, fontsize=11,
    )
    figure.text(
        0.5, -0.01,
        "T5 receives these 20 values; the 8-bin bars are not a separate "
        "model branch.",
        ha="center", color=MUTED, fontsize=8,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "belief": belief,
        "last_reliable": last_reliable,
        "confidence": confidence,
        "entropy": entropy,
        "age": age,
        "valid": valid,
    }


def save_map_top5(
    pose, positions, order, similarities, path, case_title
):
    x, y, yaw = map(float, pose)
    dx = positions[:, 0] - x
    dy = positions[:, 1] - y
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    distance = np.hypot(dx, dy)

    candidates = list(order)
    candidate_distance = distance[candidates]
    limit = max(120.0, float(candidate_distance.max()) * 1.18)

    figure = plt.figure(figsize=(10.8, 6.4), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(3.0, 1.25))
    axis = figure.add_subplot(grid[0, 0])
    table_axis = figure.add_subplot(grid[0, 1])
    table_axis.axis("off")

    axis.scatter(
        right, forward, s=8, color=GRAY, alpha=0.35,
        edgecolors="none", label="anonymous map node",
    )
    axis.scatter(
        right[candidates], forward[candidates], s=46, color=BLUE,
        edgecolors=SURFACE, linewidths=1.2, zorder=3,
        label="Top-5 retrieved candidate",
    )
    for rank, index in enumerate(candidates, start=1):
        norm = max(float(distance[index]), 1e-6)
        unit_x = float(right[index]) / norm
        unit_y = float(forward[index]) / norm
        axis.annotate(
            f"T{rank}", (right[index], forward[index]),
            textcoords="offset points",
            xytext=(unit_x * 11.0, unit_y * 11.0),
            fontsize=8, color=INK, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=SURFACE,
                      edgecolor=GRID, linewidth=0.6),
        )

    axis.scatter(
        [0.0], [0.0], marker="^", s=120, color=INK, zorder=5,
        label="UAV (forward = up)",
    )
    axis.annotate(
        "", xy=(0.0, limit * 0.10), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.2),
        zorder=4,
    )
    for radius in range(100, int(limit // 100) * 100 + 1, 100):
        axis.add_patch(
            Circle((0.0, 0.0), radius, fill=False, color=GRID,
                   linewidth=0.7, zorder=1)
        )

    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("relative right (m)")
    axis.set_ylabel("relative forward (m)")
    axis.axhline(0.0, color=GRID, linewidth=0.6, zorder=0)
    axis.axvline(0.0, color=GRID, linewidth=0.6, zorder=0)
    axis.grid(color=GRID, linewidth=0.5, alpha=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5,
               markerfacecolor=GRAY, markeredgecolor="none",
               label="anonymous map node"),
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=BLUE, markeredgecolor=SURFACE,
               label="Top-5 retrieved candidate"),
        Line2D([0], [0], marker="^", linestyle="", markersize=8,
               markerfacecolor=INK, markeredgecolor="none",
               label="UAV (forward = up)"),
    ]
    axis.legend(handles=legend_handles, loc="upper left", fontsize=8)
    axis.set_title("UAV-centric view (rings = 100 m)", color=MUTED, fontsize=9)

    table_lines = ["rank  sim      dist    bearing"]
    for rank, index in enumerate(candidates, start=1):
        bearing = math.degrees(math.atan2(right[index], forward[index]))
        table_lines.append(
            f"T{rank}   {similarities[index]:.6f} "
            f"{distance[index]:5.0f} m {bearing:+5.0f}°"
        )
    table_axis.text(
        0.0, 0.98, "Top-5 anonymous candidates\n\n" + "\n".join(table_lines),
        transform=table_axis.transAxes, va="top", ha="left",
        fontsize=9, family="monospace", color=INK,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=SURFACE,
                  edgecolor=GRID, linewidth=0.8),
    )
    table_axis.text(
        0.0, 0.42,
        "bearing 0° = forward\npositive = right\n\n"
        "CSV stores relative_forward,\nrelative_right and normalized\n"
        "distance (divide metres by 500).",
        transform=table_axis.transAxes, va="top", ha="left",
        fontsize=8, color=MUTED,
    )
    figure.suptitle(
        f"Anonymous landmark Top-5 retrieval — {case_title}",
        color=INK, fontsize=11,
    )
    figure.text(
        0.5, -0.01,
        "Top-5 selected by Goal-RGB cosine only; no goal_index, is_goal or "
        "env.goal is used.",
        ha="center", color=MUTED, fontsize=8,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_top5_csv(path, similarities, order, pose, positions):
    x, y, yaw = map(float, pose)
    rows = []
    for rank, index in enumerate(order, start=1):
        dx = float(positions[index, 0]) - x
        dy = float(positions[index, 1]) - y
        forward = (dx * math.cos(yaw) + dy * math.sin(yaw))
        right = (-dx * math.sin(yaw) + dy * math.cos(yaw))
        distance = math.hypot(dx, dy)
        rows.append({
            "rank": rank,
            "similarity": f"{float(similarities[index]):.9f}",
            "relative_forward": f"{forward / MAP_POSITION_SCALE_M:.9f}",
            "relative_right": f"{right / MAP_POSITION_SCALE_M:.9f}",
            "normalized_distance": f"{distance / MAP_POSITION_SCALE_M:.9f}",
        })
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank", "similarity", "relative_forward",
                "relative_right", "normalized_distance",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Statistics and per-case export
# ---------------------------------------------------------------------------


def array_statistics(array):
    array = np.asarray(array)
    # float16 accumulation over 221k token values can overflow; compute the
    # summary in float64 while reporting the stored dtype separately.
    values = array.astype(np.float64)
    finite = np.isfinite(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(finite.all()),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "nonzero_fraction": float(np.mean(values != 0)),
        "l2_norm": float(np.linalg.norm(values)),
    }


def build_input_summary(arrays, case_info):
    current = arrays["current_tokens"].astype(np.float32)
    goal = arrays["goal_tokens"].astype(np.float32)
    current_norm = np.linalg.norm(current, axis=-1)
    goal_norm = np.linalg.norm(goal, axis=-1)
    similarity = goal_current_similarity(
        torch.from_numpy(current).unsqueeze(0),
        torch.from_numpy(goal).unsqueeze(0),
    )[0].cpu().numpy()
    grid = similarity.reshape(3, ENCODER.grid_size, ENCODER.grid_size)
    per_scale_max = grid.reshape(3, -1).max(axis=1)
    per_scale_mean = grid.reshape(3, -1).mean(axis=1)
    peaks = [
        tuple(int(v) for v in np.unravel_index(int(np.argmax(grid[index])),
                                               grid[index].shape))
        for index in range(3)
    ]

    history_mask = arrays["history_mask"]
    history_action = arrays["history_action"].astype(np.float32)
    history_pose = arrays["history_pose"]
    valid_slots = int(history_mask.sum())
    last_action = -1
    if valid_slots:
        last_action = int(np.argmax(history_action[0]))

    gab = arrays["gab"]
    belief = gab[:8].astype(np.float64)
    last_reliable = gab[8:16].astype(np.float64)
    map_feat = arrays["map_feat"].astype(np.float64)
    top_similarities = map_feat[[0, 4, 8, 12, 16]]

    return {
        "case": case_info["case"],
        "episode_id": case_info["episode_id"],
        "step_id": case_info["step_id"],
        "trajectory_band": case_info["trajectory_band"],
        "arrays": {
            name: array_statistics(array) for name, array in arrays.items()
        },
        "validity": {
            "all_finite": bool(all(np.isfinite(array).all()
                                   for array in arrays.values())),
            "history_mask_pattern_ok": bool(
                np.array_equal(
                    history_mask,
                    (np.arange(HISTORY_STEPS) < valid_slots).astype(np.uint8),
                )
            ),
            "history_action_one_hot_under_mask": bool(
                one_hot_check(history_action[history_mask == 1])
            ) and bool(np.all(history_action[history_mask == 0] == 0)),
            "history_pose_action_block_equals_history_action": bool(
                np.allclose(history_pose[:, 6:10], history_action.astype(np.float32))
            ),
            "gab_belief_sums_to_one": bool(abs(belief.sum() - 1.0) < 1e-5),
            "gab_last_reliable_sums_to_one": bool(
                abs(last_reliable.sum() - 1.0) < 1e-5
            ),
            "gab_valid_in_0_1": bool(gab[19] in (0.0, 1.0)),
            "map_top5_similarity_sorted": bool(
                np.all(np.diff(top_similarities) <= 1e-9)
            ),
        },
        "current_goal_matching": {
            "current_token_norm_mean": float(current_norm.mean()),
            "current_token_norm_std": float(current_norm.std()),
            "goal_token_norm": [float(value) for value in goal_norm],
            "per_scale_max_cosine": [float(value) for value in per_scale_max],
            "per_scale_mean_cosine": [float(value) for value in per_scale_mean],
            "per_scale_peak_cell": [
                [int(row), int(col)] for row, col in peaks
            ],
            "global_max_cosine": float(per_scale_max.max()),
        },
        "history": {
            "valid_slots": valid_slots,
            "mask": [int(value) for value in history_mask],
            "most_recent_action": (
                ["Forward", "Left", "Right", "Stop"][last_action]
                if last_action >= 0 else None
            ),
            "history_pose_motion_range": [
                float(history_pose[history_mask == 1, 0:2].min()),
                float(history_pose[history_mask == 1, 0:2].max()),
            ] if valid_slots else [0.0, 0.0],
        },
        "gab": {
            "belief": [float(value) for value in belief],
            "last_reliable": [float(value) for value in last_reliable],
            "belief_peak_bin": int(np.argmax(belief)),
            "last_reliable_peak_bin": int(np.argmax(last_reliable)),
            "confidence": float(gab[16]),
            "entropy": float(gab[17]),
            "last_seen_age_normalized": float(gab[18]),
            "valid": float(gab[19]),
            "entropy_recomputed": normalized_entropy(belief),
        },
        "map": {
            "top5_similarity": [float(value) for value in top_similarities],
            "top1_similarity": float(map_feat[20]),
            "top1_minus_top2_margin": float(map_feat[21]),
            "top1_minus_top5_spread": float(
                top_similarities[0] - top_similarities[-1]
            ),
            "candidate_relative_forward": [
                float(value) for value in map_feat[[1, 5, 9, 13, 17]]
            ],
            "candidate_relative_right": [
                float(value) for value in map_feat[[2, 6, 10, 14, 18]]
            ],
            "candidate_normalized_distance": [
                float(value) for value in map_feat[[3, 7, 11, 15, 19]]
            ],
        },
    }


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def export_case(
    case_name,
    selection,
    scan,
    encoder,
    device,
    output_dir,
    aux_manifest,
):
    output_dir = Path(output_dir)
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    case_info = CASE_DEFINITIONS[case_name]
    index = scan["index"]
    position = selection["row_index"]
    shard_index = int(index["shard"][position])
    shard_row = int(index["row"][position])
    episode = int(index["episode"][position])
    step = int(index["step"][position])
    band_id = int(index["band"][position])
    seed = int(index["seed"][position])
    band_name = str(TRAIN.trajectory_mix[band_id][0])

    shard_path = scan["shard_paths"][shard_index]
    aux_path = scan["aux_paths"][shard_index]
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    aux = np.load(aux_path)

    arrays = {
        "current_tokens": shard["tokens"][shard_row].cpu().numpy().astype(np.float16),
        "goal_tokens": shard["goal_tokens"][shard_row].cpu().numpy().astype(np.float16),
        "history_vis": aux["history_vis"][shard_row].astype(np.float16),
        "history_pose": aux["history_pose"][shard_row].astype(np.float32),
        "history_action": aux["history_action"][shard_row].astype(np.float16),
        "history_mask": aux["history_mask"][shard_row].astype(np.uint8),
        "gab": aux["gab"][shard_row].astype(np.float32),
        "map_feat": aux["map_feat"][shard_row].astype(np.float32),
    }
    for name, spec in EXPECTED_ARRAYS.items():
        array = arrays[name]
        if tuple(array.shape) != tuple(spec["shape"]):
            raise RuntimeError(
                f"{case_name}.{name} shape {array.shape} != {spec['shape']}"
            )
        if array.dtype != spec["dtype"]:
            raise RuntimeError(
                f"{case_name}.{name} dtype {array.dtype} != {spec['dtype']}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"{case_name}.{name} contains non-finite values")

    npz_path = case_dir / "t5_inputs.npz"
    np.savez_compressed(npz_path, **arrays)
    npz_sha = sha256_file(npz_path)

    # Ordered episode sequence (used for GAB reproduction and history replay).
    shard_cache = {}
    sequence = load_episode_sequence(
        episode, scan["episode_rows"], scan["shard_paths"], shard_cache
    )
    if int(sequence["steps"][step]) != step:
        raise RuntimeError(f"{case_name}: step order mismatch")
    if sequence["steps"].tolist() != list(range(len(sequence["steps"]))):
        raise RuntimeError(f"{case_name}: non-contiguous episode steps")

    # GAB reproduction from archived heatmaps + executed actions.
    gab_config = gab_config_from_threshold(
        float(aux_manifest["gab_reliability_threshold"])
    )
    gab_features = build_gab_for_episode(
        sequence["heatmaps"],
        sequence["actions"],
        gab_config,
        step_ids=sequence["steps"],
    )
    if not np.allclose(gab_features[step], arrays["gab"], atol=1e-7, rtol=0):
        raise RuntimeError(f"{case_name}: GAB is not reproducible")
    evidence = current_evidence(sequence["heatmaps"][step])

    # Replay to the selected state and reconstruct the observation.
    env, observation, replay_mismatches = replay_episode(
        seed, band_id, sequence["actions"], step
    )
    if observation is None:
        raise RuntimeError(f"{case_name}: replay produced no observation")
    save_current_rgb(observation, case_dir / "current_rgb.png")

    goal_rgbs = env.get_goal_reference_multiscale_rgbs()
    save_goal_reference_scales(
        goal_rgbs, case_dir / "goal_reference_scales.png",
        case_info["title"],
    )
    similarity_grids = save_local_similarity(
        arrays["current_tokens"], arrays["goal_tokens"],
        case_dir / "local_similarity_24x24.png", case_info["title"],
        gab_confidence=float(arrays["gab"][16]),
    )
    save_gab_belief(
        gab_features[step], evidence,
        case_dir / "gab_bearing_belief.png", case_info["title"],
    )

    # Anonymous map: rebuild and reproduce the archived 22-D feature.
    goal_token = arrays["goal_tokens"][2].astype(np.float32)
    landmark_map = build_landmark_map(env, seed, encoder, device)
    reconstructed_map_feat = map_retrieval_features(
        env.pose,
        landmark_map["positions"],
        landmark_map["embeddings"],
        goal_token,
    )
    if not np.allclose(
        reconstructed_map_feat, arrays["map_feat"], atol=1e-5, rtol=1e-6
    ):
        raise RuntimeError(f"{case_name}: map feature is not reproducible")

    similarities = retrieval_similarities(
        landmark_map["embeddings"], goal_token
    )
    order = np.argsort(-similarities)[:MAP_TOP_K]
    save_map_top5(
        env.pose, landmark_map["positions"], order, similarities,
        case_dir / "landmark_top5_map.png", case_info["title"],
    )
    write_top5_csv(
        case_dir / "landmark_top5.csv", similarities, order,
        env.pose, landmark_map["positions"],
    )

    # Observation re-encoding check (offline only; not a model input).
    encoded = encoder(image_to_tensor(observation, device)).squeeze(0).cpu()
    archived = torch.from_numpy(arrays["current_tokens"].astype(np.float32))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            encoded.flatten(), archived.flatten(), dim=0
        )
    )
    max_abs = float((encoded - archived).abs().max())

    retrieval = retrieval_diagnostics(
        landmark_map["goal_index"], landmark_map["embeddings"], goal_token
    )
    projection = env.goal_projection()

    # Offline-only diagnostics: explicitly not model input.
    diagnostics = {
        "case": case_name,
        "offline_only": True,
        "not_model_input": True,
        "warning": (
            "Simulator ground truth / post-hoc diagnostics. Never concatenate "
            "or feed any value in this file to the MEC."
        ),
        "selection": {
            "rule": SELECTION_RULES[case_name],
            "current_confidence": selection["current_confidence"],
            "current_confidence_percentile": selection[
                "current_confidence_percentile"
            ],
            "history_valid_slots": selection["history_valid_slots"],
            "gab_confidence": selection["gab_confidence"],
            "gab_valid": selection["gab_valid"],
            "map_top5_spread": selection["map_top5_spread"],
            "map_top5_spread_percentile": selection[
                "map_top5_spread_percentile"
            ],
        },
        "episode": {
            "episode_id": episode,
            "step_id": step,
            "trajectory_band": band_name,
            "band_id": band_id,
            "episode_seed": seed,
            "episode_natural_steps": int(len(sequence["steps"])),
            "actions_executed_to_state": [
                int(value) for value in sequence["actions"][:step]
            ],
        },
        "replay_and_correctness": {
            "oracle_replay_mismatches": int(replay_mismatches),
            "pose_xy_yaw": [float(value) for value in env.pose],
            "true_distance_to_goal_m": float(env.distance_to_goal),
            "goal_visible_by_simulator_projection": bool(projection is not None),
            "goal_rank_in_retrieval": int(retrieval["goal_rank"]),
            "retrieval_top1_is_goal": int(retrieval["top1"]),
            "retrieval_top5_contains_goal": int(retrieval["top5"]),
            "goal_similarity": float(retrieval["goal_similarity"]),
            "retrieval_top1_similarity": float(retrieval["top1_similarity"]),
            "render_reencode_cosine_vs_archived": cosine,
            "render_reencode_max_abs_vs_archived": max_abs,
        },
        "current_evidence": {
            "bins": [float(value) for value in evidence["bins"]],
            "confidence": float(evidence["confidence"]),
            "entropy": float(evidence["entropy"]),
            "peak_x_column": int(evidence["peak_x"]),
            "peak_y_row": int(evidence["peak_y"]),
        },
        "local_similarity": {
            "per_scale_max": [
                float(grid.max()) for grid in similarity_grids
            ],
            "per_scale_peak_cell": [
                [int(value) for value in np.unravel_index(
                    int(np.argmax(grid)), grid.shape
                )]
                for grid in similarity_grids
            ],
        },
    }
    write_json(case_dir / "diagnostics.json", diagnostics)

    input_summary = build_input_summary(
        arrays,
        {
            "case": case_name,
            "episode_id": episode,
            "step_id": step,
            "trajectory_band": band_name,
        },
    )
    input_summary["source"] = {
        "train_dir": relative_to_project(Path(aux_manifest["_train_dir"])),
        "shard": relative_to_project(shard_path),
        "shard_row": shard_row,
        "aux_file": relative_to_project(aux_path),
        "aux_row": shard_row,
    }
    write_json(case_dir / "input_summary.json", input_summary)

    metadata = {
        "case": case_name,
        "title": case_info["title"],
        "description": case_info["description"],
        "selection_rule": SELECTION_RULES[case_name],
        "selection_values": {
            "current_confidence": selection["current_confidence"],
            "current_confidence_percentile": selection[
                "current_confidence_percentile"
            ],
            "history_valid_slots": selection["history_valid_slots"],
            "gab_confidence": selection["gab_confidence"],
            "gab_valid": selection["gab_valid"],
            "map_top1": selection["map_top1"],
            "map_top2": selection["map_top2"],
            "map_top5": selection["map_top5"],
            "map_top5_spread": selection["map_top5_spread"],
            "map_top5_margin": selection["map_top5_margin"],
        },
        "provenance": {
            "split": "train_600",
            "episode_id": episode,
            "step_id": step,
            "trajectory_band": band_name,
            "band_id": band_id,
            "episode_seed": seed,
            "shard": relative_to_project(shard_path),
            "shard_sha256": sha256_file(shard_path),
            "shard_row": shard_row,
            "aux_file": relative_to_project(aux_path),
            "aux_sha256": sha256_file(aux_path),
            "aux_row": shard_row,
            "episode_natural_steps": [
                {
                    "step": int(item[2]),
                    "shard": relative_to_project(
                        scan["shard_paths"][item[0]]
                    ),
                    "row": int(item[1]),
                    "action": int(sequence["actions"][item[2]]),
                }
                for item in sequence["items"]
            ],
        },
        "npz": {
            "file": "t5_inputs.npz",
            "sha256": npz_sha,
            "bytes": int(npz_path.stat().st_size),
        },
        "arrays": {
            name: {
                "shape": list(spec["shape"]),
                "dtype": np.dtype(spec["dtype"]).name,
                "source": spec["source"],
                "definition": spec["definition"],
            }
            for name, spec in EXPECTED_ARRAYS.items()
        },
        "input_contract": {
            "validator": "stage2j_model._validate_common_inputs",
            "batch_wrapper": "unsqueeze(0) on every array",
            "heatmap": None,
            "network_state": None,
            "model_input_arrays_only": sorted(EXPECTED_ARRAYS),
        },
        "final_test_read": False,
        "exported_utc": utc_now(),
    }
    write_json(case_dir / "case_metadata.json", metadata)

    print(
        f"[{case_name}] episode={episode} step={step} band={band_name} "
        f"conf={selection['current_confidence']:.4f} "
        f"map_spread={selection['map_top5_spread']:.6f}",
        flush=True,
    )
    return {
        "case": case_name,
        "episode_id": episode,
        "step_id": step,
        "band": band_name,
        "npz_sha256": npz_sha,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--aux-dir", type=Path, default=DEFAULT_AUX_DIR)
    parser.add_argument(
        "--aux-manifest", type=Path, default=DEFAULT_AUX_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--force", action="store_true",
        help="remove existing case directories under --output-dir",
    )
    args = parser.parse_args()

    reject_final_test(args.train_dir, args.aux_dir, args.aux_manifest)
    manifest = json.loads(args.aux_manifest.read_text(encoding="utf-8"))
    if manifest.get("final_test_read"):
        raise SystemExit("Aux manifest claims a Final Test read; refusing")
    threshold = float(manifest["gab_reliability_threshold"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in CASE_ORDER
                if (output_dir / name).exists()]
    if existing:
        if not args.force:
            raise SystemExit(
                "Refusing to overwrite existing case directories: "
                f"{[str(path) for path in existing]} (use --force)"
            )
        for path in existing:
            shutil.rmtree(path)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}", flush=True)
    encoder = load_encoder(device)

    scan = scan_train(args.train_dir, args.aux_dir)
    selection = select_cases(scan["index"], threshold)

    rules_payload = {
        "created_utc": utc_now(),
        "split": "train_600",
        "final_test_read": False,
        "pool": {
            "natural_samples": int(len(scan["index"]["episode"])),
            "episodes": int(len(scan["episode_rows"])),
            "sample_type_counts": {
                str(key): int(value)
                for key, value in sorted(scan["sample_type_counts"].items())
            },
        },
        "observable_definitions": {
            "current_match_confidence": (
                "max softmax probability of the archived 3x8x8 Goal-Current "
                "similarity heatmap after max-over-scales at temperature 0.10"
            ),
            "history_valid_slots": "sum(history_mask)",
            "gab_confidence": "gab[16] from the Stage-2i sidecar",
            "gab_valid": "gab[19] from the Stage-2i sidecar",
            "map_top5_spread": "map_feat[0] - map_feat[16]",
            "map_top5_margin": "map_feat[0] - map_feat[4]",
        },
        "rules": SELECTION_RULES,
        "gab_reliability_threshold": threshold,
        "top1_similarity_median": selection["top1_median"],
        "selected": selection["selected"],
        "constraints": {
            "navigation_success_used": False,
            "true_distance_used": False,
            "goal_coordinates_used": False,
            "oracle_action_used": False,
            "final_test_read": False,
            "distinct_episodes": True,
        },
    }
    write_json(output_dir / "selection_rules.json", rules_payload)

    manifest_for_export = dict(manifest)
    manifest_for_export["_train_dir"] = str(args.train_dir)
    results = []
    for case_name, selection_record in zip(
        CASE_ORDER, selection["selected"]
    ):
        results.append(export_case(
            case_name,
            selection_record,
            scan,
            encoder,
            device,
            output_dir,
            manifest_for_export,
        ))
    print(json.dumps(results, indent=2), flush=True)
    print("T5_INPUT_EXAMPLES_EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
