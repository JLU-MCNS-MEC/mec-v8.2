"""Lightweight RGB-only / RGB+Depth / Shuffled-Depth Arrival probes.

This is an offline probe experiment.  It does NOT modify the five MEC models,
does not train the shared encoder or Transformer-M, and does not write into
archived datasets/results.

Data policy:
  * Natural samples come from the archived V8.2 token shards.  Their
    episode/step/label metadata is replayed with ToyUAVEnv and checked against
    the archived distance / arrival label / goal visibility.  RGB features are
    the archived tokens; Depth is rendered at the verified replay state.
  * Hard-arrival samples are regenerated deterministically from the same
    Train/Val_hard episode seeds with the existing token_data hard-state
    sampler.  RGB and Depth for those samples come from the same render, so
    they cannot be misaligned.  (The archived shards do not store hard-state
    poses, so exact hard-state Depth replay is impossible.)
  * True Goal-region Depth / coordinates / horizontal distance / oracle masks
    are never inputs.  Only visual-match Depth evidence is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
from config import AUX, ENCODER, TRAIN
from depth_evidence_v83 import goal_conditioned_depth_evidence
from encoder import VisualTokenEncoder
from env import STOP, ToyUAVEnv
from gbs_model_v3 import goal_current_similarity
from gbs_network import build_random_gbs_layout
from token_data import (
    SAMPLE_BOUNDARY_NEGATIVE,
    SAMPLE_DISTRACTOR_NEGATIVE,
    SAMPLE_HARD_POSITIVE,
    SAMPLE_TYPE_NAMES,
    _collect_hard_state,
    encode_multiscale_goal_tokens,
)
from utils import image_to_tensor


ARCHIVE = Path("runs/archive_pycharm_v82_20261018")
TRAIN_DIR = Path("token_dataset_train")
VAL_DIR = Path("token_dataset_val")
VAL_HARD_DIR = Path("token_dataset_val_hard")
OUTPUT_ROOT = Path("runs/v83_depth_probe")
SEED = 20261018
RGB_FEATURE_DIM = 104 + 2  # frozen arrival features + peak/margin
DEPTH_FEATURE_DIM = 6
PROBE_DIMS = {"P0": RGB_FEATURE_DIM, "P1": RGB_FEATURE_DIM + DEPTH_FEATURE_DIM, "P2": RGB_FEATURE_DIM + DEPTH_FEATURE_DIM}


def band_ranges():
    return {
        i: (str(name), float(dmin), float(dmax))
        for i, (name, _w, dmin, dmax, _amin, _amax) in enumerate(TRAIN.trajectory_mix)
    }


class ShardCache:
    def __init__(self, files):
        self.files = list(files)
        self.cache = {}

    def get(self, path):
        if path not in self.cache:
            self.cache[path] = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        return self.cache[path]


def _to_numpy(t):
    return t.detach().cpu().numpy()


def scan_natural_groups(split_dir: Path):
    """Scan a split and group archived natural samples by episode seed.

    Returns a list of episode groups.  Each group contains all natural rows
    (including arrival-mask-excluded rows, needed to replay the oracle
    trajectory) and the selected arrival-mask rows.
    """
    files = sorted(Path(split_dir).glob("shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"No shards in {split_dir}")
    cache = ShardCache(files)
    by_seed = defaultdict(list)
    hard_counts = Counter()
    for path in files:
        d = cache.get(path)
        st = _to_numpy(d["sample_type"])
        nat_idx = np.flatnonzero(st == 0)
        hard_counts.update(st[st != 0].tolist())
        for i in nat_idx:
            rec = {
                "shard": str(path),
                "row": int(i),
                "episode_seed": int(d["episode_seeds"][i]),
                "step_id": int(d["step_ids"][i]),
                "episode_id": int(d["episode_ids"][i]),
                "band_id": int(d["trajectory_band_ids"][i]),
                "network_layout_seed": int(d["network_layout_seeds"][i]),
                "label": int(d["labels"][i]),
                "arrival_label": float(d["arrival_labels"][i]),
                "distance_to_goal_m": float(d["distance_to_goal_m"][i]),
                "goal_visible": int(d["goal_visible"][i]),
                "selected": bool(int(d["arrival_train_mask"][i])),
            }
            by_seed[rec["episode_seed"]].append(rec)
    groups = []
    for seed in sorted(by_seed):
        recs = by_seed[seed]
        episode_ids = {r["episode_id"] for r in recs}
        bands = {r["band_id"] for r in recs}
        layouts = {r["network_layout_seed"] for r in recs}
        if len(episode_ids) != 1 or len(bands) != 1 or len(layouts) != 1:
            raise RuntimeError(
                f"Episode seed {seed} has inconsistent episode/band/layout metadata"
            )
        step_map = {}
        for r in recs:
            if r["step_id"] in step_map:
                raise RuntimeError(f"Duplicate step {r['step_id']} for seed {seed}")
            step_map[r["step_id"]] = r
        groups.append({
            "seed": seed,
            "episode_id": next(iter(episode_ids)),
            "band_id": next(iter(bands)),
            "network_layout_seed": next(iter(layouts)),
            "step_map": step_map,
            "all_records": recs,
            "selected_records": [r for r in recs if r["selected"]],
        })
    return groups, cache, hard_counts


def _features_from_lists(cur_list, goal_list, heat_list, net_list, depth_list, model, device):
    """Compute 106-D RGB features and 6-D visual-match Depth features."""
    cur = torch.stack(cur_list).to(device, dtype=torch.float32)
    goal = torch.stack(goal_list).to(device, dtype=torch.float32)
    heat = torch.stack(heat_list).to(device, dtype=torch.float32)
    net = torch.stack(net_list).to(device, dtype=torch.float32)
    depth = torch.stack([torch.as_tensor(d, dtype=torch.float32) for d in depth_list])
    with torch.inference_mode():
        arrival = model.forward_components(cur, goal, heat, net)["arrival_features"]
        depth8 = goal_conditioned_depth_evidence(cur, goal, depth)
        sim = goal_current_similarity(cur, goal).amax(dim=1)
        k = min(12, int(sim.shape[1]))
        scores = torch.topk(sim, k=k, dim=1).values
        peak = ((scores[:, 0] + 1.0) * 0.5).clamp(0, 1)
        if k >= 2:
            margin = ((scores[:, 0] - scores[:, 1]) * 0.5).clamp(0, 1)
        else:
            margin = torch.zeros_like(peak)
        rgb_conf = torch.stack([peak, margin], dim=1)
        x_rgb = torch.cat([arrival, rgb_conf], dim=1)
        depth6 = depth8[:, [0, 1, 2, 3, 4, 7]]
        return (
            x_rgb.detach().cpu().numpy().astype(np.float32),
            depth6.detach().cpu().numpy().astype(np.float32),
        )


def _replay_natural_group(group, cache, model, device, band_map):
    """Replay one natural episode and extract selected samples."""
    seed = group["seed"]
    band_name, dmin, dmax = band_map[group["band_id"]]
    env = ToyUAVEnv()
    env.reset(seed=seed, goal_distance_range_m=(dmin, dmax), render_observation=False)
    step_map = group["step_map"]
    selected = {r["step_id"]: r for r in group["selected_records"]}
    max_step = max(step_map)
    cur_list, goal_list, heat_list, net_list, depth_list = [], [], [], [], []
    labels, meta = [], []
    distance_error_max = 0.0
    arrival_mismatch = 0
    goal_visible_mismatch = 0
    replayed_steps = 0
    for step in range(max_step + 1):
        rec = step_map.get(step)
        if rec is None:
            raise RuntimeError(f"Missing archived step {step} for seed {seed}")
        distance_error_max = max(
            distance_error_max,
            abs(float(env.distance_to_goal) - rec["distance_to_goal_m"]),
        )
        arrival_now = float(env.distance_to_goal < env.stop_radius)
        if int(arrival_now) != int(rec["arrival_label"] >= 0.5):
            arrival_mismatch += 1
        visible_now = int(env.goal_projection() is not None)
        if visible_now != rec["goal_visible"]:
            goal_visible_mismatch += 1
        replayed_steps += 1
        if step in selected:
            shard = cache.get(rec["shard"])
            row = rec["row"]
            cur_list.append(shard["tokens"][row].float())
            goal_list.append(shard["goal_tokens"][row].float())
            heat_list.append(shard["spatial_heatmaps"][row].float())
            net_list.append(shard["network_states"][row].float())
            depth_list.append(env.render_depth())
            labels.append(float(rec["arrival_label"]))
            meta.append({
                "episode_seed": seed,
                "episode_id": group["episode_id"],
                "step_id": step,
                "band_id": group["band_id"],
                "sample_type": 0,
                "distance_to_goal_m": rec["distance_to_goal_m"],
            })
        if step < max_step:
            env.step(rec["label"], render_observation=False)
    if len(labels) != len(group["selected_records"]):
        raise RuntimeError(
            f"Selected count mismatch seed {seed}: {len(labels)} vs "
            f"{len(group['selected_records'])}"
        )
    if not labels:
        return None, {
            "seed": seed,
            "selected": 0,
            "replayed_steps": replayed_steps,
            "distance_error_max": distance_error_max,
            "arrival_mismatch": arrival_mismatch,
            "goal_visible_mismatch": goal_visible_mismatch,
        }
    x_rgb, depth6 = _features_from_lists(
        cur_list, goal_list, heat_list, net_list, depth_list, model, device
    )
    return (x_rgb, depth6, np.asarray(labels, np.float32), meta), {
        "seed": seed,
        "selected": len(labels),
        "replayed_steps": replayed_steps,
        "distance_error_max": distance_error_max,
        "arrival_mismatch": arrival_mismatch,
        "goal_visible_mismatch": goal_visible_mismatch,
    }


def _generate_hard_group(group, cache, model, encoder, device, band_map, seed_offset=0):
    """Generate deterministic hard-arrival samples for one episode group."""
    seed = group["seed"]
    band_name, dmin, dmax = band_map[group["band_id"]]
    first = group["all_records"][0]
    shard = cache.get(first["shard"])
    row = first["row"]
    goal_token = shard["goal_tokens"][row].float()  # [3,384]
    network_nodes = build_random_gbs_layout(seed=group["network_layout_seed"], verbose=False)
    env = ToyUAVEnv()
    env.reset(seed=seed, goal_distance_range_m=(dmin, dmax), render_observation=False)
    rng = np.random.default_rng(seed * 1009 + SEED + seed_offset)
    plan = (
        [SAMPLE_HARD_POSITIVE] * int(TRAIN.arrival_hard_positive_per_episode)
        + [SAMPLE_BOUNDARY_NEGATIVE] * int(TRAIN.arrival_hard_boundary_negative_per_episode)
        + [SAMPLE_DISTRACTOR_NEGATIVE] * int(TRAIN.arrival_hard_distractor_negative_per_episode)
    )
    cur_list, goal_list, heat_list, net_list, depth_list = [], [], [], [], []
    labels, meta = [], []
    for hard_idx, sample_type in enumerate(plan):
        sample = None
        for attempt in range(4):
            sample = _collect_hard_state(
                env=env,
                sample_type=sample_type,
                rng=rng,
                encoder=encoder,
                device=device,
                goal_token=goal_token,
                network_nodes=network_nodes,
                episode_id=group["episode_id"],
                episode_seed=seed,
                network_layout_seed=group["network_layout_seed"],
                band_id=group["band_id"],
                hard_index=hard_idx * 4 + attempt,
            )
            if sample is not None:
                break
        if sample is None:
            continue
        cur_list.append(sample["current_tokens"].float())
        goal_list.append(sample["goal_token"].float())
        heat_list.append(sample["spatial_heatmap"].float())
        net_list.append(sample["network_state"].float())
        depth_list.append(env.render_depth())
        labels.append(float(sample["arrival_label"]))
        meta.append({
            "episode_seed": seed,
            "episode_id": group["episode_id"],
            "step_id": int(sample["step_id"]),
            "band_id": group["band_id"],
            "sample_type": int(sample_type),
            "distance_to_goal_m": float(sample["distance_to_goal_m"]),
        })
    if not labels:
        return None, {"seed": seed, "generated": 0, "requested": len(plan)}
    x_rgb, depth6 = _features_from_lists(
        cur_list, goal_list, heat_list, net_list, depth_list, model, device
    )
    return (x_rgb, depth6, np.asarray(labels, np.float32), meta), {
        "seed": seed,
        "generated": len(labels),
        "requested": len(plan),
    }


def build_split(split_dir, include_hard, model, encoder, device, max_episodes=None):
    groups, cache, hard_counts = scan_natural_groups(split_dir)
    if max_episodes is not None:
        groups = groups[: int(max_episodes)]
    print(f"Building {split_dir}: {len(groups)} natural episodes, include_hard={include_hard}", flush=True)
    xs, ds, ys, metas = [], [], [], []
    align = {
        "natural_episodes_checked": 0,
        "natural_samples_checked": 0,
        "distance_error_max": 0.0,
        "arrival_mismatch": 0,
        "goal_visible_mismatch": 0,
        "episode_id_mismatch": 0,
        "step_gap_episodes": [],
        "hard_requested": 0,
        "hard_generated": 0,
        "archived_hard_counts": {str(k): int(v) for k, v in hard_counts.items()},
        "note": "Natural RGB is archived; Depth is rendered at verified replay states. Hard samples are regenerated with RGB+Depth from the same pose.",
    }
    band_map = band_ranges()
    for gi, group in enumerate(groups):
        result, info = _replay_natural_group(group, cache, model, device, band_map)
        align["natural_episodes_checked"] += 1
        align["natural_samples_checked"] += int(info["selected"])
        align["distance_error_max"] = max(align["distance_error_max"], info["distance_error_max"])
        align["arrival_mismatch"] += int(info["arrival_mismatch"])
        align["goal_visible_mismatch"] += int(info["goal_visible_mismatch"])
        if info["selected"] and result is not None:
            x_rgb, depth6, y, meta = result
            xs.append(x_rgb); ds.append(depth6); ys.append(y); metas.extend(meta)
        if include_hard:
            hard_result, hard_info = _generate_hard_group(
                group, cache, model, encoder, device, band_map
            )
            align["hard_requested"] += int(hard_info["requested"])
            align["hard_generated"] += int(hard_info["generated"])
            if hard_result is not None:
                x_rgb, depth6, y, meta = hard_result
                xs.append(x_rgb); ds.append(depth6); ys.append(y); metas.extend(meta)
        if (gi + 1) % 25 == 0 or gi + 1 == len(groups):
            print(f"  processed {gi + 1}/{len(groups)} episodes", flush=True)
    if not xs:
        raise RuntimeError(f"No probe samples built for {split_dir}")
    out = (
        np.concatenate(xs, axis=0),
        np.concatenate(ds, axis=0),
        np.concatenate(ys, axis=0),
        metas,
    )
    return out, align


def _zip_meta(meta_list):
    keys = [
        "episode_seed", "episode_id", "step_id", "band_id", "sample_type",
        "distance_to_goal_m",
    ]
    return {k: np.asarray([m[k] for m in meta_list]) for k in keys}


def save_cache(path, x_rgb, depth6, y, meta):
    np.savez_compressed(
        path,
        x_rgb=x_rgb.astype(np.float32),
        depth6=depth6.astype(np.float32),
        y=y.astype(np.float32),
        **{k: v for k, v in _zip_meta(meta).items()},
    )


class ProbeMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def auroc(y, score):
    y = np.asarray(y).astype(int); score = np.asarray(score, float)
    pos = score[y == 1]; neg = score[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y, score):
    y = np.asarray(y).astype(int); score = np.asarray(score, float)
    npos = int(y.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / npos
    drecall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(drecall * precision))


def brier(y, prob):
    y = np.asarray(y, float); prob = np.asarray(prob, float)
    return float(np.mean((prob - y) ** 2))


def ece(y, prob, bins=10):
    y = np.asarray(y, float); prob = np.asarray(prob, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (prob >= lo) & (prob < hi if b < bins - 1 else prob <= hi)
        if m.any():
            total += m.mean() * abs(float(y[m].mean()) - float(prob[m].mean()))
    return float(total)


def threshold_metrics(y, prob, threshold):
    y = np.asarray(y).astype(int); pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return float(precision), float(recall)


def choose_threshold(y, prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        p, r = threshold_metrics(y, prob, t)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def metrics_dict(y, logits, threshold):
    prob = sigmoid(np.asarray(logits, float))
    precision, recall = threshold_metrics(y, prob, threshold)
    return {
        "auroc": auroc(y, prob),
        "auprc": average_precision(y, prob),
        "brier": brier(y, prob),
        "ece": ece(y, prob),
        "precision": precision,
        "recall": recall,
        "threshold": float(threshold),
        "n": int(len(y)),
        "positives": int(np.asarray(y).sum()),
        "positive_rate": float(np.asarray(y).mean()),
    }


def _prepare_inputs(x_rgb_train, d_train, x_rgb_val_nat, d_val_nat,
                    x_rgb_val_hard, d_val_hard, probe, seed):
    mu = x_rgb_train.mean(0, keepdims=True)
    sd = x_rgb_train.std(0, keepdims=True) + 1e-6
    xt = (x_rgb_train - mu) / sd
    xn = (x_rgb_val_nat - mu) / sd
    xh = (x_rgb_val_hard - mu) / sd
    if probe == "P0":
        return xt, xn, xh, {"rgb_mean": mu.tolist(), "rgb_std": sd.tolist()}
    dmu = d_train.mean(0, keepdims=True)
    dsd = d_train.std(0, keepdims=True) + 1e-6
    dt = (d_train - dmu) / dsd
    dn = (d_val_nat - dmu) / dsd
    dh = (d_val_hard - dmu) / dsd
    if probe == "P2":
        rng = np.random.default_rng(seed)
        dt = dt[rng.permutation(len(dt))]
        dn = dn[rng.permutation(len(dn))]
        dh = dh[rng.permutation(len(dh))]
    return (
        np.concatenate([xt, dt], axis=1),
        np.concatenate([xn, dn], axis=1),
        np.concatenate([xh, dh], axis=1),
        {"rgb_mean": mu.tolist(), "rgb_std": sd.tolist(),
         "depth_mean": dmu.tolist(), "depth_std": dsd.tolist()},
    )


def train_probe(name, x_train, y_train, x_val_nat, y_val_nat, x_val_hard, y_val_hard,
                seed, max_epochs=50, patience=8, batch_size=256, lr=1e-3,
                weight_decay=1e-4, device=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProbeMLP(x_train.shape[1]).to(device)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=generator)
    best_state = None
    best_epoch = 0
    best_ap = -1.0
    stale = 0
    curves = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        model.eval()
        with torch.inference_mode():
            val_logits = model(torch.from_numpy(x_val_nat.astype(np.float32)).to(device)).cpu().numpy()
        val_ap = average_precision(y_val_nat, sigmoid(val_logits))
        curves.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(n_batches, 1),
            "val_natural_auprc": val_ap,
        })
        print(f"  {name} epoch {epoch:02d} train_loss={loss_sum/max(n_batches,1):.5f} val_nat_auprc={val_ap:.5f}", flush=True)
        if val_ap > best_ap + 1e-6:
            best_ap = val_ap
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  {name} early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        nat_logits = model(torch.from_numpy(x_val_nat.astype(np.float32)).to(device)).cpu().numpy()
        hard_logits = model(torch.from_numpy(x_val_hard.astype(np.float32)).to(device)).cpu().numpy()
    nat_prob = sigmoid(nat_logits)
    threshold = choose_threshold(y_val_nat, nat_prob)
    nat_metrics = metrics_dict(y_val_nat, nat_logits, threshold)
    hard_metrics = metrics_dict(y_val_hard, hard_logits, threshold)
    elapsed = time.time() - t0
    return {
        "model": model,
        "best_epoch": int(best_epoch),
        "best_val_natural_auprc": float(best_ap),
        "train_time_s": float(elapsed),
        "curves": curves,
        "nat_metrics": nat_metrics,
        "hard_metrics": hard_metrics,
        "threshold": float(threshold),
        "input_dim": int(x_train.shape[1]),
    }


def _plot_curves(path, title, curves):
    epochs = [c["epoch"] for c in curves]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.0))
    ax1.plot(epochs, [c["train_loss"] for c in curves], label="train loss", color="tab:blue")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [c["val_natural_auprc"] for c in curves], label="val AUPRC", color="tab:orange")
    ax2.set_ylabel("val natural AUPRC", color="tab:orange")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_experiment(args):
    started = time.time()
    seed = int(args.seed)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    evaluation.CHECKPOINT_DIR = ARCHIVE / "gbs_v82_epoch_compare_results"
    device = evaluation.DEVICE
    model = evaluation.load_model("Transformer-M", 50)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    encoder = VisualTokenEncoder().to(device)
    encoder.load_state_dict(torch.load(ARCHIVE / "shared_encoder.pt", map_location=device, weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    outdir = Path(args.output_root) / ("smoke_" + stamp if args.smoke else stamp)
    outdir.mkdir(parents=True, exist_ok=False)
    cache_dir = outdir / "cache"
    cache_dir.mkdir()
    print(f"Output: {outdir}", flush=True)

    max_train = args.max_train_episodes if args.smoke else None
    max_val = args.max_val_episodes if args.smoke else None
    max_hard = args.max_hard_episodes if args.smoke else None

    train_data, align_train = build_split(
        TRAIN_DIR, True, model, encoder, device, max_train
    )
    val_nat_data, align_val = build_split(
        VAL_DIR, False, model, encoder, device, max_val
    )
    val_hard_data, align_hard = build_split(
        VAL_HARD_DIR, True, model, encoder, device, max_hard
    )
    align = {"train": align_train, "val_natural": align_val, "val_hard": align_hard,
             "generated_hard_note": "Hard samples are regenerated deterministically; RGB and Depth come from the same pose."}
    (outdir / "alignment_checks.json").write_text(json.dumps(align, indent=2), encoding="utf-8")

    x_rgb_tr, d_tr, y_tr, meta_tr = train_data
    x_rgb_vn, d_vn, y_vn, meta_vn = val_nat_data
    x_rgb_vh, d_vh, y_vh, meta_vh = val_hard_data
    save_cache(cache_dir / "train.npz", x_rgb_tr, d_tr, y_tr, meta_tr)
    save_cache(cache_dir / "val_natural.npz", x_rgb_vn, d_vn, y_vn, meta_vn)
    save_cache(cache_dir / "val_hard.npz", x_rgb_vh, d_vh, y_vh, meta_vh)
    print(
        f"Samples: train={len(y_tr)} pos={int(y_tr.sum())}, "
        f"val_natural={len(y_vn)} pos={int(y_vn.sum())}, "
        f"val_hard={len(y_vh)} pos={int(y_vh.sum())}",
        flush=True,
    )

    results = {}
    for probe in ("P0", "P1", "P2"):
        x_tr, x_vn, x_vh, scaler = _prepare_inputs(
            x_rgb_tr, d_tr, x_rgb_vn, d_vn, x_rgb_vh, d_vh, probe, seed
        )
        print(f"Training {probe} input_dim={x_tr.shape[1]}", flush=True)
        res = train_probe(
            probe, x_tr, y_tr, x_vn, y_vn, x_vh, y_vh,
            seed=seed,
            max_epochs=(2 if args.smoke else args.max_epochs),
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        results[probe] = res
        probe_dir = outdir / probe
        probe_dir.mkdir()
        torch.save(res["model"].state_dict(), probe_dir / f"{probe}_best.pt")
        with (probe_dir / "curves.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res["curves"][0]))
            w.writeheader(); w.writerows(res["curves"])
        _plot_curves(probe_dir / "curve.png", f"{probe} (input {res['input_dim']})", res["curves"])
        (probe_dir / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
        (probe_dir / "metrics.json").write_text(json.dumps({
            "probe": probe,
            "input_dim": res["input_dim"],
            "best_epoch": res["best_epoch"],
            "best_val_natural_auprc": res["best_val_natural_auprc"],
            "train_time_s": res["train_time_s"],
            "threshold": res["threshold"],
            "natural_val": res["nat_metrics"],
            "hard_val": res["hard_metrics"],
        }, indent=2), encoding="utf-8")
        del res["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for probe in ("P0", "P1", "P2"):
        r = results[probe]
        for split, m in (("natural_val", r["nat_metrics"]), ("hard_val", r["hard_metrics"])):
            rows.append({
                "probe": probe, "input_dim": r["input_dim"], "split": split,
                "best_epoch": r["best_epoch"], "auroc": m["auroc"], "auprc": m["auprc"],
                "brier": m["brier"], "ece": m["ece"], "recall": m["recall"],
                "precision": m["precision"], "threshold": m["threshold"],
                "n": m["n"], "positives": m["positives"],
            })
    with (outdir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    summary = {
        "seed": seed,
        "branch": __import__("subprocess").check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "train_samples": int(len(y_tr)),
        "train_positives": int(y_tr.sum()),
        "val_natural_samples": int(len(y_vn)),
        "val_natural_positives": int(y_vn.sum()),
        "val_hard_samples": int(len(y_vh)),
        "val_hard_positives": int(y_vh.sum()),
        "probes": {
            p: {
                "input_dim": results[p]["input_dim"],
                "best_epoch": results[p]["best_epoch"],
                "train_time_s": results[p]["train_time_s"],
                "natural_val": results[p]["nat_metrics"],
                "hard_val": results[p]["hard_metrics"],
            } for p in results
        },
        "total_time_s": time.time() - started,
        "alignment": align,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Comparison:", flush=True)
    for r in rows:
        print(r, flush=True)
    print(f"Total time: {summary['total_time_s']:.1f}s", flush=True)
    print(f"Results: {outdir}", flush=True)
    return outdir, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-train-episodes", type=int, default=2)
    p.add_argument("--max-val-episodes", type=int, default=2)
    p.add_argument("--max-hard-episodes", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
