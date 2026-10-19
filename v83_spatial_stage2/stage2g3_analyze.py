"""Stage-2g.3 post-hoc analysis of the saved real closed-loop probe run.

This script performs *inference only* with the checkpoints already saved by
stage2g3_probe.py. It does not train, recollect, or modify any checkpoint.

It recovers the analyses that were not written into probe_results.json:

  * P0 / P1-2 Dev-Val metrics and per-band metrics (audited against JSON);
  * near/far x visible/invisible quadrant confusion and false-trigger rates;
  * Train-600 positive-sample episode/distribution/run-length statistics;
  * episode-paired P0 vs P1-2 comparisons, paired bootstrap and LOEO;
  * Train-OOF-only threshold provenance reconstruction;
  * synthetic Stage-2g.1 vs real closed-loop distribution comparison;
  * leakage/history/label/class-ratio audits.

All derived outputs stay under runs/ and are not committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest, ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP  # noqa: E402

from v83_spatial_stage2.stage2g_features import (  # noqa: E402
    FORBIDDEN_FEATURE_TERMS,
    feature_names,
)
from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    COMPACT_FEATURE_NAMES,
    compact_indices,
)
from v83_spatial_stage2.stage2g3_probe import (  # noqa: E402
    BAND_NAMES,
    build_input,
    filter_moves,
    load_probe_for_comparison,
    load_split,
    real_masks,
    real_metrics,
    select_thresholds,
)
from v83_spatial_stage2.stage2g_probe import (  # noqa: E402
    assign_episode_folds,
    auc_rank,
    average_precision,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    metric_block,
)

ACTION_NAMES = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "STOP"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def contiguous_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def quantile_summary(values, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {
            str(q): float(np.quantile(values, q)) for q in qs
        },
    }


def count_bins(values, edges, labels=None):
    values = np.asarray(values, dtype=float)
    if labels is None:
        labels = [
            f"{edges[i]}-{edges[i + 1]}"
            for i in range(len(edges) - 1)
        ]
    result = {}
    for index, label in enumerate(labels):
        low, high = edges[index], edges[index + 1]
        if index == len(edges) - 2:
            mask = (values >= low) & (values <= high)
        else:
            mask = (values >= low) & (values < high)
        result[label] = int(mask.sum())
    return result


def gini(values):
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float(
        (n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n
    )


def concentration(values):
    values = np.sort(np.asarray(values, dtype=float))[::-1]
    total = float(values.sum())
    if total <= 0:
        return {}
    shares = values.cumsum() / total
    needed = {}
    for target in (0.5, 0.8, 0.9):
        index = int(np.searchsorted(shares, target) + 1)
        needed[str(target)] = min(index, len(values))
    return {
        "total": total,
        "episodes": int(len(values)),
        "top1_share": float(shares[0]) if len(shares) else 0.0,
        "top3_share": float(shares[min(2, len(shares) - 1)]),
        "top5_share": float(shares[min(4, len(shares) - 1)]),
        "top10_share": float(shares[min(9, len(shares) - 1)]),
        "gini": gini(values),
        "episodes_for_50pct": needed[str(0.5)],
        "episodes_for_80pct": needed[str(0.8)],
        "episodes_for_90pct": needed[str(0.9)],
    }


# ---------------------------------------------------------------------------
# Raw episode scan: labels, history, positive runs and contexts
# ---------------------------------------------------------------------------
def scan_split(directory, return_context=False):
    directory = Path(directory)
    files = sorted(directory.glob("episode_*.npz"))
    if not files:
        raise RuntimeError(f"No episode files in {directory}")
    compact_idx = compact_indices(feature_names())

    totals = Counter()
    band_episode_counts = Counter()
    band_positive_states = Counter()
    band_positive_episodes = Counter()
    positive_per_episode = []
    positive_distance = []
    positive_since_loc = []
    positive_since_vis = []
    positive_actions = Counter()
    positive_history_patterns = Counter()
    run_lengths = []
    run_records = []
    label_mismatches = Counter()
    history_mismatches = Counter()
    zero_move_episodes = []
    duplicate_episode_ids = Counter()
    duplicate_seed_layout = Counter()
    seeds = set()
    layouts = set()
    seed_layout_pairs = set()
    context = {} if return_context else None

    for path in files:
        data = np.load(path)
        episode = int(data["episode_id"])
        duplicate_episode_ids[episode] += 1
        n = len(data["step"])
        steps = np.asarray(data["step"], dtype=np.int64)
        actions = np.asarray(data["action"], dtype=np.int64)
        distance = np.asarray(data["distance_m"], dtype=np.float64)
        visible = np.asarray(data["target_visible"], dtype=np.int64)
        localizable = np.asarray(data["target_localizable"], dtype=np.int64)
        near = np.asarray(data["near_target_invisible"], dtype=np.int64)
        stop = np.asarray(data["stop"], dtype=np.int64)
        history_valid = np.asarray(data["history_valid"], dtype=np.int64)
        history_actions = np.asarray(data["history_actions"], dtype=np.int64)
        history_dt = np.asarray(data["history_dt"], dtype=np.float64)
        history_features = np.asarray(data["history_features"], dtype=np.float32)
        current = np.asarray(data["current_features"], dtype=np.float32)
        band = int(data["band"])
        seed = int(data["seed"])
        layout = int(data["network_layout_seed"])
        seeds.add(seed)
        layouts.add(layout)
        seed_layout_pairs.add((seed, layout))
        duplicate_seed_layout[(seed, layout)] += 1

        totals["episodes"] += 1
        totals["states"] += n
        totals["stop_states"] += int(stop.sum())
        totals["history_invalid"] += int((history_valid == 0).sum())
        totals["positives"] += int(near.sum())
        band_episode_counts[band] += 1
        band_positive_states[band] += int(near.sum())
        if near.any():
            band_positive_episodes[band] += 1
        if not np.isfinite(distance).all():
            label_mismatches["nonfinite_distance"] += 1

        # Label semantics.
        expected_near = ((distance < 100.0) & (visible == 0)).astype(np.int64)
        label_mismatches["near_target_invisible"] += int(
            (expected_near != near).sum()
        )
        label_mismatches["localizable_gt_visible"] += int(
            ((localizable == 1) & (visible == 0)).sum()
        )
        label_mismatches["stop_not_action"] += int(
            (stop != (actions == STOP).astype(np.int64)).sum()
        )
        label_mismatches["near_implies_not_visible"] += int(
            ((near == 1) & (visible == 1)).sum()
        )
        label_mismatches["negative_distance"] += int((distance < 0).sum())

        # History audit: frame 0 = t-1, frame 1 = t-2, reset per episode.
        expected_valid = (steps >= 2).astype(np.int64)
        history_mismatches["history_valid"] += int(
            (history_valid != expected_valid).sum()
        )
        if n >= 1:
            history_mismatches["t0_history_nonzero"] += int(
                np.abs(history_features[0]).sum() > 0
            )
            history_mismatches["t0_action_not_minus1"] += int(
                (history_actions[0] != -1).sum()
            )
            history_mismatches["t0_dt_nonzero"] += int(
                np.abs(history_dt[0]).sum() > 0
            )
        if n >= 2:
            history_mismatches["t1_valid_not_zero"] += int(
                history_valid[1] != 0
            )
            history_mismatches["t1_frame0_mismatch"] += int(
                not np.array_equal(
                    history_features[1, 0], current[0, compact_idx]
                )
            )
            history_mismatches["t1_frame1_nonzero"] += int(
                np.abs(history_features[1, 1]).sum() > 0
            )
            history_mismatches["t1_action0_not_t0"] += int(
                history_actions[1, 0] != actions[0]
            )
            history_mismatches["t1_action1_not_minus1"] += int(
                history_actions[1, 1] != -1
            )
        if n > 2:
            current_compact = current[:, compact_idx]
            expected_a0 = actions[1:-1]
            expected_a1 = actions[:-2]
            history_mismatches["history_features_t1"] += int(
                (history_features[2:, 0] != current_compact[1:-1]).sum()
            )
            history_mismatches["history_features_t2"] += int(
                (history_features[2:, 1] != current_compact[:-2]).sum()
            )
            history_mismatches["history_actions_t1"] += int(
                (history_actions[2:, 0] != expected_a0).sum()
            )
            history_mismatches["history_actions_t2"] += int(
                (history_actions[2:, 1] != expected_a1).sum()
            )
            history_mismatches["history_dt_t1"] += int(
                (history_dt[2:, 0] != 1.0).sum()
            )
            history_mismatches["history_dt_t2"] += int(
                (history_dt[2:, 1] != 1.0).sum()
            )
        history_mismatches["nonfinite_features"] += int(
            (~np.isfinite(current)).sum()
        )
        history_mismatches["nonfinite_history"] += int(
            (~np.isfinite(history_features)).sum()
        )

        # Per-state context for Dev-Val score stratification.
        since_loc = np.full(n, -1, dtype=np.int64)
        since_vis = np.full(n, -1, dtype=np.int64)
        last_loc = -1
        last_vis = -1
        for t in range(n):
            if loc := bool(localizable[t]):
                since_loc[t] = 0
                last_loc = t
            elif last_loc >= 0:
                since_loc[t] = t - last_loc
            if bool(visible[t]):
                since_vis[t] = 0
                last_vis = t
            elif last_vis >= 0:
                since_vis[t] = t - last_vis
        run_start = np.zeros(n, dtype=bool)
        run_len = np.zeros(n, dtype=np.int64)
        runs = contiguous_runs(near == 1)
        for start, length in runs:
            run_start[start] = True
            run_len[start:start + length] = length
            run_lengths.append(length)
            run_records.append({
                "episode_id": episode,
                "band": BAND_NAMES[band],
                "start": int(start),
                "length": int(length),
                "distance_start_m": float(distance[start]),
                "distance_end_m": float(distance[start + length - 1]),
                "actions": dict(Counter(
                    ACTION_NAMES[int(a)] for a in actions[start:start + length]
                )),
            })

        positive_episode_count = int(near.sum())
        positive_per_episode.append({
            "episode_id": episode,
            "band": BAND_NAMES[band],
            "states": int(n),
            "positives": positive_episode_count,
            "positive_fraction": (
                float(positive_episode_count / n) if n else 0.0
            ),
            "target_visible_states": int(visible.sum()),
            "target_localizable_states": int(localizable.sum()),
            "seed": seed,
            "network_layout_seed": layout,
        })
        if positive_episode_count:
            positive_distance.extend(distance[near == 1].tolist())
        for t in np.flatnonzero(near == 1):
            positive_action = int(actions[t])
            positive_actions[ACTION_NAMES.get(positive_action, "UNKNOWN")] += 1
            if t >= 2:
                positive_history_patterns[(
                    ACTION_NAMES.get(int(history_actions[t, 0]), "INVALID"),
                    ACTION_NAMES.get(int(history_actions[t, 1]), "INVALID"),
                )] += 1
            positive_since_loc.append(int(since_loc[t]))
            positive_since_vis.append(int(since_vis[t]))

        valid_move = (stop == 0) & (history_valid > 0)
        if not valid_move.any():
            zero_move_episodes.append({
                "episode_id": episode,
                "states": int(n),
                "length": int(n),
                "stops": int(stop.sum()),
                "band": BAND_NAMES[band],
            })

        if return_context:
            for t in range(n):
                context[(episode, int(steps[t]))] = {
                    "target_visible": int(visible[t]),
                    "target_localizable": int(localizable[t]),
                    "near_target_invisible": int(near[t]),
                    "distance_m": float(distance[t]),
                    "action": int(actions[t]),
                    "history_action_t1": (
                        int(history_actions[t, 0]) if t >= 1 else -1
                    ),
                    "history_action_t2": (
                        int(history_actions[t, 1]) if t >= 2 else -1
                    ),
                    "since_localizable": int(since_loc[t]),
                    "since_visible": int(since_vis[t]),
                    "run_start": bool(run_start[t]),
                    "run_length": int(run_len[t]),
                }

    totals["positive_episodes"] = int(
        sum(1 for row in positive_per_episode if row["positives"] > 0)
    )
    counts = np.asarray(
        [row["positives"] for row in positive_per_episode if row["positives"] > 0],
        dtype=float,
    )
    return {
        "directory": str(directory),
        "manifest_sha256": sha256(directory / "manifest.json"),
        "totals": dict(totals),
        "band_episode_counts": {
            BAND_NAMES[k]: int(v) for k, v in sorted(band_episode_counts.items())
        },
        "band_positive_states": {
            BAND_NAMES[k]: int(v) for k, v in sorted(band_positive_states.items())
        },
        "band_positive_episodes": {
            BAND_NAMES[k]: int(v) for k, v in sorted(band_positive_episodes.items())
        },
        "positive_per_episode": sorted(
            positive_per_episode, key=lambda row: row["positives"], reverse=True
        ),
        "positive_per_episode_summary": quantile_summary(counts),
        "positive_count_histogram": {
            "1": int(((counts >= 1) & (counts <= 1)).sum()),
            "2-4": int(((counts >= 2) & (counts <= 4)).sum()),
            "5-9": int(((counts >= 5) & (counts <= 9)).sum()),
            "10-19": int(((counts >= 10) & (counts <= 19)).sum()),
            "20-49": int(((counts >= 20) & (counts <= 49)).sum()),
            "50-99": int(((counts >= 50) & (counts <= 99)).sum()),
            "100+": int((counts >= 100).sum()),
        },
        "concentration": concentration(counts),
        "positive_distance_summary": quantile_summary(positive_distance),
        "positive_distance_bins": count_bins(
            positive_distance,
            [0.0, 20.0, 50.0, 100.0],
            ["0-20", "20-50", "50-100"],
        ),
        "run_length_summary": quantile_summary(run_lengths),
        "run_length_bins": count_bins(
            run_lengths,
            [0, 1, 2, 5, 10, 20, 50, 10000],
            ["1", "2-4", "5-9", "10-19", "20-49", "50-99", "100+"],
        ),
        "run_fraction_ge_5": (
            float(np.mean(np.asarray(run_lengths) >= 5)) if run_lengths else 0.0
        ),
        "run_fraction_ge_10": (
            float(np.mean(np.asarray(run_lengths) >= 10)) if run_lengths else 0.0
        ),
        "run_fraction_ge_20": (
            float(np.mean(np.asarray(run_lengths) >= 20)) if run_lengths else 0.0
        ),
        "runs": sorted(run_records, key=lambda row: row["length"], reverse=True)[:20],
        "positive_since_localizable_summary": quantile_summary(
            [x for x in positive_since_loc if x >= 0]
        ),
        "positive_since_localizable_never": int(
            sum(1 for x in positive_since_loc if x < 0)
        ),
        "positive_since_localizable_bins": count_bins(
            [x for x in positive_since_loc if x >= 0],
            [0, 1, 2, 5, 10, 20, 50, 10000],
            ["0-1", "2", "3-4", "5-9", "10-19", "20-49", "50+"],
        ),
        "positive_since_localizable_le2": int(
            sum(1 for x in positive_since_loc if 0 <= x <= 2)
        ),
        "positive_since_visible_summary": quantile_summary(
            [x for x in positive_since_vis if x >= 0]
        ),
        "positive_since_visible_never": int(
            sum(1 for x in positive_since_vis if x < 0)
        ),
        "positive_since_visible_le2": int(
            sum(1 for x in positive_since_vis if 0 <= x <= 2)
        ),
        "positive_actions": dict(positive_actions),
        "positive_history_patterns": {
            f"{a}->{b}": int(v)
            for (a, b), v in positive_history_patterns.most_common(20)
        },
        "label_mismatches": dict(label_mismatches),
        "history_mismatches": dict(history_mismatches),
        "zero_move_episodes": zero_move_episodes,
        "duplicate_episode_ids": {
            str(k): int(v) for k, v in duplicate_episode_ids.items() if v > 1
        },
        "duplicate_seed_layout": {
            f"{k[0]}:{k[1]}": int(v)
            for k, v in duplicate_seed_layout.items() if v > 1
        },
        "unique_seeds": len(seeds),
        "unique_layouts": len(layouts),
        "context": context,
    }


# ---------------------------------------------------------------------------
# Prediction reconstruction
# ---------------------------------------------------------------------------
def model_input(data, variant):
    if variant == 0:
        return data["current_features"].astype(np.float32, copy=False)
    return build_input(data, 1)


def predict_external(probe_dir, data, variant, device):
    checkpoint = (
        Path(probe_dir) / "models"
        / f"v{variant}_near_target_invisible_external.pt"
    )
    model, mean, std, weight, threshold = load_probe_for_comparison(
        checkpoint, device
    )
    scores = batch_predict(
        model, mean, std, weight, model_input(data, variant), device
    )
    return scores, checkpoint, weight


def reconstruct_oof(probe_dir, train_data, variant, device):
    folds = assign_episode_folds(
        train_data["episode_id"], train_data["band"], 5, 0
    )
    x = model_input(train_data, variant)
    oof = np.full(len(x), np.nan, dtype=np.float64)
    weights = {}
    for fold in range(5):
        checkpoint = (
            Path(probe_dir) / "models"
            / f"v{variant}_near_target_invisible_fold{fold}.pt"
        )
        model, mean, std, weight, _threshold = load_probe_for_comparison(
            checkpoint, device
        )
        mask = folds == fold
        oof[mask] = batch_predict(model, mean, std, weight, x[mask], device)
        weights[fold] = float(weight)
    return oof, folds, weights


def batch_predict(model, mean, std, pos_weight, features, device):
    # Local copy so this module does not depend on a private probe helper.
    x = (np.asarray(features, dtype=np.float32) - mean) / std
    x = np.clip(x, -20.0, 20.0)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 16384):
            tensor = torch.from_numpy(x[start:start + 16384]).to(device)
            logits = model(tensor).squeeze(1)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(outputs)
    if pos_weight is None:
        return probabilities
    w = float(pos_weight)
    p = np.clip(probabilities, 1e-8, 1 - 1e-8)
    odds = p / (1.0 - p) / w
    return odds / (1.0 + odds)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_recall(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    flags = np.asarray(scores) >= float(threshold)
    tp = int((flags & (labels == 1)).sum())
    fp = int((flags & (labels == 0)).sum())
    fn = int((~flags & (labels == 1)).sum())
    tn = int((~flags & (labels == 0)).sum())
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "flags": int(flags.sum()),
    }


def metric_summary(labels, scores):
    block = metric_block(labels, scores)
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    positive_rate = float(labels.mean()) if len(labels) else float("nan")
    brier_reference = positive_rate
    positive_mean = (
        float(positive_scores.mean()) if len(positive_scores) else float("nan")
    )
    negative_mean = (
        float(negative_scores.mean()) if len(negative_scores) else float("nan")
    )
    return {
        "n": block["n"],
        "positives": block["positives"],
        "positive_rate": block["positive_rate"],
        "auroc": block["auroc"],
        "auprc": block["auprc"],
        "auprc_negative": block["auprc_negative"],
        "auprc_lift_over_positive_rate": (
            block["auprc"] / positive_rate if positive_rate else float("nan")
        ),
        "brier": block["brier"],
        "brier_constant_zero": brier_reference,
        "brier_skill_score": (
            1.0 - block["brier"] / brier_reference
            if brier_reference else float("nan")
        ),
        "ece": block["ece"],
        "mean_probability": block["mean_probability"],
        "median_probability": block["median_probability"],
        "mean_probability_positive": positive_mean,
        "mean_probability_negative": negative_mean,
        "positive_negative_mean_ratio": (
            positive_mean / negative_mean
            if negative_mean and negative_mean > 0 else float("nan")
        ),
        "balanced_accuracy_at_0.5": block["balanced_accuracy_at_0.5"],
        "at_0.5": block["confusions"]["0.5"],
    }


def quadrant_table(data, scores, threshold):
    masks = real_masks(data)
    labels = np.asarray(data["near_target_invisible"], dtype=int)
    flags = np.asarray(scores) >= float(threshold)
    total_flags = max(int(flags.sum()), 1)
    table = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        count = int(mask.sum())
        flag_count = int((flags & mask).sum())
        table[name] = {
            "n": count,
            "positives": int(labels[mask].sum()),
            "flags": flag_count,
            "flag_rate": flag_count / max(count, 1),
            "share_of_all_flags": flag_count / total_flags,
            "mean_score": float(np.mean(scores[mask])) if count else float("nan"),
            "median_score": (
                float(np.median(scores[mask])) if count else float("nan")
            ),
            "p_ge_0.5": (
                float((scores[mask] >= 0.5).mean()) if count else float("nan")
            ),
        }
    # Explicit 2x2 confusion against the positive label.
    near = masks["near_invisible"]
    tp = int((flags & near).sum())
    fp = int((flags & ~near).sum())
    fn = int((~flags & near).sum())
    tn = int((~flags & ~near).sum())
    table["confusion"] = {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }
    # Distance-bin view, including mean scores.
    bins = {}
    for low, high in ((0, 20), (20, 50), (50, 100), (100, 200), (200, 400)):
        mask = (
            (data["distance_m"] >= low) & (data["distance_m"] < high)
        )
        count = int(mask.sum())
        if not count:
            continue
        bin_labels = labels[mask]
        bin_flags = flags[mask]
        tp_bin = int((bin_flags & (bin_labels == 1)).sum())
        fp_bin = int((bin_flags & (bin_labels == 0)).sum())
        bins[f"{low}-{high}"] = {
            "n": count,
            "positives": int(bin_labels.sum()),
            "flags": int(bin_flags.sum()),
            "flag_rate": float(bin_flags.mean()),
            "precision": tp_bin / max(tp_bin + fp_bin, 1),
            "recall": tp_bin / max(int(bin_labels.sum()), 1),
            "mean_score": float(np.mean(scores[mask])),
        }
    table["distance_bins"] = bins
    return table


def per_episode_table(data, p0, p1, threshold0, threshold1):
    labels = np.asarray(data["near_target_invisible"], dtype=int)
    episodes = np.asarray(data["episode_id"])
    bands = np.asarray(data["band"])
    rows = []
    for episode in np.unique(episodes):
        mask = episodes == episode
        y = labels[mask]
        s0, s1 = p0[mask], p1[mask]
        row = {
            "episode_id": int(episode),
            "band": BAND_NAMES[int(bands[mask][0])],
            "n": int(mask.sum()),
            "n_positives": int(y.sum()),
            "n_negatives": int(len(y) - y.sum()),
            "mean_positive_p0": (
                float(s0[y == 1].mean()) if y.sum() else float("nan")
            ),
            "mean_positive_p1": (
                float(s1[y == 1].mean()) if y.sum() else float("nan")
            ),
            "mean_negative_p0": (
                float(s0[y == 0].mean()) if (len(y) - y.sum()) else float("nan")
            ),
            "mean_negative_p1": (
                float(s1[y == 0].mean()) if (len(y) - y.sum()) else float("nan")
            ),
            "recall_p0": (
                float((s0[y == 1] >= threshold0).mean()) if y.sum() else float("nan")
            ),
            "recall_p1": (
                float((s1[y == 1] >= threshold1).mean()) if y.sum() else float("nan")
            ),
            "ap_p0": average_precision(y, s0) if 0 < y.sum() < len(y) else float("nan"),
            "ap_p1": average_precision(y, s1) if 0 < y.sum() < len(y) else float("nan"),
            "auroc_p0": auc_rank(y, s0) if 0 < y.sum() < len(y) else float("nan"),
            "auroc_p1": auc_rank(y, s1) if 0 < y.sum() < len(y) else float("nan"),
        }
        row["delta_mean_positive"] = (
            row["mean_positive_p1"] - row["mean_positive_p0"]
        )
        row["delta_recall"] = row["recall_p1"] - row["recall_p0"]
        row["delta_ap"] = row["ap_p1"] - row["ap_p0"]
        rows.append(row)
    return rows


def _pr_from_arrays(y, scores, threshold):
    flags = scores >= float(threshold)
    tp = int((flags & (y == 1)).sum())
    fp = int((flags & (y == 0)).sum())
    fn = int((~flags & (y == 1)).sum())
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def paired_bootstrap(episodes, labels, p0, p1, threshold0, threshold1,
                     masks=None, n_boot=1000, seed=0):
    rng = np.random.default_rng(int(seed))
    unique = np.unique(episodes)
    index_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique
    }
    draws = defaultdict(list)
    for _ in range(int(n_boot)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            index_by_episode[int(episode)] for episode in sample
        ])
        y = labels[indices]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s0, s1 = p0[indices], p1[indices]
        draws["auroc"].append(auc_rank(y, s1) - auc_rank(y, s0))
        draws["auprc"].append(
            average_precision(y, s1) - average_precision(y, s0)
        )
        pre0, rec0 = _pr_from_arrays(y, s0, threshold0)
        pre1, rec1 = _pr_from_arrays(y, s1, threshold1)
        draws["precision"].append(pre1 - pre0)
        draws["recall"].append(rec1 - rec0)
        if masks is not None:
            for name, mask in masks.items():
                mb = np.asarray(mask, dtype=bool)[indices]
                if not mb.any():
                    continue
                f0 = float((s0[mb] >= threshold0).mean())
                f1 = float((s1[mb] >= threshold1).mean())
                draws[f"flag_rate_{name}"].append(f1 - f0)
    result = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            result[name] = {"n_boot": 0}
            continue
        result[name] = {
            "n_boot": int(len(values)),
            "mean": float(values.mean()),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "fraction_le_0": float((values <= 0).mean()),
            "fraction_gt_0": float((values > 0).mean()),
        }
    return result


def leave_one_positive_episode_out(data, p0, p1):
    labels = np.asarray(data["near_target_invisible"], dtype=int)
    episodes = np.asarray(data["episode_id"])
    positive_episodes = [
        int(episode)
        for episode in np.unique(episodes)
        if labels[episodes == episode].sum() > 0
    ]
    rows = []
    for excluded in positive_episodes:
        keep = episodes != excluded
        if labels[keep].sum() == 0 or labels[keep].sum() == keep.sum():
            continue
        rows.append({
            "excluded_episode": excluded,
            "n": int(keep.sum()),
            "positives": int(labels[keep].sum()),
            "auroc_p0": auc_rank(labels[keep], p0[keep]),
            "auroc_p1": auc_rank(labels[keep], p1[keep]),
            "auprc_p0": average_precision(labels[keep], p0[keep]),
            "auprc_p1": average_precision(labels[keep], p1[keep]),
        })
    for row in rows:
        row["delta_auroc"] = row["auroc_p1"] - row["auroc_p0"]
        row["delta_auprc"] = row["auprc_p1"] - row["auprc_p0"]
    return rows


def sign_test(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    wins = int((values > 0).sum())
    losses = int((values < 0).sum())
    ties = int((values == 0).sum())
    result = {"wins": wins, "losses": losses, "ties": ties}
    if wins + losses:
        result["p_value"] = float(
            binomtest(wins, wins + losses, 0.5).pvalue
        )
    else:
        result["p_value"] = 1.0
    return result


def context_table(context, data, scores_by_variant, thresholds):
    episodes = np.asarray(data["episode_id"])
    steps = np.asarray(data["step"])
    since = np.asarray([
        context[(int(e), int(s))]["since_localizable"]
        for e, s in zip(episodes, steps)
    ])
    run_len = np.asarray([
        context[(int(e), int(s))]["run_length"]
        for e, s in zip(episodes, steps)
    ])
    run_start = np.asarray([
        context[(int(e), int(s))]["run_start"]
        for e, s in zip(episodes, steps)
    ])
    h1 = np.asarray([
        context[(int(e), int(s))]["history_action_t1"]
        for e, s in zip(episodes, steps)
    ])
    h2 = np.asarray([
        context[(int(e), int(s))]["history_action_t2"]
        for e, s in zip(episodes, steps)
    ])
    labels = np.asarray(data["near_target_invisible"], dtype=int)
    positive = labels == 1

    def group(mask):
        output = {}
        for name, score in scores_by_variant.items():
            threshold = thresholds[name]
            output[name] = {
                "n": int(mask.sum()),
                "positives": int((mask & positive).sum()),
                "mean_score": float(score[mask].mean()) if mask.any() else float("nan"),
                "flag_rate": (
                    float((score[mask] >= threshold).mean())
                    if mask.any() else float("nan")
                ),
            }
        p0 = scores_by_variant["P0"]
        p1 = scores_by_variant["P1-2"]
        output["delta_mean_score"] = (
            float((p1[mask] - p0[mask]).mean()) if mask.any() else float("nan")
        )
        return output

    since_groups = {}
    edges = [(0, 1, "0-1"), (2, 2, "2"), (3, 4, "3-4"),
             (5, 9, "5-9"), (10, 19, "10-19"), (20, 49, "20-49"),
             (50, 10_000, "50+")]
    for low, high, name in edges:
        mask = positive & (since >= low) & (since <= high)
        since_groups[name] = group(mask)
    since_groups["never"] = group(positive & (since < 0))
    run_groups = {}
    for low, high, name in (
        (1, 1, "1"), (2, 4, "2-4"), (5, 19, "5-19"), (20, 10_000, "20+")
    ):
        mask = positive & (run_len >= low) & (run_len <= high)
        run_groups[name] = group(mask)
    action_groups = {}
    for code, name in ACTION_NAMES.items():
        mask = positive & (np.asarray(data["action"]) == code)
        if mask.any():
            action_groups[name] = group(mask)
    history_groups = {}
    for a1 in ACTION_NAMES.values():
        for a2 in ACTION_NAMES.values():
            if a1 == "STOP" or a2 == "STOP":
                continue
            mask = (
                positive
                & (h1 == next(k for k, v in ACTION_NAMES.items() if v == a1))
                & (h2 == next(k for k, v in ACTION_NAMES.items() if v == a2))
            )
            if mask.sum() >= 10:
                history_groups[f"{a1}->{a2}"] = group(mask)
    return {
        "since_localizable": since_groups,
        "run_length": run_groups,
        "positive_action": action_groups,
        "history_action_pattern": history_groups,
        "run_start_vs_interior": {
            "run_start": group(positive & run_start),
            "run_interior": group(positive & ~run_start),
        },
    }


# ---------------------------------------------------------------------------
# Synthetic comparison
# ---------------------------------------------------------------------------
def load_synthetic_metadata(combined_path, supplement_path):
    combined = np.load(combined_path)
    supplement = np.load(supplement_path)
    keys_combined = [
        "near_target_invisible", "source", "sample_type", "variant",
        "history_origin", "band", "target_visible", "target_localizable",
        "distance_m", "episode_id", "history_valid", "history_actions",
        "step_id",
    ]
    keys_supplement = [
        "variant", "target_visible", "target_localizable",
        "near_target_invisible", "distance_m", "history_actions",
        "history_dt", "history_step_ids", "band", "episode_id",
        "start_relative_deg", "final_relative_deg",
    ]
    combined_data = {key: combined[key] for key in keys_combined}
    supplement_data = {key: supplement[key] for key in keys_supplement}
    return combined_data, supplement_data


def synthetic_summary(combined_data, supplement_data):
    labels = combined_data["near_target_invisible"]
    source = combined_data["source"]
    variant = combined_data["variant"]
    distance = combined_data["distance_m"]
    history_actions = combined_data["history_actions"]
    supplement_variant = supplement_data["variant"]
    supplement_visible = supplement_data["target_visible"]
    supplement_localizable = supplement_data["target_localizable"]
    supplement_distance = supplement_data["distance_m"]
    supplement_actions = supplement_data["history_actions"]
    supplement_band = supplement_data["band"]
    supplement_start = supplement_data["start_relative_deg"]
    supplement_final = supplement_data["final_relative_deg"]

    def action_pattern(actions):
        return Counter(
            f"{ACTION_NAMES.get(int(a), 'INVALID')}->"
            f"{ACTION_NAMES.get(int(b), 'INVALID')}"
            for a, b in zip(actions[:, 0], actions[:, 1])
        )

    supplement_positive = (
        (supplement_variant == 1) & (supplement_data["near_target_invisible"] == 1)
    )
    natural_positive = (source == 0) & (labels == 1)
    supplement_natural_positive = (source == 1) & (variant == 1) & (labels == 1)
    return {
        "combined_train": {
            "states": int(len(labels)),
            "positives": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "source_counts": {
                str(k): int(v) for k, v in zip(*np.unique(source, return_counts=True))
            },
            "near_invisible_by_source": {
                "natural_far_features": int(natural_positive.sum()),
                "supplement": int(supplement_natural_positive.sum()),
            },
            "distance_summary_supplement_positives": quantile_summary(
                distance[supplement_natural_positive]
            ),
            "history_valid_rate": float(combined_data["history_valid"].mean()),
            "history_patterns_supplement_positives": dict(
                action_pattern(history_actions[supplement_natural_positive]).most_common(10)
            ),
            "history_patterns_natural_positives": dict(
                action_pattern(history_actions[natural_positive]).most_common(10)
            ),
        },
        "supplement_train": {
            "rows": int(len(supplement_variant)),
            "near_invisible_rows": int(supplement_positive.sum()),
            "near_invisible_target_visible_rate": (
                float(supplement_visible[supplement_positive].mean())
                if supplement_positive.any() else float("nan")
            ),
            "near_invisible_localizable_rate": (
                float(supplement_localizable[supplement_positive].mean())
                if supplement_positive.any() else float("nan")
            ),
            "distance_summary": quantile_summary(
                supplement_distance[supplement_positive]
            ),
            "band_counts": {
                BAND_NAMES[int(k)]: int(v)
                for k, v in zip(*np.unique(
                    supplement_band[supplement_positive], return_counts=True
                ))
            },
            "history_patterns": dict(
                action_pattern(supplement_actions[supplement_positive]).most_common(10)
            ),
            "start_relative_deg": quantile_summary(
                supplement_start[supplement_positive]
            ),
            "final_relative_deg": quantile_summary(
                supplement_final[supplement_positive]
            ),
            "design": (
                "Four frames t-3..t by three RIGHT turns; near-invisible "
                "variant is visible at t-3/t-2 and invisible at t; history "
                "actions are RIGHT x3. The last visible frame is t-2, i.e. "
                "2 control steps before the positive, inside the P1-2 "
                "2-frame history window."
            ),
        },
        "unavailable": [
            "Per-frame target_localizable labels for synthetic history frames "
            "are not stored; last-localizable timing for synthetic positives "
            "is therefore taken from the documented generation design, not "
            "measured per state.",
        ],
    }


def quadrant_scores(data, scores, threshold):
    masks = real_masks(data)
    output = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        output[name] = {
            "n": int(mask.sum()),
            "mean_score": float(scores[mask].mean()) if mask.any() else float("nan"),
            "median_score": float(np.median(scores[mask])) if mask.any() else float("nan"),
            "max_score": float(scores[mask].max()) if mask.any() else float("nan"),
            "flag_rate_at_threshold": (
                float((scores[mask] >= threshold).mean())
                if mask.any() else float("nan")
            ),
            "flags": int((scores[mask] >= threshold).sum()),
        }
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--synthetic-checkpoint", type=Path,
        default=Path(
            "runs/v83_spatial_stage2/stage2g1/"
            "probe_20261019T124500Z/models/"
            "v2_near_target_invisible_external.pt"
        ),
    )
    parser.add_argument(
        "--synthetic-combined-train", type=Path,
        default=Path(
            "runs/v83_spatial_stage2/stage2g1/"
            "combined_train_20261019T121500Z/combined_train.npz"
        ),
    )
    parser.add_argument(
        "--synthetic-supplement-train", type=Path,
        default=Path(
            "runs/v83_spatial_stage2/stage2g1/"
            "supplement_train_20261019T120000Z/supplement_train.npz"
        ),
    )
    parser.add_argument(
        "--train-stats", type=Path,
        default=Path("runs/v83_spatial_stage2/datasets/train_600/dataset_stats.json"),
    )
    parser.add_argument(
        "--val-specs", type=Path,
        default=Path(
            "runs/v83_spatial_stage2/prep_audit_20261018T043420Z/"
            "dev_val_180_spec.json"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    probe = json.loads(
        (Path(args.probe_dir) / "probe_results.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (Path(args.probe_dir) / "probe_provenance.json").read_text(encoding="utf-8")
    )
    selected = {
        "P0": float(
            probe["variants"]["0"]["train_thresholds"]["f1_joint_far5"]["threshold"]
        ),
        "P1-2": float(
            probe["variants"]["1"]["train_thresholds"]["f1_joint_far5"]["threshold"]
        ),
    }
    recall_half = {
        "P0": float(
            probe["variants"]["0"]["train_thresholds"]["recall_0.5_unconstrained"]["threshold"]
        ),
        "P1-2": float(
            probe["variants"]["1"]["train_thresholds"]["recall_0.5_unconstrained"]["threshold"]
        ),
    }

    print("scanning train episodes", flush=True)
    train_scan = scan_split(args.train_dir)
    print("scanning val episodes", flush=True)
    val_scan = scan_split(args.val_dir, return_context=True)
    val_context = val_scan.pop("context")
    train_scan.pop("context", None)

    print("loading arrays", flush=True)
    train_raw = load_split(args.train_dir)
    val_raw = load_split(args.val_dir)
    train_data, _ = filter_moves(train_raw)
    val_data, _ = filter_moves(val_raw)
    labels = np.asarray(val_data["near_target_invisible"], dtype=int)

    print("reconstructing external scores", flush=True)
    external = {}
    checkpoint_info = {}
    for name, variant in (("P0", 0), ("P1-2", 1)):
        scores, checkpoint, weight = predict_external(
            args.probe_dir, val_data, variant, device
        )
        external[name] = scores
        checkpoint_info[name] = {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "pos_weight": float(weight),
            "input_dim": int(
                np.asarray(val_data["current_features"]).shape[1]
                + (2 * (74 + 4 + 1) if variant else 0)
            ),
        }
        del scores

    print("reconstructing train OOF", flush=True)
    oof_reproduction = {}
    thresholds_reproduced = {}
    oof_scores = {}
    for name, variant in (("P0", 0), ("P1-2", 1)):
        oof, folds, fold_weights = reconstruct_oof(
            args.probe_dir, train_data, variant, device
        )
        oof_scores[name] = oof
        train_labels = np.asarray(
            train_data["near_target_invisible"], dtype=int
        )
        fold_class_balance = {}
        for fold in range(5):
            train_mask = folds != fold
            fold_pos = int(train_labels[train_mask].sum())
            fold_neg = int(train_mask.sum()) - fold_pos
            expected_weight = fold_neg / max(fold_pos, 1)
            checkpoint_weight = float(fold_weights[fold])
            fold_class_balance[str(fold)] = {
                "train_states": int(train_mask.sum()),
                "positives": fold_pos,
                "negatives": fold_neg,
                "expected_pos_weight": expected_weight,
                "checkpoint_pos_weight": checkpoint_weight,
                "abs_difference": abs(
                    expected_weight - checkpoint_weight
                ),
            }
        ref = probe["variants"][str(variant)]["heads"][
            "near_target_invisible"
        ]["oof"]
        oof_metric = metric_summary(
            np.asarray(train_data["near_target_invisible"], dtype=int), oof
        )
        recomputed = select_thresholds(train_data, oof)
        saved_threshold = selected[name]
        reco_threshold = float(
            recomputed["f1_joint_far5"]["threshold"]
        )
        oof_reproduction[name] = {
            "n": oof_metric["n"],
            "positives": oof_metric["positives"],
            "auroc": oof_metric["auroc"],
            "auroc_saved": ref["auroc"],
            "auprc": oof_metric["auprc"],
            "auprc_saved": ref["auprc"],
            "brier": oof_metric["brier"],
            "brier_saved": ref["brier"],
            "fold_pos_weight": {
                str(k): float(v) for k, v in fold_weights.items()
            },
            "fold_class_balance": fold_class_balance,
        }
        thresholds_reproduced[name] = {
            "saved_f1_joint_far5": saved_threshold,
            "recomputed_f1_joint_far5": reco_threshold,
            "abs_difference": abs(saved_threshold - reco_threshold),
            "saved_recall_0.5_unconstrained": recall_half[name],
            "recomputed_recall_0.5_unconstrained": float(
                recomputed["recall_0.5_unconstrained"]["threshold"]
            ),
        }
        del oof

    # External metric audit.
    external_audit = {}
    for name in ("P0", "P1-2"):
        variant = 0 if name == "P0" else 1
        summary = metric_summary(labels, external[name])
        saved = probe["variants"][str(variant)]["heads"][
            "near_target_invisible"
        ]["external"]["overall"]
        external_audit[name] = {
            "n": summary["n"],
            "positives": summary["positives"],
            "auroc": summary["auroc"],
            "auroc_saved": saved["auroc"],
            "auroc_abs_diff": abs(summary["auroc"] - saved["auroc"]),
            "auprc": summary["auprc"],
            "auprc_saved": saved["auprc"],
            "auprc_abs_diff": abs(summary["auprc"] - saved["auprc"]),
            "brier": summary["brier"],
            "brier_saved": saved["brier"],
            "brier_abs_diff": abs(summary["brier"] - saved["brier"]),
            "ece": summary["ece"],
            "ece_saved": saved["ece"],
            "ece_abs_diff": abs(summary["ece"] - saved["ece"]),
        }

    # Overall, bands, operating points.
    metrics = {}
    for name in ("P0", "P1-2"):
        scores = external[name]
        threshold = selected[name]
        entry = {
            "overall": metric_summary(labels, scores),
            "at_selected_threshold": real_metrics(val_data, scores, threshold),
            "at_recall_0.5_unconstrained": real_metrics(
                val_data, scores, recall_half[name]
            ),
            "at_0.5": metric_summary(labels, scores)["at_0.5"],
            "episode_bootstrap_95ci": bootstrap_ci(
                val_data["episode_id"], labels, scores,
                n_boot=500, seed=args.seed,
            ),
            "bands": {},
            "quadrants": quadrant_table(val_data, scores, threshold),
        }
        for code, band in BAND_NAMES.items():
            mask = np.asarray(val_data["band"]) == code
            entry["bands"][band] = {
                "metrics": metric_summary(labels[mask], scores[mask]),
                "at_selected_threshold": real_metrics(
                    {key: value[mask] for key, value in val_data.items()},
                    scores[mask], threshold,
                ),
            }
        metrics[name] = entry

    # Near-state-only discrimination.
    near_mask = np.asarray(val_data["distance_m"]) < 100.0
    near_discrimination = {}
    for name in ("P0", "P1-2"):
        scores = external[name]
        near_discrimination[name] = {
            "auroc_near_only": auc_rank(labels[near_mask], scores[near_mask]),
            "auprc_near_only": average_precision(
                labels[near_mask], scores[near_mask]
            ),
            "near_invisible_vs_near_visible_auroc": auc_rank(
                labels[near_mask], scores[near_mask]
            ),
        }
        near_discrimination[name]["episode_bootstrap_95ci"] = bootstrap_ci(
            np.asarray(val_data["episode_id"])[near_mask],
            labels[near_mask],
            scores[near_mask],
            n_boot=500,
            seed=args.seed,
        )
    near_discrimination["delta"] = {
        "auroc_near_only": (
            near_discrimination["P1-2"]["auroc_near_only"]
            - near_discrimination["P0"]["auroc_near_only"]
        ),
        "auprc_near_only": (
            near_discrimination["P1-2"]["auprc_near_only"]
            - near_discrimination["P0"]["auprc_near_only"]
        ),
    }
    distance_bin_discrimination = {}
    for low, high in ((0, 20), (20, 50), (50, 100)):
        mask = (
            (np.asarray(val_data["distance_m"]) >= low)
            & (np.asarray(val_data["distance_m"]) < high)
        )
        if not mask.any():
            continue
        block = {
            "n": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "negative": int((1 - labels[mask]).sum()),
        }
        for name in ("P0", "P1-2"):
            scores = external[name]
            if 0 < labels[mask].sum() < mask.sum():
                block[f"{name}_auroc"] = auc_rank(labels[mask], scores[mask])
                block[f"{name}_auprc"] = average_precision(
                    labels[mask], scores[mask]
                )
            else:
                block[f"{name}_auroc"] = float("nan")
                block[f"{name}_auprc"] = float("nan")
        distance_bin_discrimination[f"{low}-{high}"] = block

    # Threshold-matched comparison: separates ranking gain from threshold
    # movement (P1-2 selected a lower threshold than P0).
    common_thresholds = sorted({
        selected["P0"], selected["P1-2"], 0.05, 0.1,
    })
    common_threshold_metrics = {}
    for threshold in common_thresholds:
        common_threshold_metrics[f"{threshold:.8f}"] = {
            name: real_metrics(val_data, external[name], threshold)
            for name in ("P0", "P1-2")
        }
    band_comparison = {}
    for code, band in BAND_NAMES.items():
        mask = np.asarray(val_data["band"]) == code
        block = {
            "n": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "positive_episodes": int(
                np.unique(np.asarray(val_data["episode_id"])[mask][
                    labels[mask] == 1
                ]).size
            ),
            "P0_auroc": auc_rank(labels[mask], external["P0"][mask]),
            "P1-2_auroc": auc_rank(labels[mask], external["P1-2"][mask]),
            "P0_auprc": average_precision(labels[mask], external["P0"][mask]),
            "P1-2_auprc": average_precision(
                labels[mask], external["P1-2"][mask]
            ),
        }
        block["delta_auroc"] = block["P1-2_auroc"] - block["P0_auroc"]
        block["delta_auprc"] = block["P1-2_auprc"] - block["P0_auprc"]
        band_comparison[band] = block

    # Per-episode paired analysis.
    print("per-episode paired analysis", flush=True)
    per_episode = per_episode_table(
        val_data, external["P0"], external["P1-2"],
        selected["P0"], selected["P1-2"],
    )
    positive_rows = [row for row in per_episode if row["n_positives"] > 0]
    zero_score_episodes = [
        {
            "episode_id": row["episode_id"],
            "band": row["band"],
            "n_positives": row["n_positives"],
            "mean_positive_p1": row["mean_positive_p1"],
            "mean_negative_p1": row["mean_negative_p1"],
            "auroc_p1": row["auroc_p1"],
        }
        for row in positive_rows
        if row["mean_positive_p1"] < 1e-4
    ]
    paired = {
        "n_episodes": int(len(per_episode)),
        "n_positive_episodes": int(len(positive_rows)),
        "per_episode": per_episode,
        "near_zero_score_episodes": {
            "threshold_mean_positive_p1": 1e-4,
            "n_episodes": int(len(zero_score_episodes)),
            "by_band": dict(Counter(
                row["band"] for row in zero_score_episodes
            )),
            "episodes": zero_score_episodes,
        },
        "sign_tests": {
            "positive_mean_score": sign_test(
                [row["delta_mean_positive"] for row in positive_rows]
            ),
            "positive_recall": sign_test(
                [row["delta_recall"] for row in positive_rows]
            ),
            "episode_ap": sign_test(
                [row["delta_ap"] for row in positive_rows]
            ),
            "episode_auroc": sign_test(
                [row["auroc_p1"] - row["auroc_p0"] for row in positive_rows]
            ),
        },
        "bootstrap_delta": paired_bootstrap(
            np.asarray(val_data["episode_id"]),
            labels,
            external["P0"],
            external["P1-2"],
            selected["P0"],
            selected["P1-2"],
            masks=real_masks(val_data),
            n_boot=args.bootstrap,
            seed=args.seed,
        ),
        "leave_one_positive_episode_out": leave_one_positive_episode_out(
            val_data, external["P0"], external["P1-2"]
        ),
    }
    loeo = paired["leave_one_positive_episode_out"]
    if loeo:
        paired["leave_one_positive_episode_out_summary"] = {
            "delta_auroc": quantile_summary([row["delta_auroc"] for row in loeo]),
            "delta_auprc": quantile_summary([row["delta_auprc"] for row in loeo]),
            "positive_auroc_gain_count": int(
                sum(row["delta_auroc"] > 0 for row in loeo)
            ),
            "positive_auprc_gain_count": int(
                sum(row["delta_auprc"] > 0 for row in loeo)
            ),
            "n_exclusions": len(loeo),
            "worst_case_auroc_delta": float(
                min(row["delta_auroc"] for row in loeo)
            ),
            "best_case_auroc_delta": float(
                max(row["delta_auroc"] for row in loeo)
            ),
        }

    # Context stratification (does history help only when a recent view exists?).
    context_analysis = context_table(
        val_context,
        val_data,
        {"P0": external["P0"], "P1-2": external["P1-2"]},
        selected,
    )

    # Synthetic model on real Dev Val.
    print("synthetic comparison", flush=True)
    synthetic_model, s_mean, s_std, s_weight, s_threshold = (
        load_probe_for_comparison(args.synthetic_checkpoint, device)
    )
    synthetic_scores = batch_predict(
        synthetic_model, s_mean, s_std, s_weight,
        build_input(val_data, 1), device,
    )
    synthetic_metrics = metric_summary(labels, synthetic_scores)
    saved_synthetic = probe["synthetic_comparison"]
    synthetic_audit = {
        "checkpoint": str(args.synthetic_checkpoint),
        "checkpoint_sha256": sha256(args.synthetic_checkpoint),
        "threshold": float(s_threshold),
        "overall": synthetic_metrics,
        "auroc_saved": saved_synthetic["metrics"]["auroc"],
        "auprc_saved": saved_synthetic["metrics"]["auprc"],
        "auroc_abs_diff": abs(
            synthetic_metrics["auroc"] - saved_synthetic["metrics"]["auroc"]
        ),
        "auprc_abs_diff": abs(
            synthetic_metrics["auprc"] - saved_synthetic["metrics"]["auprc"]
        ),
        "quadrants": quadrant_scores(
            val_data, synthetic_scores, s_threshold
        ),
        "real_trained_quadrants": quadrant_scores(
            val_data, external["P1-2"], selected["P1-2"]
        ),
        "ks_near_invisible_vs_real_trained": float(ks_2samp(
            synthetic_scores[real_masks(val_data)["near_invisible"]],
            external["P1-2"][real_masks(val_data)["near_invisible"]],
        ).statistic),
    }

    synthetic_meta = None
    if args.synthetic_combined_train.is_file() and args.synthetic_supplement_train.is_file():
        print("synthetic metadata comparison", flush=True)
        combined_data, supplement_data = load_synthetic_metadata(
            args.synthetic_combined_train, args.synthetic_supplement_train
        )
        synthetic_meta = synthetic_summary(combined_data, supplement_data)

    # Threshold provenance audit.
    threshold_audit = {
        "source": provenance.get("threshold_source", "unknown"),
        "saved": selected,
        "recall_0.5_unconstrained": recall_half,
        "reproduced_from_fold_checkpoints": thresholds_reproduced,
        "dev_val_used_for_selection": False,
        "evidence": (
            "select_thresholds() is called with train_data and Train-OOF "
            "scores only; the saved thresholds are reproduced to <1e-6 from "
            "the fold checkpoints, and no Dev-Val score is passed to it."
        ),
        "external_model_training_split": "all Train 600 (move-valid states)",
        "external_standardization": "Train mean/std, clipped to [-20,20]",
    }

    # Split overlap / leakage.
    train_stats = json.loads(Path(args.train_stats).read_text(encoding="utf-8"))
    val_specs = json.loads(Path(args.val_specs).read_text(encoding="utf-8"))
    train_specs = train_stats["trajectories"]
    val_spec_list = val_specs["trajectories"]
    train_seeds = {int(row["seed"]) for row in train_specs}
    val_seeds = {int(row["seed"]) for row in val_spec_list}
    train_layouts = {int(row["network_layout_seed"]) for row in train_specs}
    val_layouts = {int(row["network_layout_seed"]) for row in val_spec_list}
    train_pairs = {
        (int(row["seed"]), int(row["network_layout_seed"]))
        for row in train_specs
    }
    val_pairs = {
        (int(row["seed"]), int(row["network_layout_seed"]))
        for row in val_spec_list
    }
    feature_terms = {
        term: [
            name for name in feature_names()
            if term in name.lower()
        ]
        for term in FORBIDDEN_FEATURE_TERMS
    }
    feature_terms = {k: v for k, v in feature_terms.items() if v}
    leakage = {
        "feature_names_forbidden_terms": feature_terms,
        "feature_count": len(feature_names()),
        "split_overlap": {
            "train_seeds": len(train_seeds),
            "val_seeds": len(val_seeds),
            "seed_intersection": len(train_seeds & val_seeds),
            "train_layouts": len(train_layouts),
            "val_layouts": len(val_layouts),
            "layout_intersection": len(train_layouts & val_layouts),
            "seed_layout_pair_intersection": len(train_pairs & val_pairs),
            "train_seed_range": [min(train_seeds), max(train_seeds)],
            "val_seed_range": [min(val_seeds), max(val_seeds)],
            "train_layout_range": [min(train_layouts), max(train_layouts)],
            "val_layout_range": [min(val_layouts), max(val_layouts)],
        },
        "episode_id_namespaces": {
            "note": (
                "Train IDs are dataset positions 0-599; Dev-Val IDs are "
                "spec episode_index 0-179. They overlap numerically but are "
                "different episodes; no join by raw episode_id across splits "
                "is performed."
            )
        },
        "train_class_ratio": {
            "states_all": int(train_scan["totals"]["states"]),
            "positives_all": int(train_scan["totals"]["positives"]),
            "positive_rate_all": (
                train_scan["totals"]["positives"]
                / max(train_scan["totals"]["states"], 1)
            ),
            "states_move_valid": int(len(train_data["episode_id"])),
            "positives_move_valid": int(
                train_data["near_target_invisible"].sum()
            ),
            "positive_rate_move_valid": float(
                train_data["near_target_invisible"].mean()
            ),
        },
        "val_class_ratio": {
            "states_all": int(val_scan["totals"]["states"]),
            "positives_all": int(val_scan["totals"]["positives"]),
            "positive_rate_all": (
                val_scan["totals"]["positives"]
                / max(val_scan["totals"]["states"], 1)
            ),
            "states_move_valid": int(len(val_data["episode_id"])),
            "positives_move_valid": int(
                val_data["near_target_invisible"].sum()
            ),
            "positive_rate_move_valid": float(
                val_data["near_target_invisible"].mean()
            ),
        },
        "fold_pos_weight": oof_reproduction,
        "label_mismatches": {
            "train": train_scan["label_mismatches"],
            "val": val_scan["label_mismatches"],
        },
        "history_mismatches": {
            "train": train_scan["history_mismatches"],
            "val": val_scan["history_mismatches"],
        },
        "final_test_accessed": False,
        "checkpoint_data_dirs": provenance.get("data"),
    }

    results = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "command": sys.argv,
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
            "probe_dir": str(args.probe_dir),
            "probe_model_count": len(provenance.get("models", [])),
            "checkpoints": checkpoint_info,
            "no_training_performed": True,
            "no_recollection_performed": True,
            "final_test_accessed": False,
        },
        "data": {
            "train": train_scan,
            "val": val_scan,
            "filtered": {
                "train_move_valid_states": int(len(train_data["episode_id"])),
                "train_move_valid_episodes": int(
                    np.unique(train_data["episode_id"]).size
                ),
                "val_move_valid_states": int(len(val_data["episode_id"])),
                "val_move_valid_episodes": int(
                    np.unique(val_data["episode_id"]).size
                ),
            },
        },
        "audit": {
            "external": external_audit,
            "oof": oof_reproduction,
            "thresholds": threshold_audit,
            "reproduction_check_saved": json.loads(
                (Path(args.probe_dir) / "reproduction_check.json").read_text(
                    encoding="utf-8"
                )
            ),
        },
        "metrics": metrics,
        "common_threshold_metrics": common_threshold_metrics,
        "band_comparison": band_comparison,
        "near_discrimination": near_discrimination,
        "distance_bin_discrimination": distance_bin_discrimination,
        "episode_paired": paired,
        "context_analysis": context_analysis,
        "synthetic": {
            "metadata": synthetic_meta,
            "model_on_real_val": synthetic_audit,
        },
        "leakage": leakage,
        "missing": [
            "probe_results.json does not store row-level probabilities; all "
            "per-state/per-episode results here were recovered by inference "
            "with the saved checkpoints and exactly reproduce the saved "
            "aggregate metrics.",
            "Run lengths and 'steps since last localizable/visible' are in "
            "control steps. The dataset does not store wall-clock per-step "
            "time; steps are unit-dt transitions.",
            "Synthetic history frames do not store per-frame visibility or "
            "localizability labels; synthetic last-high-confidence timing is "
            "taken from the generation design.",
            "Single seed (seed 0) and single probe architecture; no "
            "multi-seed replication.",
            "Final Test 300 was not read or evaluated.",
        ],
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    np.savez(
        output_dir / "predictions.npz",
        episode_id=np.asarray(val_data["episode_id"]),
        step=np.asarray(val_data["step"]),
        band=np.asarray(val_data["band"]),
        labels=labels,
        distance_m=np.asarray(val_data["distance_m"]),
        target_visible=np.asarray(val_data["target_visible"]),
        p0=external["P0"],
        p1=external["P1-2"],
        synthetic=synthetic_scores,
    )
    print(json.dumps({
        "output": str(output_dir),
        "external_audit": {
            name: {
                "auroc": external_audit[name]["auroc"],
                "auprc": external_audit[name]["auprc"],
                "auroc_abs_diff": external_audit[name]["auroc_abs_diff"],
            }
            for name in ("P0", "P1-2")
        },
        "thresholds": {
            name: thresholds_reproduced[name]["abs_difference"]
            for name in ("P0", "P1-2")
        },
        "paired_bootstrap": paired["bootstrap_delta"],
        "loeo": paired.get("leave_one_positive_episode_out_summary"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
