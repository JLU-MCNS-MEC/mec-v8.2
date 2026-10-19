"""Stage-2g light target-presence / localizability probes.

Two input modes:

  token  : frozen current RGB tokens, goal tokens, Goal-Current heatmaps and
           parameter-free baseline/coherent maps from Stage-2g features.
  scalar : observable scalar features from the Stage-2f Dev-Val M0 rollout
           states (a closed-loop failure-regime distribution check).

Labels are offline only:

  target_visible      = occlusion-aware visible_pixels >= 1
  target_localizable  = target_visible AND goal_half_size_px >= 10 AND
                        visible_pixels >= 16

For the Stage-2f scalar mode, `visible_pixels` is unavailable; localizable is
`goal_visible AND goal_half_size_px >= 10` (occlusion differs by ~0.6% in the
shard audit).

Models are linear / one-hidden-layer MLP probes on frozen features. Splits are
grouped by episode (5-fold OOF), so adjacent frames of an episode are always on
the same side of a split. Class weighting is applied only through the loss and
is undone by the prior-correction formula before calibration metrics.

No MEC training, no checkpoint modification, Final Test 300 not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BAND_NAMES = {0: "short", 1: "medium", 2: "long"}
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)
SCALAR_FORBIDDEN_EXACT = {
    "episode_index", "step", "seed", "band", "trajectory",
    "goal_visible", "goal_px", "goal_py", "goal_half_size_px",
    "distance_m", "heading_error_deg", "oracle_move_action", "stop_ready",
    "coherent_peak_error_px", "baseline_peak_error_px",
    "logged_stop", "logged_arrival_probability",
    "owner_move_action", "owner_arrival_probability",
    "executed_action", "previous_action",
    "m0_arrival_probability", "m0_move_top1", "m0_move_margin",
    "m0_move_action", "movement_label",
}
SCALAR_FORBIDDEN_TERMS = (
    "visible", "distance", "heading", "oracle", "half_size",
    "goal_px", "goal_py", "peak_error", "movement_label",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc_rank(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    cum_tp = np.cumsum(labels)
    cum_fp = np.cumsum(1 - labels)
    ap = 0.0
    prev_recall = 0.0
    start = 0
    n = len(scores)
    while start < n:
        end = start + 1
        while end < n and scores[end] == scores[start]:
            end += 1
        tp = float(cum_tp[end - 1])
        fp = float(cum_fp[end - 1])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        start = end
    return float(ap)


def brier_score(labels, probabilities):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - labels) ** 2))


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(confidence - accuracy)
    return float(ece)


def confusion(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "negative_recall": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
    }


def negative_operating_points(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    negative = 1 - labels
    n_neg = int(negative.sum())
    if n_neg == 0:
        return {}
    unique = np.unique(probabilities)
    # Coarse grid is enough for the diagnostic and avoids 55k full passes.
    grid = np.unique(np.quantile(
        probabilities, np.linspace(0.0, 1.0, 201)
    ))
    recall_points = {}
    for target in (0.3, 0.5, 0.7):
        chosen = None
        for threshold in grid:
            predicted = probabilities <= threshold
            tp = int((predicted & (negative == 1)).sum())
            fp = int((predicted & (labels == 1)).sum())
            recall = tp / n_neg
            if recall >= target:
                precision = tp / max(tp + fp, 1)
                fpr = fp / max(int(labels.sum()), 1)
                chosen = {
                    "threshold": float(threshold),
                    "recall": float(recall),
                    "precision": float(precision),
                    "false_positive_rate": float(fpr),
                    "flagged": int(tp + fp),
                }
                break
        recall_points[str(target)] = chosen
    precision_points = {}
    for target in (0.5, 0.8, 0.9):
        chosen = None
        for threshold in grid[::-1]:
            predicted = probabilities <= threshold
            tp = int((predicted & (negative == 1)).sum())
            fp = int((predicted & (labels == 1)).sum())
            precision = tp / max(tp + fp, 1)
            if precision >= target and tp > 0:
                chosen = {
                    "threshold": float(threshold),
                    "recall": float(tp / n_neg),
                    "precision": float(precision),
                    "false_positive_rate": float(
                        fp / max(int(labels.sum()), 1)
                    ),
                    "flagged": int(tp + fp),
                }
                break
        precision_points[str(target)] = chosen
    return {
        "at_negative_recall": recall_points,
        "at_negative_precision": precision_points,
    }


def metric_block(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    values = {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negative_rate": float(1.0 - labels.mean()) if len(labels) else float("nan"),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "auroc": auc_rank(labels, probabilities),
        "auprc": average_precision(labels, probabilities),
        "auprc_negative": average_precision(1 - labels, 1.0 - probabilities),
        "brier": brier_score(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "mean_probability": float(probabilities.mean()) if len(labels) else float("nan"),
        "median_probability": float(np.median(probabilities)) if len(labels) else float("nan"),
        "confusions": {
            str(threshold): confusion(labels, probabilities, threshold)
            for threshold in THRESHOLDS
        },
    }
    values["balanced_accuracy_at_0.5"] = values["confusions"]["0.5"][
        "balanced_accuracy"
    ]
    values["negative_operating_points"] = negative_operating_points(
        labels, probabilities
    )
    return values


def bootstrap_ci(episodes, labels, probabilities, n_boot=300, seed=0,
                 metrics=("auroc", "auprc", "brier")):
    rng = np.random.default_rng(seed)
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
        labels_b = labels[indices]
        probs_b = probabilities[indices]
        if labels_b.min() == labels_b.max():
            continue
        draws["auroc"].append(auc_rank(labels_b, probs_b))
        draws["auprc"].append(average_precision(labels_b, probs_b))
        draws["brier"].append(brier_score(labels_b, probs_b))
    result = {}
    for name in metrics:
        values = np.asarray(draws[name], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            result[name] = {
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "n_boot": int(len(values)),
            }
        else:
            result[name] = {"low": float("nan"), "high": float("nan"),
                            "n_boot": 0}
    return result


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_token_split(features_dir, split):
    path = Path(features_dir) / f"features_{split}.npz"
    data = np.load(path)
    return {
        "features": data["features"].astype(np.float32),
        "episode_id": data["episode_id"].astype(np.int64),
        "step_id": data["step_id"].astype(np.int64),
        "band": data["band"].astype(np.int64),
        "target_visible": data["target_visible"].astype(np.int64),
        "target_localizable": data["target_localizable"].astype(np.int64),
        "distance_m": data["distance_m"].astype(np.float32),
        "goal_half_size_px": data["goal_half_size_px"].astype(np.float32),
        "sample_type": data["sample_type"].astype(np.int64),
        "feature_names": data["feature_names"].tolist(),
    }


def scalar_feature_columns(columns):
    selected = []
    for name in columns:
        if name in SCALAR_FORBIDDEN_EXACT:
            continue
        lowered = name.lower()
        if any(term in lowered for term in SCALAR_FORBIDDEN_TERMS):
            continue
        selected.append(name)
    return selected


def load_scalar_split(features_csv, trajectory="M0"):
    frame = pd.read_csv(features_csv)
    frame = frame[frame["trajectory"] == trajectory].reset_index(drop=True)
    columns = scalar_feature_columns(frame.columns)
    features = frame[columns].to_numpy(dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    visible = (frame["goal_visible"].to_numpy() > 0.5).astype(np.int64)
    half = frame["goal_half_size_px"].to_numpy(dtype=np.float32)
    localizable = (
        (visible == 1) & np.isfinite(half) & (half >= 10.0)
    ).astype(np.int64)
    return {
        "features": features,
        "episode_id": frame["episode_index"].to_numpy(dtype=np.int64),
        "step_id": frame["step"].to_numpy(dtype=np.int64),
        "band": frame["band"].map(
            {"short": 0, "medium": 1, "long": 2}
        ).to_numpy(dtype=np.int64),
        "target_visible": visible,
        "target_localizable": localizable,
        "distance_m": frame["distance_m"].to_numpy(dtype=np.float32),
        "goal_half_size_px": half,
        "sample_type": np.zeros(len(frame), dtype=np.int64),
        "feature_names": columns,
    }


# ---------------------------------------------------------------------------
# Episode folds and training
# ---------------------------------------------------------------------------

def assign_episode_folds(episodes, bands, n_folds, seed):
    rng = np.random.default_rng(int(seed))
    folds = np.full(len(episodes), -1, dtype=np.int64)
    episode_band = {}
    for episode, band in zip(episodes.tolist(), bands.tolist()):
        episode_band.setdefault(int(episode), int(band))
    for band in sorted(set(episode_band.values())):
        members = [ep for ep, b in episode_band.items() if b == band]
        rng.shuffle(members)
        for index, episode in enumerate(members):
            folds[episodes == episode] = index % int(n_folds)
    if np.any(folds < 0):
        raise RuntimeError("Some states did not receive a fold")
    return folds


def standardize(train_x, valid_x):
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    np.clip(train_x, -20.0, 20.0, out=train_x)
    np.clip(valid_x, -20.0, 20.0, out=valid_x)
    return train_x, valid_x


def build_model(kind, dim, hidden, seed):
    torch.manual_seed(int(seed))
    if kind == "linear":
        return nn.Linear(dim, 1)
    if kind == "mlp":
        return nn.Sequential(
            nn.Linear(dim, int(hidden)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(hidden), 1),
        )
    raise ValueError(kind)


def train_and_predict(
    train_x, train_y, valid_x, kind, hidden, epochs, batch_size,
    learning_rate, weight_decay, pos_weight, seed, device,
):
    train_x = torch.from_numpy(train_x).to(device)
    train_y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    valid_x = torch.from_numpy(valid_x).to(device)
    model = build_model(kind, train_x.shape[1], hidden, seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(
            [float(pos_weight)], device=device
        )
    )
    n = train_x.shape[0]
    batch_size = int(batch_size) if int(batch_size) > 0 else n
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    last_loss = float("nan")
    for _epoch in range(int(epochs)):
        model.train()
        permutation = torch.randperm(n, generator=generator).to(device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            logits = model(train_x[indices]).squeeze(1)
            loss = loss_function(logits, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        last_loss = epoch_loss / max(n, 1)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, valid_x.shape[0], 16384):
            logits = model(valid_x[start:start + 16384]).squeeze(1)
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(predictions).astype(np.float64)
    return probabilities, {"final_train_loss": float(last_loss)}


def prior_correct(probabilities, pos_weight):
    if pos_weight is None:
        return np.asarray(probabilities, dtype=float)
    w = float(pos_weight)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    odds = p / (1.0 - p) / w
    return odds / (1.0 + odds)


def model_specs(pos_weight, include_plain=True):
    specs = []
    if include_plain:
        specs.append({"name": "linear_plain", "kind": "linear",
                      "weighted": False})
    specs.append({"name": "linear_balanced", "kind": "linear",
                  "weighted": True})
    specs.append({"name": "mlp_balanced", "kind": "mlp", "weighted": True})
    return specs


# ---------------------------------------------------------------------------
# Evaluation and subsets
# ---------------------------------------------------------------------------

def subset_masks(data):
    distance = data["distance_m"]
    visible = data["target_visible"]
    localizable = data["target_localizable"]
    return {
        "near_invisible": (distance < 100.0) & (visible == 0),
        "far_visible": (distance >= 100.0) & (visible == 1),
        "far_not_localizable": (distance >= 100.0) & (localizable == 0),
        "far_invisible": (distance >= 100.0) & (visible == 0),
        "near_localizable": (distance < 100.0) & (localizable == 1),
    }


def subset_report(data, probabilities, head):
    masks = subset_masks(data)
    labels = data[head]
    report = {}
    for name, mask in masks.items():
        if not np.any(mask):
            report[name] = {"n": 0}
            continue
        y = labels[mask]
        p = probabilities[mask]
        entry = {
            "n": int(mask.sum()),
            "positive_rate": float(y.mean()),
            "mean_probability": float(p.mean()),
            "median_probability": float(np.median(p)),
            "fraction_ge_0.5": float((p >= 0.5).mean()),
        }
        if len(np.unique(y)) == 2:
            entry["auroc"] = auc_rank(y, p)
            entry["auprc"] = average_precision(y, p)
        report[name] = entry
    return report


def combined_subset_auc(data, probabilities, name_a, name_b):
    masks = subset_masks(data)
    mask_a, mask_b = masks[name_a], masks[name_b]
    mask = mask_a | mask_b
    if not np.any(mask_a) or not np.any(mask_b):
        return float("nan")
    # Label 1 for the far-visible regime, 0 for the near-invisible regime.
    y = np.concatenate([
        np.zeros(int(mask_a.sum()), dtype=int),
        np.ones(int(mask_b.sum()), dtype=int),
    ])
    p = np.concatenate([probabilities[mask_a], probabilities[mask_b]])
    return auc_rank(y, p)


def evaluate_predictions(data, probabilities, episodes, head, bootstrap):
    result = {
        "overall": metric_block(data[head], probabilities),
        "bands": {},
        "subsets": subset_report(data, probabilities, head),
        "subset_auc_far_visible_vs_near_invisible": combined_subset_auc(
            data, probabilities, "near_invisible", "far_visible"
        ),
        "subset_auc_far_invisible_vs_near_invisible": combined_subset_auc(
            data, probabilities, "near_invisible", "far_invisible"
        ),
    }
    for code, name in BAND_NAMES.items():
        mask = data["band"] == code
        if np.any(mask):
            result["bands"][name] = metric_block(
                data[head][mask], probabilities[mask]
            )
    if bootstrap:
        result["overall"]["episode_bootstrap_95ci"] = bootstrap_ci(
            episodes, data[head], probabilities, n_boot=bootstrap
        )
    return result


def _pos_weight(labels):
    positives = float(np.asarray(labels).sum())
    negatives = float(len(labels) - positives)
    return negatives / max(positives, 1.0)


def run_mode(data, args, device, label_prefix):
    labels = ("target_visible", "target_localizable")
    specs = model_specs(0.0)
    folds = assign_episode_folds(
        data["episode_id"], data["band"], args.folds, args.seed
    )
    output = {
        "samples": int(len(data["episode_id"])),
        "episodes": int(np.unique(data["episode_id"]).size),
        "feature_dim": int(data["features"].shape[1]),
        "feature_names": data["feature_names"],
        "folds": {
            str(fold): int((folds == fold).sum())
            for fold in sorted(set(folds.tolist()))
        },
        "heads": {},
    }
    for head in labels:
        output["heads"][head] = {
            "positive_rate": float(data[head].mean()),
            "pos_weight": _pos_weight(data[head]),
            "models": {},
        }
    predictions = {
        (head, spec["name"]): np.full(len(data["episode_id"]), np.nan)
        for head in labels for spec in specs
    }
    fold_info = {
        (head, spec["name"]): [] for head in labels for spec in specs
    }
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        print(
            f"  [{label_prefix}] fold {fold} "
            f"train={int(train_mask.sum())} valid={int(valid_mask.sum())}",
            flush=True,
        )
        train_x, valid_x = standardize(
            data["features"][train_mask], data["features"][valid_mask]
        )
        for head in labels:
            train_y = data[head][train_mask]
            for spec in specs:
                pos_weight = _pos_weight(train_y) if spec["weighted"] else None
                probabilities, info = train_and_predict(
                    train_x, train_y, valid_x, spec["kind"], args.hidden,
                    args.epochs, args.batch_size, args.lr,
                    args.weight_decay, pos_weight,
                    args.seed + int(fold), device,
                )
                probabilities = prior_correct(probabilities, pos_weight)
                predictions[(head, spec["name"])][valid_mask] = probabilities
                fold_info[(head, spec["name"])].append({
                    "fold": int(fold),
                    "train": int(train_mask.sum()),
                    "valid": int(valid_mask.sum()),
                    "pos_weight": pos_weight,
                    **info,
                })
    for head in labels:
        for spec in specs:
            key = (head, spec["name"])
            oof = predictions[key]
            if np.any(~np.isfinite(oof)):
                raise RuntimeError(f"Missing OOF predictions for {key}")
            metrics = evaluate_predictions(
                data, oof, data["episode_id"], head, args.bootstrap
            )
            metrics["folds"] = fold_info[key]
            output["heads"][head]["models"][spec["name"]] = {
                "cv": metrics,
            }
    return output, predictions, folds


def run_external_all(train_data, valid_data, args, device):
    labels = ("target_visible", "target_localizable")
    specs = model_specs(0.0)
    train_x, valid_x = standardize(
        train_data["features"], valid_data["features"]
    )
    output = {"heads": {}}
    predictions = {}
    for head in labels:
        output["heads"][head] = {
            "positive_rate": float(valid_data[head].mean()),
            "models": {},
        }
        train_y = train_data[head]
        for spec in specs:
            pos_weight = _pos_weight(train_y) if spec["weighted"] else None
            print(f"  [external] {head} {spec['name']}", flush=True)
            probabilities, info = train_and_predict(
                train_x, train_y, valid_x, spec["kind"], args.hidden,
                args.epochs, args.batch_size, args.lr, args.weight_decay,
                pos_weight, args.seed + 1000, device,
            )
            probabilities = prior_correct(probabilities, pos_weight)
            result = evaluate_predictions(
                valid_data, probabilities, valid_data["episode_id"], head,
                args.bootstrap,
            )
            result["train_info"] = info
            result["pos_weight"] = pos_weight
            output["heads"][head]["models"][spec["name"]] = result
            predictions[(head, spec["name"])] = probabilities
    return output, predictions


def write_predictions(path, data, probabilities, folds, head):
    np.savez_compressed(
        path,
        episode_id=data["episode_id"],
        step_id=data["step_id"],
        band=data["band"],
        target_visible=data["target_visible"],
        target_localizable=data["target_localizable"],
        distance_m=data["distance_m"],
        goal_half_size_px=data["goal_half_size_px"],
        sample_type=data["sample_type"],
        fold=folds,
        probability=probabilities,
    )


def render_tables(results):
    lines = ["# Stage-2g probe tables", ""]
    for mode in ("token", "scalar"):
        if mode not in results:
            continue
        mode_result = results[mode]
        lines += [f"## {mode} mode", ""]
        lines += [
            f"- Samples: {mode_result['samples']}, episodes: "
            f"{mode_result['episodes']}, feature dim: "
            f"{mode_result['feature_dim']}"
        ]
        lines += [""]
        for head, head_result in mode_result["heads"].items():
            lines += [f"### {head}", ""]
            lines += [
                "| Model | AUROC | AUPRC | Brier | ECE | P@0.5 | R@0.5 | F1@0.5 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for model_name, model_result in head_result["models"].items():
                overall = model_result["cv"]["overall"]
                confusion = overall["confusions"]["0.5"]
                lines.append(
                    f"| {model_name} | {overall['auroc']:.3f} | "
                    f"{overall['auprc']:.3f} | {overall['brier']:.3f} | "
                    f"{overall['ece']:.3f} | "
                    f"{confusion['precision']:.3f} | "
                    f"{confusion['recall']:.3f} | "
                    f"{confusion['f1']:.3f} |"
                )
            lines += [""]
            lines += [
                "| Model | Band | AUROC | AUPRC | Brier | ECE | P@0.5 | R@0.5 |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            for model_name, model_result in head_result["models"].items():
                for band, band_result in model_result["cv"]["bands"].items():
                    confusion = band_result["confusions"]["0.5"]
                    lines.append(
                        f"| {model_name} | {band} | "
                        f"{band_result['auroc']:.3f} | "
                        f"{band_result['auprc']:.3f} | "
                        f"{band_result['brier']:.3f} | "
                        f"{band_result['ece']:.3f} | "
                        f"{confusion['precision']:.3f} | "
                        f"{confusion['recall']:.3f} |"
                    )
            lines += [""]
            lines += [
                "| Model | Subset | n | Positive rate | Mean p | Median p | p>=0.5 | AUROC | AUPRC |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for model_name, model_result in head_result["models"].items():
                for subset, entry in model_result["cv"]["subsets"].items():
                    lines.append(
                        f"| {model_name} | {subset} | {entry.get('n', 0)} | "
                        f"{entry.get('positive_rate', float('nan')):.3f} | "
                        f"{entry.get('mean_probability', float('nan')):.3f} | "
                        f"{entry.get('median_probability', float('nan')):.3f} | "
                        f"{entry.get('fraction_ge_0.5', float('nan')):.3f} | "
                        f"{entry.get('auroc', float('nan')):.3f} | "
                        f"{entry.get('auprc', float('nan')):.3f} |"
                    )
            lines += [""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["token", "scalar", "both"],
                        default="both")
    parser.add_argument("--features-dir", type=Path, default=Path(
        "runs/v83_spatial_stage2/stage2g/features_20261019T120000Z"
    ))
    parser.add_argument("--stage2f-features", type=Path, default=Path(
        "runs/v83_spatial_stage2/stage2f/"
        "collect_20261019T024000Z/features.csv"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if args.mode in ("token", "both"):
        train_data = load_token_split(args.features_dir, "train")
        valid_data = load_token_split(args.features_dir, "val")
        print("token train", train_data["features"].shape,
              "valid", valid_data["features"].shape, flush=True)
        token_result, token_predictions, token_folds = run_mode(
            train_data, args, device, "token"
        )
        external, external_predictions = run_external_all(
            train_data, valid_data, args, device
        )
        token_result["external_val"] = external
        results["token"] = token_result
        for (head, spec_name), oof in token_predictions.items():
            write_predictions(
                output_dir / f"token_oof_{head}_{spec_name}.npz",
                train_data, oof, token_folds, head,
            )
        for (head, spec_name), probabilities in external_predictions.items():
            write_predictions(
                output_dir / f"token_val_{head}_{spec_name}.npz",
                valid_data, probabilities,
                np.full(len(probabilities), -1), head,
            )
        # Fold assignment for the validation split is not used; -1 marks it.

    if args.mode in ("scalar", "both"):
        scalar_data = load_scalar_split(args.stage2f_features)
        print("scalar", scalar_data["features"].shape, flush=True)
        scalar_result, scalar_predictions, scalar_folds = run_mode(
            scalar_data, args, device, "scalar"
        )
        results["scalar"] = scalar_result
        for (head, spec_name), oof in scalar_predictions.items():
            write_predictions(
                output_dir / f"scalar_oof_{head}_{spec_name}.npz",
                scalar_data, oof, scalar_folds, head,
            )

    (output_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "tables.md").write_text(
        render_tables(results), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        "modes": list(results),
        "token_samples": results.get("token", {}).get("samples"),
        "scalar_samples": results.get("scalar", {}).get("samples"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
