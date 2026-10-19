"""Stage-2g.1 near-target probe: P0 current vs P1 short history.

P0: current 2,858 observable features.
P1-k: current + k previous frames (k=1,2,3), each with 74 compact visual
features, 4-way action one-hot and dt, plus no true distance/visibility.

Episode-level 5-fold OOF on Train 600 and external validation on Dev Val 180.
All metrics are computed on the common valid-history subset. The trigger
operating point is selected on Train OOF only; Dev Val is held out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g_probe import (  # noqa: E402
    auc_rank,
    average_precision,
    bootstrap_ci,
    build_model,
    confusion,
    expected_calibration_error,
    metric_block,
    assign_episode_folds,
    prior_correct,
    standardize,
    train_and_predict,
)
from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    goal_indices,
    one_hot_actions,
)

HEADS = ("target_visible", "target_localizable", "near_target_invisible")
HEAD_SHORT = {
    "target_visible": "visible",
    "target_localizable": "localizable",
    "near_target_invisible": "near_invisible",
}
BAND_NAMES = {0: "short", 1: "medium", 2: "long"}
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7)
PER_STATE = (
    "current_features", "history_features", "history_actions", "history_dt",
    "history_origin", "episode_id", "step_id", "band", "target_visible",
    "target_localizable", "near_target_invisible", "distance_m",
    "goal_half_size_px", "source", "variant", "pair_id", "sample_type",
)


def load_combined(path):
    z = np.load(path)
    valid = z["history_valid"] > 0
    data = {key: z[key][valid] for key in PER_STATE}
    data["feature_names"] = [str(x) for x in z["feature_names"]]
    data["compact_feature_names"] = [
        str(x) for x in z["compact_feature_names"]
    ]
    return data


def build_input(data, variant):
    if variant == 0:
        return data["current_features"].astype(np.float32)
    parts = [data["current_features"].astype(np.float32)]
    for frame in range(int(variant)):
        parts.append(data["history_features"][:, frame, :])
        parts.append(one_hot_actions(data["history_actions"][:, frame]))
        parts.append(data["history_dt"][:, frame:frame + 1])
    return np.concatenate(parts, axis=1)


def pos_weight(labels):
    labels = np.asarray(labels)
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    return negatives / max(positives, 1.0)


def score_for_head(probabilities, head):
    # Higher score always means "flag for near-target recovery".
    if head == "near_target_invisible":
        return np.asarray(probabilities, dtype=float)
    return 1.0 - np.asarray(probabilities, dtype=float)


def subset_masks(data):
    source = data["source"]
    variant = data["variant"]
    distance = data["distance_m"]
    visible = data["target_visible"]
    localizable = data["target_localizable"]
    near = data["near_target_invisible"]
    return {
        "near_visible": (source == 1) & (variant == 0),
        "near_localizable": (
            (source == 1) & (variant == 0) & (localizable == 1)
        ),
        "near_invisible": near == 1,
        "near_supplement_invisible": (source == 1) & (variant == 1),
        "far_visible": (
            (source == 0) & (distance >= 100.0) & (visible == 1)
        ),
        "far_invisible": (
            (source == 0) & (distance >= 100.0) & (visible == 0)
        ),
    }


def trigger_metrics(data, score, threshold):
    flags = score >= float(threshold)
    masks = subset_masks(data)
    near = masks["near_invisible"]
    far_visible = masks["far_visible"]
    far_invisible = masks["far_invisible"]
    near_visible = masks["near_visible"]
    n_flags = int(flags.sum())
    tp = int((flags & near).sum())
    fv = int((flags & far_visible).sum())
    fi = int((flags & far_invisible).sum())
    nv = int((flags & near_visible).sum())
    return {
        "threshold": float(threshold),
        "flags": n_flags,
        "near_target_invisible_precision": tp / max(n_flags, 1),
        "near_target_invisible_recall": tp / max(int(near.sum()), 1),
        "near_target_invisible_flags": tp,
        "far_visible_false_trigger_rate": fv / max(int(far_visible.sum()), 1),
        "far_visible_flags": fv,
        "far_invisible_false_trigger_rate": fi / max(int(far_invisible.sum()), 1),
        "far_invisible_flags": fi,
        "near_visible_false_trigger_rate": nv / max(int(near_visible.sum()), 1),
        "near_visible_flags": nv,
        "flag_composition": {
            "near_target_invisible": tp / max(n_flags, 1),
            "far_visible": fv / max(n_flags, 1),
            "far_invisible": fi / max(n_flags, 1),
            "near_visible": nv / max(n_flags, 1),
            "other": max(
                n_flags - tp - fv - fi - nv, 0
            ) / max(n_flags, 1),
        },
    }


def threshold_grid(score):
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, 401))
    return np.unique(np.concatenate([quantiles, [0.0, 1.0]]))


def select_thresholds(data, score):
    candidates = []
    for threshold in threshold_grid(score):
        metrics = trigger_metrics(data, score, threshold)
        candidates.append(metrics)
    # For score >= threshold, increasing the threshold lowers recall. The
    # highest threshold that still meets the recall target gives the best
    # precision at that recall.
    by_recall = None
    for entry in sorted(
        candidates, key=lambda x: x["threshold"], reverse=True
    ):
        if entry["near_target_invisible_recall"] >= 0.5:
            by_recall = entry
            break
    # For score >= threshold, increasing the threshold raises precision. Keep
    # the highest threshold with precision >= 0.8 and at least one flag.
    by_precision = None
    for entry in sorted(candidates, key=lambda x: x["threshold"]):
        if (
            entry["flags"] > 0
            and entry["near_target_invisible_precision"] >= 0.8
        ):
            by_precision = entry
    by_f1 = None
    for entry in candidates:
        if entry["flags"] == 0:
            continue
        if entry["far_visible_false_trigger_rate"] > 0.05:
            continue
        precision = entry["near_target_invisible_precision"]
        recall = entry["near_target_invisible_recall"]
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        entry = dict(entry)
        entry["f1"] = f1
        if by_f1 is None or (
            f1, precision, entry["threshold"]
        ) > (
            by_f1["f1"], by_f1["near_target_invisible_precision"],
            by_f1["threshold"],
        ):
            by_f1 = entry
    by_precision_far5 = None
    for entry in candidates:
        if entry["flags"] == 0:
            continue
        if entry["far_visible_false_trigger_rate"] > 0.05:
            continue
        if by_precision_far5 is None or (
            entry["near_target_invisible_precision"],
            entry["near_target_invisible_recall"],
            entry["threshold"],
        ) > (
            by_precision_far5["near_target_invisible_precision"],
            by_precision_far5["near_target_invisible_recall"],
            by_precision_far5["threshold"],
        ):
            by_precision_far5 = entry
    return {
        "recall_0.5": by_recall,
        "precision_0.8": by_precision,
        "f1_far_visible_5pct": by_f1,
        "precision_far_visible_5pct": by_precision_far5,
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


def near_far_discrimination(data, score, mask_a, mask_b):
    """AUC with mask_a (near) as positive and mask_b (far) as negative."""
    if not np.any(mask_a) or not np.any(mask_b):
        return float("nan")
    y = np.concatenate([
        np.ones(int(mask_a.sum()), dtype=int),
        np.zeros(int(mask_b.sum()), dtype=int),
    ])
    values = np.concatenate([score[mask_a], score[mask_b]])
    return auc_rank(y, values)


def paired_metrics(data, score, threshold):
    source = data["source"]
    pair_id = data["pair_id"]
    variant = data["variant"]
    labels = data["target_visible"]
    rows = []
    for pair in np.unique(pair_id[source == 1]):
        mask_visible = (source == 1) & (pair_id == pair) & (variant == 0)
        mask_invisible = (source == 1) & (pair_id == pair) & (variant == 1)
        if not np.any(mask_visible) or not np.any(mask_invisible):
            continue
        rows.append((
            float(score[mask_invisible][0]),
            float(score[mask_visible][0]),
            int(data["episode_id"][mask_invisible][0]),
        ))
    if not rows:
        return {"pairs": 0}
    invisible = np.asarray([row[0] for row in rows])
    visible = np.asarray([row[1] for row in rows])
    y = np.concatenate([np.ones(len(rows)), np.zeros(len(rows))])
    values = np.concatenate([invisible, visible])
    return {
        "pairs": int(len(rows)),
        "paired_auroc": auc_rank(y, values),
        "paired_accuracy_at_threshold": float(
            ((invisible >= threshold) & (visible < threshold)).mean()
        ),
        "fraction_invisible_above_visible": float(
            (invisible > visible).mean()
        ),
        "mean_score_difference": float((invisible - visible).mean()),
    }


def bootstrap_trigger(episodes, data, score, threshold, n_boot=200):
    rng = np.random.default_rng(0)
    unique = np.unique(episodes)
    index_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique
    }
    masks = subset_masks(data)
    near = masks["near_invisible"]
    far_visible = masks["far_visible"]
    far_invisible = masks["far_invisible"]
    flags = score >= float(threshold)
    draws = defaultdict(list)
    for _ in range(int(n_boot)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            index_by_episode[int(episode)] for episode in sample
        ])
        f = flags[indices]
        n = near[indices]
        fv = far_visible[indices]
        fi = far_invisible[indices]
        if n.sum() == 0 or fv.sum() == 0:
            continue
        tp = int((f & n).sum())
        draws["precision"].append(tp / max(int(f.sum()), 1))
        draws["recall"].append(tp / max(int(n.sum()), 1))
        draws["far_visible_false_trigger"].append(
            float((f & fv).sum()) / max(int(fv.sum()), 1)
        )
        if fi.sum() > 0:
            draws["far_invisible_false_trigger"].append(
                float((f & fi).sum()) / max(int(fi.sum()), 1)
            )
    result = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        if len(values):
            result[name] = {
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "n_boot": int(len(values)),
            }
    return result


def bootstrap_near_far(episodes, data, score, mask_a, mask_b, n_boot=200,
                       seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(episodes)
    index_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique
    }
    draws = []
    for _ in range(int(n_boot)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            index_by_episode[int(episode)] for episode in sample
        ])
        a = mask_a[indices]
        b = mask_b[indices]
        s = score[indices]
        if a.sum() == 0 or b.sum() == 0:
            continue
        y = np.concatenate([
            np.ones(int(a.sum()), dtype=int),
            np.zeros(int(b.sum()), dtype=int),
        ])
        values = np.concatenate([s[a], s[b]])
        draws.append(auc_rank(y, values))
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return {"low": float("nan"), "high": float("nan"), "n_boot": 0}
    return {
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "n_boot": int(len(draws)),
    }


def bootstrap_paired(episodes, data, score, threshold, n_boot=200, seed=0):
    # Build pair rows with their episode, then resample episodes.
    source = data["source"]
    pair_id = data["pair_id"]
    variant = data["variant"]
    pair_rows = []
    for pair in np.unique(pair_id[source == 1]):
        mask_visible = (source == 1) & (pair_id == pair) & (variant == 0)
        mask_invisible = (source == 1) & (pair_id == pair) & (variant == 1)
        if not np.any(mask_visible) or not np.any(mask_invisible):
            continue
        pair_rows.append((
            int(data["episode_id"][mask_invisible][0]),
            float(score[mask_invisible][0]),
            float(score[mask_visible][0]),
        ))
    pair_rows = np.asarray(pair_rows, dtype=float)
    if not len(pair_rows):
        return {}
    rng = np.random.default_rng(seed)
    by_episode = defaultdict(list)
    for row in pair_rows:
        by_episode[int(row[0])].append(row)
    episodes_unique = np.asarray(sorted(by_episode), dtype=int)
    draws_auc = []
    draws_acc = []
    for _ in range(int(n_boot)):
        sample = rng.choice(episodes_unique, size=len(episodes_unique),
                            replace=True)
        rows = np.concatenate([by_episode[int(e)] for e in sample], axis=0)
        invisible = rows[:, 1]
        visible = rows[:, 2]
        y = np.concatenate([
            np.ones(len(rows), dtype=int), np.zeros(len(rows), dtype=int)
        ])
        values = np.concatenate([invisible, visible])
        draws_auc.append(auc_rank(y, values))
        draws_acc.append(
            float(((invisible >= threshold) & (visible < threshold)).mean())
        )
    return {
        "paired_auroc": {
            "low": float(np.percentile(draws_auc, 2.5)),
            "high": float(np.percentile(draws_auc, 97.5)),
        },
        "paired_accuracy": {
            "low": float(np.percentile(draws_acc, 2.5)),
            "high": float(np.percentile(draws_acc, 97.5)),
        },
    }


def run_variant(train_data, valid_data, variant, folds, args, device):
    train_x = build_input(train_data, variant)
    valid_x = build_input(valid_data, variant)
    oof = {head: np.full(len(train_x), np.nan) for head in HEADS}
    external = {}
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_std, valid_std = standardize(
            train_x[train_mask], train_x[valid_mask]
        )
        for head in HEADS:
            train_y = train_data[head][train_mask]
            weight = pos_weight(train_y)
            probabilities, _info = train_and_predict(
                train_std, train_y, valid_std, "mlp", args.hidden,
                args.epochs, args.batch_size, args.lr, args.weight_decay,
                weight, args.seed + int(fold), device,
            )
            oof[head][valid_mask] = prior_correct(probabilities, weight)
    train_std, valid_std = standardize(train_x, valid_x)
    for head in HEADS:
        weight = pos_weight(train_data[head])
        probabilities, info = train_and_predict(
            train_std, train_data[head], valid_std, "mlp", args.hidden,
            args.epochs, args.batch_size, args.lr, args.weight_decay,
            weight, args.seed + 1000, device,
        )
        external[head] = {
            "probabilities": prior_correct(probabilities, weight),
            "train_info": info,
            "pos_weight": weight,
        }
    return {
        "variant": int(variant),
        "oof": oof,
        "external": external,
        "folds": folds,
    }


def evaluate_result(data, probabilities, head, args, train_data=None,
                    train_probabilities=None):
    labels = data[head]
    result = {
        "overall": metric_block(labels, probabilities),
        "bands": {},
        "subsets": subset_report(data, probabilities, head),
        "episode_bootstrap_95ci": bootstrap_ci(
            data["episode_id"], labels, probabilities,
            n_boot=args.bootstrap,
        ),
    }
    for code, name in BAND_NAMES.items():
        mask = data["band"] == code
        if np.any(mask):
            result["bands"][name] = metric_block(
                labels[mask], probabilities[mask]
            )
    score = score_for_head(probabilities, head)
    masks = subset_masks(data)
    result["near_invisible_vs_far_invisible_auroc"] = (
        near_far_discrimination(
            data, score, masks["near_invisible"], masks["far_invisible"]
        )
    )
    result["near_invisible_vs_far_visible_auroc"] = (
        near_far_discrimination(
            data, score, masks["near_invisible"], masks["far_visible"]
        )
    )
    result["near_invisible_vs_far_invisible_ci"] = bootstrap_near_far(
        data["episode_id"], data, score, masks["near_invisible"],
        masks["far_invisible"], n_boot=args.bootstrap,
    )
    hard_far_invisible = masks["far_invisible"] & (
        data["history_origin"] == 1
    )
    natural_far_invisible = masks["far_invisible"] & (
        data["history_origin"] == 0
    )
    result["near_invisible_vs_hard_far_invisible_auroc"] = (
        near_far_discrimination(
            data, score, masks["near_invisible"], hard_far_invisible
        )
    )
    result["near_invisible_vs_natural_far_invisible_auroc"] = (
        near_far_discrimination(
            data, score, masks["near_invisible"], natural_far_invisible
        )
    )
    result["paired"] = paired_metrics(
        data, score, threshold=0.5
    )
    result["paired_ci"] = bootstrap_paired(
        data["episode_id"], data, score, threshold=0.5,
        n_boot=args.bootstrap,
    )
    if train_data is not None and train_probabilities is not None:
        train_score = score_for_head(train_probabilities, head)
        result["thresholds"] = select_thresholds(
            train_data, train_score
        )
        # Apply every Train-selected threshold to the held-out split.
        for name, entry in result["thresholds"].items():
            if entry is None:
                continue
            val_metrics = trigger_metrics(data, score, entry["threshold"])
            val_metrics["ci"] = bootstrap_trigger(
                data["episode_id"], data, score, entry["threshold"],
                n_boot=args.bootstrap,
            )
            val_metrics["paired_at_trigger"] = paired_metrics(
                data, score, entry["threshold"]
            )
            result.setdefault("trigger_val", {})[name] = val_metrics
        # Diagnostic only: best threshold on this split (not a deployable
        # selection; reported for headroom analysis).
        result["trigger_oracle_diagnostic"] = select_thresholds(
            data, score
        )
    return result


def run_baseline(train_data, valid_data, head, feature_matrix, folds,
                 args, device):
    oof = np.full(len(feature_matrix["train"]), np.nan)
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_std, valid_std = standardize(
            feature_matrix["train"][train_mask],
            feature_matrix["train"][valid_mask],
        )
        weight = pos_weight(train_data[head][train_mask])
        probabilities, _info = train_and_predict(
            train_std, train_data[head][train_mask], valid_std, "mlp",
            args.hidden, args.epochs, args.batch_size, args.lr,
            args.weight_decay, weight, args.seed + int(fold), device,
        )
        oof[valid_mask] = prior_correct(probabilities, weight)
    train_std, valid_std = standardize(
        feature_matrix["train"], feature_matrix["valid"]
    )
    weight = pos_weight(train_data[head])
    probabilities, info = train_and_predict(
        train_std, train_data[head], valid_std, "mlp", args.hidden,
        args.epochs, args.batch_size, args.lr, args.weight_decay,
        weight, args.seed + 1000, device,
    )
    probabilities = prior_correct(probabilities, weight)
    result = evaluate_result(
        valid_data, probabilities, head, args,
        train_data=train_data, train_probabilities=oof,
    )
    return {"oof": oof, "valid": probabilities, "result": result,
            "train_info": info, "pos_weight": weight}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", type=int,
                        default=[0, 1, 2, 3])
    args = parser.parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = load_combined(args.train)
    valid_data = load_combined(args.val)
    folds = assign_episode_folds(
        train_data["episode_id"], train_data["band"], args.folds, args.seed
    )
    results = {
        "data": {
            "train_states": int(len(train_data["episode_id"])),
            "train_episodes": int(np.unique(train_data["episode_id"]).size),
            "val_states": int(len(valid_data["episode_id"])),
            "val_episodes": int(np.unique(valid_data["episode_id"]).size),
            "train_folds": {
                str(f): int((folds == f).sum())
                for f in sorted(set(folds.tolist()))
            },
            "labels_train": {
                head: float(train_data[head].mean()) for head in HEADS
            },
            "labels_val": {
                head: float(valid_data[head].mean()) for head in HEADS
            },
        },
        "variants": {},
    }
    predictions = {}
    for variant in args.variants:
        print(f"running variant {variant}", flush=True)
        variant_result = run_variant(
            train_data, valid_data, variant, folds, args, device
        )
        variant_output = {"variant": int(variant), "heads": {}}
        for head in HEADS:
            print(f"  evaluating variant {variant} {head}", flush=True)
            oof_metrics = evaluate_result(
                train_data, variant_result["oof"][head], head, args,
                train_data=train_data,
                train_probabilities=variant_result["oof"][head],
            )
            external_metrics = evaluate_result(
                valid_data, variant_result["external"][head]["probabilities"],
                head, args,
                train_data=train_data,
                train_probabilities=variant_result["oof"][head],
            )
            variant_output["heads"][head] = {
                "oof": oof_metrics,
                "external": external_metrics,
                "external_train_info": variant_result["external"][head][
                    "train_info"
                ],
            }
            predictions[(variant, head, "oof")] = variant_result["oof"][head]
            predictions[(variant, head, "val")] = variant_result[
                "external"
            ][head]["probabilities"]
        results["variants"][str(variant)] = variant_output

    # Goal-token-only baseline (observable scene prior control).
    goal_idx = goal_indices(train_data["feature_names"])
    goal_matrix = {
        "train": train_data["current_features"][:, goal_idx],
        "valid": valid_data["current_features"][:, goal_idx],
    }
    print("running goal-only baseline", flush=True)
    goal_baseline = run_baseline(
        train_data, valid_data, "near_target_invisible",
        goal_matrix, folds, args, device,
    )
    results["goal_only_baseline"] = goal_baseline["result"]

    # Distance-only diagnostic: no model, no deployable input. This only
    # quantifies how much the true distance alone explains the label.
    train_masks = subset_masks(train_data)
    valid_masks = subset_masks(valid_data)
    distance_auc_train = near_far_discrimination(
        train_data, -train_data["distance_m"],
        train_masks["near_invisible"], train_masks["far_invisible"],
    )
    distance_auc_val = near_far_discrimination(
        valid_data, -valid_data["distance_m"],
        valid_masks["near_invisible"], valid_masks["far_invisible"],
    )
    results["distance_only_diagnostic"] = {
        "note": "offline diagnostic only; true distance is not a model input",
        "near_invisible_vs_far_invisible_auroc_train": distance_auc_train,
        "near_invisible_vs_far_invisible_auroc_val": distance_auc_val,
        "paired_auroc": 0.5,
    }

    # History-length summary.
    history_summary = {}
    for variant in args.variants:
        key = str(variant)
        if key not in results["variants"]:
            continue
        heads = results["variants"][key]["heads"]
        history_summary[key] = {}
        for head in HEADS:
            external = heads[head]["external"]
            trigger = external.get("trigger_val", {}).get(
                "f1_far_visible_5pct"
            )
            history_summary[key][head] = {
                "auroc": external["overall"]["auroc"],
                "auprc": external["overall"]["auprc"],
                "near_inv_vs_far_inv_auroc": external[
                    "near_invisible_vs_far_invisible_auroc"
                ],
                "paired_auroc": external["paired"].get("paired_auroc"),
                "trigger": trigger,
            }
    results["history_length"] = history_summary

    (output_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    for (variant, head, split), probabilities in predictions.items():
        np.savez_compressed(
            output_dir / f"predictions_v{variant}_{head}_{split}.npz",
            probability=probabilities,
        )
    print(json.dumps({
        "output": str(output_dir),
        "variants": list(results["variants"]),
        "data": results["data"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
