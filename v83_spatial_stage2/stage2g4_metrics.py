"""Stage-2g.4 event metrics, thresholds and episode-level inference."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

from v83_spatial_stage2.stage2g_probe import (
    auc_rank,
    average_precision,
    metric_block,
)
from v83_spatial_stage2.stage2g4_data import EVENT_KS


def fast_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0)
        / (n_pos * n_neg)
    )


def fast_average_precision(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(s[1:] != s[:-1]) + 1,
        len(s),
    ]
    ends = boundaries[1:] - 1
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp[ends] / np.maximum(tp[ends] + fp[ends], 1)
    recall = tp[ends] / n_pos
    return float(np.sum(precision * np.diff(np.r_[0.0, recall])))


def event_masks(data, k):
    event = np.asarray(data["events"][k], dtype=bool)
    near = np.asarray(data["near_target_invisible"]) == 1
    visible = np.asarray(data["target_visible"]) == 1
    distance = np.asarray(data["distance_m"])
    return {
        "event": event,
        "sustained_near_inv": near & ~event,
        "near_visible": (distance < 100.0) & visible,
        "far_visible": (distance >= 100.0) & visible,
        "far_invisible": (distance >= 100.0) & ~visible,
    }


def event_metrics(data, k, scores, threshold):
    scores = np.asarray(scores, dtype=float)
    masks = event_masks(data, k)
    event = masks["event"]
    flags = scores >= float(threshold)
    tp = int((flags & event).sum())
    fp = int((flags & ~event).sum())
    fn = int((~flags & event).sum())
    tn = int((~flags & ~event).sum())
    episodes = np.asarray(data["episode_id"])
    negative_flags = flags & ~event
    _, episode_codes = np.unique(episodes, return_inverse=True)
    per_episode = np.bincount(
        episode_codes,
        weights=negative_flags.astype(np.float64),
        minlength=int(episode_codes.max()) + 1,
    )
    positive_flags_per_episode = np.bincount(
        episode_codes,
        weights=(flags & event).astype(np.float64),
        minlength=int(episode_codes.max()) + 1,
    )
    total_flags = max(int(flags.sum()), 1)
    subsets = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        flag_count = int((flags & mask).sum())
        subsets[name] = {
            "n": count,
            "positives": int((event & mask).sum()),
            "flags": flag_count,
            "flag_rate": flag_count / max(count, 1),
            "share_of_all_flags": flag_count / total_flags,
            "mean_score": (
                float(scores[mask].mean()) if count else float("nan")
            ),
            "median_score": (
                float(np.median(scores[mask])) if count else float("nan")
            ),
            "p_ge_0.5": (
                float((scores[mask] >= 0.5).mean()) if count else float("nan")
            ),
        }
    negative_states = max(int((~event).sum()), 1)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flags": int(flags.sum()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "f1": 2 * (tp / max(tp + fp, 1)) * (tp / max(tp + fn, 1))
        / max(
            (tp / max(tp + fp, 1)) + (tp / max(tp + fn, 1)),
            1e-12,
        ),
        "false_positive_rate": fp / negative_states,
        "false_positives_per_1000_negative": 1000.0 * fp / negative_states,
        "per_episode_false_positives": {
            "mean": float(per_episode.mean()),
            "median": float(np.median(per_episode)),
            "p90": float(np.quantile(per_episode, 0.9)),
            "max": float(per_episode.max()),
            "episodes_with_any": int((per_episode > 0).sum()),
            "episodes": int(len(per_episode)),
        },
        "per_episode_true_positives": {
            "mean": float(np.mean(positive_flags_per_episode)),
            "median": float(np.median(positive_flags_per_episode)),
            "max": float(np.max(positive_flags_per_episode)),
        },
        "composition": {
            name: subsets[name]["flags"] / total_flags
            for name in masks
        },
        "subsets": subsets,
    }


def per_episode_table(data, k, scores, threshold):
    """Episode-level TP/FP counts, sorted by FP (for absolute-budget audit)."""
    event = np.asarray(data["events"][k], dtype=bool)
    flags = np.asarray(scores) >= float(threshold)
    episodes = np.asarray(data["episode_id"])
    rows = []
    for episode in np.unique(episodes):
        mask = episodes == episode
        rows.append({
            "episode_id": int(episode),
            "n": int(mask.sum()),
            "positives": int((event & mask).sum()),
            "tp": int((flags & event & mask).sum()),
            "fp": int((flags & ~event & mask).sum()),
        })
    return sorted(rows, key=lambda row: row["fp"], reverse=True)


def threshold_grid(scores, points=401):
    scores = np.asarray(scores, dtype=float)
    grid = np.unique(np.concatenate([
        np.quantile(scores, np.linspace(0.0, 1.0, points)),
        [0.0, 1.0],
    ]))
    return grid


def select_event_thresholds(data, k, scores, constraints=None):
    """Select event thresholds on Train OOF only.

    Constraints bound each negative subset's flag rate; the primary point
    maximizes F1 among constrained thresholds. Unconstrained and
    precision/recall-target points are also reported.
    """
    if constraints is None:
        constraints = {
            "near_visible": 0.02,
            "far_visible": 0.02,
            "far_invisible": 0.02,
            "sustained_near_inv": 0.05,
        }
    candidates = []
    for threshold in threshold_grid(scores):
        entry = event_metrics(data, k, scores, threshold)
        if entry["flags"] == 0:
            continue
        entry["constrained"] = all(
            entry["subsets"][name]["flag_rate"] <= limit
            for name, limit in constraints.items()
        )
        candidates.append(entry)
    def best(entries, key):
        if not entries:
            return None
        return max(entries, key=key)
    constrained = [entry for entry in candidates if entry["constrained"]]
    f1_joint = best(
        constrained,
        key=lambda e: (e["f1"], e["precision"], e["threshold"]),
    )
    f1_unconstrained = best(
        candidates,
        key=lambda e: (e["f1"], e["precision"], e["threshold"]),
    )
    precision_half = best(
        [entry for entry in candidates if entry["precision"] >= 0.5],
        key=lambda e: (e["recall"], e["threshold"]),
    )
    recall_half = best(
        [entry for entry in candidates if entry["recall"] >= 0.5],
        key=lambda e: (e["precision"], e["threshold"]),
    )
    precision_half_constrained = best(
        [entry for entry in constrained if entry["precision"] >= 0.5],
        key=lambda e: (e["recall"], e["threshold"]),
    )
    recall_half_constrained = best(
        [entry for entry in constrained if entry["recall"] >= 0.5],
        key=lambda e: (e["precision"], e["threshold"]),
    )
    return {
        "constraints": constraints,
        "f1_joint": f1_joint,
        "f1_unconstrained": f1_unconstrained,
        "precision_0.5": precision_half,
        "recall_0.5": recall_half,
        "precision_0.5_constrained": precision_half_constrained,
        "recall_0.5_constrained": recall_half_constrained,
    }


def stratified_event_vs_sustained(data, k, scores):
    """Event-vs-sustained AUROC within distance bins (static-distance control)."""
    masks = event_masks(data, k)
    event = masks["event"]
    sustained = masks["sustained_near_inv"]
    distance = np.asarray(data["distance_m"])
    rows = {}
    values_all = []
    weighted = []
    weights = []
    for low, high in ((0, 20), (20, 50), (50, 100)):
        bin_mask = (distance >= low) & (distance < high)
        pos = event & bin_mask
        neg = sustained & bin_mask
        if pos.any() and neg.any():
            y = np.concatenate([
                np.ones(int(pos.sum())), np.zeros(int(neg.sum()))
            ])
            values = np.concatenate([scores[pos], scores[neg]])
            value = auc_rank(y, values)
        else:
            value = float("nan")
        rows[f"{low}-{high}"] = {
            "event": int(pos.sum()),
            "sustained": int(neg.sum()),
            "auroc": value,
        }
        if np.isfinite(value):
            values_all.append(value)
            weight = int(min(pos.sum(), neg.sum()))
            weighted.append(value * weight)
            weights.append(weight)
    rows["macro_auroc"] = (
        float(np.mean(values_all)) if values_all else float("nan")
    )
    rows["weighted_auroc"] = (
        float(sum(weighted) / sum(weights)) if weights else float("nan")
    )
    return rows


def subset_aucs(data, k, scores):
    masks = event_masks(data, k)
    event = masks["event"]
    output = {}
    for name in ("sustained_near_inv", "near_visible", "far_visible", "far_invisible"):
        negative = masks[name]
        if not event.any() or not negative.any():
            output[name] = float("nan")
            continue
        y = np.concatenate([
            np.ones(int(event.sum())), np.zeros(int(negative.sum()))
        ])
        values = np.concatenate([scores[event], scores[negative]])
        output[name] = auc_rank(y, values)
    return output


def metric_summary(data, k, scores):
    labels = np.asarray(data["events"][k], dtype=int)
    scores = np.asarray(scores, dtype=float)
    block = metric_block(labels, scores)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    positive_rate = float(labels.mean()) if len(labels) else float("nan")
    return {
        "n": block["n"],
        "positives": block["positives"],
        "positive_rate": block["positive_rate"],
        "auroc": block["auroc"],
        "auprc": block["auprc"],
        "auprc_lift_over_positive_rate": (
            block["auprc"] / positive_rate if positive_rate else float("nan")
        ),
        "brier": block["brier"],
        "brier_constant_zero": positive_rate,
        "brier_skill_score": (
            1.0 - block["brier"] / positive_rate
            if positive_rate else float("nan")
        ),
        "ece": block["ece"],
        "mean_probability": block["mean_probability"],
        "mean_probability_positive": (
            float(positive_scores.mean()) if len(positive_scores) else float("nan")
        ),
        "mean_probability_negative": (
            float(negative_scores.mean()) if len(negative_scores) else float("nan")
        ),
        "positive_negative_mean_ratio": (
            float(positive_scores.mean() / negative_scores.mean())
            if len(negative_scores) and negative_scores.mean() > 0
            else float("nan")
        ),
        "at_0.5": block["confusions"]["0.5"],
    }


def episode_bootstrap(data, k, scores, threshold, n_boot=1000, seed=0):
    episodes = np.asarray(data["episode_id"])
    labels = np.asarray(data["events"][k], dtype=int)
    unique = np.unique(episodes)
    index_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique
    }
    rng = np.random.default_rng(int(seed))
    draws = defaultdict(list)
    for _ in range(int(n_boot)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            index_by_episode[int(episode)] for episode in sample
        ])
        y = labels[indices]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = scores[indices]
        draws["auroc"].append(fast_auc(y, s))
        draws["auprc"].append(fast_average_precision(y, s))
        flags = s >= float(threshold)
        tp = int((flags & (y == 1)).sum())
        fp = int((flags & (y == 0)).sum())
        fn = int((~flags & (y == 1)).sum())
        draws["precision"].append(tp / max(tp + fp, 1))
        draws["recall"].append(tp / max(tp + fn, 1))
        draws["false_positives"].append(fp)
    output = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            output[name] = {"n_boot": 0}
            continue
        output[name] = {
            "n_boot": int(len(values)),
            "mean": float(values.mean()),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
    return output


def paired_bootstrap(data, k, scores_a, scores_b, threshold_a, threshold_b,
                     n_boot=1000, seed=0):
    episodes = np.asarray(data["episode_id"])
    labels = np.asarray(data["events"][k], dtype=int)
    unique = np.unique(episodes)
    index_by_episode = {
        int(episode): np.flatnonzero(episodes == episode)
        for episode in unique
    }
    rng = np.random.default_rng(int(seed))
    draws = defaultdict(list)
    for _ in range(int(n_boot)):
        sample = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([
            index_by_episode[int(episode)] for episode in sample
        ])
        y = labels[indices]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        sa, sb = scores_a[indices], scores_b[indices]
        draws["auroc"].append(fast_auc(y, sb) - fast_auc(y, sa))
        draws["auprc"].append(
            fast_average_precision(y, sb)
            - fast_average_precision(y, sa)
        )
        fa = sa >= float(threshold_a)
        fb = sb >= float(threshold_b)
        for name, flags in (("a", fa), ("b", fb)):
            tp = int((flags & (y == 1)).sum())
            fp = int((flags & (y == 0)).sum())
            fn = int((~flags & (y == 1)).sum())
            draws[f"{name}_precision"].append(tp / max(tp + fp, 1))
            draws[f"{name}_recall"].append(tp / max(tp + fn, 1))
        draws["precision"].append(
            draws["b_precision"][-1] - draws["a_precision"][-1]
        )
        draws["recall"].append(
            draws["b_recall"][-1] - draws["a_recall"][-1]
        )
    output = {}
    for name, values in draws.items():
        if name in ("a_precision", "b_precision", "a_recall", "b_recall"):
            continue
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            output[name] = {"n_boot": 0}
            continue
        output[name] = {
            "n_boot": int(len(values)),
            "mean": float(values.mean()),
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
            "fraction_le_0": float((values <= 0).mean()),
            "fraction_gt_0": float((values > 0).mean()),
        }
    return output


def leave_one_out(data, k, scores, threshold, exclude_episodes):
    episodes = np.asarray(data["episode_id"])
    rows = []
    for excluded in exclude_episodes:
        keep = episodes != int(excluded)
        subset = {
            key: (
                {name: value[keep] for name, value in value.items()}
                if key == "events"
                else value[keep]
            )
            for key, value in data.items()
        }
        metrics = event_metrics(subset, k, scores[keep], threshold)
        summary = metric_summary(subset, k, scores[keep])
        rows.append({
            "excluded_episode": int(excluded),
            "n": int(keep.sum()),
            "positives": int(np.asarray(data["events"][k])[keep].sum()),
            "auroc": summary["auroc"],
            "auprc": summary["auprc"],
            **{
                f"at_threshold_{key}": metrics[key]
                for key in (
                    "tp", "fp", "fn", "precision", "recall",
                    "f1", "flags",
                )
            },
        })
    return rows
