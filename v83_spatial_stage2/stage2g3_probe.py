"""Stage-2g.3 real closed-loop P0/P1-2 target-presence probe.

Trains only lightweight probes on the real M0@0.7 closed-loop states collected
by stage2g3_collect.py. P0 uses the current 2,858 features; P1-2 adds two
previous frames (74 compact visual features + action one-hot + dt).

5-fold episode-level OOF on Train 600; Dev Val 180 is independent validation.
The threshold is selected on Train OOF only. No MEC training, no checkpoint
modification, Final Test 300 sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import ks_2samp  # noqa: E402

from v83_spatial_stage2.stage2g_probe import (  # noqa: E402
    assign_episode_folds,
    auc_rank,
    average_precision,
    bootstrap_ci,
    brier_score,
    build_model,
    expected_calibration_error,
    metric_block,
    prior_correct,
    standardize,
    train_and_predict,
)
from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    COMPACT_FEATURE_NAMES,
    compact_indices,
    goal_indices,
    one_hot_actions,
)

BANDS = ("short", "medium", "long")
BAND_NAMES = {0: "short", 1: "medium", 2: "long"}
HEADS = ("target_visible", "target_localizable", "near_target_invisible")
HEAD_SHORT = {
    "target_visible": "visible",
    "target_localizable": "localizable",
    "near_target_invisible": "near_inv",
}
PER_STATE = (
    "episode_id", "seed", "band", "step", "action", "stop", "current_features",
    "history_features", "history_actions", "history_dt", "history_valid",
    "distance_m", "goal_visible_geom", "goal_half_size_px", "visible_pixels",
    "target_visible", "target_localizable", "near_target_invisible",
    "heading_error_deg", "oracle_move_action", "arrival_probability",
)


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


def load_split(directory):
    files = sorted(Path(directory).glob("episode_*.npz"))
    if not files:
        raise RuntimeError(f"No episode files in {directory}")
    parts = defaultdict(list)
    for path in files:
        data = np.load(path)
        n = len(data["step"])
        for key in PER_STATE:
            if key in ("episode_id", "seed", "band"):
                continue
            parts[key].append(data[key])
        parts["episode_id"].append(
            np.full(n, int(data["episode_id"]), dtype=np.int64)
        )
        parts["seed"].append(np.full(n, int(data["seed"]), dtype=np.int64))
        parts["band"].append(np.full(n, int(data["band"]), dtype=np.int64))
    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


def filter_moves(data):
    mask = (data["stop"] == 0) & (data["history_valid"] > 0)
    return {key: value[mask] for key, value in data.items()}, mask


def build_input(data, variant):
    if variant == 0:
        return data["current_features"].astype(np.float32)
    parts = [data["current_features"].astype(np.float32)]
    for frame in range(2):
        parts.append(data["history_features"][:, frame, :])
        parts.append(one_hot_actions(data["history_actions"][:, frame]))
        parts.append(data["history_dt"][:, frame:frame + 1])
    return np.concatenate(parts, axis=1)


def real_masks(data):
    distance = data["distance_m"]
    visible = data["target_visible"]
    return {
        "near_invisible": data["near_target_invisible"] == 1,
        "far_visible": (distance >= 100.0) & (visible == 1),
        "far_invisible": (distance >= 100.0) & (visible == 0),
        "near_visible": (distance < 100.0) & (visible == 1),
    }


def real_metrics(data, score, threshold):
    masks = real_masks(data)
    flags = np.asarray(score) >= float(threshold)
    near = masks["near_invisible"]
    far_visible = masks["far_visible"]
    far_invisible = masks["far_invisible"]
    near_visible = masks["near_visible"]
    tp = int((flags & near).sum())
    fp = int((flags & ~near).sum())
    fn = int((~flags & near).sum())
    tn = int((~flags & ~near).sum())
    n_flags = int(flags.sum())
    return {
        "threshold": float(threshold),
        "flags": n_flags,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "far_visible_false_trigger": int(
            (flags & far_visible).sum()
        ) / max(int(far_visible.sum()), 1),
        "far_invisible_false_trigger": int(
            (flags & far_invisible).sum()
        ) / max(int(far_invisible.sum()), 1),
        "near_visible_false_trigger": int(
            (flags & near_visible).sum()
        ) / max(int(near_visible.sum()), 1),
        "composition": {
            "near_invisible": int((flags & near).sum()) / max(n_flags, 1),
            "far_visible": int((flags & far_visible).sum()) / max(n_flags, 1),
            "far_invisible": int(
                (flags & far_invisible).sum()
            ) / max(n_flags, 1),
            "near_visible": int(
                (flags & near_visible).sum()
            ) / max(n_flags, 1),
        },
    }


def select_thresholds(data, score):
    candidates = []
    grid = np.unique(np.concatenate([
        np.quantile(score, np.linspace(0.0, 1.0, 401)), [0.0, 1.0]
    ]))
    for threshold in grid:
        entry = real_metrics(data, score, threshold)
        if entry["flags"] == 0:
            continue
        precision = entry["precision"]
        recall = entry["recall"]
        entry["f1"] = 2 * precision * recall / max(
            precision + recall, 1e-12
        )
        candidates.append(entry)
    def constrained(entry):
        return (
            entry["far_visible_false_trigger"] <= 0.05
            and entry["far_invisible_false_trigger"] <= 0.05
        )
    f1_joint = None
    for entry in candidates:
        if not constrained(entry):
            continue
        if f1_joint is None or (
            entry["f1"], entry["precision"], entry["threshold"]
        ) > (
            f1_joint["f1"], f1_joint["precision"], f1_joint["threshold"]
        ):
            f1_joint = entry
    recall_half = None
    for entry in sorted(
        candidates, key=lambda x: x["threshold"], reverse=True
    ):
        if entry["recall"] >= 0.5 and constrained(entry):
            recall_half = entry
            break
    recall_half_unconstrained = None
    for entry in sorted(
        candidates, key=lambda x: x["threshold"], reverse=True
    ):
        if entry["recall"] >= 0.5:
            recall_half_unconstrained = entry
            break
    precision_high = None
    for entry in sorted(
        candidates, key=lambda x: x["threshold"]
    ):
        if entry["precision"] >= 0.8 and constrained(entry):
            precision_high = entry
    return {
        "f1_joint_far5": f1_joint,
        "recall_0.5_far5": recall_half,
        "recall_0.5_unconstrained": recall_half_unconstrained,
        "precision_0.8_far5": precision_high,
    }


def bootstrap_real(episodes, data, score, threshold, n_boot=200, seed=0):
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
        subset = {key: value[indices] for key, value in data.items()}
        metrics = real_metrics(subset, score[indices], threshold)
        if int(real_masks(subset)["near_invisible"].sum()) == 0:
            continue
        draws["precision"].append(metrics["precision"])
        draws["recall"].append(metrics["recall"])
        draws["far_visible_false_trigger"].append(
            metrics["far_visible_false_trigger"]
        )
        draws["far_invisible_false_trigger"].append(
            metrics["far_invisible_false_trigger"]
        )
    output = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        if len(values):
            output[name] = {
                "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "n_boot": int(len(values)),
            }
    return output


def save_checkpoint(path, model, mean, std, weight, variant, head, fold,
                    input_dim, feature_names, compact_names, data_dirs,
                    config, seed):
    checkpoint = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "clip": 20.0,
        "pos_weight": float(weight),
        "input_dim": int(input_dim),
        "feature_names": list(feature_names),
        "compact_feature_names": list(compact_names),
        "history_order": ["t-1", "t-2"],
        "action_encoding": {
            "FORWARD": 0, "LEFT": 1, "RIGHT": 2, "STOP": 3,
            "one_hot_dim": 4,
        },
        "variant": int(variant),
        "head": head,
        "fold": int(fold),
        "seed": int(seed),
        "config": dict(config),
        "data": dict(data_dirs),
        "git_commit": git_commit(),
    }
    torch.save(checkpoint, path)


def run_variant(train_data, valid_data, variant, folds, args, device,
                models_dir, feature_names, compact_names, data_dirs):
    train_x = build_input(train_data, variant)
    valid_x = build_input(valid_data, variant)
    oof = {head: np.full(len(train_x), np.nan) for head in HEADS}
    external = {}
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_std, valid_std, mean, std = standardize(
            train_x[train_mask], train_x[valid_mask], return_stats=True
        )
        for head in HEADS:
            train_y = train_data[head][train_mask]
            weight = float(
                (len(train_y) - train_y.sum())
                / max(float(train_y.sum()), 1.0)
            )
            probabilities, _info, model = train_and_predict(
                train_std, train_y, valid_std, "mlp", args.hidden,
                args.epochs, args.batch_size, args.lr,
                args.weight_decay, weight, args.seed + int(fold),
                device, return_model=True,
            )
            oof[head][valid_mask] = prior_correct(probabilities, weight)
            if args.save_models:
                save_checkpoint(
                    models_dir / f"v{variant}_{head}_fold{fold}.pt",
                    model, mean, std, weight, variant, head, fold,
                    train_x.shape[1], feature_names, compact_names,
                    data_dirs, {
                        "kind": "mlp", "hidden": int(args.hidden),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                    }, args.seed + int(fold),
                )
    train_std, valid_std, mean, std = standardize(
        train_x, valid_x, return_stats=True
    )
    for head in HEADS:
        weight = float(
            (len(train_data[head]) - train_data[head].sum())
            / max(float(train_data[head].sum()), 1.0)
        )
        probabilities, info, model = train_and_predict(
            train_std, train_data[head], valid_std, "mlp", args.hidden,
            args.epochs, args.batch_size, args.lr, args.weight_decay,
            weight, args.seed + 1000, device, return_model=True,
        )
        external[head] = {
            "probabilities": prior_correct(probabilities, weight),
            "pos_weight": weight,
            "train_info": info,
        }
        if args.save_models:
            save_checkpoint(
                models_dir / f"v{variant}_{head}_external.pt",
                model, mean, std, weight, variant, head, -1,
                train_x.shape[1], feature_names, compact_names,
                data_dirs, {
                    "kind": "mlp", "hidden": int(args.hidden),
                    "epochs": int(args.epochs),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "weight_decay": float(args.weight_decay),
                }, args.seed + 1000,
            )
    return {
        "variant": int(variant),
        "oof": oof,
        "external": external,
        "folds": folds,
        "input_dim": int(train_x.shape[1]),
    }


def load_probe_for_comparison(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    seed = int(
        checkpoint.get("seed", checkpoint["config"].get("seed", 0))
    )
    model = build_model(
        "mlp", int(checkpoint["input_dim"]),
        int(checkpoint["config"]["hidden"]), seed,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return (
        model,
        np.asarray(checkpoint["mean"], dtype=np.float32),
        np.asarray(checkpoint["std"], dtype=np.float32),
        float(checkpoint["pos_weight"]),
        float(checkpoint.get("threshold", 0.13694876571430528)),
    )


def batch_predict(model, mean, std, pos_weight, features, device):
    x = (features.astype(np.float32) - mean) / std
    x = np.clip(x, -20.0, 20.0)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 16384):
            tensor = torch.from_numpy(
                x[start:start + 16384]
            ).to(device)
            logits = model(tensor).squeeze(1)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return prior_correct(np.concatenate(outputs), pos_weight)


def subset_report(data, probabilities):
    masks = real_masks(data)
    report = {}
    for name, mask in masks.items():
        y = data["near_target_invisible"][mask]
        p = probabilities[mask]
        report[name] = {
            "n": int(mask.sum()),
            "mean_probability": float(p.mean()) if len(p) else float("nan"),
            "median_probability": float(np.median(p)) if len(p) else float("nan"),
            "p_ge_0.5": float((p >= 0.5).mean()) if len(p) else float("nan"),
            "positive_rate": float(y.mean()) if len(y) else float("nan"),
        }
    return report


def near_far_auc(data, score, name_a, name_b):
    masks = real_masks(data)
    a, b = masks[name_a], masks[name_b]
    if not np.any(a) or not np.any(b):
        return float("nan")
    y = np.concatenate([np.ones(int(a.sum())), np.zeros(int(b.sum()))])
    values = np.concatenate([score[a], score[b]])
    return auc_rank(y, values)


def episode_stats(data, threshold):
    near = data["near_target_invisible"] == 1
    episodes = np.unique(data["episode_id"])
    counts = []
    for episode in episodes:
        mask = data["episode_id"] == episode
        counts.append(int((near & mask).sum()))
    bands = {}
    for code, name in BAND_NAMES.items():
        mask = data["band"] == code
        bands[name] = {
            "states": int(mask.sum()),
            "near_invisible": int((near & mask).sum()),
            "episodes": int(np.unique(data["episode_id"][mask]).size),
        }
    distance_bins = {}
    for low, high in ((0, 20), (20, 50), (50, 100)):
        mask = (data["distance_m"] >= low) & (data["distance_m"] < high)
        distance_bins[f"{low}-{high}"] = {
            "states": int(mask.sum()),
            "near_invisible": int((near & mask).sum()),
            "episodes": int(np.unique(data["episode_id"][mask]).size),
        }
    return {
        "episodes": int(len(episodes)),
        "episodes_with_near_invisible": int(sum(c > 0 for c in counts)),
        "near_invisible_per_episode": {
            "min": int(min(counts)) if counts else 0,
            "median": float(np.median(counts)) if counts else 0.0,
            "max": int(max(counts)) if counts else 0,
            "mean": float(np.mean(counts)) if counts else 0.0,
        },
        "bands": bands,
        "distance_bins": distance_bins,
    }


def run_single_input(train_data, valid_data, folds, args, device,
                     train_x, valid_x, head="near_target_invisible"):
    oof = np.full(len(train_x), np.nan)
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_std, valid_std = standardize(
            train_x[train_mask], train_x[valid_mask]
        )
        train_y = train_data[head][train_mask]
        weight = float(
            (len(train_y) - train_y.sum()) / max(float(train_y.sum()), 1.0)
        )
        probabilities, _info = train_and_predict(
            train_std, train_y, valid_std, "mlp", args.hidden,
            args.epochs, args.batch_size, args.lr, args.weight_decay,
            weight, args.seed + int(fold), device,
        )
        oof[valid_mask] = prior_correct(probabilities, weight)
    train_std, valid_std = standardize(train_x, valid_x)
    train_y = train_data[head]
    weight = float(
        (len(train_y) - train_y.sum()) / max(float(train_y.sum()), 1.0)
    )
    probabilities, _info = train_and_predict(
        train_std, train_y, valid_std, "mlp", args.hidden,
        args.epochs, args.batch_size, args.lr, args.weight_decay,
        weight, args.seed + 1000, device,
    )
    probabilities = prior_correct(probabilities, weight)
    metrics = metric_block(valid_data[head], probabilities)
    metrics["near_invisible_vs_far_invisible_auroc"] = near_far_auc(
        valid_data, probabilities, "near_invisible", "far_invisible"
    )
    metrics["near_inv_vs_near_visible_auroc"] = near_far_auc(
        valid_data, probabilities, "near_invisible", "near_visible"
    )
    thresholds = select_thresholds(train_data, oof)
    trigger = {}
    for name, entry in thresholds.items():
        if entry is None:
            trigger[name] = None
            continue
        value = real_metrics(valid_data, probabilities, entry["threshold"])
        value["train_threshold"] = entry["threshold"]
        trigger[name] = value
    return {
        "overall": metrics,
        "thresholds": thresholds,
        "trigger_points": trigger,
    }


def render_tables(results):
    lines = ["# Stage-2g.3 real closed-loop probe tables", ""]
    for variant in ("0", "1"):
        if variant not in results["variants"]:
            continue
        lines += [f"## P{variant}", ""]
        for head in HEADS:
            entry = results["variants"][variant]["heads"][head]["external"]
            lines += [f"### {head}", "",
                      "| Group | AUROC | AUPRC | Brier | ECE | P@0.5 | R@0.5 |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for group in ("overall", "short", "medium", "long"):
                block = entry["overall"] if group == "overall" else entry["bands"][group]
                confusion = block["confusions"]["0.5"]
                lines.append(
                    f"| {group} | {block['auroc']:.3f} | "
                    f"{block['auprc']:.3f} | {block['brier']:.3f} | "
                    f"{block['ece']:.3f} | {confusion['precision']:.3f} | "
                    f"{confusion['recall']:.3f} |"
                )
            lines.append("")
    lines += ["## Trigger points (Dev Val, Train-OOF selected)", ""]
    lines += [
        "| Variant | Point | Threshold | Precision | Recall | Far-vis FT | Far-inv FT | Flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("0", "1"):
        if variant not in results["variants"]:
            continue
        points = results["variants"][variant]["trigger_points"]
        for name, entry in points.items():
            if entry is None:
                continue
            lines.append(
                f"| P{variant} | {name} | {entry['threshold']:.4f} | "
                f"{entry['precision']:.3f} | {entry['recall']:.3f} | "
                f"{entry['far_visible_false_trigger']:.4f} | "
                f"{entry['far_invisible_false_trigger']:.4f} | "
                f"{entry['flags']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-checkpoint", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    train_raw = load_split(args.train_dir)
    valid_raw = load_split(args.val_dir)
    train_data, train_mask = filter_moves(train_raw)
    valid_data, valid_mask = filter_moves(valid_raw)
    canonical_names = list(COMPACT_FEATURE_NAMES)
    from v83_spatial_stage2.stage2g_features import (
        feature_names as canonical_feature_names,
    )
    feature_names = canonical_feature_names()
    if len(train_data["current_features"][0]) != len(feature_names):
        raise RuntimeError("feature dimension mismatch")
    folds = assign_episode_folds(
        train_data["episode_id"], train_data["band"], args.folds, args.seed
    )
    data_dirs = {
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "train_manifest_sha256": sha256(
            Path(args.train_dir) / "manifest.json"
        ),
        "val_manifest_sha256": sha256(
            Path(args.val_dir) / "manifest.json"
        ),
    }
    results = {
        "data": {
            "train_states_all": int(len(train_raw["episode_id"])),
            "train_states_move_valid": int(len(train_data["episode_id"])),
            "train_episodes": int(np.unique(train_data["episode_id"]).size),
            "val_states_all": int(len(valid_raw["episode_id"])),
            "val_states_move_valid": int(len(valid_data["episode_id"])),
            "val_episodes": int(np.unique(valid_data["episode_id"]).size),
            "train_near_invisible_all": int(
                train_raw["near_target_invisible"].sum()
            ),
            "train_near_invisible_move": int(
                train_data["near_target_invisible"].sum()
            ),
            "val_near_invisible_all": int(
                valid_raw["near_target_invisible"].sum()
            ),
            "val_near_invisible_move": int(
                valid_data["near_target_invisible"].sum()
            ),
            "train_stop_states": int(train_raw["stop"].sum()),
            "val_stop_states": int(valid_raw["stop"].sum()),
            "folds": {
                str(f): int((folds == f).sum())
                for f in sorted(set(folds.tolist()))
            },
        },
        "variants": {},
    }
    predictions = {}
    for variant in (0, 1):
        print(f"training P{variant}", flush=True)
        variant_result = run_variant(
            train_data, valid_data, variant, folds, args, device,
            models_dir, feature_names, canonical_names, data_dirs,
        )
        variant_output = {"variant": variant, "input_dim": variant_result["input_dim"], "heads": {}}
        for head in HEADS:
            print(f"  evaluating P{variant} {head}", flush=True)
            oof_metrics = metric_block(
                train_data[head], variant_result["oof"][head]
            )
            external_metrics = {
                "overall": metric_block(
                    valid_data[head],
                    variant_result["external"][head]["probabilities"],
                ),
                "bands": {},
            }
            for code, name in BAND_NAMES.items():
                mask = valid_data["band"] == code
                external_metrics["bands"][name] = metric_block(
                    valid_data[head][mask],
                    variant_result["external"][head]["probabilities"][mask],
                )
            external_metrics["episode_bootstrap_95ci"] = bootstrap_ci(
                valid_data["episode_id"], valid_data[head],
                variant_result["external"][head]["probabilities"],
                n_boot=args.bootstrap,
            )
            external_metrics["subsets"] = subset_report(
                valid_data,
                variant_result["external"][head]["probabilities"],
            )
            direction_score = (
                variant_result["external"][head]["probabilities"]
                if head == "near_target_invisible"
                else 1.0 - variant_result["external"][head]["probabilities"]
            )
            external_metrics["near_invisible_vs_far_invisible_auroc"] = near_far_auc(
                valid_data, direction_score,
                "near_invisible", "far_invisible",
            )
            external_metrics["near_invisible_vs_far_visible_auroc"] = near_far_auc(
                valid_data, direction_score,
                "near_invisible", "far_visible",
            )
            external_metrics["near_inv_vs_near_visible_auroc"] = near_far_auc(
                valid_data, direction_score,
                "near_invisible", "near_visible",
            )
            external_metrics["far_inv_vs_far_visible_auroc"] = near_far_auc(
                valid_data, direction_score,
                "far_invisible", "far_visible",
            )
            variant_output["heads"][head] = {
                "oof": oof_metrics,
                "external": external_metrics,
            }
            predictions[(variant, head, "oof")] = variant_result["oof"][head]
            predictions[(variant, head, "val")] = variant_result["external"][head]["probabilities"]
        # Near-inv trigger thresholds selected on Train OOF only.
        train_score = variant_result["oof"]["near_target_invisible"]
        valid_score = variant_result["external"]["near_target_invisible"]["probabilities"]
        train_thresholds = select_thresholds(train_data, train_score)
        trigger_points = {}
        for name, entry in train_thresholds.items():
            if entry is None:
                trigger_points[name] = None
                continue
            val_entry = real_metrics(valid_data, valid_score, entry["threshold"])
            val_entry["ci"] = bootstrap_real(
                valid_data["episode_id"], valid_data, valid_score,
                entry["threshold"], n_boot=args.bootstrap,
            )
            val_entry["train_threshold"] = entry["threshold"]
            val_entry["bands"] = {}
            for code, band_name in BAND_NAMES.items():
                band_mask = valid_data["band"] == code
                if not np.any(band_mask):
                    continue
                band_data = {
                    key: value[band_mask]
                    for key, value in valid_data.items()
                }
                val_entry["bands"][band_name] = real_metrics(
                    band_data, valid_score[band_mask],
                    entry["threshold"],
                )
            val_entry["distance_bins"] = {}
            for low, high in ((0, 20), (20, 50), (50, 100),
                              (100, 200), (200, 400)):
                bin_mask = (
                    (valid_data["distance_m"] >= low)
                    & (valid_data["distance_m"] < high)
                )
                if not np.any(bin_mask):
                    continue
                bin_data = {
                    key: value[bin_mask]
                    for key, value in valid_data.items()
                }
                val_entry["distance_bins"][f"{low}-{high}"] = real_metrics(
                    bin_data, valid_score[bin_mask], entry["threshold"]
                )
            trigger_points[name] = val_entry
        variant_output["trigger_points"] = trigger_points
        variant_output["train_thresholds"] = train_thresholds
        variant_output["episode_stats"] = episode_stats(
            valid_data, trigger_points.get("f1_joint_far5", {}).get("threshold", 0.5)
        )
        results["variants"][str(variant)] = variant_output

    # Goal-token-only baseline: observable scene/appearance prior control.
    print("evaluating goal-only baseline", flush=True)
    goal_idx = goal_indices(feature_names)
    goal_only = run_single_input(
        train_data, valid_data, folds, args, device,
        train_data["current_features"][:, goal_idx],
        valid_data["current_features"][:, goal_idx],
    )
    results["goal_only_baseline"] = goal_only

    # Synthetic model comparison on the same real Dev-Val states.
    synthetic = None
    if args.synthetic_checkpoint is not None:
        print("evaluating synthetic P1-2 comparison", flush=True)
        synthetic_model, s_mean, s_std, s_weight, s_threshold = (
            load_probe_for_comparison(args.synthetic_checkpoint, device)
        )
        synthetic_input = build_input(valid_data, 1)
        synthetic_score = batch_predict(
            synthetic_model, s_mean, s_std, s_weight,
            synthetic_input, device,
        )
        synthetic_metrics = metric_block(
            valid_data["near_target_invisible"], synthetic_score
        )
        synthetic_metrics["trigger"] = real_metrics(
            valid_data, synthetic_score, s_threshold
        )
        real_score = predictions[(1, "near_target_invisible", "val")]
        near = real_masks(valid_data)["near_invisible"]
        synthetic = {
            "checkpoint": str(args.synthetic_checkpoint),
            "threshold": float(s_threshold),
            "metrics": synthetic_metrics,
            "score_quantiles": {
                str(q): float(np.quantile(synthetic_score, q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
            "real_trained_score_quantiles": {
                str(q): float(np.quantile(real_score, q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            },
            "ks_statistic_near_invisible": float(ks_2samp(
                real_score[near], synthetic_score[near]
            ).statistic),
            "near_inv_vs_far_inv_auroc_synthetic": near_far_auc(
                valid_data, synthetic_score, "near_invisible",
                "far_invisible",
            ),
            "near_inv_vs_far_inv_auroc_real": near_far_auc(
                valid_data, real_score, "near_invisible", "far_invisible",
            ),
        }
    results["synthetic_comparison"] = synthetic

    # Gate decision for P1-2 (primary) and P0.
    def gate(variant_name):
        points = results["variants"][variant_name]["trigger_points"]
        candidates = [
            points.get("f1_joint_far5"),
            points.get("recall_0.5_far5"),
        ]
        candidates = [entry for entry in candidates if entry is not None]
        if not candidates:
            return {"pass": False, "reason": "no constrained threshold"}
        best = max(
            candidates,
            key=lambda entry: (
                entry["precision"] >= 0.5,
                entry["recall"],
                entry["precision"],
            ),
        )
        stats = episode_stats(valid_data, best["threshold"])
        near_total = max(
            int(valid_data["near_target_invisible"].sum()), 1
        )
        band_counts = [
            int(item["near_invisible"]) for item in stats["bands"].values()
        ]
        distance_counts = [
            int(item["near_invisible"])
            for item in stats["distance_bins"].values()
        ]
        largest_band_share = max(band_counts) / near_total
        largest_distance_share = max(distance_counts) / near_total
        dominance = {
            "near_invisible_total": near_total,
            "episodes_with_near_invisible": stats[
                "episodes_with_near_invisible"
            ],
            "largest_band_share": float(largest_band_share),
            "largest_distance_share": float(largest_distance_share),
            "bands": stats["bands"],
            "distance_bins": stats["distance_bins"],
            "ok": bool(
                largest_band_share <= 0.80
                and largest_distance_share <= 0.80
                and stats["episodes_with_near_invisible"] >= 5
            ),
        }
        checks = {
            "precision_ge_0.50": best["precision"] >= 0.5,
            "recall_ge_0.50": best["recall"] >= 0.5,
            "far_visible_ft_le_0.05": best["far_visible_false_trigger"] <= 0.05,
            "far_invisible_ft_le_0.05": best["far_invisible_false_trigger"] <= 0.05,
            "not_dominated_by_few_episodes_or_distance": dominance["ok"],
        }
        return {
            "pass": all(checks.values()),
            "checks": checks,
            "selected_point": best,
            "dominance": dominance,
        }
    results["gate"] = {
        "P0": gate("0"),
        "P1-2": gate("1"),
    }
    p0_overall = results["variants"]["0"]["heads"][
        "near_target_invisible"
    ]["external"]["overall"]
    p1_overall = results["variants"]["1"]["heads"][
        "near_target_invisible"
    ]["external"]["overall"]
    auroc_gain = float(p1_overall["auroc"] - p0_overall["auroc"])
    auprc_gain = float(p1_overall["auprc"] - p0_overall["auprc"])
    history_gain = {
        "P0_auroc": float(p0_overall["auroc"]),
        "P1-2_auroc": float(p1_overall["auroc"]),
        "auroc_gain": auroc_gain,
        "P0_auprc": float(p0_overall["auprc"]),
        "P1-2_auprc": float(p1_overall["auprc"]),
        "auprc_gain": auprc_gain,
        "clear_gain": bool(auroc_gain >= 0.01 or auprc_gain >= 0.01),
    }
    results["gate"]["history_gain"] = history_gain
    results["gate"]["P1-2"]["checks"]["history_gain"] = history_gain[
        "clear_gain"
    ]
    results["gate"]["P1-2"]["pass"] = bool(
        results["gate"]["P1-2"]["pass"] and history_gain["clear_gain"]
    )

    # Fixed-sample reproduction of the saved P1-2 external model.
    reproduction = None
    if args.save_models:
        checkpoint_path = models_dir / "v1_near_target_invisible_external.pt"
        model, mean, std, weight, _threshold = load_probe_for_comparison(
            checkpoint_path, device
        )
        sample = build_input(valid_data, 1)[:256]
        sample_prob = batch_predict(model, mean, std, weight, sample, device)
        reference = predictions[(1, "near_target_invisible", "val")][:256]
        reproduction = {
            "checkpoint": str(checkpoint_path),
            "states": int(len(sample)),
            "max_abs_difference": float(np.max(np.abs(sample_prob - reference))),
            "mean_abs_difference": float(np.mean(np.abs(sample_prob - reference))),
        }
        (output_dir / "reproduction_check.json").write_text(
            json.dumps(reproduction, indent=2), encoding="utf-8"
        )
    results["reproduction"] = reproduction
    if args.save_models:
        thresholds = {
            str(v): {
                "near_target_invisible": results["variants"][str(v)][
                    "train_thresholds"
                ]
            }
            for v in (0, 1)
        }
        provenance = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "command": sys.argv,
            "data": data_dirs,
            "counts": results["data"],
            "config": {
                "folds": int(args.folds), "epochs": int(args.epochs),
                "hidden": int(args.hidden),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "seed": int(args.seed),
                "class_weight": "pos_weight = n_neg / n_pos",
                "standardize": "mean/std from training split; clip [-20,20]",
            },
            "feature_names": feature_names,
            "compact_feature_names": canonical_names,
            "history_order": ["t-1", "t-2"],
            "action_encoding": {
                "FORWARD": 0, "LEFT": 1, "RIGHT": 2, "STOP": 3,
                "one_hot_dim": 4,
            },
            "threshold_source": "Train OOF only",
            "thresholds": thresholds,
            "models": sorted(
                str(path.relative_to(output_dir))
                for path in models_dir.glob("*.pt")
            ),
        }
        (output_dir / "probe_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
    (output_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "tables.md").write_text(
        render_tables(results), encoding="utf-8"
    )
    print(json.dumps({
        "data": results["data"],
        "gate": {
            "P0": results["gate"]["P0"].get("pass"),
            "P1-2": results["gate"]["P1-2"].get("pass"),
            "history_gain": results["gate"]["history_gain"],
        },
        "P1-2_near_inv": {
            "auroc": results["variants"]["1"]["heads"]["near_target_invisible"]["external"]["overall"]["auroc"],
            "auprc": results["variants"]["1"]["heads"]["near_target_invisible"]["external"]["overall"]["auprc"],
        },
        "output": str(output_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
