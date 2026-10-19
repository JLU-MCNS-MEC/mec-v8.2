"""Stage-2g.4 long-horizon target-loss event probe.

Trains only lightweight MLP probes on existing Stage-2g.3 real closed-loop
states. No M0/M1 checkpoint is modified, no MEC is trained, no controller is
run and no Final Test data is read.

Variants:
    P0          current 2,858 observable features
    P1-2        current + 2-step compact history
    P1-8        current + 8-step compact history
    P1-16       current + 16-step compact history
    P1-16+mem   P1-16 + label-free observable last-seen memory

Each variant is trained separately for event labels K=4, 8, 16. Thresholds
are selected on Train 5-fold episode OOF only; Dev Val is evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g_probe import (  # noqa: E402
    assign_episode_folds,
    metric_block,
    prior_correct,
    standardize,
    train_and_predict,
)
from v83_spatial_stage2.stage2g4_data import (  # noqa: E402
    EVENT_KS,
    MEMORY_NAMES,
    build_variant,
    confidence_stats,
    event_episode_concentration,
    event_stats,
    load_event_split,
    view_mask,
)
from v83_spatial_stage2.stage2g4_metrics import (  # noqa: E402
    event_metrics,
    metric_summary,
    select_event_thresholds,
    subset_aucs,
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


def save_checkpoint(path, model, mean, std, weight, variant, k, fold,
                    input_dim, input_names, args, data_dirs, memory_stats):
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
        "input_names": list(input_names),
        "variant": variant,
        "event_k": int(k),
        "fold": int(fold),
        "seed": int(args.seed),
        "config": {
            "kind": "mlp",
            "hidden": int(args.hidden),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "event_definition": (
            "distance<100 AND target_visible==0 AND "
            "target_localizable within previous K steps"
        ),
        "memory": (
            None if memory_stats is None else {
                "components": memory_stats["components"],
                "percentile": memory_stats["percentile"],
                "threshold": memory_stats["threshold"],
                "mean": memory_stats["mean"],
                "std": memory_stats["std"],
                "names": list(MEMORY_NAMES),
            }
        ),
        "data": dict(data_dirs),
        "git_commit": git_commit(),
    }
    torch.save(checkpoint, path)


def train_condition(train_x, train_y, val_x, val_y, folds, args, device,
                    output_dir, models_dir, variant, k, input_names,
                    data_dirs, memory_stats):
    output = {
        "variant": variant,
        "k": int(k),
        "input_dim": int(train_x.shape[1]),
        "train_states": int(len(train_x)),
        "train_positives": int(train_y.sum()),
        "val_states": int(len(val_x)),
        "val_positives": int(val_y.sum()),
        "train_positive_rate": float(train_y.mean()),
        "val_positive_rate": float(val_y.mean()),
        "fold_positives": {},
        "oof": None,
        "thresholds": None,
        "external": {},
        "checkpoints": [],
    }
    oof = np.full(len(train_x), np.nan, dtype=np.float64)
    fold_weights = {}
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        valid_mask = folds == fold
        train_std, valid_std, mean, std = standardize(
            train_x[train_mask], train_x[valid_mask], return_stats=True
        )
        y_train = train_y[train_mask]
        weight = float(
            (len(y_train) - y_train.sum()) / max(float(y_train.sum()), 1.0)
        )
        probabilities, info, model = train_and_predict(
            train_std, y_train, valid_std, "mlp", args.hidden,
            args.epochs, args.batch_size, args.lr, args.weight_decay,
            weight, args.seed + int(fold), device, return_model=True,
        )
        oof[valid_mask] = prior_correct(probabilities, weight)
        fold_weights[str(fold)] = float(weight)
        output["fold_positives"][str(fold)] = {
            "train_positives": int(y_train.sum()),
            "train_states": int(len(y_train)),
            "pos_weight": float(weight),
            "train_info": info,
        }
        checkpoint_path = (
            models_dir / f"{variant}_K{k}_fold{fold}.pt"
        )
        if args.save_models:
            save_checkpoint(
                checkpoint_path, model, mean, std, weight, variant, k, fold,
                train_x.shape[1], input_names, args, data_dirs, memory_stats,
            )
            output["checkpoints"].append(str(checkpoint_path.relative_to(output_dir)))
    train_std, val_std, mean, std = standardize(
        train_x, val_x, return_stats=True
    )
    weight = float(
        (len(train_y) - train_y.sum()) / max(float(train_y.sum()), 1.0)
    )
    probabilities, info, model = train_and_predict(
        train_std, train_y, val_std, "mlp", args.hidden,
        args.epochs, args.batch_size, args.lr, args.weight_decay,
        weight, args.seed + 1000, device, return_model=True,
    )
    val_prob = prior_correct(probabilities, weight)
    checkpoint_path = models_dir / f"{variant}_K{k}_external.pt"
    if args.save_models:
        save_checkpoint(
            checkpoint_path, model, mean, std, weight, variant, k, -1,
            train_x.shape[1], input_names, args, data_dirs, memory_stats,
        )
        output["checkpoints"].append(str(checkpoint_path.relative_to(output_dir)))
    output["external_train_info"] = {
        "pos_weight": float(weight),
        "train_info": info,
    }
    return oof, val_prob, output, fold_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+",
        default=["P0", "P1-2", "P1-8", "P1-16", "P1-16+mem"],
    )
    parser.add_argument("--ks", nargs="+", type=int, default=list(EVENT_KS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--exclude-episodes", nargs="*", type=int, default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    print("loading splits", flush=True)
    train_raw = load_event_split(args.train_dir)
    val_raw = load_event_split(args.val_dir)
    data_dirs = {
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "train_manifest_sha256": sha256(Path(args.train_dir) / "manifest.json"),
        "val_manifest_sha256": sha256(Path(args.val_dir) / "manifest.json"),
    }
    if args.exclude_episodes:
        excluded = set(int(x) for x in args.exclude_episodes)
        keep = ~np.isin(train_raw["episode_id"], list(excluded))
        train_raw = view_mask(train_raw, keep)
        print(
            f"excluded train episodes {sorted(excluded)}; "
            f"remaining states {len(train_raw['step'])}",
            flush=True,
        )
    train_common = train_raw["common_mask"]
    val_common = val_raw["common_mask"]
    train_data = view_mask(train_raw, train_common)
    val_data = view_mask(val_raw, val_common)
    train_x_index = np.flatnonzero(train_common)
    val_x_index = np.flatnonzero(val_common)
    folds = assign_episode_folds(
        train_data["episode_id"], train_data["band"], args.folds, args.seed
    )
    memory_stats = confidence_stats(train_raw) if "P1-16+mem" in args.variants else None

    results = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "command": sys.argv,
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
            "excluded_train_episodes": (
                sorted(int(x) for x in args.exclude_episodes)
                if args.exclude_episodes else []
            ),
            "no_training_of_mec": True,
            "no_controller": True,
            "final_test_accessed": False,
        },
        "config": {
            "variants": list(args.variants),
            "ks": [int(k) for k in args.ks],
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(device),
            "common_state_mask": "step>=16 and action!=STOP",
            "threshold_source": "Train OOF only",
            "class_weight": "pos_weight = n_neg / n_pos",
            "standardize": "Train mean/std; clip [-20,20]",
        },
        "data": {
            "train_raw_states": int(len(train_raw["step"])),
            "train_common_states": int(len(train_data["step"])),
            "train_episodes": int(np.unique(train_data["episode_id"]).size),
            "val_raw_states": int(len(val_raw["step"])),
            "val_common_states": int(len(val_data["step"])),
            "val_episodes": int(np.unique(val_data["episode_id"]).size),
            "train_manifest_sha256": data_dirs["train_manifest_sha256"],
            "val_manifest_sha256": data_dirs["val_manifest_sha256"],
        },
        "conditions": {},
        "memory": None if memory_stats is None else {
            **memory_stats,
            "names": list(MEMORY_NAMES),
            "built_from": "Train observable features only; labels not read",
        },
    }
    # Event stats are computed on the unmasked raw split so coverage is
    # comparable with the Stage-2g.3 report.
    results["event_stats"] = {
        "train": event_stats(train_raw),
        "val": event_stats(val_raw),
    }
    if not args.exclude_episodes:
        results["event_stats"]["train"]["concentration"] = {
            str(k): event_episode_concentration(train_raw, k)
            for k in args.ks
        }
        results["event_stats"]["val"]["concentration"] = {
            str(k): event_episode_concentration(val_raw, k)
            for k in args.ks
        }

    train_history_cache = {}
    val_history_cache = {}
    predictions = {
        "episode_id": np.asarray(val_data["episode_id"]),
        "band": np.asarray(val_data["band"]),
        "step": np.asarray(val_data["step"]),
        "distance_m": np.asarray(val_data["distance_m"]),
        "target_visible": np.asarray(val_data["target_visible"]),
        "target_localizable": np.asarray(val_data["target_localizable"]),
        "near_target_invisible": np.asarray(val_data["near_target_invisible"]),
    }
    for k in args.ks:
        predictions[f"event_K{k}"] = np.asarray(val_data["events"][k])

    for variant in args.variants:
        print(f"building variant {variant}", flush=True)
        train_matrix, input_names = build_variant(
            train_raw, variant, train_history_cache, memory_stats
        )
        val_matrix, val_names = build_variant(
            val_raw, variant, val_history_cache, memory_stats
        )
        if list(input_names) != list(val_names):
            raise RuntimeError("Train/Val input names differ")
        train_x = np.asarray(train_matrix[train_x_index], dtype=np.float32)
        val_x = np.asarray(val_matrix[val_x_index], dtype=np.float32)
        if train_x.shape[1] != len(input_names):
            raise RuntimeError(
                f"{variant}: input_dim {train_x.shape[1]} != "
                f"names {len(input_names)}"
            )
        del train_matrix, val_matrix
        for k in args.ks:
            print(f"  training {variant} K={k}", flush=True)
            train_y = np.asarray(train_data["events"][k], dtype=int)
            val_y = np.asarray(val_data["events"][k], dtype=int)
            oof, val_prob, condition, fold_weights = train_condition(
                train_x, train_y, val_x, val_y, folds, args, device,
                output_dir, models_dir, variant, k, input_names,
                data_dirs, memory_stats,
            )
            condition["oof"] = metric_summary(
                train_data, k, oof
            )
            condition["thresholds"] = select_event_thresholds(
                train_data, k, oof
            )
            external_train_info = condition.pop("external_train_info")
            condition["external"] = metric_summary(
                val_data, k, val_prob
            )
            condition["external"].update(external_train_info)
            condition["trigger_points"] = {}
            for name, entry in condition["thresholds"].items():
                if name == "constraints" or entry is None:
                    continue
                value = event_metrics(
                    val_data, k, val_prob, entry["threshold"]
                )
                value["train_threshold"] = entry["threshold"]
                condition["trigger_points"][name] = value
            condition["subset_aucs"] = subset_aucs(
                val_data, k, val_prob
            )
            condition["fold_weights"] = fold_weights
            condition["memory"] = (
                None if memory_stats is None else {
                    "threshold": memory_stats["threshold"],
                    "percentile": memory_stats["percentile"],
                }
            )
            results["conditions"][f"{variant}|K{k}"] = condition
            predictions[f"{variant}|K{k}|oof"] = oof
            predictions[f"{variant}|K{k}|val"] = val_prob
        del train_x, val_x

    results["data"]["train_common_states_used"] = int(
        len(train_data["step"])
    )
    results["data"]["val_common_states_used"] = int(len(val_data["step"]))
    results["data"]["folds"] = {
        str(fold): int((folds == fold).sum())
        for fold in sorted(set(folds.tolist()))
    }
    (output_dir / "probe_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    provenance = {
        "created_utc": results["meta"]["created_utc"],
        "git_commit": results["meta"]["git_commit"],
        "command": sys.argv,
        "data": data_dirs,
        "config": results["config"],
        "data_counts": results["data"],
        "event_stats": results["event_stats"],
        "memory": results["memory"],
        "models": sorted(
            str(path.relative_to(output_dir))
            for path in models_dir.glob("*.pt")
        ) if args.save_models else [],
        "notes": (
            "No distance/visibility/localizability label is part of any "
            "model input. Memory is a deterministic function of observable "
            "features and actions, with a threshold fitted on Train only."
        ),
    }
    (output_dir / "probe_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        "data": results["data"],
        "conditions": len(results["conditions"]),
        "memory": results["memory"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
