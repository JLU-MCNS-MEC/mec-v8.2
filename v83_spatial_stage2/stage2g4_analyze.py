"""Stage-2g.4 post-hoc analysis and frozen-gate evaluation.

Reads the probe results and saved predictions, recomputes every reported
metric from predictions, adds episode-level CIs, paired comparisons, LOEO,
legacy Stage-2g.3 baselines, observable-memory diagnostics and the frozen
Stage-2g.4 gate. No training and no new data collection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g_probe import (  # noqa: E402
    assign_episode_folds,
    auc_rank,
    average_precision,
)
from v83_spatial_stage2.stage2g3_probe import (  # noqa: E402
    batch_predict,
    load_probe_for_comparison,
)
from v83_spatial_stage2.stage2g4_data import (  # noqa: E402
    BAND_NAMES,
    EVENT_KS,
    MEMORY_NAMES,
    build_history,
    build_memory,
    confidence_stats,
    event_episode_concentration,
    event_stats,
    history_input,
    load_event_split,
    view_mask,
)
from v83_spatial_stage2.stage2g4_metrics import (  # noqa: E402
    episode_bootstrap,
    event_masks,
    event_metrics,
    leave_one_out,
    metric_summary,
    paired_bootstrap,
    per_episode_table,
    select_event_thresholds,
    stratified_event_vs_sustained,
    subset_aucs,
    threshold_grid,
)


def git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def take(data, mask):
    mask = np.asarray(mask, dtype=bool)
    output = {}
    for key, value in data.items():
        if key == "events":
            output[key] = {
                name: np.asarray(item)[mask]
                for name, item in value.items()
            }
        elif isinstance(value, np.ndarray) and value.shape[0] == len(mask):
            output[key] = value[mask]
        else:
            output[key] = value
    return output


def band_metrics(data, k, scores, threshold):
    output = {}
    for code, name in BAND_NAMES.items():
        mask = np.asarray(data["band"]) == code
        if not mask.any():
            continue
        subset = take(data, mask)
        output[name] = {
            "n": int(mask.sum()),
            "positives": int(np.asarray(data["events"][k])[mask].sum()),
            "metrics": metric_summary(subset, k, scores[mask]),
            "at_threshold": event_metrics(
                subset, k, scores[mask], threshold
            ),
        }
    return output


def condition_analysis(data, k, scores, saved, threshold):
    metrics = metric_summary(data, k, scores)
    trigger = event_metrics(data, k, scores, threshold)
    return {
        "metrics": metrics,
        "at_threshold": trigger,
        "threshold": float(threshold),
        "bands": band_metrics(data, k, scores, threshold),
        "subset_aucs": subset_aucs(data, k, scores),
        "subset_aucs_stratified": stratified_event_vs_sustained(
            data, k, scores
        ),
        "per_episode_top_false_positives": per_episode_table(
            data, k, scores, threshold
        )[:10],
        "saved_auroc": saved["external"]["auroc"],
        "saved_auprc": saved["external"]["auprc"],
        "auroc_abs_diff": abs(metrics["auroc"] - saved["external"]["auroc"]),
        "auprc_abs_diff": abs(metrics["auprc"] - saved["external"]["auprc"]),
    }


def legacy_scores(train_raw, val_raw, train_data, val_data, train_common,
                  val_common, probe_dir):
    """Apply the saved Stage-2g.3 P0/P1-2 models to the event label."""
    device = torch.device("cpu")
    close = train_raw["close_mask"]
    full_folds = assign_episode_folds(
        train_raw["episode_id"][close],
        train_raw["band"][close],
        5,
        0,
    )
    fold_by_state = np.full(len(close), -1, dtype=np.int64)
    fold_by_state[close] = full_folds
    outputs = {}
    for name, variant in (("legacy_P0", 0), ("legacy_P1-2", 1)):
        if variant == 0:
            train_x = train_data["current_features"]
            val_x = val_data["current_features"]
        else:
            h_train, a_train, dt_train = build_history(train_raw, 2)
            h_val, a_val, dt_val = build_history(val_raw, 2)
            train_x = history_input(train_raw, h_train, a_train, dt_train)[
                train_common
            ]
            val_x = history_input(val_raw, h_val, a_val, dt_val)[val_common]
        oof = np.full(len(train_x), np.nan, dtype=np.float64)
        for fold in range(5):
            checkpoint = (
                Path(probe_dir) / "models"
                / f"v{variant}_near_target_invisible_fold{fold}.pt"
            )
            model, mean, std, weight, _ = load_probe_for_comparison(
                checkpoint, device
            )
            mask = fold_by_state[train_common] == fold
            oof[mask] = batch_predict(
                model, mean, std, weight, train_x[mask], device
            )
        checkpoint = (
            Path(probe_dir) / "models"
            / f"v{variant}_near_target_invisible_external.pt"
        )
        model, mean, std, weight, _ = load_probe_for_comparison(
            checkpoint, device
        )
        val_score = batch_predict(
            model, mean, std, weight, val_x, device
        )
        outputs[name] = {"oof": oof, "val": val_score}
    return outputs


def oracle_threshold_sweep(data, k, scores):
    """Best achievable Dev-Val operating points; diagnostic only."""
    rows = []
    for threshold in threshold_grid(scores):
        metrics = event_metrics(data, k, scores, threshold)
        if metrics["flags"]:
            rows.append(metrics)
    if not rows:
        return {}
    best_f1 = max(rows, key=lambda row: row["f1"])
    best_precision = max(rows, key=lambda row: row["precision"])
    precision_half = [
        row for row in rows if row["precision"] >= 0.5
    ]
    recall_half = [
        row for row in rows if row["recall"] >= 0.5
    ]
    return {
        "max_f1": {
            "f1": best_f1["f1"],
            "precision": best_f1["precision"],
            "recall": best_f1["recall"],
            "threshold": best_f1["threshold"],
        },
        "max_precision": {
            "precision": best_precision["precision"],
            "recall": best_precision["recall"],
            "threshold": best_precision["threshold"],
        },
        "max_recall_at_precision_0.5": (
            max(precision_half, key=lambda row: row["recall"])["recall"]
            if precision_half else None
        ),
        "max_precision_at_recall_0.5": (
            max(recall_half, key=lambda row: row["precision"])["precision"]
            if recall_half else None
        ),
    }


def oracle_diagnostics(data, k):
    event = np.asarray(data["events"][k], dtype=bool)
    distance = np.asarray(data["distance_m"], dtype=float)
    visible = np.asarray(data["target_visible"], dtype=int)
    since = np.asarray(data["since_localizable"], dtype=int)
    masks = event_masks(data, k)
    scores = {
        "distance_only": -distance,
        "visibility_only": 1.0 - visible,
        "since_localizable_only": -np.where(since >= 0, since, 999),
    }
    output = {}
    for name, score in scores.items():
        entry = {"event_vs_all": auc_rank(event.astype(int), score)}
        for negative_name in ("sustained_near_inv", "far_visible", "far_invisible"):
            negative = masks[negative_name]
            if event.any() and negative.any():
                y = np.concatenate([
                    np.ones(int(event.sum())),
                    np.zeros(int(negative.sum())),
                ])
                values = np.concatenate([
                    score[event], score[negative]
                ])
                entry[f"event_vs_{negative_name}"] = auc_rank(y, values)
        entry["event_vs_sustained_stratified"] = (
            stratified_event_vs_sustained(data, k, score)
        )
        output[name] = entry
    return output


def gate_checks(data, k, primary_name, primary_score, p0, p1_2, p1_16,
                thresholds, primary_thresholds, n_boot=1000):
    threshold = float(thresholds[primary_name])
    metrics = event_metrics(data, k, primary_score, threshold)
    summary = metric_summary(data, k, primary_score)
    aucs = subset_aucs(data, k, primary_score)
    g1 = primary_thresholds["precision_0.5_constrained"] is not None
    g2 = metrics["precision"] >= 0.5 and metrics["recall"] >= 0.5
    g3 = (
        metrics["fp"] <= metrics["tp"]
        and metrics["per_episode_false_positives"]["mean"] <= 1.0
    )
    g4 = (
        aucs.get("sustained_near_inv", float("-inf")) >= 0.70
        and aucs.get("far_invisible", float("-inf")) >= 0.75
    )
    pair_p0 = paired_bootstrap(
        data, k, p0, primary_score,
        float(thresholds["P0"]), threshold,
        n_boot=n_boot,
    ) if "P0" in thresholds else None
    pair_p12 = paired_bootstrap(
        data, k, p1_2, primary_score,
        float(thresholds["P1-2"]), threshold,
        n_boot=n_boot,
    ) if "P1-2" in thresholds else None
    over_p0 = False
    over_p12 = False
    if pair_p0:
        over_p0 = (
            (pair_p0["auroc"]["mean"] >= 0.03 and pair_p0["auroc"]["low"] > 0)
            or (pair_p0["auprc"]["mean"] >= 0.03 and pair_p0["auprc"]["low"] > 0)
        )
    if pair_p12:
        over_p12 = (
            (pair_p12["auroc"]["mean"] >= 0.02 and pair_p12["auroc"]["low"] > 0)
            or (pair_p12["auprc"]["mean"] >= 0.02 and pair_p12["auprc"]["low"] > 0)
        )
    g5 = bool(over_p0 and over_p12)
    episodes = np.asarray(data["episode_id"])
    event = np.asarray(data["events"][k]) == 1
    counts = defaultdict(int)
    for episode in episodes[event]:
        counts[int(episode)] += 1
    top_episode = max(counts, key=counts.get) if counts else None
    loeo_rows = leave_one_out(
        data, k, primary_score, threshold,
        [top_episode] if top_episode is not None else [],
    )
    g6 = False
    loeo = None
    if loeo_rows:
        loeo = loeo_rows[0]
        g6 = (
            loeo["at_threshold_precision"] >= 0.5
            and loeo["at_threshold_recall"] >= 0.5
            and loeo["at_threshold_fp"] <= loeo["at_threshold_tp"]
            and (summary["auroc"] - loeo["auroc"]) < 0.05
        )
    return {
        "k": int(k),
        "primary": primary_name,
        "threshold": float(threshold),
        "checks": {
            "G1_train_precision_0.5_constrained_exists": bool(g1),
            "G2_dev_precision_recall_ge_0.5": bool(g2),
            "G3_absolute_fp_budget": bool(g3),
            "G4_event_vs_sustained_and_far_invisible": bool(g4),
            "G5_long_history_gain": bool(g5),
            "G6_leave_one_extreme_episode_out": bool(g6),
        },
        "evidence": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fp_per_episode_mean": metrics["per_episode_false_positives"]["mean"],
            "auroc": summary["auroc"],
            "auprc": summary["auprc"],
            "event_vs_sustained_auroc": aucs.get("sustained_near_inv"),
            "event_vs_far_invisible_auroc": aucs.get("far_invisible"),
            "paired_vs_P0": pair_p0,
            "paired_vs_P1-2": pair_p12,
            "top_episode": top_episode,
            "top_episode_positives": (
                int(counts[top_episode]) if top_episode is not None else 0
            ),
            "loeo_top_episode": loeo,
        },
        "pass": bool(g1 and g2 and g3 and g4 and g5 and g6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--stage2g3-probe-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--paired-bootstrap", type=int, default=1000)
    parser.add_argument("--ablation-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading data and predictions", flush=True)
    train_raw = load_event_split(args.train_dir)
    val_raw = load_event_split(args.val_dir)
    train_common = train_raw["common_mask"]
    val_common = val_raw["common_mask"]
    train_data = view_mask(train_raw, train_common)
    val_data = view_mask(val_raw, val_common)
    results = json.loads(
        (Path(args.probe_dir) / "probe_results.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = np.load(Path(args.probe_dir) / "predictions.npz")
    conditions = results["conditions"]
    ks = [int(k) for k in results["config"]["ks"]]
    variants = list(results["config"]["variants"])

    def event_distance_bins(data, k):
        event = np.asarray(data["events"][k]) & data["common_mask"]
        distance = np.asarray(data["distance_m"])
        output = {}
        for low, high in ((0, 20), (20, 50), (50, 100)):
            mask = event & (distance >= low) & (distance < high)
            output[f"{low}-{high}"] = int(mask.sum())
        output["total"] = int(event.sum())
        episodes = np.asarray(data["episode_id"])[event]
        output["episodes"] = int(np.unique(episodes).size)
        return output

    analysis = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
            "probe_dir": str(args.probe_dir),
            "train_dir": str(args.train_dir),
            "val_dir": str(args.val_dir),
            "no_training_performed": True,
            "final_test_accessed": False,
        },
        "event_stats": {
            "train": event_stats(train_raw),
            "val": event_stats(val_raw),
        },
        "data": {
            "train_common_states": int(train_common.sum()),
            "val_common_states": int(val_common.sum()),
            "train_common_episodes": int(
                np.unique(train_data["episode_id"]).size
            ),
            "val_common_episodes": int(
                np.unique(val_data["episode_id"]).size
            ),
        },
        "conditions": {},
        "paired": {},
        "loeo": {},
        "loeo_train": {},
        "legacy": {},
        "oracle": {},
        "oracle_threshold_sweep": {},
        "memory": {},
        "ablation": {},
    }
    for k in ks:
        analysis["event_stats"][f"concentration_train_K{k}"] = (
            event_episode_concentration(train_raw, k)
        )
        analysis["event_stats"][f"concentration_val_K{k}"] = (
            event_episode_concentration(val_raw, k)
        )
        analysis["event_stats"][f"distance_bins_train_K{k}"] = (
            event_distance_bins(train_raw, k)
        )
        analysis["event_stats"][f"distance_bins_val_K{k}"] = (
            event_distance_bins(val_raw, k)
        )

    print("condition metrics", flush=True)
    for key, saved in conditions.items():
        variant = saved["variant"]
        k = int(saved["k"])
        oof = predictions[f"{variant}|K{k}|oof"]
        val = predictions[f"{variant}|K{k}|val"]
        threshold_entry = saved["thresholds"].get("f1_joint")
        threshold = (
            float(threshold_entry["threshold"])
            if threshold_entry is not None else 0.5
        )
        entry = condition_analysis(
            val_data, k, val, saved, threshold
        )
        entry["variant"] = variant
        entry["k"] = k
        entry["oof_metrics"] = metric_summary(train_data, k, oof)
        entry["oof_audit"] = {
            "auroc_saved": saved["oof"]["auroc"],
            "auroc_recomputed": entry["oof_metrics"]["auroc"],
            "auprc_saved": saved["oof"]["auprc"],
            "auprc_recomputed": entry["oof_metrics"]["auprc"],
        }
        entry["bootstrap"] = episode_bootstrap(
            val_data, k, val, threshold,
            n_boot=args.bootstrap,
        )
        analysis["conditions"][key] = entry

    print("paired comparisons", flush=True)
    for k in ks:
        base = f"P0|K{k}"
        if base not in analysis["conditions"]:
            continue
        p0 = predictions[f"P0|K{k}|val"]
        p0_thr = analysis["conditions"][base]["threshold"]
        for variant in variants:
            key = f"{variant}|K{k}"
            if key not in analysis["conditions"] or variant == "P0":
                continue
            score = predictions[f"{variant}|K{k}|val"]
            threshold = analysis["conditions"][key]["threshold"]
            analysis["paired"][f"{variant}_vs_P0_K{k}"] = paired_bootstrap(
                val_data, k, p0, score, p0_thr, threshold,
                n_boot=args.paired_bootstrap,
            )
        if f"P1-2|K{k}" in analysis["conditions"]:
            p12 = predictions[f"P1-2|K{k}|val"]
            p12_thr = analysis["conditions"][f"P1-2|K{k}"]["threshold"]
            for variant in ("P1-8", "P1-16", "P1-16+mem"):
                key = f"{variant}|K{k}"
                if key not in analysis["conditions"]:
                    continue
                score = predictions[f"{variant}|K{k}|val"]
                threshold = analysis["conditions"][key]["threshold"]
                analysis["paired"][f"{variant}_vs_P1-2_K{k}"] = (
                    paired_bootstrap(
                        val_data, k, p12, score, p12_thr, threshold,
                        n_boot=args.paired_bootstrap,
                    )
                )
        if (
            f"P1-16|K{k}" in analysis["conditions"]
            and f"P1-16+mem|K{k}" in analysis["conditions"]
        ):
            base_score = predictions[f"P1-16|K{k}|val"]
            base_thr = analysis["conditions"][f"P1-16|K{k}"]["threshold"]
            score = predictions[f"P1-16+mem|K{k}|val"]
            threshold = analysis["conditions"][f"P1-16+mem|K{k}"]["threshold"]
            analysis["paired"][f"P1-16+mem_vs_P1-16_K{k}"] = (
                paired_bootstrap(
                    val_data, k, base_score, score, base_thr, threshold,
                    n_boot=args.paired_bootstrap,
                )
            )

    print("LOEO", flush=True)
    for k in ks:
        for variant in variants:
            key = f"{variant}|K{k}"
            if key not in analysis["conditions"]:
                continue
            threshold = analysis["conditions"][key]["threshold"]
            val_score = predictions[f"{variant}|K{k}|val"]
            top = event_episode_concentration(val_raw, k)["top_episodes"]
            episodes = [row["episode_id"] for row in top[:5]]
            analysis["loeo"][key] = leave_one_out(
                val_data, k, val_score, threshold, episodes
            )
            train_top = event_episode_concentration(train_raw, k)[
                "top_episodes"
            ]
            train_episodes = [row["episode_id"] for row in train_top[:5]]
            if 299 not in train_episodes:
                train_episodes.append(299)
            analysis.setdefault("loeo_train", {})[key] = leave_one_out(
                train_data, k, predictions[f"{variant}|K{k}|oof"],
                threshold, train_episodes,
            )

    print("legacy baselines", flush=True)
    if args.stage2g3_probe_dir is not None:
        legacy = legacy_scores(
            train_raw, val_raw, train_data, val_data,
            train_common, val_common, args.stage2g3_probe_dir,
        )
        for name, scores in legacy.items():
            for k in ks:
                thresholds = select_event_thresholds(
                    train_data, k, scores["oof"]
                )
                entry = thresholds["f1_joint"]
                if entry is None:
                    threshold = 0.5
                else:
                    threshold = float(entry["threshold"])
                analysis["legacy"][f"{name}|K{k}"] = {
                    "thresholds": thresholds,
                    "oof": metric_summary(train_data, k, scores["oof"]),
                    "val": metric_summary(val_data, k, scores["val"]),
                    "at_threshold": event_metrics(
                        val_data, k, scores["val"], threshold
                    ),
                    "subset_aucs": subset_aucs(
                        val_data, k, scores["val"]
                    ),
                    "subset_aucs_stratified": (
                        stratified_event_vs_sustained(
                            val_data, k, scores["val"]
                        )
                    ),
                }

    print("oracle and memory diagnostics", flush=True)
    for k in ks:
        analysis["oracle"][f"K{k}"] = oracle_diagnostics(val_data, k)
        for variant in ("P1-16", "P1-16+mem"):
            key = f"{variant}|K{k}"
            if f"{key}|val" in predictions:
                analysis["oracle_threshold_sweep"][key] = (
                    oracle_threshold_sweep(
                        val_data, k, predictions[f"{key}|val"]
                    )
                )
    memory_stats = confidence_stats(train_raw)
    memory = build_memory(val_raw, memory_stats)
    memory_common = memory[val_common]
    age = memory_common[:, MEMORY_NAMES.index("mem_age_high")]
    for k in ks:
        event = np.asarray(val_data["events"][k], dtype=bool)
        masks = event_masks(val_data, k)
        entry = {}
        for name in ("sustained_near_inv", "far_invisible", "far_visible"):
            negative = masks[name]
            if event.any() and negative.any():
                y = np.concatenate([
                    np.ones(int(event.sum())),
                    np.zeros(int(negative.sum())),
                ])
                values = np.concatenate([-age[event], -age[negative]])
                entry[f"event_vs_{name}_age_auroc"] = auc_rank(y, values)
        entry["event_age_quantiles"] = {
            str(q): float(np.quantile(age[event], q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        } if event.any() else {}
        entry["sustained_age_quantiles"] = {
            str(q): float(np.quantile(age[masks["sustained_near_inv"]], q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        } if masks["sustained_near_inv"].any() else {}
        analysis["memory"][f"K{k}"] = entry
    analysis["memory"]["threshold"] = memory_stats
    analysis["memory"]["names"] = list(MEMORY_NAMES)

    print("gate", flush=True)
    primary = {}
    for k in ks:
        candidates = [
            f"{name}|K{k}" for name in ("P1-16", "P1-16+mem")
            if f"{name}|K{k}" in analysis["conditions"]
        ]
        if not candidates:
            continue
        primary[k] = max(
            candidates,
            key=lambda key: analysis["conditions"][key]["oof_metrics"]["auprc"],
        )
    analysis["primary"] = {str(k): v for k, v in primary.items()}
    gate = {}
    for k, key in primary.items():
        if k != 16:
            continue
        variant = analysis["conditions"][key]["variant"]
        threshold_map = {}
        for name in ("P0", "P1-2", "P1-16", "P1-16+mem"):
            condition_key = f"{name}|K{k}"
            if condition_key not in analysis["conditions"]:
                continue
            threshold_map[name] = analysis["conditions"][condition_key]["threshold"]
        gate[f"K{k}"] = gate_checks(
            val_data, k,
            variant,
            predictions[f"{variant}|K{k}|val"],
            predictions["P0|K16|val"],
            predictions["P1-2|K16|val"],
            predictions["P1-16|K16|val"],
            threshold_map,
            results["conditions"][key]["thresholds"],
            n_boot=args.paired_bootstrap,
        )
    analysis["gate"] = gate
    passed = any(entry["pass"] for entry in gate.values())
    analysis["decision"] = {
        "pass": bool(passed),
        "decision": (
            "PASS: report controlled MEC integration ablation plan only"
            if passed else
            "STOP the Recovery probe line; pivot to explicit target spatial "
            "belief / coarse navigation; do not stack MLP or history frames"
        ),
        "final_test_accessed": False,
    }

    print("ablation", flush=True)
    if args.ablation_dir is not None and Path(args.ablation_dir).is_dir():
        ablation = json.loads(
            (Path(args.ablation_dir) / "probe_results.json").read_text(
                encoding="utf-8"
            )
        )
        ablation_pred = np.load(
            Path(args.ablation_dir) / "predictions.npz"
        )
        for key, saved in ablation["conditions"].items():
            variant = saved["variant"]
            k = int(saved["k"])
            if k != 16:
                continue
            threshold_entry = saved["thresholds"].get("f1_joint")
            threshold = (
                float(threshold_entry["threshold"])
                if threshold_entry is not None else 0.5
            )
            score = ablation_pred[f"{variant}|K{k}|val"]
            analysis["ablation"][key] = {
                "threshold": threshold,
                "metrics": metric_summary(val_data, k, score),
                "at_threshold": event_metrics(
                    val_data, k, score, threshold
                ),
                "subset_aucs": subset_aucs(val_data, k, score),
                "subset_aucs_stratified": stratified_event_vs_sustained(
                    val_data, k, score
                ),
            }
    else:
        analysis["ablation"] = {
            "status": "not_run_or_missing",
            "expected": "separate probe run excluding extreme train episodes",
        }

    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir),
        "primary": analysis["primary"],
        "gate": analysis["gate"],
        "decision": analysis["decision"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
