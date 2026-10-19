"""Stage-2d R1/R2/R3 off-policy Arrival calibration.

R1: Platt scaling on the frozen base logit (2 parameters).
R2: 104-D -> 32 -> 1 residual on the frozen base logit, trained with hard-mined
    far negatives (B0 p >= 0.3) from Train-480 rollouts.
R3: identical R2 architecture/initialization/budget, but an equal-size random
    far-negative control instead of hard mining.

The original Arrival Head, backbone, context encoder and movement classifier
are frozen. Calibration only sees the observable 104-D Arrival feature and the
original B0 logit. True distance is used only to form labels and to select
training rows; it is never a model input.

Protocol:
  * episode split 480 train / 120 internal validation, fixed seed;
  * LR / epoch / config selection on the 120 internal-validation episodes;
  * Dev Val 180 is touched only after the configuration is fixed, for the
    development closed-loop evaluation;
  * Final Test 300 is sealed.

Weights and caches stay under the gitignored runs directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2d_common import (
    BANDS,
    M0_CKPT,
    SPLIT_SEED,
    STAGE2B_DEV_FEATURES,
    STAGE2C_ROLLOUT,
    TRAIN_EPISODES,
    VAL_SPECS,
    StopRule,
    arrival_head_hash,
    arrival_offline_metrics,
    bernoulli_kl,
    build_calibrator,
    build_oracle_cache,
    build_training_rows,
    calibrate_probabilities,
    device_summary,
    evaluate_policy,
    feature_normalization,
    file_sha256,
    head_logits,
    hazard_from_steps,
    load_dev_val_features,
    load_dev_specs,
    load_encoder,
    load_m0,
    load_rollout_steps,
    movement_hash,
    paired_by_band,
    parameter_count,
    per_decision_hazard,
    risk_rows,
    selection_score,
    sigmoid_numpy,
    split_episodes,
    stop_event_summary,
    summarize_by_band,
    verify_b0_replay,
    write_csv,
    write_json,
)
from v83_spatial_stage2.stage2d_common import episode_mask


TRAIN_SEED = 20264017
R1_LRS = (1e-2, 1e-1)
R1_ANCHORS = (1.0, 10.0)
R23_LRS = (1e-3, 3e-3, 1e-2)
R23_ANCHORS = (1.0, 10.0)
R23_RESIDUALS = (1e-3, 1e-2)
VARIANTS = ("R1", "R2", "R3")


def _config_id(kind, config):
    return (
        f"{kind}_lr{config['lr']:g}_anchor{config['lambda_anchor']:g}"
        f"_res{config['lambda_residual']:g}"
    )


def _variant_configs(kind, args):
    if kind == "R1":
        return [
            {"lr": float(lr), "lambda_anchor": float(anchor),
             "lambda_residual": 0.0}
            for lr in args.r1_lrs for anchor in args.r1_anchors
        ]
    return [
        {"lr": float(lr), "lambda_anchor": float(anchor),
         "lambda_residual": float(residual)}
        for lr in args.r23_lrs
        for anchor in args.r23_anchors
        for residual in args.r23_residuals
    ]


def _train_tensors(data, rows, anchor_mask, device):
    return {
        "features": torch.from_numpy(
            np.asarray(data["features"][rows], dtype=np.float32)
        ).to(device),
        "base_logit": torch.from_numpy(
            np.asarray(data["arrival_logit"][rows], dtype=np.float32)
        ).to(device),
        "labels": torch.from_numpy(
            np.asarray(data["arrival_label"][rows], dtype=np.float32)
        ).to(device),
        "anchor": torch.from_numpy(
            np.asarray(anchor_mask, dtype=bool)
        ).to(device),
    }


def _val_tensors(data, val_mask, device):
    return {
        "features": torch.from_numpy(
            np.asarray(data["features"][val_mask], dtype=np.float32)
        ).to(device),
        "base_logit": torch.from_numpy(
            np.asarray(data["arrival_logit"][val_mask], dtype=np.float32)
        ).to(device),
        "labels": np.asarray(data["arrival_label"][val_mask], dtype=np.float32),
        "distance": np.asarray(data["distance_m"][val_mask], dtype=np.float64),
        "episode": np.asarray(data["episode_index"][val_mask], dtype=np.int64),
    }


def _evaluate_validation(calibrator, val, b0_near_recall, threshold,
                         selection_smooth_weight, near_tolerance, penalty):
    calibrator.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(
            calibrator(val["features"], val["base_logit"])
        ).cpu().numpy()
    hazard = per_decision_hazard(
        probability, val["distance"], val["episode"], threshold=threshold
    )
    metrics = arrival_offline_metrics(
        val["labels"], probability, val["distance"],
        sample_type=None, threshold=threshold,
    )
    score = selection_score(
        hazard, b0_near_recall,
        near_tolerance=near_tolerance,
        penalty=penalty,
        smooth_weight=selection_smooth_weight,
    )
    return score, hazard, metrics, probability


def train_one_config(kind, config, data, rows_info, val, b0_near_recall,
                     feature_mean, feature_std, args, device):
    torch.manual_seed(int(args.seed))
    calibrator = build_calibrator(kind, feature_mean, feature_std).to(device)
    train = _train_tensors(data, rows_info["row_indices"],
                           rows_info["anchor_mask"], device)
    with torch.inference_mode():
        initial_logit = calibrator(train["features"][:64], train["base_logit"][:64])
        init_max_abs_diff = float(
            torch.max(torch.abs(initial_logit - train["base_logit"][:64])).item()
        )
    if init_max_abs_diff != 0.0:
        raise RuntimeError(
            f"{kind} does not reproduce B0 at initialization: {init_max_abs_diff}"
        )
    optimizer = torch.optim.AdamW(
        calibrator.parameters(), lr=float(config["lr"]),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator().manual_seed(int(args.seed))
    best_score = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(
        {key: value.detach().cpu().clone()
         for key, value in calibrator.state_dict().items()}
    )
    stale = 0
    history = []
    n_rows = int(train["features"].shape[0])
    start_time = time.time()

    for epoch in range(1, int(args.max_epochs) + 1):
        calibrator.train()
        permutation = torch.randperm(n_rows, generator=generator)
        loss_sum = 0.0
        batches = 0
        for start in range(0, n_rows, int(args.batch_size)):
            indices = permutation[start:start + int(args.batch_size)].to(device)
            features = train["features"][indices]
            base_logit = train["base_logit"][indices]
            labels = train["labels"][indices]
            anchor = train["anchor"][indices]
            calibrated_logit = calibrator(features, base_logit)
            # Deliberately plain BCE: no pos_weight (Stage-2b regression).
            loss = F.binary_cross_entropy_with_logits(calibrated_logit, labels)
            if anchor.any() and float(config["lambda_anchor"]) > 0.0:
                loss = loss + float(config["lambda_anchor"]) * bernoulli_kl(
                    base_logit[anchor], calibrated_logit[anchor]
                )
            if float(config["lambda_residual"]) > 0.0:
                residual = calibrated_logit - base_logit
                loss = loss + float(config["lambda_residual"]) * (
                    residual * residual
                ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss {kind} {config} epoch {epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
        score, hazard, metrics, _ = _evaluate_validation(
            calibrator, val, b0_near_recall, args.threshold,
            args.selection_smooth_weight,
            args.selection_near_tolerance,
            args.selection_penalty,
        )
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(batches, 1),
            "selection_score": score,
            "near20_recall": hazard["near20_stop_rate"],
            "far40_stop_rate": hazard["far40_stop_rate"],
            "far40_stops_per_episode": hazard[
                "expected_false_stop_decisions_per_episode"
            ],
            "predicted_far40_stops_per_episode": hazard[
                "predicted_false_stop_decisions_per_episode"
            ],
            "boundary_stop_rate": hazard["boundary_stop_rate"],
            "natural_f1": metrics["natural"]["f1"],
            "natural_brier": metrics["natural"]["brier"],
        })
        improved = score < best_score - 1e-9
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu().clone()
                 for key, value in calibrator.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"  {kind} {config['lr']:g}/{config['lambda_anchor']:g}/"
                f"{config['lambda_residual']:g} epoch {epoch:03d} "
                f"loss={history[-1]['train_loss']:.5f} "
                f"near={hazard['near20_stop_rate']:.3f} "
                f"far40={hazard['far40_stop_rate']:.5f} "
                f"far40stop/ep={hazard['expected_false_stop_decisions_per_episode']:.3f} "
                f"score={score:.4f}",
                flush=True,
            )
        if stale >= int(args.patience):
            break

    final = build_calibrator(kind, feature_mean, feature_std)
    final.load_state_dict(best_state)
    final.to(device)
    final.eval()
    score, hazard, metrics, _ = _evaluate_validation(
        final, val, b0_near_recall, args.threshold,
        args.selection_smooth_weight,
        args.selection_near_tolerance,
        args.selection_penalty,
    )
    return {
        "calibrator": final,
        "best_state": best_state,
        "config": config,
        "best_epoch": int(best_epoch),
        "best_score": float(score),
        "validation_hazard": hazard,
        "validation_metrics": metrics,
        "history": history,
        "init_max_abs_diff": init_max_abs_diff,
        "train_time_s": float(time.time() - start_time),
        "parameter_count": parameter_count(final),
        "train_rows": int(len(rows_info["row_indices"])),
        "anchor_rows": int(rows_info["anchor_mask"].sum()),
        "final_train_loss": history[-1]["train_loss"] if history else None,
    }


def train_variant(kind, data, rows_info, val, b0_near_recall, feature_mean,
                  feature_std, args, device, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    configs = _variant_configs(kind, args)
    grid_rows = []
    grid_history = []
    results = []
    for position, config in enumerate(configs, start=1):
        config_id = _config_id(kind, config)
        print(
            f"[{kind} {position}/{len(configs)}] {config_id}", flush=True,
        )
        result = train_one_config(
            kind, config, data, rows_info, val, b0_near_recall,
            feature_mean, feature_std, args, device,
        )
        results.append(result)
        grid_rows.append({
            "variant": kind,
            "config_id": config_id,
            "lr": config["lr"],
            "lambda_anchor": config["lambda_anchor"],
            "lambda_residual": config["lambda_residual"],
            "best_epoch": result["best_epoch"],
            "selection_score": result["best_score"],
            "near20_recall": result["validation_hazard"]["near20_stop_rate"],
            "far40_stop_rate": result["validation_hazard"]["far40_stop_rate"],
            "far40_stops_per_episode": result["validation_hazard"][
                "expected_false_stop_decisions_per_episode"
            ],
            "predicted_far40_stops_per_episode": result["validation_hazard"][
                "predicted_false_stop_decisions_per_episode"
            ],
            "boundary_stop_rate": result["validation_hazard"][
                "boundary_stop_rate"
            ],
            "natural_f1": result["validation_metrics"]["natural"]["f1"],
            "natural_brier": result["validation_metrics"]["natural"]["brier"],
            "parameter_count": result["parameter_count"],
            "init_max_abs_diff": result["init_max_abs_diff"],
            "train_time_s": result["train_time_s"],
        })
        for row in result["history"]:
            grid_history.append({"config_id": config_id, **row})
    write_csv(outdir / "grid_summary.csv", grid_rows)
    write_csv(outdir / "grid_history.csv", grid_history)

    # Deterministic tie-break: lower score, then earlier config in the grid.
    best_index = min(
        range(len(results)), key=lambda index: (results[index]["best_score"], index)
    )
    selected = results[best_index]
    write_csv(outdir / "history.csv", selected["history"])
    torch.save(
        {
            "format": "stage2d_calibration_v1",
            "kind": kind,
            "state_dict": selected["best_state"],
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "selected_config": selected["config"],
            "best_epoch": selected["best_epoch"],
            "selection_score": selected["best_score"],
            "validation_hazard": selected["validation_hazard"],
            "validation_metrics": selected["validation_metrics"],
            "init_max_abs_diff": selected["init_max_abs_diff"],
            "m0_checkpoint": str(M0_CKPT),
        },
        outdir / "calibration.pt",
    )
    base_state = torch.load(M0_CKPT, map_location="cpu", weights_only=True)
    torch.save(
        {
            "format": "stage2d_full_checkpoint_v1",
            "base_state": base_state,
            "calibration": torch.load(
                outdir / "calibration.pt", map_location="cpu",
                weights_only=False,
            ),
        },
        outdir / "best.pt",
    )
    result = {
        "variant": kind,
        "selected_config": selected["config"],
        "best_epoch": selected["best_epoch"],
        "selection_score": selected["best_score"],
        "parameter_count": selected["parameter_count"],
        "train_rows": selected["train_rows"],
        "anchor_rows": selected["anchor_rows"],
        "init_max_abs_diff": selected["init_max_abs_diff"],
        "train_time_s": selected["train_time_s"],
        "validation_hazard": selected["validation_hazard"],
        "validation_metrics": selected["validation_metrics"],
        "grid_size": len(configs),
        "grid_summary": grid_rows,
        "calibration_path": str(outdir / "calibration.pt"),
        "checkpoint_path": str(outdir / "best.pt"),
        "movement_hash": movement_hash(base_state),
        "arrival_head_hash": arrival_head_hash(base_state),
    }
    if kind == "R1":
        result["platt_scale"] = float(
            1.0 + selected["best_state"]["scale_delta"].item()
        )
        result["platt_bias"] = float(selected["best_state"]["bias"].item())
    else:
        result["residual_first_layer_norm"] = float(
            selected["best_state"]["net.0.weight"].norm().item()
        )
        result["residual_last_layer_norm"] = float(
            selected["best_state"]["net.2.weight"].norm().item()
        )
        result["residual_last_layer_bias"] = float(
            selected["best_state"]["net.2.bias"].item()
        )
    write_json(outdir / "metrics.json", result)
    print(
        f"Selected {kind}: {_config_id(kind, selected['config'])} "
        f"epoch={selected['best_epoch']} score={selected['best_score']:.4f} "
        f"near={selected['validation_hazard']['near20_stop_rate']:.4f} "
        f"far40={selected['validation_hazard']['far40_stop_rate']:.5f}",
        flush=True,
    )
    return selected["calibrator"], result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, default=STAGE2C_ROLLOUT)
    parser.add_argument("--dev-features", type=Path, default=STAGE2B_DEV_FEATURES)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--max-episodes", type=int, default=180)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-p", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--selection-near-tolerance", type=float, default=0.05)
    parser.add_argument("--selection-penalty", type=float, default=20.0)
    parser.add_argument("--selection-smooth-weight", type=float, default=0.02)
    parser.add_argument("--r1-lrs", nargs="+", type=float, default=list(R1_LRS))
    parser.add_argument("--r1-anchors", nargs="+", type=float, default=list(R1_ANCHORS))
    parser.add_argument("--r23-lrs", nargs="+", type=float, default=list(R23_LRS))
    parser.add_argument("--r23-anchors", nargs="+", type=float, default=list(R23_ANCHORS))
    parser.add_argument(
        "--r23-residuals", nargs="+", type=float, default=list(R23_RESIDUALS)
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 3)
        args.patience = min(args.patience, 3)
        args.r1_lrs = [1e-2]
        args.r1_anchors = [1.0]
        args.r23_lrs = [1e-3]
        args.r23_anchors = [1.0]
        args.r23_residuals = [1e-3]
        args.max_episodes = min(args.max_episodes, 6)
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "steps").mkdir()
    (args.output_dir / "stops").mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_rollout_steps(args.rollout)
    train_episodes, val_episodes = split_episodes(
        data["episode_index"], seed=args.split_seed,
        n_train=args.train_episodes,
    )
    val_mask = episode_mask(data["episode_index"], val_episodes)
    train_mask = episode_mask(data["episode_index"], train_episodes)
    feature_mean, feature_std = feature_normalization(
        data["features"], train_mask
    )
    hard_rows = build_training_rows(
        data["distance_m"], data["arrival_label"], data["arrival_probability"],
        data["episode_index"], train_episodes, kind="hard",
        hard_p=args.hard_p, seed=args.split_seed,
    )
    random_rows = build_training_rows(
        data["distance_m"], data["arrival_label"], data["arrival_probability"],
        data["episode_index"], train_episodes, kind="random",
        hard_p=args.hard_p, seed=args.split_seed,
    )
    if hard_rows["stats"]["total_rows"] != random_rows["stats"]["total_rows"]:
        raise RuntimeError("R2/R3 training row counts differ")
    if hard_rows["stats"]["added_far_rows"] != random_rows["stats"]["added_far_rows"]:
        raise RuntimeError("R2/R3 added far-negative counts differ")
    if set(hard_rows["stats"]) != set(random_rows["stats"]):
        raise RuntimeError("R2/R3 mining stats schema mismatch")

    write_json(args.output_dir / "split.json", {
        "split_seed": int(args.split_seed),
        "train_episodes": [int(x) for x in train_episodes],
        "validation_episodes": [int(x) for x in val_episodes],
        "train_episode_count": int(len(train_episodes)),
        "validation_episode_count": int(len(val_episodes)),
        "overlap": int(
            len(set(train_episodes.tolist()) & set(val_episodes.tolist()))
        ),
    })
    write_json(args.output_dir / "mining_stats.json", {
        "hard": hard_rows["stats"],
        "random": random_rows["stats"],
        "same_total_rows": True,
        "hard_probability_threshold": float(args.hard_p),
    })
    b0_val_probability = np.asarray(
        data["arrival_probability"][val_mask], dtype=np.float64
    )
    b0_val_hazard = per_decision_hazard(
        b0_val_probability,
        np.asarray(data["distance_m"][val_mask], dtype=np.float64),
        np.asarray(data["episode_index"][val_mask], dtype=np.int64),
        threshold=args.threshold,
    )
    b0_val_metrics = arrival_offline_metrics(
        np.asarray(data["arrival_label"][val_mask]),
        b0_val_probability,
        np.asarray(data["distance_m"][val_mask]),
        sample_type=None, threshold=args.threshold,
    )
    b0_near_recall = float(b0_val_hazard["near20_stop_rate"])
    write_json(args.output_dir / "calibration_val_b0.json", {
        "b0_near20_recall": b0_near_recall,
        "hazard": b0_val_hazard,
        "metrics": b0_val_metrics,
    })
    print(
        f"Split {len(train_episodes)}/{len(val_episodes)} episodes; "
        f"train rows {hard_rows['stats']['total_rows']} "
        f"(hard={hard_rows['stats']['added_far_rows']}, "
        f"random={random_rows['stats']['added_far_rows']}); "
        f"B0 val near20 recall={b0_near_recall:.4f}",
        flush=True,
    )

    calibrators = {}
    variant_results = {}
    val = _val_tensors(data, val_mask, device)
    for kind, rows_info in (
        ("R1", hard_rows), ("R2", hard_rows), ("R3", random_rows),
    ):
        calibrator, result = train_variant(
            kind, data, rows_info, val, b0_near_recall,
            feature_mean, feature_std, args, device,
            args.output_dir / kind,
        )
        calibrators[kind] = calibrator
        variant_results[kind] = result

    # Calibration-validation metrics for B0/R1/R2/R3 (internal, on-policy).
    val_features = torch.from_numpy(
        np.asarray(data["features"][val_mask], dtype=np.float32)
    ).to(device)
    val_logits = torch.from_numpy(
        np.asarray(data["arrival_logit"][val_mask], dtype=np.float32)
    ).to(device)
    val_labels = np.asarray(data["arrival_label"][val_mask], dtype=np.float32)
    val_distance = np.asarray(data["distance_m"][val_mask], dtype=np.float64)
    val_episode = np.asarray(data["episode_index"][val_mask], dtype=np.int64)
    val_probabilities = {"B0": b0_val_probability}
    for kind in VARIANTS:
        calibrators[kind].eval()
        with torch.inference_mode():
            val_probabilities[kind] = torch.sigmoid(
                calibrators[kind](val_features, val_logits)
            ).cpu().numpy()
    calibration_val_metrics = {}
    for name, probability in val_probabilities.items():
        calibration_val_metrics[name] = {
            "hazard": per_decision_hazard(
                probability, val_distance, val_episode, threshold=args.threshold
            ),
            "metrics": arrival_offline_metrics(
                val_labels, probability, val_distance,
                sample_type=None, threshold=args.threshold,
            ),
        }
    write_json(
        args.output_dir / "calibration_val_metrics.json",
        calibration_val_metrics,
    )

    # Dev Val 180 offline Arrival metrics (evaluation only; config is fixed).
    dev = load_dev_val_features(args.dev_features)
    _, _, b0_head = load_m0(device)
    dev_base_logits = head_logits(b0_head, dev["features"], device)
    dev_probabilities = {"B0": sigmoid_numpy(dev_base_logits)}
    for kind in VARIANTS:
        dev_probabilities[kind] = calibrate_probabilities(
            calibrators[kind], dev["features"], dev_base_logits, device
        )
    np.savez_compressed(
        args.output_dir / "dev_val_offline_probabilities.npz",
        **dev_probabilities,
    )
    dev_offline_metrics = {
        name: arrival_offline_metrics(
            dev["arrival_label"], probability, dev["distance"],
            sample_type=dev["sample_type"], threshold=args.threshold,
        )
        for name, probability in dev_probabilities.items()
    }
    write_json(args.output_dir / "dev_val_offline_metrics.json", {
        "threshold": float(args.threshold),
        "metrics": dev_offline_metrics,
    })

    # Dev Val 180 closed loop, every variant re-run from scratch.
    specs = load_dev_specs(args.val_specs, args.max_episodes)
    encoder = load_encoder(device)
    oracle_cache = build_oracle_cache(specs)
    model, _, _ = load_m0(device)
    episodes_by_variant = {}
    steps_by_variant = {}
    stops_by_variant = {}
    for name in ("B0",) + VARIANTS:
        rule = StopRule(name, args.threshold, 1, 1)
        calibrator = None if name == "B0" else calibrators[name]
        print(f"Closed-loop {name}", flush=True)
        episodes, steps, stops = evaluate_policy(
            model, specs, oracle_cache, encoder, device, rule,
            calibrator=calibrator, progress_every=30,
        )
        episodes_by_variant[name] = episodes
        steps_by_variant[name] = steps
        stops_by_variant[name] = stops
        write_csv(args.output_dir / "steps" / f"{name}.csv", steps)
        write_csv(args.output_dir / "stops" / f"{name}.csv", stops)
        write_csv(args.output_dir / f"{name}_per_episode.csv", episodes)
        print(
            f"  {name}: SR={np.mean([r['success'] for r in episodes]):.4f} "
            f"OSR={np.mean([r['osr'] for r in episodes]):.4f} "
            f"premature={np.mean([r['premature_stop'] for r in episodes]):.4f}",
            flush=True,
        )
    replay_check = verify_b0_replay(episodes_by_variant["B0"])
    write_json(args.output_dir / "replay_check.json", replay_check)
    if replay_check["outcome_mismatch_count"] != 0:
        raise SystemExit(
            "B0 replay outcome mismatch: "
            f"{replay_check['outcome_mismatch_count']} fields"
        )

    all_episodes = [
        row for name in ("B0",) + VARIANTS
        for row in episodes_by_variant[name]
    ]
    write_csv(args.output_dir / "closedloop_per_episode.csv", all_episodes)
    closedloop_metrics = {}
    paired = {}
    for name in ("B0",) + VARIANTS:
        summary = summarize_by_band(episodes_by_variant[name])
        closedloop_metrics[name] = {
            "overall": summary["all"],
            "by_band": summary,
            "stop_events": {
                band: stop_event_summary([
                    row for row in stops_by_variant[name]
                    if band == "all" or str(row["band"]) == band
                ])
                for band in ("all",) + BANDS
            },
            "hazard": {
                band: hazard_from_steps(
                    steps_by_variant[name] if band == "all" else [
                        row for row in steps_by_variant[name]
                        if str(row["band"]) == band
                    ],
                    threshold=args.threshold,
                )
                for band in ("all",) + BANDS
            },
        }
        if name != "B0":
            paired[name] = paired_by_band(
                {"B0": episodes_by_variant["B0"], name: episodes_by_variant[name]},
                base="B0", other=name,
            )
    risk = risk_rows(
        steps_by_variant,
        threshold_map={name: args.threshold for name in steps_by_variant},
    )
    write_csv(args.output_dir / "risk.csv", risk)
    write_json(args.output_dir / "closedloop_metrics.json", {
        "threshold": float(args.threshold),
        "metrics": closedloop_metrics,
        "paired_vs_b0": paired,
        "risk": risk,
    })
    write_json(args.output_dir / "paired.json", paired)

    base_state = torch.load(M0_CKPT, map_location="cpu", weights_only=True)
    m0_movement_hash = movement_hash(base_state)
    m0_arrival_head_hash = arrival_head_hash(base_state)
    for kind in VARIANTS:
        if variant_results[kind]["movement_hash"] != m0_movement_hash:
            raise RuntimeError(f"{kind} movement hash differs from M0")
        if variant_results[kind]["arrival_head_hash"] != m0_arrival_head_hash:
            raise RuntimeError(f"{kind} Arrival-head hash differs from M0")
    write_json(args.output_dir / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "V8.3 Stage-2d R1/R2/R3 off-policy calibration",
        "command": sys.argv,
        "m0_checkpoint": str(M0_CKPT),
        "m0_checkpoint_sha256": file_sha256(M0_CKPT),
        "rollout": str(args.rollout),
        "rollout_sha256": file_sha256(args.rollout),
        "dev_features": str(args.dev_features),
        "dev_features_sha256": file_sha256(args.dev_features),
        "val_specs": str(args.val_specs),
        "val_specs_sha256": file_sha256(args.val_specs),
        "split_seed": int(args.split_seed),
        "train_episodes": int(len(train_episodes)),
        "validation_episodes": int(len(val_episodes)),
        "threshold": float(args.threshold),
        "hard_probability_threshold": float(args.hard_p),
        "seed": int(args.seed),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "weight_decay": float(args.weight_decay),
        "grids": {
            "R1_lrs": [float(x) for x in args.r1_lrs],
            "R1_anchors": [float(x) for x in args.r1_anchors],
            "R23_lrs": [float(x) for x in args.r23_lrs],
            "R23_anchors": [float(x) for x in args.r23_anchors],
            "R23_residuals": [float(x) for x in args.r23_residuals],
        },
        "movement_hash_m0": m0_movement_hash,
        "arrival_head_hash_m0": m0_arrival_head_hash,
        "variant_hashes": {
            kind: {
                "movement_hash": variant_results[kind]["movement_hash"],
                "arrival_head_hash": variant_results[kind]["arrival_head_hash"],
            }
            for kind in VARIANTS
        },
        "device": device_summary(device),
        "variants": {
            kind: {
                "selected_config": variant_results[kind].get(
                    "selected_config", variant_results[kind].get("config")
                ),
                "best_epoch": variant_results[kind].get("best_epoch"),
                "parameter_count": variant_results[kind].get("parameter_count"),
            }
            for kind in VARIANTS
        },
        "note": (
            "Frozen M0; calibration uses only 104-D Arrival features and the "
            "original logit; true distance is labels/diagnostics only; "
            "Dev Val closed loop re-run per variant; Final Test not read."
        ),
    })
    print(json.dumps({
        name: closedloop_metrics[name]["overall"]
        for name in ("B0",) + VARIANTS
    }, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
