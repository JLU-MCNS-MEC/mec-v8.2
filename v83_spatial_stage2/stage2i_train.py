"""Stage-2i single-variant MEC training (Transformer-M, frozen encoder).

Fixed protocol for all six variants:
  Train 600 / Dev Val 180, max 50 epochs, early-stopping patience 8,
  batch 48, lr 6e-5, AdamW wd 1e-4, ReduceLROnPlateau(0.5, patience 3),
  seed 20264017, original MEC loss (movement CE + arrival BCE, no pos_weight),
  checkpoint score = 0.7 * macro-F1 + 0.3 * mean(natural, hard) arrival F1.

Only the movement/arrival heads and the Stage-2i branches are trained; the
shared RGB encoder is frozen and consumed as pre-computed tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AUX, TRAIN
from gbs_model_v3 import arrival_gated_decision
from v83_spatial_stage2.stage2i_common import (
    MODEL_DIRS, model_dir, read_json, reject_final_test, update_progress,
    utc_now,
)
from v83_spatial_stage2.stage2i_data import (
    load_shard_bundle, movement_class_counts, soft_class_weights, sorted_shards,
)
from v83_spatial_stage2.stage2i_model import (
    VARIANTS, build_stage2i_model, parameter_count,
)


SEED = 20264017
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 48
VAL_BATCH_SIZE = 96
INITIAL_LR = 6e-5
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-6
USE_AMP = bool(TRAIN.use_amp)

AUX_FIELDS = {
    "T0": (),
    "T1": (),
    "T2": (),
    "T3": (),
    "T4": ("history_vis", "history_pose", "history_action",
           "history_mask", "gab"),
    "T5": ("history_vis", "history_pose", "history_action",
           "history_mask", "gab", "map_feat"),
}


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()


def aux_path_for(aux_root, split, shard_index):
    return Path(aux_root) / split / f"aux_shard_{shard_index:04d}.npz"


def build_batch(bundle, indices, device, aux_fields):
    index = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    tokens = bundle["tokens"][index].to(device, dtype=torch.float32,
                                        non_blocking=True)
    goal = bundle["goal_tokens"][index].to(device, dtype=torch.float32,
                                          non_blocking=True)
    heat = bundle["spatial_heatmaps"][index].to(device, dtype=torch.float32,
                                               non_blocking=True)
    network = bundle["network_states"][index].to(device, dtype=torch.float32,
                                                 non_blocking=True)
    labels = bundle["labels"][index].to(device, dtype=torch.long,
                                        non_blocking=True)
    arrival = bundle["arrival_labels"][index].to(device, dtype=torch.float32,
                                                 non_blocking=True)
    move_valid = bundle["move_valid"][index].to(device, dtype=torch.bool,
                                                non_blocking=True)
    arrival_mask = bundle["arrival_train_mask"][index].to(device, dtype=torch.bool,
                                                         non_blocking=True)
    sample_type = bundle["sample_type"][index].to(device, dtype=torch.long,
                                                  non_blocking=True)
    extras = {}
    for field in aux_fields:
        extras[field] = bundle[field][index].to(device, non_blocking=True)
    return {
        "tokens": tokens, "goal": goal, "heat": heat, "network": network,
        "labels": labels, "arrival": arrival, "move_valid": move_valid,
        "arrival_mask": arrival_mask, "sample_type": sample_type,
        "extras": extras,
    }


def forward_batch(model, variant, batch):
    extras = batch["extras"]
    return model.forward_components(
        batch["tokens"], batch["goal"],
        heatmap=batch["heat"] if VARIANTS[variant]["use_heatmap"] else None,
        network_state=batch["network"] if VARIANTS[variant]["use_network"] else None,
        history_vis=extras.get("history_vis"),
        history_pose=extras.get("history_pose"),
        history_action=extras.get("history_action"),
        history_mask=extras.get("history_mask"),
        gab=extras.get("gab"),
        map_feat=extras.get("map_feat"),
    )


def compute_loss(components, batch, move_weights):
    move_logits = components["move_logits"]
    arrival_logit = components["arrival_logit"]
    labels = batch["labels"]
    valid = batch["move_valid"]
    if bool(valid.any()):
        move_loss = F.cross_entropy(
            move_logits[valid], labels[valid], weight=move_weights
        )
    else:
        move_loss = move_logits.sum() * 0.0
    mask = batch["arrival_mask"]
    if bool(mask.any()):
        arrival_loss = F.binary_cross_entropy_with_logits(
            arrival_logit[mask], batch["arrival"][mask]
        )
    else:
        arrival_loss = arrival_logit.sum() * 0.0
    total = move_loss + float(TRAIN.arrival_loss_weight) * arrival_loss
    return total, move_loss, arrival_loss


def recalls_from_confusion(confusion):
    return [
        float(confusion[i, i] / confusion[i].sum().clamp_min(1))
        for i in range(4)
    ]


def macro_f1_from_confusion(confusion):
    values = []
    for i in range(4):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - confusion[i, i])
        fn = float(confusion[i, :].sum() - confusion[i, i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return float(np.mean(values))


def binary_metrics(y_true, y_pred):
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(model, variant, shard_paths, aux_root, aux_fields, device,
             batch_size=VAL_BATCH_SIZE):
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.long)
    arrival_true, arrival_pred = [], []
    hard_true, hard_pred = [], []
    distances = []
    total_natural = 0
    use_amp = bool(USE_AMP and device.type == "cuda")
    for shard_index, path in enumerate(shard_paths):
        aux_path = aux_path_for(aux_root, "val", shard_index) if aux_fields else None
        bundle = load_shard_bundle(path, aux_path, aux_fields)
        n = bundle["num_samples"]
        for start in range(0, n, batch_size):
            indices = np.arange(start, min(start + batch_size, n))
            batch = build_batch(bundle, indices, device, aux_fields)
            with amp_context(device, use_amp):
                comp = forward_batch(model, variant, batch)
            natural = batch["sample_type"] == 0
            if bool(natural.any()):
                decision = arrival_gated_decision(
                    comp["move_logits"], comp["arrival_logit"]
                )
                pred = decision["action"][natural]
                gt = batch["labels"][natural]
                for a, b in zip(gt.cpu().tolist(), pred.cpu().tolist()):
                    confusion[int(a), int(b)] += 1
                total_natural += int(natural.sum())
                prob = torch.sigmoid(comp["arrival_logit"][natural])
                arrival_true.append(batch["arrival"][natural].cpu())
                arrival_pred.append((prob >= 0.5).cpu())
                distances.append(bundle["distance_to_goal_m"][
                    torch.as_tensor(indices)][natural.cpu()].float())
            hard = batch["sample_type"] != 0
            if bool(hard.any()):
                prob_hard = torch.sigmoid(comp["arrival_logit"][hard])
                hard_true.append(batch["arrival"][hard].cpu())
                hard_pred.append((prob_hard >= 0.5).cpu())
        del bundle

    recalls = recalls_from_confusion(confusion)
    total = int(confusion.sum())
    accuracy = float(confusion.diag().sum() / max(total, 1))
    y_true = torch.cat(arrival_true).bool()
    y_pred = torch.cat(arrival_pred).bool()
    arr = binary_metrics(y_true, y_pred)
    dist = torch.cat(distances).float()
    stop_pred = torch.cat(arrival_pred).bool()
    far40 = dist >= 40.0
    far100 = dist >= 100.0
    hard_arr = (
        binary_metrics(torch.cat(hard_true).bool(), torch.cat(hard_pred).bool())
        if hard_true else {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                           "tp": 0, "fp": 0, "fn": 0}
    )
    hard_negative = 0
    hard_negative_false = 0
    if hard_true:
        ht = torch.cat(hard_true).bool()
        hp = torch.cat(hard_pred).bool()
        hard_negative = int((~ht).sum())
        hard_negative_false = int((hp & ~ht).sum())
    return {
        "accuracy": accuracy,
        "balanced4": float(np.mean(recalls)),
        "macro_f1": macro_f1_from_confusion(confusion),
        "recalls": recalls,
        "natural_arrival": arr,
        "hard_arrival": hard_arr,
        "hard_negative_false_stop_rate": (
            hard_negative_false / hard_negative if hard_negative else 0.0
        ),
        "far40_false_stop_rate": (
            float(stop_pred[far40].float().mean()) if bool(far40.any()) else 0.0
        ),
        "far100_false_stop_rate": (
            float(stop_pred[far100].float().mean()) if bool(far100.any()) else 0.0
        ),
        "near20_miss_stop_rate": (
            float((~stop_pred[y_true]).float().mean()) if bool(y_true.any()) else 0.0
        ),
        "natural_samples": int(total_natural),
    }


def save_history(path, history):
    if not history:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run_smoke(variant, train_shards, aux_root, aux_fields, device, output_dir):
    set_seed(SEED)
    model = build_stage2i_model(variant).to(device)
    model.train()
    bundle = load_shard_bundle(
        train_shards[0],
        aux_path_for(aux_root, "train", 0) if aux_fields else None,
        aux_fields,
    )
    indices = np.arange(min(8, bundle["num_samples"]))
    batch = build_batch(bundle, indices, device, aux_fields)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR)
    move_weights = torch.ones(3, device=device)
    comp = forward_batch(model, variant, batch)
    loss, move_loss, arr_loss = compute_loss(comp, batch, move_weights)
    loss.backward()
    grad_report = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad_report[name] = float(parameter.grad.norm())
    optimizer.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "loss": float(loss.item()),
        "move_loss": float(move_loss.item()),
        "arrival_loss": float(arr_loss.item()),
        "parameters": parameter_count(model),
        "gradient_parameters": len(grad_report),
        "gradient_nonzero": int(sum(1 for v in grad_report.values() if v > 0)),
        "grad_norms": grad_report,
        "finite": bool(torch.isfinite(loss).item()),
        "device": str(device),
    }
    (output_dir / f"smoke_{variant}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in payload.items()
                      if k != "grad_norms"}, indent=2), flush=True)
    torch.save(model.state_dict(), output_dir / f"smoke_{variant}.pt")
    return payload


def mov_weights(aux_root, device):
    counts = read_json(Path(aux_root) / "train_class_counts.json")
    if counts is None:
        raise SystemExit(
            f"Missing {Path(aux_root) / 'train_class_counts.json'}; run stage2i_prepare.py"
        )
    weights = soft_class_weights(
        torch.tensor(counts["counts"], dtype=torch.float64),
        float(counts.get("gamma", TRAIN.movement_class_balance_gamma)),
    )
    return weights.to(device), counts["counts"]


def train_one(args):
    variant = args.variant
    if variant not in MODEL_DIRS:
        raise SystemExit(f"Unknown variant {variant}")
    reject_final_test(args.train_dir)
    reject_final_test(args.val_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config_payload["variant_flags"] = VARIANTS[variant]
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    aux_fields = AUX_FIELDS[variant]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_shards = sorted_shards(args.train_dir)
    val_shards = sorted_shards(args.val_dir)
    print(f"Variant={variant} device={device} train_shards={len(train_shards)} "
          f"val_shards={len(val_shards)}", flush=True)

    if args.smoke:
        return run_smoke(variant, train_shards, args.aux_root, aux_fields,
                         device, output_dir)

    set_seed(args.seed)
    model = build_stage2i_model(variant).to(device)
    params = parameter_count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,} (trainable {trainable:,})", flush=True)
    for name, module in model.named_children():
        print(f"  module {name}: {sum(p.numel() for p in module.parameters()):,}",
              flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )
    use_amp = bool(USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    move_weights, class_counts = mov_weights(args.aux_root, device)
    print(f"Movement class counts F/L/R={class_counts}; "
          f"weights={move_weights.detach().cpu().tolist()}", flush=True)

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.csv"
    history = []
    start_epoch = 1
    best_score = -1.0
    best_epoch = -1
    best_metrics = None
    stale = 0
    elapsed_before = 0.0

    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        best_metrics = checkpoint.get("best_metrics")
        stale = int(checkpoint.get("stale", 0))
        elapsed_before = float(checkpoint.get("elapsed_s", 0.0))
        print(f"Resuming from epoch {checkpoint['epoch']} "
              f"(best {best_epoch}, score {best_score:.4f})", flush=True)
    elif args.resume and best_path.is_file():
        raise SystemExit("Only best.pt exists; cannot resume exact optimizer state")

    update_progress(
        variant, train_status="training", train_started_utc=utc_now(),
        config={
            "variant": variant, "seed": int(args.seed),
            "max_epochs": int(args.max_epochs), "patience": int(args.patience),
            "batch_size": int(args.batch_size), "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "parameters": params, "trainable_parameters": trainable,
            "train_dir": str(args.train_dir), "val_dir": str(args.val_dir),
            "aux_root": str(args.aux_root),
            "aux_fields": list(aux_fields),
            "loss": "movement CE (soft class weights) + arrival BCE; no pos_weight",
            "checkpoint_score": "0.7*macro_f1 + 0.3*mean(natural,hard)_arrival_f1",
            "stop_threshold_for_metrics": 0.5,
            "frozen_shared_encoder": True,
        },
    )

    train_started = time.time()
    for epoch in range(start_epoch, int(args.max_epochs) + 1):
        epoch_started = time.time()
        model.train()
        order = list(range(len(train_shards)))
        random.Random(int(args.seed) + epoch).shuffle(order)
        total_sum = move_sum = arr_sum = 0.0
        batches = 0
        natural_correct = 0
        natural_total = 0
        epoch_samples = 0
        for shard_position, shard_index in enumerate(order):
            aux_path = (aux_path_for(args.aux_root, "train", shard_index)
                        if aux_fields else None)
            bundle = load_shard_bundle(
                train_shards[shard_index], aux_path, aux_fields
            )
            n = bundle["num_samples"]
            epoch_samples += int(n)
            rng = np.random.default_rng(
                int(args.seed) + epoch * 100003 + shard_position
            )
            permutation = rng.permutation(n)
            for start in range(0, n, args.batch_size):
                indices = permutation[start:start + args.batch_size]
                batch = build_batch(bundle, indices, device, aux_fields)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, use_amp):
                    comp = forward_batch(model, variant, batch)
                    loss, move_loss, arr_loss = compute_loss(
                        comp, batch, move_weights
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}, batches {batches}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                batches += 1
                total_sum += float(loss.item())
                move_sum += float(move_loss.item())
                arr_sum += float(arr_loss.item())
                natural = batch["sample_type"] == 0
                if bool(natural.any()):
                    with torch.no_grad():
                        decision = arrival_gated_decision(
                            comp["move_logits"], comp["arrival_logit"]
                        )
                        pred = decision["action"][natural]
                        natural_correct += int(
                            (pred == batch["labels"][natural]).sum()
                        )
                        natural_total += int(natural.sum())
            del bundle

        val = evaluate(
            model, variant, val_shards, args.aux_root, aux_fields, device
        )
        arrival_selection_f1 = 0.5 * (
            float(val["natural_arrival"]["f1"])
            + float(val["hard_arrival"]["f1"])
        )
        score = (
            float(TRAIN.checkpoint_macro_f1_weight) * val["macro_f1"]
            + float(TRAIN.checkpoint_arrival_f1_weight) * arrival_selection_f1
        )
        scheduler.step(score)
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_total_loss": total_sum / max(batches, 1),
            "train_move_loss": move_sum / max(batches, 1),
            "train_arrival_loss": arr_sum / max(batches, 1),
            "train_natural_acc": natural_correct / max(natural_total, 1),
            "val_acc": val["accuracy"],
            "val_balanced4": val["balanced4"],
            "val_macro_f1": val["macro_f1"],
            "val_natural_arrival_f1": val["natural_arrival"]["f1"],
            "val_hard_arrival_f1": val["hard_arrival"]["f1"],
            "val_arrival_selection_f1": arrival_selection_f1,
            "val_far40_false_stop_rate": val["far40_false_stop_rate"],
            "val_far100_false_stop_rate": val["far100_false_stop_rate"],
            "val_near20_miss_stop_rate": val["near20_miss_stop_rate"],
            "checkpoint_score": score,
            "lr": lr,
            "epoch_time_s": time.time() - epoch_started,
        }
        history.append(row)
        save_history(history_path, history)
        print(
            f"{variant} Epoch {epoch:02d}/{args.max_epochs} | "
            f"Loss={row['train_total_loss']:.4f} "
            f"(M={row['train_move_loss']:.4f}, A={row['train_arrival_loss']:.4f}) | "
            f"TrainAcc={100*row['train_natural_acc']:5.2f}% | "
            f"ValMacroF1={100*row['val_macro_f1']:5.2f}% | "
            f"ArrF1 nat/hard={100*row['val_natural_arrival_f1']:5.2f}/"
            f"{100*row['val_hard_arrival_f1']:5.2f}% | "
            f"Score={100*score:5.2f}% | LR={lr:.2e} | "
            f"{row['epoch_time_s']:.1f}s",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_metrics = {
                "score": score,
                **{key: val[key] for key in (
                    "accuracy", "balanced4", "macro_f1", "recalls",
                    "natural_arrival", "hard_arrival",
                    "hard_negative_false_stop_rate",
                    "far40_false_stop_rate", "far100_false_stop_rate",
                    "near20_miss_stop_rate", "natural_samples",
                )},
            }
            torch.save({
                "variant": variant,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "score": score,
                "metrics": best_metrics,
                "parameters": params,
                "config": vars(args),
            }, best_path)
            stale = 0
            print(f"  saved best -> {best_path}", flush=True)
        else:
            stale += 1
        torch.save({
            "variant": variant,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "stale": stale,
            "elapsed_s": elapsed_before + time.time() - train_started,
        }, last_path)
        if stale >= int(args.patience):
            print(f"  early stop at epoch {epoch} (best {best_epoch})",
                  flush=True)
            break

    training_time = elapsed_before + time.time() - train_started
    result = {
        "variant": variant,
        "model": "Transformer-M",
        "seed": int(args.seed),
        "parameters": params,
        "trainable_parameters": trainable,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "training_time_s": training_time,
        "checkpoint": str(best_path),
        "history": str(history_path),
        "batch_size": int(args.batch_size),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "lr": float(args.lr),
        "train_samples_per_epoch": int(epoch_samples),
        "final_test_read": False,
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    update_progress(
        variant,
        train_status="trained",
        train_finished_utc=utc_now(),
        best_epoch=int(best_epoch),
        best_score=float(best_score),
        checkpoint=str(best_path),
        training_time_s=float(training_time),
        parameters=int(params),
        trainable_parameters=int(trainable),
        best_offline_metrics=best_metrics,
    )
    print("TRAINING_COMPLETE", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=list(MODEL_DIRS))
    parser.add_argument("--train-dir", type=Path,
                        default=Path("runs/v83_spatial_stage2/datasets/train_600"))
    parser.add_argument("--val-dir", type=Path,
                        default=Path("runs/v83_spatial_stage2/datasets/dev_val_180"))
    parser.add_argument("--aux-root", type=Path,
                        default=Path("runs/v83_spatial_stage2/stage2i/aux"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=INITIAL_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
