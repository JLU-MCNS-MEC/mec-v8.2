"""Stage-2j training for the four new MEC architectures.

Protocol is identical to Stage-2i T5:
  Train 600 / Dev Val 180, max 50 epochs, patience 8, batch 48, lr 6e-5,
  AdamW wd 1e-4, seed 20264017, frozen shared RGB encoder, movement CE
  (soft class weights) + Arrival BCE (no pos_weight), and the same checkpoint
  score.  Only the model builder differs.

Transformer-M is not trained here.  ``--reuse-stage2i-t5`` registers the
existing Stage-2i T5 checkpoint after strict state-dict verification.

No Final Test file is read.
"""

from __future__ import annotations

import argparse
import json
import os
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

from config import TRAIN
from gbs_model_v3 import arrival_gated_decision
from v83_spatial_stage2.stage2i_train import (
    amp_context,
    aux_path_for,
    binary_metrics,
    build_batch,
    compute_loss,
    macro_f1_from_confusion,
    mov_weights,
    recalls_from_confusion,
    save_history,
    set_seed,
)
from v83_spatial_stage2.stage2i_data import load_shard_bundle, sorted_shards
from v83_spatial_stage2.stage2j_common import (
    AUX_ROOT,
    MODEL_DIRS,
    T5_CHECKPOINT,
    T5_DIR,
    T5_EXPECTED_SHA256,
    T5_TRAINING_RESULT,
    TRAIN_DIR,
    TRAIN_PROTOCOL,
    VAL_DIR,
    file_sha256,
    model_dir,
    read_json as read_json_j,
    reject_final_test,
    update_progress,
    utc_now,
)
from v83_spatial_stage2.stage2j_model import (
    STAGE2J_CONFIGS,
    build_stage2j_model,
    parameter_count,
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

AUX_FIELDS = (
    "history_vis", "history_pose", "history_action", "history_mask",
    "gab", "map_feat",
)


def forward_batch(model, batch):
    extras = batch["extras"]
    return model.forward_components(
        batch["tokens"], batch["goal"],
        heatmap=None,
        network_state=None,
        history_vis=extras["history_vis"],
        history_pose=extras["history_pose"],
        history_action=extras["history_action"],
        history_mask=extras["history_mask"],
        gab=extras["gab"],
        map_feat=extras["map_feat"],
    )


@torch.no_grad()
def evaluate_model(model, shard_paths, aux_root, device,
                   batch_size=VAL_BATCH_SIZE):
    """Dev Val offline metrics used for checkpoint selection.

    Identical to Stage-2i's evaluator: natural samples produce the 4-way
    confusion matrix, natural/hard arrival F1 use threshold 0.5.
    """
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.long)
    arrival_true, arrival_pred = [], []
    hard_true, hard_pred = [], []
    distances = []
    total_natural = 0
    use_amp = bool(USE_AMP and device.type == "cuda")
    for shard_index, path in enumerate(shard_paths):
        aux_path = aux_path_for(aux_root, "val", shard_index)
        bundle = load_shard_bundle(path, aux_path, AUX_FIELDS)
        n = bundle["num_samples"]
        for start in range(0, n, batch_size):
            indices = np.arange(start, min(start + batch_size, n))
            batch = build_batch(bundle, indices, device, AUX_FIELDS)
            with amp_context(device, use_amp):
                comp = forward_batch(model, batch)
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


def run_smoke(model_name, train_shards, aux_root, device, output_dir):
    set_seed(SEED)
    model = build_stage2j_model(model_name).to(device)
    model.train()
    bundle = load_shard_bundle(
        train_shards[0], aux_path_for(aux_root, "train", 0), AUX_FIELDS
    )
    indices = np.arange(min(8, bundle["num_samples"]))
    batch = build_batch(bundle, indices, device, AUX_FIELDS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR)
    move_weights = torch.ones(3, device=device)
    comp = forward_batch(model, batch)
    loss, move_loss, arr_loss = compute_loss(comp, batch, move_weights)
    loss.backward()
    grad_report = {
        name: float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "architecture": STAGE2J_CONFIGS[model_name]["architecture"],
        "loss": float(loss.item()),
        "move_loss": float(move_loss.item()),
        "arrival_loss": float(arr_loss.item()),
        "parameters": parameter_count(model),
        "gradient_parameters": len(grad_report),
        "gradient_nonzero": int(sum(1 for v in grad_report.values() if v > 0)),
        "grad_norms": grad_report,
        "finite": bool(torch.isfinite(loss).item()),
        "device": str(device),
        "final_test_read": False,
    }
    (output_dir / f"smoke_{model_name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in payload.items()
                      if k != "grad_norms"}, indent=2), flush=True)
    torch.save(model.state_dict(), output_dir / f"smoke_{model_name}.pt")
    return payload


def register_transformer_m(args):
    """Register the existing Stage-2i T5 checkpoint without training."""
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(
            f"Refusing to overwrite non-empty {output_dir}; use --resume"
        )
    if not T5_CHECKPOINT.is_file():
        raise SystemExit(f"Missing Stage-2i T5 checkpoint {T5_CHECKPOINT}")
    sha = file_sha256(T5_CHECKPOINT)
    if sha != T5_EXPECTED_SHA256:
        raise SystemExit(
            f"Stage-2i T5 checkpoint SHA256 mismatch: {sha} != "
            f"{T5_EXPECTED_SHA256}"
        )
    checkpoint = torch.load(T5_CHECKPOINT, map_location="cpu",
                            weights_only=False)
    model = build_stage2j_model("Transformer-M")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    params = parameter_count(model)
    source_result = read_json_j(T5_TRAINING_RESULT, {})
    output_dir.mkdir(parents=True, exist_ok=True)

    link = output_dir / "best.pt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(T5_CHECKPOINT, start=output_dir))

    source_history = T5_DIR / "history.csv"
    if source_history.is_file():
        (output_dir / "history.csv").write_text(
            source_history.read_text(encoding="utf-8"), encoding="utf-8"
        )

    config_payload = {
        "model": "Transformer-M",
        "architecture": STAGE2J_CONFIGS["Transformer-M"]["architecture"],
        "native_config": STAGE2J_CONFIGS["Transformer-M"],
        "native_gbs_config": "GBS_MODEL_CONFIGS['Transformer-M']",
        "parameters": params,
        "trainable_parameters": params,
        "reused_stage2i_checkpoint": True,
        "source_checkpoint": str(T5_CHECKPOINT),
        "source_checkpoint_sha256": sha,
        "source_training_result": str(T5_TRAINING_RESULT),
        "protocol": TRAIN_PROTOCOL,
        "aux_fields": list(AUX_FIELDS),
        "final_test_read": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    result = {
        "model": "Transformer-M",
        "architecture": STAGE2J_CONFIGS["Transformer-M"]["architecture"],
        "seed": int(source_result.get("seed", 20264017)),
        "parameters": params,
        "trainable_parameters": params,
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_score": float(checkpoint.get("score", -1.0)),
        "best_metrics": checkpoint.get("metrics"),
        "training_time_s": float(source_result.get("training_time_s", 0.0)),
        "checkpoint": str(link),
        "source_checkpoint": str(T5_CHECKPOINT),
        "source_checkpoint_sha256": sha,
        "reused_stage2i_t5": True,
        "history": str(output_dir / "history.csv"),
        "batch_size": int(source_result.get("batch_size", BATCH_SIZE)),
        "max_epochs": int(source_result.get("max_epochs", MAX_EPOCHS)),
        "patience": int(source_result.get("patience", PATIENCE)),
        "lr": float(source_result.get("lr", INITIAL_LR)),
        "final_test_read": False,
    }
    (output_dir / "training_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    update_progress(
        "Transformer-M",
        architecture=STAGE2J_CONFIGS["Transformer-M"]["architecture"],
        parameters=params,
        trainable_parameters=params,
        train_status="reused",
        best_epoch=result["best_epoch"],
        best_score=result["best_score"],
        checkpoint=str(link),
        checkpoint_sha256=sha,
        training_time_s=result["training_time_s"],
        training_time_min=result["training_time_s"] / 60.0,
        best_offline_metrics=result["best_metrics"],
        train_finished_utc=utc_now(),
        config=config_payload,
        final_test_read=False,
    )
    print("REUSE_COMPLETE", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def train_one(args):
    model_name = args.model
    if model_name not in MODEL_DIRS:
        raise SystemExit(f"Unknown model {model_name}")
    if args.reuse_stage2i_t5:
        if model_name != "Transformer-M":
            raise SystemExit("--reuse-stage2i-t5 is only valid for Transformer-M")
        return register_transformer_m(args)

    reject_final_test(args.train_dir)
    reject_final_test(args.val_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and (output_dir / "training_result.json").is_file() \
            and not args.resume:
        raise SystemExit(
            f"{output_dir}/training_result.json already exists; use --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_shards = sorted_shards(args.train_dir)
    val_shards = sorted_shards(args.val_dir)
    print(f"Model={model_name} device={device} "
          f"train_shards={len(train_shards)} val_shards={len(val_shards)}",
          flush=True)

    if args.smoke:
        return run_smoke(model_name, train_shards, args.aux_root, device,
                         output_dir)

    set_seed(args.seed)
    model = build_stage2j_model(model_name).to(device)
    params = parameter_count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config_payload = {
        "model": model_name,
        "architecture": STAGE2J_CONFIGS[model_name]["architecture"],
        "native_config": STAGE2J_CONFIGS[model_name],
        "parameters": params,
        "trainable_parameters": trainable,
        "train_dir": str(args.train_dir),
        "val_dir": str(args.val_dir),
        "aux_root": str(args.aux_root),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "val_batch_size": int(args.val_batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "optimizer": "AdamW",
        "lr_scheduler": TRAIN_PROTOCOL["lr_scheduler"],
        "aux_fields": list(AUX_FIELDS),
        "loss": TRAIN_PROTOCOL["loss"],
        "checkpoint_score": TRAIN_PROTOCOL["checkpoint_score"],
        "stop_threshold_for_offline_metrics": 0.5,
        "frozen_shared_encoder": True,
        "protocol": TRAIN_PROTOCOL,
        "final_test_read": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    print(f"Parameters: {params:,} (trainable {trainable:,})", flush=True)
    for name, module in model.named_children():
        print(f"  module {name}: "
              f"{sum(p.numel() for p in module.parameters()):,}", flush=True)

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

    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"Cannot resume: missing {last_path}")
        checkpoint = torch.load(last_path, map_location=device,
                                weights_only=False)
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

    update_progress(
        model_name,
        architecture=STAGE2J_CONFIGS[model_name]["architecture"],
        parameters=params,
        trainable_parameters=trainable,
        train_status="training",
        train_started_utc=utc_now(),
        config=config_payload,
        final_test_read=False,
    )

    train_started = time.time()
    epoch_samples = 0
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
            aux_path = aux_path_for(args.aux_root, "train", shard_index)
            bundle = load_shard_bundle(
                train_shards[shard_index], aux_path, AUX_FIELDS
            )
            n = bundle["num_samples"]
            epoch_samples += int(n)
            rng = np.random.default_rng(
                int(args.seed) + epoch * 100003 + shard_position
            )
            permutation = rng.permutation(n)
            for start in range(0, n, args.batch_size):
                indices = permutation[start:start + args.batch_size]
                batch = build_batch(bundle, indices, device, AUX_FIELDS)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, use_amp):
                    comp = forward_batch(model, batch)
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

        val = evaluate_model(
            model, val_shards, args.aux_root, device
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
            f"{model_name} Epoch {epoch:02d}/{args.max_epochs} | "
            f"Loss={row['train_total_loss']:.4f} "
            f"(M={row['train_move_loss']:.4f}, "
            f"A={row['train_arrival_loss']:.4f}) | "
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
                "model": model_name,
                "architecture": STAGE2J_CONFIGS[model_name]["architecture"],
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
            "model": model_name,
            "architecture": STAGE2J_CONFIGS[model_name]["architecture"],
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
        "model": model_name,
        "architecture": STAGE2J_CONFIGS[model_name]["architecture"],
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
        model_name,
        architecture=STAGE2J_CONFIGS[model_name]["architecture"],
        parameters=params,
        trainable_parameters=trainable,
        train_status="trained",
        train_finished_utc=utc_now(),
        best_epoch=int(best_epoch),
        best_score=float(best_score),
        checkpoint=str(best_path),
        checkpoint_sha256=file_sha256(best_path),
        training_time_s=float(training_time),
        training_time_min=float(training_time) / 60.0,
        best_offline_metrics=best_metrics,
        config=config_payload,
        final_test_read=False,
    )
    print("TRAINING_COMPLETE", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_DIRS))
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--aux-root", type=Path, default=AUX_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    parser.add_argument("--reuse-stage2i-t5", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = model_dir(args.model)
    train_one(args)


if __name__ == "__main__":
    main()
