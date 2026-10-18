"""Independent V8.3 M0/M1/M2 training entry point.

Based on the deterministic V8.2 epoch-compare training logic:
  * fixed model seed; per-epoch shard shuffle and per-shard loader seeds;
  * same AdamW/ReduceLROnPlateau, batch size, loss and 70/30 checkpoint score;
  * same movement class weighting.

M0 = original V8.2 model/inputs.
M1 = M0 + learnable fusion of baseline24/coherent24 into the Movement Head.
M2 = M1 + temporal24/innovation24 with an explicit validity mask.

No original V8.2 script is modified. This entry point is used for the first
Transformer-M M0/M1 comparison; M2 is implemented but not trained unless M1
shows a substantial closed-loop development-validation gain.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import random
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

from config import AUX, ENCODER, NETWORK, NUM_ACTIONS, NUM_MOVE_ACTIONS, TRAIN
from gbs_model_v3 import (
    arrival_gated_decision,
    build_gbs_model,
    combine_move_and_arrival,
)
from spatial_data import (
    SpatialCache,
    build_spatial_loader,
    move_spatial_batch_to_device,
)
from spatial_mec import SpatialAwareMEC, parameter_count


SEED = 20264017
TRAIN_DIR = Path("token_dataset_train")
VAL_DIR = Path("token_dataset_val")
VAL_HARD_DIR = Path("token_dataset_val_hard")
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 48
VAL_BATCH_SIZE = 96
INITIAL_LR = 6e-5
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-6
USE_AMP = bool(TRAIN.use_amp)
OUT_ROOT = Path("runs/v83_spatial_training")


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sorted_shards(folder):
    paths = sorted(Path(folder).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards in {folder}")
    return paths


def movement_class_counts(shard_paths):
    counts = torch.zeros(NUM_MOVE_ACTIONS, dtype=torch.float64)
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        labels = data["labels"].long()
        valid = data["move_valid"].bool()
        for i in range(NUM_MOVE_ACTIONS):
            counts[i] += ((labels == i) & valid).sum().double()
    return counts


def soft_class_weights(counts, gamma=0.5):
    freq = counts / counts.sum().clamp_min(1.0)
    weights = (freq.clamp_min(1e-6)).pow(-float(gamma))
    return (weights / weights.mean().clamp_min(1e-6)).float()


def recalls_from_confusion(confusion):
    out = []
    for i in range(NUM_ACTIONS):
        denom = confusion[i].sum().item()
        out.append(float(confusion[i, i].item() / denom) if denom else 0.0)
    return out


def macro_f1_from_confusion(confusion):
    f1s = []
    for i in range(NUM_ACTIONS):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - confusion[i, i])
        fn = float(confusion[i, :].sum() - confusion[i, i])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return float(np.mean(f1s))


def binary_metrics(y_true, y_pred):
    y_true = y_true.bool(); y_pred = y_pred.bool()
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def compute_training_loss(comp, labels, arrival_labels, move_valid, arrival_mask, move_weights):
    move_logits = comp["move_logits"]
    arrival_logit = comp["arrival_logit"]
    if bool(move_valid.any()):
        move_loss = F.cross_entropy(move_logits[move_valid], labels[move_valid], weight=move_weights)
    else:
        move_loss = move_logits.sum() * 0.0
    if bool(arrival_mask.any()):
        arrival_loss = F.binary_cross_entropy_with_logits(
            arrival_logit[arrival_mask], arrival_labels[arrival_mask]
        )
    else:
        arrival_loss = arrival_logit.sum() * 0.0
    total = move_loss + float(TRAIN.arrival_loss_weight) * arrival_loss
    return total, move_loss, arrival_loss


def amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_streaming_spatial(model, shard_paths, batch_size, device, use_amp, cache):
    model.eval()
    confusion = torch.zeros(NUM_ACTIONS, NUM_ACTIONS, dtype=torch.long)
    nll_sum = 0.0
    total_natural = 0
    arrival_true_all, arrival_pred_all, arrival_prob_all, distance_all = [], [], [], []
    for shard_index, shard_path in enumerate(shard_paths):
        loader = build_spatial_loader(
            shard_path, batch_size, False, 3100000 + shard_index, cache
        )
        for batch in loader:
            (tokens, goal, heat, network, labels, arrival, move_valid,
             arrival_mask, distances, sample_type, maps, valid) = move_spatial_batch_to_device(batch, device)
            with amp_context(device, use_amp):
                comp = model.forward_components(
                    tokens, goal, heat, network,
                    spatial_maps=maps, spatial_valid=valid,
                )
                log_probs = combine_move_and_arrival(comp["move_logits"], comp["arrival_logit"])
            natural = sample_type == 0
            if bool(natural.any()):
                decision = arrival_gated_decision(comp["move_logits"], comp["arrival_logit"])
                pred = decision["action"][natural]
                gt = labels[natural]
                nll_sum += float(F.nll_loss(log_probs[natural], gt, reduction="sum").item())
                total_natural += int(gt.numel())
                for a, b in zip(gt.detach().cpu().tolist(), pred.detach().cpu().tolist()):
                    confusion[int(a), int(b)] += 1
                arr_prob = torch.sigmoid(comp["arrival_logit"][natural])
                arrival_true_all.append(arrival[natural].detach().cpu())
                arrival_prob_all.append(arr_prob.detach().cpu())
                arrival_pred_all.append((arr_prob >= 0.5).detach().cpu())
                distance_all.append(distances[natural].detach().cpu())
    recalls = recalls_from_confusion(confusion)
    total = int(confusion.sum())
    acc = float(confusion.diag().sum() / max(total, 1))
    y_true = torch.cat(arrival_true_all).bool()
    y_pred = torch.cat(arrival_pred_all).bool()
    y_prob = torch.cat(arrival_prob_all).float()
    distances = torch.cat(distance_all).float()
    arr = binary_metrics(y_true, y_pred)
    far40 = distances >= 40.0
    far100 = distances >= 100.0
    return {
        "nll": nll_sum / max(total_natural, 1),
        "accuracy": acc,
        "balanced4": float(np.mean(recalls)),
        "directional": float(np.mean(recalls[:NUM_MOVE_ACTIONS])),
        "macro_f1": macro_f1_from_confusion(confusion),
        "recalls": recalls,
        "confusion": confusion,
        "arrival": arr,
        "arrival_mean_prob": float(y_prob.mean()),
        "far40_false_stop_rate": float(y_pred[far40].float().mean()) if bool(far40.any()) else 0.0,
        "far100_false_stop_rate": float(y_pred[far100].float().mean()) if bool(far100.any()) else 0.0,
        "near20_miss_stop_rate": float((~y_pred[y_true]).float().mean()) if bool(y_true.any()) else 0.0,
    }


@torch.no_grad()
def evaluate_hard_arrival_spatial(model, shard_paths, batch_size, device, use_amp, cache):
    model.eval()
    y_true_all, y_prob_all, distances_all, types_all = [], [], [], []
    for shard_index, shard_path in enumerate(shard_paths):
        loader = build_spatial_loader(
            shard_path, batch_size, False, 4100000 + shard_index, cache
        )
        for batch in loader:
            (tokens, goal, heat, network, labels, arrival, move_valid,
             arrival_mask, distances, sample_type, maps, valid) = move_spatial_batch_to_device(batch, device)
            hard = (sample_type != 0) & arrival_mask
            if not bool(hard.any()):
                continue
            with amp_context(device, use_amp):
                comp = model.forward_components(
                    tokens[hard], goal[hard], heat[hard], network[hard],
                    spatial_maps=maps[hard], spatial_valid=valid[hard],
                )
                probs = torch.sigmoid(comp["arrival_logit"])
            y_true_all.append(arrival[hard].detach().cpu().bool())
            y_prob_all.append(probs.detach().cpu().float())
            distances_all.append(distances[hard].detach().cpu().float())
            types_all.append(sample_type[hard].detach().cpu().long())
    if not y_true_all:
        raise RuntimeError("No hard-arrival validation samples found")
    y_true = torch.cat(y_true_all).bool()
    y_prob = torch.cat(y_prob_all).float()
    y_pred = y_prob >= 0.5
    distances = torch.cat(distances_all).float()
    sample_types = torch.cat(types_all).long()
    metrics = binary_metrics(y_true, y_pred)
    hard_negative = ~y_true
    by_type = {}
    for type_id, type_name in {1: "hard_positive", 2: "boundary_negative", 3: "distractor_negative"}.items():
        mask = sample_types == type_id
        if not bool(mask.any()):
            continue
        by_type[type_name] = {
            "count": int(mask.sum()),
            "arrival_f1": binary_metrics(y_true[mask], y_pred[mask])["f1"],
            "positive_recall": float(y_pred[mask][y_true[mask]].float().mean()) if bool(y_true[mask].any()) else None,
            "false_stop_rate": float(y_pred[mask][~y_true[mask]].float().mean()) if bool((~y_true[mask]).any()) else None,
        }
    return {
        "count": int(y_true.numel()),
        "arrival": metrics,
        "hard_negative_false_stop_rate": float(y_pred[hard_negative].float().mean()) if bool(hard_negative.any()) else 0.0,
        "positive_recall": metrics["recall"],
        "mean_probability": float(y_prob.mean()),
        "by_type": by_type,
    }


def checkpoint_snapshot(val, hard_val, score, epoch):
    return {
        "epoch": epoch,
        "score": score,
        "accuracy": val["accuracy"],
        "balanced_accuracy": val["balanced4"],
        "directional_balanced_accuracy": val["directional"],
        "macro_f1": val["macro_f1"],
        "arrival": val["arrival"],
        "hard_arrival": hard_val["arrival"],
        "hard_arrival_count": hard_val["count"],
        "hard_negative_false_stop_rate": hard_val["hard_negative_false_stop_rate"],
        "far40_false_stop_rate": val["far40_false_stop_rate"],
        "far100_false_stop_rate": val["far100_false_stop_rate"],
        "near20_miss_stop_rate": val["near20_miss_stop_rate"],
        "recalls": val["recalls"],
        "confusion": val["confusion"].tolist(),
    }


def save_history(path, history):
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)


def train_one(mode, model_name, train_shards, val_shards, val_hard_shards,
              train_cache, val_cache, hard_cache, outdir, args, device):
    set_seed(args.seed)
    base = build_gbs_model(model_name)
    model = SpatialAwareMEC(base, mode).to(device)
    params = parameter_count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*80}\nTraining {model_name} {mode} | params={params:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    use_amp = bool(USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    move_weights = soft_class_weights(movement_class_counts(train_shards)).to(device)
    best_score = -1.0
    best_epoch = -1
    best_metrics = None
    stale = 0
    history = []
    t0 = time.time()
    ckpt = outdir / "best.pt"
    for epoch in range(1, args.max_epochs + 1):
        epoch_t0 = time.time()
        model.train()
        shards = list(train_shards)
        random.Random(int(args.seed) + epoch).shuffle(shards)
        total_loss_sum = move_loss_sum = arr_loss_sum = 0.0
        batches = 0
        natural_correct = 0
        natural_total = 0
        for shard_index, path in enumerate(shards):
            loader_seed = int(args.seed) + epoch * 100003 + shard_index
            loader = build_spatial_loader(path, args.batch_size, True, loader_seed, train_cache)
            for batch in loader:
                (tokens, goal, heat, network, labels, arrival, move_valid,
                 arrival_mask, _distances, sample_type, maps, valid) = move_spatial_batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, use_amp):
                    comp = model.forward_components(
                        tokens, goal, heat, network,
                        spatial_maps=maps, spatial_valid=valid,
                    )
                    loss, move_loss, arr_loss = compute_training_loss(
                        comp, labels, arrival, move_valid, arrival_mask, move_weights
                    )
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                batches += 1
                total_loss_sum += float(loss.item())
                move_loss_sum += float(move_loss.item())
                arr_loss_sum += float(arr_loss.item())
                natural = sample_type == 0
                if bool(natural.any()):
                    decision = arrival_gated_decision(comp["move_logits"], comp["arrival_logit"])
                    pred = decision["action"][natural]
                    natural_correct += int((pred == labels[natural]).sum())
                    natural_total += int(natural.sum())
        val = evaluate_streaming_spatial(model, val_shards, args.val_batch_size, device, use_amp, val_cache)
        hard_val = evaluate_hard_arrival_spatial(model, val_hard_shards, args.val_batch_size, device, use_amp, hard_cache)
        arrival_selection_f1 = 0.5 * (float(val["arrival"]["f1"]) + float(hard_val["arrival"]["f1"]))
        score = (
            float(TRAIN.checkpoint_macro_f1_weight) * val["macro_f1"]
            + float(TRAIN.checkpoint_arrival_f1_weight) * arrival_selection_f1
        )
        scheduler.step(score)
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_total_loss": total_loss_sum / max(batches, 1),
            "train_move_loss": move_loss_sum / max(batches, 1),
            "train_arrival_loss": arr_loss_sum / max(batches, 1),
            "train_natural_acc": natural_correct / max(natural_total, 1),
            "val_nll": val["nll"],
            "val_acc": val["accuracy"],
            "val_bal4": val["balanced4"],
            "val_directional": val["directional"],
            "val_macro_f1": val["macro_f1"],
            "val_natural_arrival_f1": val["arrival"]["f1"],
            "val_hard_arrival_f1": hard_val["arrival"]["f1"],
            "val_hard_arrival_count": hard_val["count"],
            "val_hard_positive_recall": hard_val["positive_recall"],
            "val_hard_negative_false_stop_rate": hard_val["hard_negative_false_stop_rate"],
            "val_far40_false_stop_rate": val["far40_false_stop_rate"],
            "val_far100_false_stop_rate": val["far100_false_stop_rate"],
            "val_near20_miss_stop_rate": val["near20_miss_stop_rate"],
            "arrival_selection_f1": arrival_selection_f1,
            "checkpoint_score": score,
            "lr": lr,
            "epoch_time_s": time.time() - epoch_t0,
        }
        history.append(row)
        print(
            f"{mode} Epoch {epoch:03d}/{args.max_epochs} | "
            f"Loss={row['train_total_loss']:.4f} | "
            f"ValMacroF1={100*val['macro_f1']:5.2f}% | "
            f"ArrF1 nat/hard={100*val['arrival']['f1']:5.2f}/{100*hard_val['arrival']['f1']:5.2f}% | "
            f"Score={100*score:5.2f}% | LR={lr:.2e} | {row['epoch_time_s']:.1f}s",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_metrics = checkpoint_snapshot(val, hard_val, score, epoch)
            torch.save(model.state_dict(), ckpt)
            stale = 0
            print(f"  saved best -> {ckpt}", flush=True)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"  early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break
        save_history(outdir / "history.csv", history)
    save_history(outdir / "history.csv", history)
    total_time = time.time() - t0
    result = {
        "model": model_name,
        "mode": mode,
        "seed": int(args.seed),
        "parameters": params,
        "trainable_parameters": trainable,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "train_time_s": total_time,
        "checkpoint": str(ckpt),
        "history": str(outdir / "history.csv"),
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
    }
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_smoke(args, train_shards, train_cache, device):
    results = {}
    for mode in args.models:
        set_seed(args.seed)
        base = build_gbs_model(args.model_name)
        if mode != "M0":
            base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        model = SpatialAwareMEC(base, mode).to(device)
        model.train()
        shard = train_shards[0]
        loader = build_spatial_loader(shard, min(args.batch_size, 8), True, 1, train_cache)
        batch = next(iter(loader))
        (tokens, goal, heat, network, labels, arrival, move_valid,
         arrival_mask, _distances, sample_type, maps, valid) = move_spatial_batch_to_device(batch, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, bool(USE_AMP and device.type == "cuda")):
            comp = model.forward_components(
                tokens, goal, heat, network,
                spatial_maps=maps, spatial_valid=valid,
            )
            loss, move_loss, arr_loss = compute_training_loss(
                comp, labels, arrival, move_valid, arrival_mask,
                soft_class_weights(movement_class_counts(train_shards)).to(device),
            )
        loss.backward()
        finite = bool(torch.isfinite(loss).item())
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grad_finite = all(bool(torch.isfinite(g).all()) for g in grads)
        loss_value = float(loss.detach())
        print(
            f"SMOKE {mode}: params={parameter_count(model):,} loss={loss_value:.4f} "
            f"finite={finite} grad_finite={grad_finite} maps={tuple(maps.shape)}",
            flush=True,
        )
        results[mode] = {
            "parameters": parameter_count(model),
            "loss": loss_value,
            "finite": finite,
            "grad_finite": grad_finite,
            "spatial_maps_shape": list(maps.shape),
        }
        if mode != "M0":
            max_diff = 0.0
            state = model.base.state_dict()
            for k, v in base_state.items():
                max_diff = max(
                    max_diff,
                    float((state[k].detach().cpu() - v).abs().max().item()),
                )
            print(f"SMOKE {mode}: base-init max diff vs M0 base = {max_diff:.3e}", flush=True)
            results[mode]["base_init_max_diff"] = max_diff
        smoke_dir = Path(args.output_dir) / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), smoke_dir / f"{mode}.pt")
    (Path(args.output_dir) / "smoke").mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "smoke" / "smoke.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Transformer-M")
    p.add_argument("--models", nargs="+", default=["M0", "M1"])
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=INITIAL_LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    for mode in args.models:
        if mode not in ("M0", "M1", "M2"):
            raise SystemExit(f"Unknown mode {mode}")
    cache_dir = Path(args.cache_dir)
    for split in ("train", "val", "val_hard"):
        if not (cache_dir / f"{split}.npz").is_file():
            raise SystemExit(f"Missing cache {cache_dir / f'{split}.npz'}")
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = OUT_ROOT / stamp
    args.output_dir.mkdir(parents=True, exist_ok=False if not args.smoke else True)
    train_shards = sorted_shards(TRAIN_DIR)
    val_shards = sorted_shards(VAL_DIR)
    hard_shards = sorted_shards(VAL_HARD_DIR)
    train_cache = SpatialCache(cache_dir / "train.npz")
    val_cache = SpatialCache(cache_dir / "val.npz")
    hard_cache = SpatialCache(cache_dir / "val_hard.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        run_smoke(args, train_shards, train_cache, device)
        return
    results = {}
    for mode in args.models:
        outdir = Path(args.output_dir) / args.model_name / mode
        outdir.mkdir(parents=True, exist_ok=True)
        results[mode] = train_one(
            mode, args.model_name, train_shards, val_shards, hard_shards,
            train_cache, val_cache, hard_cache, outdir, args, device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(results, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
