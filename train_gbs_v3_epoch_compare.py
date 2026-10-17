"""
V8.2 MEC training.

Only the MEC input interface is redesigned. The V8.1 supervision/training
protocol is retained for a fair modality comparison:
  - frozen shared RGB encoder
  - expert movement labels for F/L/R
  - independent binary Arrival/Stop supervision
  - six hard Arrival states per training episode
  - 300 train / 60 natural validation / 60 hard-arrival validation / 60 test episodes
  - one controlled 100-epoch run per model, saving best checkpoints within epochs 1-50
    and within epochs 1-100 for a paired training-budget comparison
  - model-specific fixed seeds and deterministic per-shard data order

No success reward or MEC capability/model-quality information is used as input.
"""

from __future__ import annotations

import contextlib
import csv
import json
import random
from pathlib import Path

from tensorboard_logging import make_writer, new_run_id, write_epoch_metrics

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import (
    AUX,
    ENCODER,
    GBS_MODEL_CONFIGS,
    NETWORK,
    NUM_ACTIONS,
    NUM_MOVE_ACTIONS,
    TRAIN,
)
from gbs_model_v3 import (
    arrival_gated_decision,
    build_gbs_model,
    combine_move_and_arrival,
)


SEED = 20261017
MODEL_SEEDS = {
    "Small": 20261017,
    "Transformer-S": 20262017,
    "Medium": 20263017,
    "Transformer-M": 20264017,
    "Large": 20265017,
}
TRAIN_DIR = Path("token_dataset_train")
VAL_DIR = Path("token_dataset_val")
VAL_HARD_DIR = Path("token_dataset_val_hard")
OUT_DIR = Path("gbs_v82_epoch_compare_results")

MAX_EPOCHS = 100
COMPARE_CUTOFF = 50
LR_PATIENCE = 3
LR_FACTOR = 0.5
MIN_LR = 1e-6
USE_AMP = bool(TRAIN.use_amp)

BATCH_SIZES = {
    "Small": 64,
    "Transformer-S": 64,
    "Medium": 64,
    "Transformer-M": 48,
    "Large": 32,
}
VAL_BATCH_SIZES = {
    "Small": 128,
    "Transformer-S": 128,
    "Medium": 128,
    "Transformer-M": 96,
    "Large": 64,
}
INITIAL_LR = {
    "Small": 1e-4,
    "Transformer-S": 8e-5,
    "Medium": 8e-5,
    "Transformer-M": 6e-5,
    "Large": 5e-5,
}

MOVE_NAMES = ["Forward", "Left", "Right"]
ACTION_NAMES = ["Forward", "Left", "Right", "Stop"]


def safe_name(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # Improve repeatability on a fixed machine/GPU. Some CUDA kernels may still
    # have small platform-dependent nondeterminism, but model order no longer
    # changes the initialization or data order of another model.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sorted_shards(folder):
    paths = sorted(Path(folder).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No .pt files found in {Path(folder).resolve()}"
        )
    return paths


def load_shard(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = [
        "tokens",
        "goal_tokens",
        "spatial_heatmaps",
        "network_states",
        "labels",
        "arrival_labels",
        "move_valid",
        "arrival_train_mask",
        "distance_to_goal_m",
        "sample_type",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"{path} missing V8.2 fields {missing}. "
            "Regenerate token_dataset_train/val/test."
        )

    tensors = (
        data["tokens"].float(),
        data["goal_tokens"].float(),
        data["spatial_heatmaps"].float(),
        data["network_states"].float(),
        data["labels"].long(),
        data["arrival_labels"].float(),
        data["move_valid"].bool(),
        data["arrival_train_mask"].bool(),
        data["distance_to_goal_m"].float(),
        data["sample_type"].long(),
    )

    tokens, goal, heatmap, network_state, *_ = tensors
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
        ENCODER.num_tokens,
        ENCODER.token_dim,
    ):
        raise ValueError(f"tokens shape error: {tuple(tokens.shape)}")
    if goal.ndim != 3 or tuple(goal.shape[1:]) != (
        AUX.goal_token_count,
        ENCODER.token_dim,
    ):
        raise ValueError(f"goal_tokens shape error: {tuple(goal.shape)}")
    if heatmap.ndim != 4 or tuple(heatmap.shape[1:]) != (
        AUX.spatial_heatmap_channels,
        AUX.spatial_heatmap_grid,
        AUX.spatial_heatmap_grid,
    ):
        raise ValueError(f"spatial_heatmaps shape error: {tuple(heatmap.shape)}")
    if network_state.ndim != 2 or network_state.shape[1] != NETWORK.network_state_dim:
        raise ValueError(f"network_states shape error: {tuple(network_state.shape)}")
    return tensors


def build_loader(path, batch_size, shuffle, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*load_shard(path)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def move_batch_to_device(batch, device):
    (
        tokens,
        goal_tokens,
        spatial_heatmaps,
        network_states,
        labels,
        arrival_labels,
        move_valid,
        arrival_train_mask,
        distance_to_goal_m,
        sample_type,
    ) = batch
    return (
        tokens.to(device, dtype=torch.float32, non_blocking=True),
        goal_tokens.to(device, dtype=torch.float32, non_blocking=True),
        spatial_heatmaps.to(device, dtype=torch.float32, non_blocking=True),
        network_states.to(device, dtype=torch.float32, non_blocking=True),
        labels.to(device, dtype=torch.long, non_blocking=True),
        arrival_labels.to(device, dtype=torch.float32, non_blocking=True),
        move_valid.to(device, dtype=torch.bool, non_blocking=True),
        arrival_train_mask.to(device, dtype=torch.bool, non_blocking=True),
        distance_to_goal_m.to(device, dtype=torch.float32, non_blocking=True),
        sample_type.to(device, dtype=torch.long, non_blocking=True),
    )


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
    return weights / weights.mean().clamp_min(1e-6)


def recalls_from_confusion(confusion):
    out = []
    for i in range(NUM_ACTIONS):
        denom = confusion[i].sum().item()
        out.append(float(confusion[i, i].item() / denom) if denom else 0.0)
    return out


def balanced_accuracy(recalls):
    return float(np.mean(recalls))


def directional_balanced_accuracy(recalls):
    return float(np.mean(recalls[:NUM_MOVE_ACTIONS]))


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
            if precision + recall
            else 0.0
        )
    return float(np.mean(f1s))


def binary_metrics(y_true, y_pred):
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_training_loss(
    components,
    labels,
    arrival_labels,
    move_valid,
    arrival_mask,
    move_weights,
):
    move_logits = components["move_logits"]
    arrival_logit = components["arrival_logit"]

    if bool(move_valid.any()):
        move_loss = F.cross_entropy(
            move_logits[move_valid],
            labels[move_valid],
            weight=move_weights,
        )
    else:
        move_loss = move_logits.sum() * 0.0

    if bool(arrival_mask.any()):
        arrival_loss = F.binary_cross_entropy_with_logits(
            arrival_logit[arrival_mask],
            arrival_labels[arrival_mask],
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
def evaluate_streaming(model, shard_paths, batch_size, device, use_amp):
    model.eval()
    confusion = torch.zeros(NUM_ACTIONS, NUM_ACTIONS, dtype=torch.long)
    nll_sum = 0.0
    total_natural = 0

    arrival_true_all = []
    arrival_pred_all = []
    arrival_prob_all = []
    distance_all = []

    for shard_index, shard_path in enumerate(shard_paths):
        for batch in build_loader(
            shard_path, batch_size, shuffle=False,
            seed=3100000 + int(shard_index),
        ):
            (
                tokens,
                goal_tokens,
                heatmaps,
                network_states,
                labels,
                arrival_labels,
                move_valid,
                arrival_mask,
                distances,
                sample_type,
            ) = move_batch_to_device(batch, device)

            with amp_context(device, use_amp):
                comp = model.forward_components(
                    tokens,
                    goal_tokens,
                    heatmaps,
                    network_states,
                )
                log_probs = combine_move_and_arrival(
                    comp["move_logits"], comp["arrival_logit"]
                )

            natural = sample_type == 0
            if bool(natural.any()):
                decision = arrival_gated_decision(
                    comp["move_logits"], comp["arrival_logit"]
                )
                pred = decision["action"][natural]
                gt = labels[natural]
                nll_sum += float(
                    F.nll_loss(
                        log_probs[natural], gt, reduction="sum"
                    ).item()
                )
                total_natural += int(gt.numel())
                for a, b in zip(
                    gt.detach().cpu().tolist(),
                    pred.detach().cpu().tolist(),
                ):
                    confusion[int(a), int(b)] += 1

                arr_prob = torch.sigmoid(comp["arrival_logit"][natural])
                arrival_true_all.append(
                    arrival_labels[natural].detach().cpu()
                )
                arrival_prob_all.append(arr_prob.detach().cpu())
                arrival_pred_all.append(
                    (arr_prob >= 0.5).detach().cpu()
                )
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
    far40_fp_rate = (
        float(y_pred[far40].float().mean()) if bool(far40.any()) else 0.0
    )
    far100_fp_rate = (
        float(y_pred[far100].float().mean())
        if bool(far100.any())
        else 0.0
    )
    near20_miss_rate = (
        float((~y_pred[y_true]).float().mean())
        if bool(y_true.any())
        else 0.0
    )

    return {
        "nll": nll_sum / max(total_natural, 1),
        "accuracy": acc,
        "balanced4": balanced_accuracy(recalls),
        "directional": directional_balanced_accuracy(recalls),
        "macro_f1": macro_f1_from_confusion(confusion),
        "recalls": recalls,
        "confusion": confusion,
        "arrival": arr,
        "arrival_mean_prob": float(y_prob.mean()),
        "far40_false_stop_rate": far40_fp_rate,
        "far100_false_stop_rate": far100_fp_rate,
        "near20_miss_stop_rate": near20_miss_rate,
    }


@torch.no_grad()
def evaluate_hard_arrival(model, shard_paths, batch_size, device, use_amp):
    """Evaluate only synthetic hard-arrival samples, never their movement labels."""
    model.eval()
    y_true_all = []
    y_prob_all = []
    distances_all = []
    types_all = []

    for shard_index, shard_path in enumerate(shard_paths):
        for batch in build_loader(
            shard_path, batch_size, shuffle=False,
            seed=4100000 + int(shard_index),
        ):
            (
                tokens, goal_tokens, heatmaps, network_states, labels,
                arrival_labels, move_valid, arrival_mask, distances, sample_type,
            ) = move_batch_to_device(batch, device)
            hard = (sample_type != 0) & arrival_mask
            if not bool(hard.any()):
                continue
            with amp_context(device, use_amp):
                comp = model.forward_components(
                    tokens[hard], goal_tokens[hard], heatmaps[hard], network_states[hard]
                )
                probs = torch.sigmoid(comp["arrival_logit"])
            y_true_all.append(arrival_labels[hard].detach().cpu().bool())
            y_prob_all.append(probs.detach().cpu().float())
            distances_all.append(distances[hard].detach().cpu().float())
            types_all.append(sample_type[hard].detach().cpu().long())

    if not y_true_all:
        raise RuntimeError(
            "No hard-arrival validation samples found. Regenerate token_dataset_val_hard."
        )

    y_true = torch.cat(y_true_all).bool()
    y_prob = torch.cat(y_prob_all).float()
    y_pred = y_prob >= 0.5
    distances = torch.cat(distances_all).float()
    sample_types = torch.cat(types_all).long()
    metrics = binary_metrics(y_true, y_pred)
    hard_negative = ~y_true
    false_stop_rate = (
        float(y_pred[hard_negative].float().mean())
        if bool(hard_negative.any()) else 0.0
    )
    by_type = {}
    type_names = {1: "hard_positive", 2: "boundary_negative", 3: "distractor_negative"}
    for type_id, type_name in type_names.items():
        mask = sample_types == int(type_id)
        if not bool(mask.any()):
            continue
        t_true = y_true[mask]
        t_pred = y_pred[mask]
        by_type[type_name] = {
            "count": int(mask.sum()),
            "arrival_f1": binary_metrics(t_true, t_pred)["f1"],
            "positive_recall": (
                float(t_pred[t_true].float().mean()) if bool(t_true.any()) else None
            ),
            "false_stop_rate": (
                float(t_pred[~t_true].float().mean()) if bool((~t_true).any()) else None
            ),
        }

    return {
        "count": int(y_true.numel()),
        "arrival": metrics,
        "hard_negative_false_stop_rate": false_stop_rate,
        "positive_recall": metrics["recall"],
        "mean_probability": float(y_prob.mean()),
        "by_type": by_type,
    }


def checkpoint_metrics(val, hard_val, score, epoch):
    return {
        "epoch": int(epoch),
        "score": float(score),
        "accuracy": float(val["accuracy"]),
        "balanced_accuracy": float(val["balanced4"]),
        "directional_balanced_accuracy": float(val["directional"]),
        "macro_f1": float(val["macro_f1"]),
        "arrival": val["arrival"],
        "hard_arrival": hard_val["arrival"],
        "hard_arrival_count": int(hard_val["count"]),
        "hard_negative_false_stop_rate": float(hard_val["hard_negative_false_stop_rate"]),
        "far40_false_stop_rate": float(val["far40_false_stop_rate"]),
        "far100_false_stop_rate": float(val["far100_false_stop_rate"]),
        "near20_miss_stop_rate": float(val["near20_miss_stop_rate"]),
        "recalls": val["recalls"],
        "confusion": val["confusion"].tolist(),
    }


def save_history_csv(model_name, history):
    path = OUT_DIR / f"{safe_name(model_name)}_training_history.csv"
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_plots(model_name, history):
    if not history:
        return
    epochs = [r["epoch"] for r in history]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(epochs, [r["train_total_loss"] for r in history], label="Train total")
    plt.plot(epochs, [r["train_move_loss"] for r in history], label="Move")
    plt.plot(epochs, [r["train_arrival_loss"] for r in history], label="Arrival")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"V8.2 training losses — {model_name}")
    plt.grid(alpha=0.25)
    plt.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{safe_name(model_name)}_loss.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        [100 * r["val_macro_f1"] for r in history],
        label="Macro-F1",
    )
    plt.plot(
        epochs,
        [100 * r["val_natural_arrival_f1"] for r in history],
        label="Natural Arrival-F1",
    )
    plt.plot(
        epochs,
        [100 * r["val_hard_arrival_f1"] for r in history],
        label="Hard Arrival-F1",
    )
    plt.plot(
        epochs,
        [100 * r["val_directional"] for r in history],
        label="Directional BA",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation (%)")
    plt.title(f"V8.2 validation quality — {model_name}")
    plt.grid(alpha=0.25)
    plt.legend()
    fig.tight_layout()
    fig.savefig(
        OUT_DIR / f"{safe_name(model_name)}_validation.png",
        dpi=220,
    )
    plt.close(fig)


def train_one_model(model_name, train_shards, val_shards, val_hard_shards, device, move_weights):
    model_seed = int(MODEL_SEEDS[model_name])
    set_seed(model_seed)

    print("\n" + "=" * 112)
    print(f"Training V8.2 MEC model: {model_name}")
    print(f"Fixed model seed: {model_seed}")
    print("=" * 112)

    model = build_gbs_model(model_name).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR[model_name],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    use_amp = bool(USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    move_weights = move_weights.to(device)

    best_50_score = -1.0
    best_50_epoch = -1
    best_50_metrics = None
    best_100_score = -1.0
    best_100_epoch = -1
    best_100_metrics = None
    history = []
    tb_writer = make_writer(OUT_DIR.name, model_name, run_id=new_run_id())
    name = safe_name(model_name)
    best_50_ckpt = OUT_DIR / f"gbs_{name}_best50.pt"
    best_100_ckpt = OUT_DIR / f"gbs_{name}_best100.pt"

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            shards = list(train_shards)
            random.Random(model_seed + epoch).shuffle(shards)

            total_loss_sum = 0.0
            move_loss_sum = 0.0
            arr_loss_sum = 0.0
            batches = 0
            natural_correct = 0
            natural_total = 0

            for shard_index, path in enumerate(shards):
                # Explicit per-epoch/per-shard generator makes the first 50 epochs
                # identical to the first 50 epochs of any same-seed comparison run.
                loader_seed = model_seed + epoch * 100003 + shard_index
                for batch in build_loader(
                    path,
                    BATCH_SIZES[model_name],
                    shuffle=True,
                    seed=loader_seed,
                ):
                    (
                        tokens,
                        goal_tokens,
                        heatmaps,
                        network_states,
                        labels,
                        arrival_labels,
                        move_valid,
                        arrival_mask,
                        _distances,
                        sample_type,
                    ) = move_batch_to_device(batch, device)

                    optimizer.zero_grad(set_to_none=True)
                    with amp_context(device, use_amp):
                        comp = model.forward_components(
                            tokens,
                            goal_tokens,
                            heatmaps,
                            network_states,
                        )
                        loss, move_loss, arr_loss = compute_training_loss(
                            comp,
                            labels,
                            arrival_labels,
                            move_valid,
                            arrival_mask,
                            move_weights,
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
                        decision = arrival_gated_decision(
                            comp["move_logits"], comp["arrival_logit"]
                        )
                        pred = decision["action"][natural]
                        natural_correct += int((pred == labels[natural]).sum())
                        natural_total += int(natural.sum())

            val = evaluate_streaming(
                model,
                val_shards,
                VAL_BATCH_SIZES[model_name],
                device,
                use_amp,
            )
            hard_val = evaluate_hard_arrival(
                model,
                val_hard_shards,
                VAL_BATCH_SIZES[model_name],
                device,
                use_amp,
            )
            # Preserve the existing 70/30 movement-arrival weighting, but evaluate
            # Arrival on both natural states and held-out hard-arrival states.
            arrival_selection_f1 = 0.5 * (
                float(val["arrival"]["f1"]) + float(hard_val["arrival"]["f1"])
            )
            score = (
                float(TRAIN.checkpoint_macro_f1_weight) * val["macro_f1"]
                + float(TRAIN.checkpoint_arrival_f1_weight) * arrival_selection_f1
            )
            scheduler.step(score)
            lr = float(optimizer.param_groups[0]["lr"])

            recalls = val["recalls"]
            arr = val["arrival"]
            hard_arr = hard_val["arrival"]
            row = {
                "epoch": epoch,
                "model_seed": model_seed,
                "train_total_loss": total_loss_sum / max(batches, 1),
                "train_move_loss": move_loss_sum / max(batches, 1),
                "train_arrival_loss": arr_loss_sum / max(batches, 1),
                "train_natural_acc": natural_correct / max(natural_total, 1),
                "val_nll": val["nll"],
                "val_acc": val["accuracy"],
                "val_bal4": val["balanced4"],
                "val_directional": val["directional"],
                "val_macro_f1": val["macro_f1"],
                "val_natural_arrival_f1": arr["f1"],
                "val_hard_arrival_f1": hard_arr["f1"],
                "val_hard_arrival_count": hard_val["count"],
                "val_hard_positive_recall": hard_val["positive_recall"],
                "val_hard_negative_false_stop_rate": hard_val["hard_negative_false_stop_rate"],
                "val_far40_false_stop_rate": val["far40_false_stop_rate"],
                "val_far100_false_stop_rate": val["far100_false_stop_rate"],
                "val_near20_miss_stop_rate": val["near20_miss_stop_rate"],
                "arrival_selection_f1": arrival_selection_f1,
                "checkpoint_score": score,
                "lr": lr,
            }
            history.append(row)

            print(
                f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
                f"Loss={row['train_total_loss']:.4f} "
                f"(Move={row['train_move_loss']:.4f}, Arr={row['train_arrival_loss']:.4f}) | "
                f"TrainNatAcc={100*row['train_natural_acc']:6.2f}% | "
                f"ValMacroF1={100*val['macro_f1']:5.2f}% | "
                f"ArrF1 natural/hard={100*arr['f1']:5.2f}/{100*hard_arr['f1']:5.2f}% | "
                f"HardNegFalseStop={100*hard_val['hard_negative_false_stop_rate']:5.2f}% | "
                f"Score={100*score:5.2f}% | LR={lr:.2e}"
            )

            snapshot = checkpoint_metrics(val, hard_val, score, epoch)
            if score > best_100_score + 1e-6:
                best_100_score = score
                best_100_epoch = epoch
                best_100_metrics = snapshot
                torch.save(model.state_dict(), best_100_ckpt)
                print(f"  Saved best-100 candidate -> {best_100_ckpt}")

            if epoch <= COMPARE_CUTOFF and score > best_50_score + 1e-6:
                best_50_score = score
                best_50_epoch = epoch
                best_50_metrics = snapshot
                torch.save(model.state_dict(), best_50_ckpt)
                print(f"  Saved best-50 candidate -> {best_50_ckpt}")

            save_history_csv(model_name, history)
            write_epoch_metrics(tb_writer, row, epoch)

    finally:
        tb_writer.close()
    save_plots(model_name, history)
    return {
        "model": model_name,
        "architecture": GBS_MODEL_CONFIGS[model_name]["architecture"],
        "parameters": params,
        "model_seed": model_seed,
        "best_50_epoch": best_50_epoch,
        "best_50_score": best_50_score,
        "best_50_metrics": best_50_metrics,
        "best_50_checkpoint": str(best_50_ckpt),
        "best_100_epoch": best_100_epoch,
        "best_100_score": best_100_score,
        "best_100_metrics": best_100_metrics,
        "best_100_checkpoint": str(best_100_ckpt),
    }


def save_quality_scores(results):
    out = {}
    for name, result in results.items():
        out[name] = {
            "model_seed": result["model_seed"],
            "best_50": {
                "epoch": result["best_50_epoch"],
                "score": result["best_50_score"],
                "metrics": result["best_50_metrics"],
            },
            "best_100": {
                "epoch": result["best_100_epoch"],
                "score": result["best_100_score"],
                "metrics": result["best_100_metrics"],
            },
        }
    with (OUT_DIR / "epoch_compare_validation_scores.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def print_dataset_summary(shards, name):
    total = 0
    natural = 0
    hard = 0
    for path in shards:
        data = torch.load(path, map_location="cpu", weights_only=True)
        n = int(data["labels"].shape[0])
        total += n
        natural += int((data["sample_type"] == 0).sum())
        hard += n - int((data["sample_type"] == 0).sum())
    print(
        f"{name}: shards={len(shards)}, total={total}, "
        f"natural={natural}, hard_arrival={hard}"
    )


def main():
    if OUT_DIR.exists() and any(OUT_DIR.glob("gbs_*_best*.pt")):
        raise FileExistsError(
            f"{OUT_DIR} already contains checkpoints. Back it up or remove it before a fresh controlled run; "
            "the script refuses to mix checkpoints from different runs."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("V8.2 controlled epoch-budget comparison")
    print("Input: Current RGB 576x384 | Goal 3x384 | Heatmap 3x8x8 | Network 5D")
    print("Budgets: best checkpoint within epochs 1-50 versus epochs 1-100")
    print("Model-specific fixed seeds; explicit deterministic shard samplers")
    print("Checkpoint score: 0.70*natural Macro-F1 + 0.30*mean(natural/hard Arrival-F1)")
    print(f"Train={TRAIN_DIR}, ValNatural={VAL_DIR}, ValHard={VAL_HARD_DIR}, Output={OUT_DIR}")

    for directory in (TRAIN_DIR, VAL_DIR, VAL_HARD_DIR):
        if not directory.exists():
            raise FileNotFoundError(
                f"Missing {directory}. See README_V82_EPOCH_COMPARISON.md for generation order."
            )

    hard_stats_path = VAL_HARD_DIR / "dataset_stats.json"
    with hard_stats_path.open("r", encoding="utf-8") as f:
        hard_stats = json.load(f)
    if not hard_stats.get("include_hard_states") or int(hard_stats.get("hard_arrival_samples", 0)) <= 0:
        raise RuntimeError(
            f"{hard_stats_path} does not contain hard-arrival validation samples. "
            "Run generate_val_hard_token_dataset.py."
        )

    train_shards = sorted_shards(TRAIN_DIR)
    val_shards = sorted_shards(VAL_DIR)
    val_hard_shards = sorted_shards(VAL_HARD_DIR)
    print_dataset_summary(train_shards, "TRAIN")
    print_dataset_summary(val_shards, "VALIDATION_NATURAL")
    print_dataset_summary(val_hard_shards, "VALIDATION_HARD_SPLIT")

    move_counts = movement_class_counts(train_shards)
    move_weights = soft_class_weights(move_counts, TRAIN.movement_class_balance_gamma).float()
    print("\nMovement soft weights (F/L/R only):")
    for i, name in enumerate(MOVE_NAMES):
        print(f"  {name:>8s}: count={move_counts[i].item():8.0f}, weight={move_weights[i].item():.4f}")

    results = {}
    for model_name in GBS_MODEL_CONFIGS:
        results[model_name] = train_one_model(
            model_name,
            train_shards,
            val_shards,
            val_hard_shards,
            device,
            move_weights,
        )
        with (OUT_DIR / "training_summary_epoch_compare.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        save_quality_scores(results)

    print("\n" + "=" * 112)
    print("V8.2 CONTROLLED EPOCH COMPARISON TRAINING FINISHED")
    print("=" * 112)
    for name, result in results.items():
        print(
            f"{name:>13s} | params={result['parameters']:,} | seed={result['model_seed']} | "
            f"best50@{result['best_50_epoch']:03d} score={100*result['best_50_score']:5.2f}% | "
            f"best100@{result['best_100_epoch']:03d} score={100*result['best_100_score']:5.2f}%"
        )
    print(f"\nOutputs: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
