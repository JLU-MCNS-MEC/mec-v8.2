"""Stage-2b Arrival-only A0/A1 ablation.

A0: 3% far-negative retention (the Stage-2 dataset mask).
A1: 10% far-negative retention, built as A0 plus additional far negatives.

Both variants:
  * initialize the Arrival Head from the same Stage-2a M0 checkpoint;
  * freeze the movement/backbone entirely (arrival features are cached once);
  * use the same seed, shuffle schedule, optimizer, max epochs and patience;
  * use BCEWithLogitsLoss(pos_weight=neg/pos) so total positive/negative loss
    weight is comparable despite the different negative count.

Only the Arrival Head is optimized. No original token dataset, Stage-2a
checkpoint, or Final Test data is modified/read.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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

import eval_fixed_mec_epoch_compare as evaluation
from config import ARRIVAL_STOP_THRESHOLD, ENCODER
from gbs_model_v3 import ArrivalReasoningHead, build_gbs_model
from v83_spatial_training.eval_closedloop_spatial import evaluate_mode, load_encoder
from v83_spatial_training.spatial_mec import SpatialAwareMEC
from v83_spatial_stage2.analyze_stage2a import summarize
from v83_spatial_training.eval_closedloop_spatial import paired_comparison


STAGE2_ROOT = Path("runs/v83_spatial_stage2")
DATA_ROOT = STAGE2_ROOT / "datasets"
M0_CKPT = STAGE2_ROOT / "stage2a_tm_20261018T044255Z/Transformer-M/M0/best.pt"
VAL_SPECS = STAGE2_ROOT / "prep_audit_20261018T043420Z/dev_val_180_spec.json"
OUT_ROOT = STAGE2_ROOT / "stage2b"
BANDS = ("short", "medium", "long")
FAR_NEG_RADIUS_M = 60.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _brier(y, p):
    return float(np.mean((p - y) ** 2))


def _ece(y, p, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
        if m.any():
            total += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(total)


def _f1(y, pred):
    y = np.asarray(y).astype(bool); pred = np.asarray(pred).astype(bool)
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _arrival_metrics(features, labels, sample_type, distance, head, device):
    with torch.inference_mode():
        logits = head.net(torch.from_numpy(features).float().to(device)).squeeze(-1).cpu().numpy()
    probs = _sigmoid(logits)
    y = labels.astype(bool)
    pred = probs >= 0.5
    natural = sample_type == 0
    hard = ~natural
    far40 = natural & (distance >= 40.0)
    far100 = natural & (distance >= 100.0)
    near_pos = natural & y
    return {
        "natural_f1": _f1(y[natural], pred[natural]),
        "hard_f1": _f1(y[hard], pred[hard]),
        "natural_brier": _brier(y[natural].astype(float), probs[natural]),
        "hard_brier": _brier(y[hard].astype(float), probs[hard]),
        "natural_ece": _ece(y[natural].astype(float), probs[natural]),
        "hard_ece": _ece(y[hard].astype(float), probs[hard]),
        "far40_false_stop": float(pred[far40].mean()) if far40.any() else 0.0,
        "far100_false_stop": float(pred[far100].mean()) if far100.any() else 0.0,
        "near20_miss_stop": float((~pred[near_pos]).mean()) if near_pos.any() else 0.0,
        "natural_n": int(natural.sum()),
        "hard_n": int(hard.sum()),
        "natural_positive": int(y[natural].sum()),
    }


def _hash_state(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _load_m0_and_head(device):
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0").to(device)
    state = torch.load(M0_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    head = ArrivalReasoningHead()
    head_state = {
        k[len("base.arrival_head."):]: v
        for k, v in state.items() if k.startswith("base.arrival_head.")
    }
    head.load_state_dict(head_state)
    head.to(device)
    return model, state, head


def extract_features(folder, model, device, max_shards=None, batch_size=128):
    base = model.base
    files = sorted(Path(folder).glob("shard_*.pt"))
    if max_shards is not None:
        files = files[: int(max_shards)]
    feats, labels, distance, sample_type, episode, step, band, mask = [], [], [], [], [], [], [], []
    for p in files:
        d = torch.load(p, map_location="cpu", weights_only=True)
        tokens = d["tokens"].float()
        goal = d["goal_tokens"].float()
        heat = d["spatial_heatmaps"].float()
        net = d["network_states"].float()
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], batch_size):
                end = min(start + batch_size, tokens.shape[0])
                comp = base.forward_components(
                    tokens[start:end].to(device),
                    goal[start:end].to(device),
                    heat[start:end].to(device),
                    net[start:end].to(device),
                )
                feats.append(comp["arrival_features"].detach().cpu().numpy())
        labels.append(d["arrival_labels"].numpy())
        distance.append(d["distance_to_goal_m"].numpy())
        sample_type.append(d["sample_type"].numpy())
        episode.append(d["episode_ids"].numpy())
        step.append(d["step_ids"].numpy())
        # The shards use trajectory_band_ids; keep it for reporting.
        band.append(d["trajectory_band_ids"].numpy())
        mask.append(d["arrival_train_mask"].numpy())
    return {
        "features": np.concatenate(feats, axis=0).astype(np.float32),
        "arrival_label": np.concatenate(labels).astype(np.float32),
        "distance": np.concatenate(distance).astype(np.float32),
        "sample_type": np.concatenate(sample_type).astype(np.int64),
        "episode_id": np.concatenate(episode).astype(np.int64),
        "step_id": np.concatenate(step).astype(np.int64),
        "band_id": np.concatenate(band).astype(np.int64),
        "arrival_mask": np.concatenate(mask).astype(bool),
    }


def build_a1_mask(train):
    a0 = train["arrival_mask"].astype(bool)
    y = train["arrival_label"]
    st = train["sample_type"]
    dist = train["distance"]
    far = (st == 0) & (y < 0.5) & (dist > FAR_NEG_RADIUS_M)
    target = int(round(0.10 * far.sum()))
    current = int((a0 & far).sum())
    extra_needed = max(0, target - current)
    candidates = np.flatnonzero(far & ~a0)
    rng = np.random.default_rng(20261018)
    if extra_needed > 0:
        chosen = rng.choice(candidates, size=min(extra_needed, len(candidates)), replace=False)
        a1 = a0.copy(); a1[chosen] = True
    else:
        a1 = a0.copy()
    # Positives, near negatives and hard samples must remain identical.
    protected = (y >= 0.5) | ((st == 0) & (dist <= FAR_NEG_RADIUS_M)) | (st != 0)
    if not np.array_equal(a0[protected], a1[protected]):
        raise RuntimeError("A1 changed protected positive/near/hard mask entries")
    if not np.all(a1 >= a0):
        raise RuntimeError("A1 is not a superset of A0")
    stats = {}
    for name, mask in (("A0", a0), ("A1", a1)):
        pos = int(((y >= 0.5) & mask).sum())
        neg = int(((y < 0.5) & mask).sum())
        stats[name] = {
            "included": int(mask.sum()),
            "positives": pos,
            "negatives": neg,
            "pos_weight_neg_over_pos": neg / max(pos, 1),
            "far_included": int((mask & far).sum()),
            "far_total": int(far.sum()),
            "near_included": int((mask & (st == 0) & (dist <= FAR_NEG_RADIUS_M)).sum()),
            "hard_included": int((mask & (st != 0)).sum()),
        }
    return a0, a1, stats


def train_head(name, train, val, mask, seed, max_epochs, patience, batch_size, lr, device, outdir):
    _, m0_state, head = _load_m0_and_head(device)
    head.net.train()
    features = train["features"]
    labels = train["arrival_label"]
    y = labels >= 0.5
    indices = np.flatnonzero(mask)
    pos = int(((labels >= 0.5) & mask).sum())
    neg = int(((labels < 0.5) & mask).sum())
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.net.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_state = copy.deepcopy(head.net.state_dict())
    best_score = -1.0; best_epoch = -1; stale = 0; history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        head.net.train()
        perm = torch.randperm(len(indices), generator=generator).numpy()
        loss_sum = 0.0; batches = 0
        for start in range(0, len(perm), batch_size):
            take = indices[perm[start:start + batch_size]]
            xb = torch.from_numpy(features[take]).float().to(device)
            yb = torch.from_numpy(labels[take]).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head.net(xb).squeeze(-1)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Arrival loss {name} epoch {epoch}")
            loss.backward(); optimizer.step()
            loss_sum += float(loss.item()); batches += 1
        metrics = _arrival_metrics(
            val["features"], val["arrival_label"], val["sample_type"],
            val["distance"], head, device,
        )
        score = 0.5 * (metrics["natural_f1"] + metrics["hard_f1"])
        history.append({"epoch": epoch, "train_loss": loss_sum / max(batches, 1),
                        "natural_f1": metrics["natural_f1"], "hard_f1": metrics["hard_f1"],
                        "selection_f1": score})
        print(f"{name} epoch {epoch:03d} loss={history[-1]['train_loss']:.5f} "
              f"natF1={100*metrics['natural_f1']:.2f} hardF1={100*metrics['hard_f1']:.2f} "
              f"sel={100*score:.2f}", flush=True)
        if score > best_score + 1e-6:
            best_score = score; best_epoch = epoch
            best_state = copy.deepcopy(head.net.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"{name} early stop epoch {epoch} (best {best_epoch})", flush=True)
                break
    head.net.load_state_dict(best_state); head.net.eval()
    metrics = _arrival_metrics(
        val["features"], val["arrival_label"], val["sample_type"],
        val["distance"], head, device,
    )
    # Build a full M0 checkpoint with the trained head; all other weights stay byte-identical.
    full = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0")
    full.load_state_dict(m0_state)
    movement_hash = _hash_state({k: v for k, v in m0_state.items() if not k.startswith("base.arrival_head.")})
    full.base.arrival_head.net.load_state_dict(best_state)
    full_state = copy.deepcopy(full.state_dict())
    movement_hash_after = _hash_state({k: v for k, v in full_state.items() if not k.startswith("base.arrival_head.")})
    if movement_hash != movement_hash_after:
        raise RuntimeError("Movement weights changed during Arrival-only training")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, outdir / "arrival_head.pt")
    torch.save(full_state, outdir / "best.pt")
    with (outdir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    result = {
        "variant": name, "best_epoch": best_epoch, "best_selection_f1": best_score,
        "train_time_s": time.time() - t0, "train_samples": int(mask.sum()),
        "positives": pos, "negatives": neg, "pos_weight": float(pos_weight.item()),
        "metrics": metrics, "movement_unchanged": True,
        "checkpoint": str(outdir / "best.pt"),
    }
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage", choices=["prepare", "train", "eval", "all"], default="all")
    p.add_argument("--seed", type=int, default=20264017)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    p.add_argument("--max-episodes", type=int, default=180)
    p.add_argument("--threshold", type=float, default=ARRIVAL_STOP_THRESHOLD)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_dir = args.output_dir / "features"
    features_dir.mkdir()
    model, m0_state, head = _load_m0_and_head(device)
    train = extract_features(DATA_ROOT / "train_600", model, device, args.max_shards)
    val = extract_features(DATA_ROOT / "dev_val_180", model, device, args.max_shards)
    np.savez(features_dir / "train.npz", **train)
    np.savez(features_dir / "dev_val.npz", **val)
    a0, a1, mask_stats = build_a1_mask(train)
    (args.output_dir / "mask_stats.json").write_text(json.dumps(mask_stats, indent=2), encoding="utf-8")
    print("Feature shapes", train["features"].shape, val["features"].shape, flush=True)
    print("Mask stats", json.dumps(mask_stats, indent=2), flush=True)
    if args.stage in ("train", "all"):
        results = {}
        for name, mask in (("A0", a0), ("A1", a1)):
            results[name] = train_head(
                name, train, val, mask, args.seed, args.max_epochs,
                args.patience, args.batch_size, args.lr, device,
                args.output_dir / name,
            )
        (args.output_dir / "train_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.stage in ("eval", "all") and not args.smoke:
        evaluation.TEST_STATS_PATH = args.val_specs
        evaluation.NUM_EPISODES = args.max_episodes
        specs = evaluation.load_test_episode_specs()
        encoder = load_encoder(device)
        oracle_cache = {
            spec["episode_index"]: evaluation.build_oracle_reference(spec) for spec in specs
        }
        rows_by = {}
        for name in ("A0", "A1"):
            ckpt = args.output_dir / name / "best.pt"
            print(f"Closed-loop {name} {ckpt}", flush=True)
            rows_by[name] = evaluate_mode(
                "M0", "Transformer-M", ckpt, specs, oracle_cache, encoder,
                device, args.threshold, args.max_episodes,
            )
            with (args.output_dir / f"{name}_per_episode.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_by[name][0])); w.writeheader(); w.writerows(rows_by[name])
        analysis = {
            "metrics": {name: summarize(rows_by[name]) for name in rows_by},
            "by_band": {
                name: {b: summarize([r for r in rows_by[name] if r["band"] == b]) for b in BANDS}
                for name in rows_by
            },
            "paired_A1_vs_A0": paired_comparison(rows_by, base="A0", other="A1"),
            "per_band_paired": {
                b: paired_comparison(
                    {n: [r for r in rows_by[n] if r["band"] == b] for n in ("A0", "A1")},
                    base="A0", other="A1",
                ) for b in BANDS
            },
        }
        (args.output_dir / "closedloop_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(json.dumps(analysis, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
