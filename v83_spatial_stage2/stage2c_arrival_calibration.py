"""Stage-2c Arrival calibration controls B0/B1/B2.

B0: original Stage-2a M0 Arrival Head, no training (reference).
B1: M0 head re-trained on 3% far-negative mask with plain BCE (no pos_weight).
B2: M0 head re-trained on 10% far-negative mask with plain BCE (no pos_weight).

B1/B2 share the frozen M0 backbone/features, initialization, seed, optimizer,
max epochs, patience and Arrival-F1 checkpoint selection. Movement weights are
never optimized. B0 closed-loop rows are the existing Stage-2a M0 evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
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

import eval_fixed_mec_epoch_compare as evaluation
from gbs_model_v3 import build_gbs_model
from v83_spatial_training.eval_closedloop_spatial import (
    evaluate_mode,
    load_encoder,
    paired_comparison,
)
from v83_spatial_training.spatial_mec import SpatialAwareMEC
from v83_spatial_stage2.analyze_stage2a import summarize
from v83_spatial_stage2.stage2b_arrival_ablation import (
    M0_CKPT,
    _arrival_metrics,
    _hash_state,
    _load_m0_and_head,
    build_a1_mask,
)


STAGE2_ROOT = Path("runs/v83_spatial_stage2")
STAGE2B = STAGE2_ROOT / "stage2b/arrival_20261018T150031Z"
STAGE2A_EVAL = STAGE2_ROOT / "stage2a_eval_20261018T070139Z"
VAL_SPECS = STAGE2_ROOT / "prep_audit_20261018T043420Z/dev_val_180_spec.json"
BANDS = ("short", "medium", "long")


def verify_cached_logits(dev_features, m0_state, device):
    """Compare cached-feature head logits with full M0 model logits, same order."""
    _, _, head = _load_m0_and_head(device)
    with torch.inference_mode():
        cached = head.net(torch.from_numpy(dev_features).float().to(device)).squeeze(-1).cpu().numpy()
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0").to(device)
    model.load_state_dict(m0_state); model.eval()
    full = []
    for p in sorted((STAGE2_ROOT / "datasets/dev_val_180").glob("shard_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=True)
        with torch.inference_mode():
            for start in range(0, d["tokens"].shape[0], 128):
                end = min(start + 128, d["tokens"].shape[0])
                comp = model.forward_components(
                    d["tokens"][start:end].float().to(device),
                    d["goal_tokens"][start:end].float().to(device),
                    d["spatial_heatmaps"][start:end].float().to(device),
                    d["network_states"][start:end].float().to(device),
                )
                full.append(comp["arrival_logit"].detach().cpu().numpy())
    full = np.concatenate(full)
    if full.shape != cached.shape:
        raise RuntimeError(f"Cached/full shape mismatch {cached.shape} vs {full.shape}")
    diff = np.abs(full - cached)
    result = {
        "n": int(len(cached)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_1e-6": bool(np.allclose(full, cached, atol=1e-6)),
        "allclose_1e-5": bool(np.allclose(full, cached, atol=1e-5)),
    }
    if result["max_abs_diff"] > 1e-4:
        raise RuntimeError(f"Cached feature/head logits mismatch: {result}")
    return result


def _prob_distributions(features, labels, sample_type, distance, head, device):
    with torch.inference_mode():
        p = torch.sigmoid(head.net(torch.from_numpy(features).float().to(device)).squeeze(-1)).cpu().numpy()
    y = labels >= 0.5
    natural = sample_type == 0
    hard = ~natural
    far = natural & (distance > 60.0)
    return {
        "natural_pos_mean": float(p[natural & y].mean()) if (natural & y).any() else None,
        "natural_neg_mean": float(p[natural & ~y].mean()) if (natural & ~y).any() else None,
        "far_neg_mean": float(p[far & ~y].mean()) if (far & ~y).any() else None,
        "hard_pos_mean": float(p[hard & y].mean()) if (hard & y).any() else None,
        "hard_neg_mean": float(p[hard & ~y].mean()) if (hard & ~y).any() else None,
    }


def train_plain_bce(name, train, val, mask, seed, max_epochs, patience, batch_size, lr, device, outdir):
    _, m0_state, head = _load_m0_and_head(device)
    head.net.train()
    features = train["features"]
    labels = train["arrival_label"]
    indices = np.flatnonzero(mask)
    pos = int(((labels >= 0.5) & mask).sum())
    neg = int(((labels < 0.5) & mask).sum())
    # Deliberately plain BCE, as in the original V8.2 arrival loss.
    criterion = torch.nn.BCEWithLogitsLoss()
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
                raise RuntimeError(f"Non-finite plain-BCE loss {name} epoch {epoch}")
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
    metrics["distributions"] = _prob_distributions(
        val["features"], val["arrival_label"], val["sample_type"],
        val["distance"], head, device,
    )
    full = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0")
    full.load_state_dict(m0_state)
    before = _hash_state({k: v for k, v in m0_state.items() if not k.startswith("base.arrival_head.")})
    full.base.arrival_head.net.load_state_dict(best_state)
    full_state = copy.deepcopy(full.state_dict())
    after = _hash_state({k: v for k, v in full_state.items() if not k.startswith("base.arrival_head.")})
    if before != after:
        raise RuntimeError("Movement weights changed in plain-BCE head training")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, outdir / "arrival_head.pt")
    torch.save(full_state, outdir / "best.pt")
    with (outdir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    result = {
        "variant": name, "loss": "plain_BCE_no_pos_weight",
        "best_epoch": best_epoch, "best_selection_f1": best_score,
        "train_time_s": time.time() - t0, "train_samples": int(mask.sum()),
        "positives": pos, "negatives": neg, "metrics": metrics,
        "movement_unchanged": True, "checkpoint": str(outdir / "best.pt"),
    }
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20264017)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    p.add_argument("--max-episodes", type=int, default=180)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = np.load(STAGE2B / "features/train.npz")
    val = np.load(STAGE2B / "features/dev_val.npz")
    train = {k: train[k] for k in train.files}
    val = {k: val[k] for k in val.files}
    _, m0_state, b0_head = _load_m0_and_head(device)
    verification = verify_cached_logits(val["features"], m0_state, device)
    (args.output_dir / "feature_logit_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print("Feature/logit verification", json.dumps(verification), flush=True)
    a0, a1, mask_stats = build_a1_mask(train)
    (args.output_dir / "mask_stats.json").write_text(json.dumps(mask_stats, indent=2), encoding="utf-8")

    b0_metrics = _arrival_metrics(
        val["features"], val["arrival_label"], val["sample_type"],
        val["distance"], b0_head, device,
    )
    b0_metrics["distributions"] = _prob_distributions(
        val["features"], val["arrival_label"], val["sample_type"],
        val["distance"], b0_head, device,
    )
    b0_result = {"variant": "B0", "loss": "original_no_training", "best_epoch": None,
                 "metrics": b0_metrics, "checkpoint": str(M0_CKPT)}
    (args.output_dir / "B0").mkdir()
    (args.output_dir / "B0/metrics.json").write_text(json.dumps(b0_result, indent=2), encoding="utf-8")

    results = {"B0": b0_result}
    for name, mask in (("B1", a0), ("B2", a1)):
        results[name] = train_plain_bce(
            name, train, val, mask, args.seed, args.max_epochs,
            args.patience, args.batch_size, args.lr, device,
            args.output_dir / name,
        )
    (args.output_dir / "train_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    evaluation.TEST_STATS_PATH = args.val_specs
    evaluation.NUM_EPISODES = args.max_episodes
    specs = evaluation.load_test_episode_specs()
    encoder = load_encoder(device)
    oracle_cache = {spec["episode_index"]: evaluation.build_oracle_reference(spec) for spec in specs}
    rows_by = {}
    # B0: existing Stage-2a M0 closed-loop rows, same 180 episodes/threshold.
    b0_rows = [r for r in csv.DictReader(open(STAGE2A_EVAL / "per_episode.csv")) if r["mode"] == "M0"]
    if len(b0_rows) != len(specs):
        raise RuntimeError(f"B0 closed-loop row count {len(b0_rows)} != {len(specs)}")
    for r in b0_rows:
        r["variant"] = "B0"
    rows_by["B0"] = b0_rows
    for name in ("B1", "B2"):
        print(f"Closed-loop {name}", flush=True)
        rows = evaluate_mode(
            "M0", "Transformer-M", args.output_dir / name / "best.pt",
            specs, oracle_cache, encoder, device, args.threshold, args.max_episodes,
        )
        for r in rows:
            r["variant"] = name
        rows_by[name] = rows
        with (args.output_dir / f"{name}_per_episode.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    analysis = {
        "feature_logit_verification": verification,
        "mask_stats": mask_stats,
        "arrival_metrics": {n: results[n]["metrics"] for n in ("B0", "B1", "B2")},
        "closedloop": {
            "metrics": {n: summarize(rows_by[n]) for n in rows_by},
            "by_band": {n: {b: summarize([r for r in rows_by[n] if r["band"] == b]) for b in BANDS} for n in rows_by},
        },
        "paired": {
            "B1_vs_B0": paired_comparison(rows_by, base="B0", other="B1"),
            "B2_vs_B0": paired_comparison(rows_by, base="B0", other="B2"),
            "B1_vs_B2": paired_comparison(rows_by, base="B1", other="B2"),
        },
        "paired_by_band": {
            b: {
                "B1_vs_B0": paired_comparison({n: [r for r in rows_by[n] if r["band"] == b] for n in ("B0", "B1")}, base="B0", other="B1"),
                "B2_vs_B0": paired_comparison({n: [r for r in rows_by[n] if r["band"] == b] for n in ("B0", "B2")}, base="B0", other="B2"),
            } for b in BANDS
        },
    }
    (args.output_dir / "stage2c_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
