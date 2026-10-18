"""Audit Stage-2 splits, seeds, far-negative retention, M1 Long failures and M1 head direction.

Read-only except for the new outputs under the specified --output-dir. The
Final Test dataset is used only for isolation hashes/statistics and is never
used for model selection or closed-loop tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_training.spatial_mec import SpatialFusionHead


DATA_ROOT = Path("runs/v83_spatial_stage2/datasets")
V83_DEV_SPEC = Path("runs/v83_validation/closedloop_val_specs.json")
M1_PER_EPISODE = Path(
    "runs/v83_spatial_training/run_20261018T023830Z/"
    "closedloop_20261018T032039385722Z/per_episode.csv"
)
BAND_RANGES = {
    "short": (150.0, 250.0),
    "medium": (250.0, 400.0),
    "long": (400.0, 550.0),
}
FAR_BINS = ((60, 100), (100, 200), (200, 1e9))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stats_seeds(stats):
    traj = set(); layout = set()
    for t in stats.get("trajectories", []):
        traj.add(int(t["seed"]))
        if "network_layout_seed" in t:
            layout.add(int(t["network_layout_seed"]))
    return traj, layout


def spec_seeds(spec):
    traj = {int(t["seed"]) for t in spec.get("trajectories", [])}
    layout = {int(t["network_layout_seed"]) for t in spec.get("trajectories", []) if "network_layout_seed" in t}
    return traj, layout


def data_summary(stats, path, expected_bands):
    bands = stats.get("trajectory_band_counts", {})
    trajectories = stats.get("trajectories", [])
    by_band = {}
    for band in ("short", "medium", "long"):
        starts = [float(t["start_goal_distance_m"]) for t in trajectories if t["band"] == band]
        by_band[band] = {
            "episodes": len(starts),
            "start_min_m": min(starts) if starts else None,
            "start_mean_m": float(np.mean(starts)) if starts else None,
            "start_max_m": max(starts) if starts else None,
        }
    size = sum(f.stat().st_size for f in Path(path).glob("*") if f.is_file())
    return {
        "path": str(path),
        "version": stats.get("version"),
        "episodes": int(stats.get("num_episodes", 0)),
        "samples": int(stats.get("total_samples", 0)),
        "band_counts": bands,
        "band_counts_match": bands == expected_bands,
        "by_band": by_band,
        "arrival_label_counts": stats.get("arrival_label_counts"),
        "arrival_train_mask_counts": stats.get("arrival_train_mask_counts"),
        "sample_type_counts": stats.get("sample_type_counts"),
        "disk_bytes": int(size),
        "sealed": (Path(path) / "SEALED.json").is_file(),
    }


def far_negative_audit(train_dir):
    stats = defaultdict(Counter)
    positive = 0
    natural = 0
    hard = 0
    for p in sorted(Path(train_dir).glob("shard_*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        st = d["sample_type"].numpy()
        dist = d["distance_to_goal_m"].numpy()
        al = d["arrival_labels"].numpy()
        mask = d["arrival_train_mask"].numpy().astype(bool)
        natural += int((st == 0).sum())
        hard += int((st != 0).sum())
        positive += int((al >= 0.5).sum())
        nat = st == 0
        for lo, hi in FAR_BINS:
            m = nat & (dist >= lo) & (dist < hi)
            stats[(lo, hi)]["total"] += int(m.sum())
            stats[(lo, hi)]["included"] += int((m & mask).sum())
        near = nat & (dist <= 60.0)
        stats[("near", 60)]["total"] += int(near.sum())
        stats[("near", 60)]["included"] += int((near & mask).sum())
        del d
    far_total = sum(stats[(lo, hi)]["total"] for lo, hi in FAR_BINS)
    far_included = sum(stats[(lo, hi)]["included"] for lo, hi in FAR_BINS)
    near_total = stats[("near", 60)]["total"]
    near_included = stats[("near", 60)]["included"]
    return {
        "natural": natural,
        "hard": hard,
        "positive": positive,
        "near_total": near_total,
        "near_included": near_included,
        "far_total": far_total,
        "far_included": far_included,
        "far_retention": far_included / max(far_total, 1),
        "by_far_bin": {
            f"{lo}-{hi if hi < 1e9 else 'inf'}": {
                "total": stats[(lo, hi)]["total"],
                "included": stats[(lo, hi)]["included"],
                "retention": stats[(lo, hi)]["included"] / max(stats[(lo, hi)]["total"], 1),
            } for lo, hi in FAR_BINS
        },
        "proposed_far_counts": {
            "A1_10pct": int(round(0.10 * far_total)),
            "A2_25pct": int(round(0.25 * far_total)),
            "A3_distance_stratified_10pct_each": {
                f"{lo}-{hi if hi < 1e9 else 'inf'}": int(round(0.10 * stats[(lo, hi)]["total"]))
                for lo, hi in FAR_BINS
            },
        },
    }


def m1_long_failures(per_episode_path, spec_path):
    rows = [r for r in csv.DictReader(open(per_episode_path)) if r["mode"] == "M1"]
    spec = load_json(spec_path)
    starts = [float(t["start_goal_distance_m"]) for t in spec["trajectories"]]
    long_rows = []
    for r in rows:
        if r["band"] != "long":
            continue
        i = int(r["episode_index"])
        outcome = "premature_stop" if int(r["premature_stop"]) else (
            "timeout" if int(r["max_steps_termination"]) else "other"
        )
        long_rows.append({
            "episode_index": i,
            "seed": int(r["seed"]),
            "start_goal_distance_m": starts[i],
            "success": int(r["success"]),
            "osr": int(r["osr"]),
            "outcome": outcome,
            "steps": int(r["steps"]),
            "min_distance_m": float(r["min_distance_m"]),
            "spl": float(r["spl"]),
            "ndtw": float(r["ndtw"]),
        })
    long_rows.sort(key=lambda x: x["episode_index"])
    return {
        "count": len(long_rows),
        "outcome_counts": dict(Counter(r["outcome"] for r in long_rows)),
        "rows": long_rows,
    }


def direction_audit():
    torch.manual_seed(20261018)
    # Symmetric left/right localized stimuli: a direction-aware head should give
    # different pooled features; a global average should not.
    x = torch.zeros(2, 2, 24, 24)
    x[0, :, 8:16, 4:10] = 1.0   # left blob
    x[1, :, 8:16, 14:20] = 1.0  # mirror right blob
    head = SpatialFusionHead(2)
    head.eval()
    with torch.no_grad():
        feat = head.net[:-1](x)
        head.net[-1].weight.normal_()
        y = head(x)
    current_feature_diff = float((feat[0] - feat[1]).abs().max().item())
    current_output_diff = float((y[0] - y[1]).abs().max().item())

    class Head2x2(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.GELU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            return self.net(x)

    h2 = Head2x2(); h2.eval()
    with torch.no_grad():
        feat2 = h2.net[:-1](x)
        h2.net[-1].weight.normal_()
        z = h2(x)
    proposed_feature_diff = float((feat2[0] - feat2[1]).abs().max().item())
    proposed_output_diff = float((z[0] - z[1]).abs().max().item())
    current_params = sum(p.numel() for p in head.parameters())
    proposed_params = sum(p.numel() for p in h2.parameters())
    return {
        "current_head": "AdaptiveAvgPool2d(1) global average",
        "current_params": current_params,
        "current_left_right_feature_max_diff": current_feature_diff,
        "current_left_right_output_max_diff": current_output_diff,
        "current_left_right_invariant": current_feature_diff < 1e-5,
        "proposed_head": "AdaptiveAvgPool2d((2,2)) coarse directional grid",
        "proposed_params": proposed_params,
        "proposed_param_delta": proposed_params - current_params,
        "proposed_left_right_feature_max_diff": proposed_feature_diff,
        "proposed_left_right_output_max_diff": proposed_output_diff,
        "proposed_direction_sensitive": proposed_feature_diff > 1e-3,
        "m1_total_params_current": 2724839,
        "m1_total_params_proposed": 2724839 + (proposed_params - current_params),
    }


def write_dev_spec(stats_path, out_path):
    stats = load_json(stats_path)
    out_path = Path(out_path)
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite {out_path}")
    trajectories = []
    for i, t in enumerate(stats["trajectories"]):
        trajectories.append({
            "episode_index": i,
            "seed": int(t["seed"]),
            "band": t["band"],
            "steps": int(t["steps"]),
            "start_goal_distance_m": float(t["start_goal_distance_m"]),
            "network_layout_seed": int(t["network_layout_seed"]),
        })
    doc = {
        "version": stats.get("version", "v8.2-input-redesign"),
        "purpose": "stage2-dev-val",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_is_final_test": False,
        "source_stats": str(stats_path),
        "trajectories": trajectories,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--m1-per-episode", type=Path, default=M1_PER_EPISODE)
    p.add_argument("--v83-dev-spec", type=Path, default=V83_DEV_SPEC)
    args = p.parse_args()
    out = args.output_dir
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)

    specs = {
        "train_600": (args.data_root / "train_600", {"short": 200, "medium": 200, "long": 200}),
        "dev_val_180": (args.data_root / "dev_val_180", {"short": 60, "medium": 60, "long": 60}),
        "final_test_300": (args.data_root / "final_test_300", {"short": 100, "medium": 100, "long": 100}),
    }
    summaries = {}
    seed_sets = {}
    layout_sets = {}
    for name, (path, expected) in specs.items():
        stats = load_json(path / "dataset_stats.json")
        summaries[name] = data_summary(stats, path, expected)
        seed_sets[name], layout_sets[name] = stats_seeds(stats)
    existing = {
        "old_train": Path("token_dataset_train"),
        "old_val": Path("token_dataset_val"),
        "old_val_hard": Path("token_dataset_val_hard"),
        "old_test": Path("token_dataset_test"),
    }
    for name, path in existing.items():
        if (path / "dataset_stats.json").is_file():
            stats = load_json(path / "dataset_stats.json")
            seed_sets[name], layout_sets[name] = stats_seeds(stats)
    if args.v83_dev_spec.is_file():
        seed_sets["v83_dev"], layout_sets["v83_dev"] = spec_seeds(load_json(args.v83_dev_spec))

    names = sorted(seed_sets)
    intersections = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            si = len(seed_sets[a] & seed_sets[b])
            li = len(layout_sets[a] & layout_sets[b])
            if si or li:
                intersections.append({"a": a, "b": b, "trajectory_seed_overlap": si,
                                      "network_layout_overlap": li})

    far = far_negative_audit(args.data_root / "train_600")
    long = m1_long_failures(args.m1_per_episode, args.v83_dev_spec)
    direction = direction_audit()
    dev_spec_path = write_dev_spec(
        args.data_root / "dev_val_180" / "dataset_stats.json",
        out / "dev_val_180_spec.json",
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_summaries": summaries,
        "seed_set_sizes": {k: len(v) for k, v in seed_sets.items()},
        "layout_set_sizes": {k: len(v) for k, v in layout_sets.items()},
        "intersections": intersections,
        "far_negative_audit": far,
        "m1_long_failures": long,
        "direction_audit": direction,
        "dev_val_spec": str(dev_spec_path),
        "note": "Final Test is used only for isolation/hashes; never for tuning.",
    }
    (out / "stage2_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "m1_long_failures.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(long["rows"][0]))
        w.writeheader(); w.writerows(long["rows"])
    print(json.dumps({
        "data_summaries": summaries,
        "intersections": intersections,
        "far_negative_audit": far,
        "m1_long_failures": long,
        "direction_audit": direction,
        "dev_val_spec": str(dev_spec_path),
    }, indent=2), flush=True)
    print(f"Audit: {out}", flush=True)


if __name__ == "__main__":
    main()
