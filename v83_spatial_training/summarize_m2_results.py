"""Render the M0/M1/M2 development comparison from training/eval artifacts.

This script only reads archived run outputs; it never modifies checkpoints.
Metrics are recomputed from per_episode.csv and cross-checked against the
evaluator's summary.json (which remains the canonical JSON artifact).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


MODES = ("M0", "M1", "M2")
BANDS = ("short", "medium", "long")


def load_metrics(base_dir, m2_dir):
    out = {}
    for mode in ("M0", "M1"):
        out[mode] = json.loads(
            (Path(base_dir) / "Transformer-M" / mode / "metrics.json").read_text()
        )
    out["M2"] = json.loads(
        (Path(m2_dir) / "Transformer-M" / "M2" / "metrics.json").read_text()
    )
    return out


def load_rows(eval_dir):
    with (Path(eval_dir) / "per_episode.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ints = {"episode_index", "seed", "success", "osr", "premature_stop",
            "max_steps_termination", "steps", "timed"}
    floats = {"spl", "ndtw", "min_distance_m", "map_ms_per_step",
              "model_ms_per_step"}
    for row in rows:
        for key in ints:
            if key in row:
                row[key] = int(row[key])
        for key in floats:
            if key in row:
                row[key] = float(row[key])
    return rows


def rows_by_mode(rows):
    return {mode: [r for r in rows if r["mode"] == mode] for mode in MODES}


def mean(values):
    return float(np.mean(values)) if values else 0.0


def sign_test_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def bootstrap_ci(diffs, n_boot=10000, seed=20261018):
    values = np.asarray(diffs, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(rows_a, rows_b, key, higher_is_better=True):
    a = {r["episode_index"]: r for r in rows_a}
    b = {r["episode_index"]: r for r in rows_b}
    indices = sorted(set(a) & set(b))
    diffs = [float(b[i][key]) - float(a[i][key]) for i in indices]
    better = sum(
        (d > 1e-12) if higher_is_better else (d < -1e-12) for d in diffs
    )
    worse = sum(
        (d < -1e-12) if higher_is_better else (d > 1e-12) for d in diffs
    )
    low, high = bootstrap_ci(diffs)
    return {
        "better": int(better),
        "worse": int(worse),
        "ties": int(len(diffs) - better - worse),
        "delta": mean(diffs),
        "ci95": [low, high],
        "sign_test_p": sign_test_p(
            sum(d > 1e-12 for d in diffs), sum(d < -1e-12 for d in diffs)
        ),
    }


def fmt_pct(value, digits=2):
    return f"{100.0 * value:.{digits}f}%"


def macro_f1(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    f1s = []
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return float(np.mean(f1s))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--m2-dir", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    metrics = load_metrics(args.base_dir, args.m2_dir)
    rows = load_rows(args.eval_dir)
    by_mode = rows_by_mode(rows)
    summary = json.loads((args.eval_dir / "summary.json").read_text())

    models = {}
    for mode in MODES:
        m = metrics[mode]
        bm = m["best_metrics"]
        models[mode] = {
            "parameters": m["parameters"],
            "best_epoch": m["best_epoch"],
            "best_score": m["best_score"],
            "train_time_s": m["train_time_s"],
            "val_macro_f1": bm["macro_f1"],
            # Macro-F1 over Forward/Left/Right only (Stop excluded).
            "val_movement_macro_f1": macro_f1(
                np.asarray(bm["confusion"], dtype=np.float64)[:3, :3]
            ),
            "val_directional_balanced_accuracy": bm["directional_balanced_accuracy"],
            "val_natural_arrival_f1": bm["arrival"]["f1"],
            "val_hard_arrival_f1": bm["hard_arrival"]["f1"],
        }
    base_params = models["M0"]["parameters"]

    closedloop = {}
    for mode in MODES:
        rr = by_mode[mode]
        timed = [r for r in rr if r.get("timed", 1)]
        map_ms = [r["map_ms_per_step"] for r in timed]
        model_ms = [r["model_ms_per_step"] for r in timed]
        total_ms = [a + b for a, b in zip(map_ms, model_ms)]
        closedloop[mode] = {
            "episodes": len(rr),
            "sr": mean([r["success"] for r in rr]),
            "osr": mean([r["osr"] for r in rr]),
            "spl": mean([r["spl"] for r in rr]),
            "ndtw": mean([r["ndtw"] for r in rr]),
            "premature_stop": mean([r["premature_stop"] for r in rr]),
            "timeout": mean([r["max_steps_termination"] for r in rr]),
            "steps": mean([r["steps"] for r in rr]),
            "min_distance_m": mean([r["min_distance_m"] for r in rr]),
            "latency_episodes": len(timed),
            "map_ms_per_step": mean(map_ms),
            "model_ms_per_step": mean(model_ms),
            "total_ms_per_step": mean(total_ms),
            "total_ms_per_step_p95": float(np.percentile(total_ms, 95)) if total_ms else 0.0,
        }

    # Cross-check the recomputed rates against summary.json.
    for mode in MODES:
        official = summary["summaries"][mode]
        for key in ("sr", "osr", "spl", "ndtw", "premature_stop", "timeout"):
            if abs(float(official[key]) - closedloop[mode][key]) > 1e-12:
                raise SystemExit(
                    f"summary.json mismatch for {mode}.{key}: "
                    f"{official[key]} vs {closedloop[mode][key]}"
                )

    paired_stats = {}
    for base, other in (("M0", "M1"), ("M0", "M2"), ("M1", "M2")):
        paired_stats[f"{other}_vs_{base}"] = {
            "success": paired(by_mode[base], by_mode[other], "success", True),
            "osr": paired(by_mode[base], by_mode[other], "osr", True),
            "premature_stop": paired(
                by_mode[base], by_mode[other], "premature_stop", False
            ),
            "timeout": paired(
                by_mode[base], by_mode[other], "max_steps_termination", False
            ),
            "spl": paired(by_mode[base], by_mode[other], "spl", True),
            "ndtw": paired(by_mode[base], by_mode[other], "ndtw", True),
        }

    band_stats = {}
    for mode in MODES:
        band_stats[mode] = {}
        for band in BANDS:
            rr = [r for r in by_mode[mode] if r["band"] == band]
            if not rr:
                continue
            band_stats[mode][band] = {
                "episodes": len(rr),
                "sr": mean([r["success"] for r in rr]),
                "osr": mean([r["osr"] for r in rr]),
                "spl": mean([r["spl"] for r in rr]),
            }

    payload = {
        "models": models,
        "closedloop": closedloop,
        "paired": paired_stats,
        "by_band": band_stats,
        "sources": {
            "base_dir": str(args.base_dir),
            "m2_dir": str(args.m2_dir),
            "eval_dir": str(args.eval_dir),
        },
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("## Model / parameters / training\n")
    print("| Model | Parameters | vs M0 | Best epoch | Best score | Train time |")
    print("|---|---:|---:|---:|---:|---:|")
    for mode in MODES:
        m = models[mode]
        delta = m["parameters"] - base_params
        pct = 100.0 * delta / base_params
        print(
            f"| {mode} | {m['parameters']:,} | "
            f"{delta:+,} ({pct:+.4f}%) | {m['best_epoch']} | "
            f"{100*m['best_score']:.2f}% | {m['train_time_s']/60:.1f} min |"
        )

    print("\n## Offline validation at best epoch\n")
    print("| Model | Movement Macro-F1 (3-way) | Macro-F1 (4-way) | Directional BA | Natural Arrival-F1 | Hard Arrival-F1 |")
    print("|---|---:|---:|---:|---:|---:|")
    for mode in MODES:
        m = models[mode]
        print(
            f"| {mode} | {100*m['val_movement_macro_f1']:.2f}% | "
            f"{100*m['val_macro_f1']:.2f}% | "
            f"{100*m['val_directional_balanced_accuracy']:.2f}% | "
            f"{100*m['val_natural_arrival_f1']:.2f}% | "
            f"{100*m['val_hard_arrival_f1']:.2f}% |"
        )

    print("\n## Closed-loop 60-episode development validation (Stop=0.5)\n")
    print("| Metric | M0 | M1 | M2 |")
    print("|---|---:|---:|---:|")
    labels = [
        ("SR", "sr", fmt_pct), ("OSR", "osr", fmt_pct), ("SPL", "spl", lambda v: f"{v:.4f}"),
        ("nDTW", "ndtw", lambda v: f"{v:.4f}"),
        ("Premature-Stop", "premature_stop", fmt_pct),
        ("Timeout", "timeout", fmt_pct),
        ("Mean steps", "steps", lambda v: f"{v:.1f}"),
        ("Mean min distance (m)", "min_distance_m", lambda v: f"{v:.1f}"),
        ("Model ms/step", "model_ms_per_step", lambda v: f"{v:.3f}"),
        ("Map ms/step", "map_ms_per_step", lambda v: f"{v:.3f}"),
        ("Total ms/step", "total_ms_per_step", lambda v: f"{v:.3f}"),
    ]
    for label, key, fmt in labels:
        print(f"| {label} | " + " | ".join(fmt(closedloop[m][key]) for m in MODES) + " |")

    print("\n## Paired per-episode comparisons\n")
    print("| Comparison | Metric | Better | Worse | Ties | Delta | 95% CI | Sign-test p |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, stats in paired_stats.items():
        for metric in ("success", "osr", "premature_stop", "timeout", "spl", "ndtw"):
            s = stats[metric]
            if metric in ("success", "osr", "premature_stop", "timeout"):
                delta = fmt_pct(s["delta"])
            else:
                delta = f"{s['delta']:+.4f}"
            print(
                f"| {name} | {metric} | {s['better']} | {s['worse']} | "
                f"{s['ties']} | {delta} | "
                f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | "
                f"{s['sign_test_p']:.4f} |"
            )

    print("\n## SR / OSR by distance band\n")
    print("| Model | Band | Episodes | SR | OSR | SPL |")
    print("|---|---|---:|---:|---:|---:|")
    for mode in MODES:
        for band in BANDS:
            s = band_stats[mode].get(band)
            if not s:
                continue
            print(
                f"| {mode} | {band} | {s['episodes']} | {fmt_pct(s['sr'])} | "
                f"{fmt_pct(s['osr'])} | {s['spl']:.4f} |"
            )
    print(f"\nArtifacts: {args.eval_dir}")


if __name__ == "__main__":
    main()
