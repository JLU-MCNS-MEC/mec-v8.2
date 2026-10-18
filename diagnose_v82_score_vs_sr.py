"""Compare offline checkpoint selection scores against closed-loop SR/OSR.

Read-only source files; writes a unique timestamped report folder under logs/.
No model loading, CUDA, training, checkpoints, or dataset regeneration.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_PATHS = {
    "train200": Path("gbs_v82_input_redesign_results/training_summary_v82.json"),
    "train100": Path("gbs_v82_epoch_compare_results/training_summary_epoch_compare.json"),
    "eval200": Path("fixed_mec_capability_v82_results/fixed_mec_summary.json"),
    "eval100": Path("fixed_mec_epoch_compare_results/epoch_compare_summary.json"),
}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_eval(path):
    raw = read_json(path)
    if isinstance(raw, dict):
        # Safely support either list or wrapped list objects.
        for key in ("results", "summary", "models"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Evaluation summary must contain a list: {path}")
    return raw


def compare(train200, train100, eval200, eval100):
    records = []
    for model_name, result in train200.items():
        found = [r for r in eval200 if r.get("model") == model_name]
        if len(found) != 1:
            raise ValueError(f"200-epoch evaluation row missing/duplicated: {model_name}")
        row = found[0]
        records.append({
            "protocol": "legacy200",
            "model": model_name,
            "best_epoch": result["best_epoch"],
            "offline_score": float(result["best_score"]),
            "sr": float(row["success_rate"]),
            "osr": float(row["osr"]),
            "premature_stop": float(row["premature_stop_rate"]),
            "spl": float(row["mean_spl"]),
            "ndtw": float(row["mean_ndtw"]),
        })
    for model_name, result in train100.items():
        for budget in (50, 100):
            found = [
                r for r in eval100
                if r.get("model") == model_name and int(r.get("budget", -1)) == budget
            ]
            if len(found) != 1:
                raise ValueError(
                    f"Budget-{budget} evaluation row missing/duplicated: {model_name}"
                )
            row = found[0]
            records.append({
                "protocol": f"controlled{budget}",
                "model": model_name,
                "best_epoch": result[f"best_{budget}_epoch"],
                "offline_score": float(result[f"best_{budget}_score"]),
                "sr": float(row["success_rate"]),
                "osr": float(row["osr"]),
                "premature_stop": float(row["premature_stop_rate"]),
                "spl": float(row["mean_spl"]),
                "ndtw": float(row["mean_ndtw"]),
            })
    return records


def summarize(rows):
    grouped = sorted(set(row["protocol"] for row in rows))
    lines = [
        "# Offline checkpoint score vs closed-loop navigation",
        "",
        "Scores are NOT success rates. Five models per protocol are too few for robust correlations.",
        "Different training protocols and seeds are not directly controlled.",
        "",
    ]
    for protocol in grouped:
        group = [r for r in rows if r["protocol"] == protocol]
        x = np.array([r["offline_score"] for r in group], dtype=float)
        y = np.array([r["sr"] for r in group], dtype=float)
        if len(group) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            p = float(np.corrcoef(x, y)[0, 1])
            p_text = f"{p:.3f}"
        else:
            p_text = "undefined"
        lines.extend([
            f"## {protocol}",
            f"Across-model Pearson(offline_score, SR): {p_text}",
            "",
            "| Model | best epoch | Offline score | SR | OSR | PremStop |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for r in sorted(group, key=lambda v: v["sr"], reverse=True):
            lines.append(
                f"| {r['model']} | {r['best_epoch']} "
                f"| {r['offline_score']:.2%} | {r['sr']:.2%} "
                f"| {r['osr']:.2%} | {r['premature_stop']:.2%} |"
            )
        lines.append("")
    lines.extend([
        "Do not pick checkpoints on the final test split.",
        "For new training, select candidates using a distinct closed-loop validation split.",
    ])
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, value in DEFAULT_PATHS.items():
        ap.add_argument("--" + key.replace("_", "-"), type=Path, default=value)
    ap.add_argument("--output-root", type=Path, default=Path("logs/diagnostics"))
    args = ap.parse_args()
    paths = {key: getattr(args, key) for key in DEFAULT_PATHS}
    for key, path in paths.items():
        if not path.is_file():
            raise SystemExit(f"Missing input {key}: {path}. Migrate results first.")
    rows = compare(
        read_json(paths["train200"]),
        read_json(paths["train100"]),
        _load_eval(paths["eval200"]),
        _load_eval(paths["eval100"]),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = args.output_root / ("v82_score_sr_" + stamp)
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "scores_vs_closed_loop.csv").open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (folder / "diagnosis.md").write_text(summarize(rows), encoding="utf-8")
    print(f"Analyzed {len(rows)} model/checkpoint records.")
    print(f"Report: {folder / 'diagnosis.md'}")
    print(f"CSV: {folder / 'scores_vs_closed_loop.csv'}")
    print("No training/checkpoint/old result was modified.")


if __name__ == "__main__":
    main()
