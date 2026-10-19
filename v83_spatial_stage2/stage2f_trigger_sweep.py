"""Stage-2f: offline threshold sweep for the observable recovery trigger.

Pure diagnostic: scans `heatmap_max_top1` / `heatmap_max_entropy` thresholds
and the consecutive-low-confidence length K, then reports precision/recall
against offline invisibility/unreliable-localization labels. Episodes are
split by parity: thresholds are selected on even episodes and reported on odd
episodes. True labels are evaluation targets only; the policy itself never
reads them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2d_common import write_csv, write_json  # noqa: E402
from v83_spatial_stage2.stage2f_analyze import read_features  # noqa: E402


BANDS = ("all", "short", "medium", "long")
STOP_THRESHOLD = 0.7
LOW_TOP1_VALUES = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
LOW_ENTROPY_VALUES = (0.85, 0.90, 0.95, 0.98)
K_VALUES = (1, 2, 3)


def group_sequences(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["trajectory"], row["episode_index"])].append(row)
    for sequence in grouped.values():
        sequence.sort(key=lambda row: row["step"])
    return list(grouped.values())


def label_rows(rows):
    for row in rows:
        row["move_state"] = int(row["m0_arrival_probability"] < STOP_THRESHOLD)
        row["invisible"] = int(row["goal_visible"] < 0.5)
        row["unreliable"] = int(
            row["invisible"]
            or (
                row["goal_visible"] > 0.5
                and row["coherent_peak_error_px"] > 42.0
            )
        )


def evaluate(sequences, low_top1, low_entropy, k, split):
    predicted, invisible, unreliable = [], [], []
    for sequence in sequences:
        low_run = 0
        for row in sequence:
            if split == "even" and row["episode_index"] % 2 != 0:
                continue
            if split == "odd" and row["episode_index"] % 2 != 1:
                continue
            low = (
                row["heatmap_max_top1"] < low_top1
                or row["heatmap_max_entropy"] > low_entropy
            )
            low_run = low_run + 1 if low else 0
            if not row["move_state"]:
                continue
            predicted.append(int(low_run >= k))
            invisible.append(int(row["invisible"]))
            unreliable.append(int(row["unreliable"]))
    if not predicted:
        return None
    predicted = np.asarray(predicted, dtype=int)
    invisible = np.asarray(invisible, dtype=int)
    unreliable = np.asarray(unreliable, dtype=int)

    def metrics(actual):
        tp = int(((predicted == 1) & (actual == 1)).sum())
        fp = int(((predicted == 1) & (actual == 0)).sum())
        fn = int(((predicted == 0) & (actual == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {
            "trigger_rate": float(predicted.mean()),
            "base_rate": float(actual.mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return {
        "low_top1": float(low_top1),
        "low_entropy": float(low_entropy),
        "k": int(k),
        "split": split,
        "states": int(len(predicted)),
        "invisible": metrics(invisible),
        "unreliable": metrics(unreliable),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = read_features(args.features)
    label_rows(rows)
    sequences = group_sequences(rows)
    rows_out = []
    for low_top1 in LOW_TOP1_VALUES:
        for low_entropy in LOW_ENTROPY_VALUES:
            for k in K_VALUES:
                for split in ("even", "odd"):
                    result = evaluate(
                        sequences, low_top1, low_entropy, k, split
                    )
                    if result is not None:
                        rows_out.append(result)

    write_csv(args.output_dir / "sweep.csv", rows_out)
    write_json(args.output_dir / "summary.json", {
        "features": str(args.features),
        "note": (
            "True labels are evaluation targets only. Threshold selection is "
            "on even episodes; odd-episode metrics are held out."
        ),
        "sweep": rows_out,
    })

    # Select on even episodes by invisible F1, subject to a trigger-rate cap.
    train = [
        row for row in rows_out
        if row["split"] == "even" and row["invisible"]["trigger_rate"] <= 0.30
    ]
    if train:
        best = max(train, key=lambda row: row["invisible"]["f1"])
        held = next(
            row for row in rows_out
            if row["split"] == "odd"
            and row["low_top1"] == best["low_top1"]
            and row["low_entropy"] == best["low_entropy"]
            and row["k"] == best["k"]
        )
        print("Selected on even episodes:", best, flush=True)
        print("Held-out odd episodes:", held, flush=True)
    else:
        print("No threshold with trigger rate <= 30% on even episodes.", flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
