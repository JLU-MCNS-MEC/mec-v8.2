"""Stage-2g.1 reporting analysis from saved probe predictions.

Reads the probe JSON, the combined Dev-Val valid data and the saved prediction
arrays. Writes analysis.json and analysis.md with the gate table, distance-band
behaviour, operating-point confusions and history-length comparison.

No training, no labels used as model inputs; this is analysis only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g_probe import auc_rank  # noqa: E402

PER_STATE = (
    "source", "variant", "pair_id", "episode_id", "distance_m",
    "target_visible", "target_localizable", "near_target_invisible",
    "history_origin",
)
HEADS = ("target_visible", "target_localizable", "near_target_invisible")
HEAD_SHORT = {
    "target_visible": "visible",
    "target_localizable": "localizable",
    "near_target_invisible": "near_inv",
}


def load_val(path):
    z = np.load(path)
    valid = z["history_valid"] > 0
    return {key: z[key][valid] for key in PER_STATE}


def score_for(probabilities, head):
    return (
        np.asarray(probabilities, dtype=float)
        if head == "near_target_invisible"
        else 1.0 - np.asarray(probabilities, dtype=float)
    )


def distance_band(distance):
    distance = np.asarray(distance, dtype=float)
    return np.select(
        [distance < 20, distance < 50, distance < 100,
         distance < 200, distance < 400],
        ["5-20", "20-50", "50-100", "100-200", "200-400"],
        default="400+",
    )


def operating_confusion(near, flags):
    tp = int((flags & near).sum())
    fp = int((flags & ~near).sum())
    fn = int((~flags & near).sum())
    tn = int((~flags & ~near).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "flags": int(flags.sum()),
    }


def rate(mask_num, mask_den):
    return float((mask_num & mask_den).sum() / max(int(mask_den.sum()), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--combined-val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(
        (args.probe_dir / "probe_results.json").read_text(encoding="utf-8")
    )
    data = load_val(args.combined_val)
    bands = distance_band(data["distance_m"])
    source = data["source"]
    variant = data["variant"]
    pair_id = data["pair_id"]
    near = data["near_target_invisible"] == 1
    near_supp = (source == 1) & (variant == 1)
    near_vis = (source == 1) & (variant == 0)
    far_visible = (
        (source == 0) & (data["distance_m"] >= 100)
        & (data["target_visible"] == 1)
    )
    far_invisible = (
        (source == 0) & (data["distance_m"] >= 100)
        & (data["target_visible"] == 0)
    )

    output = {
        "data": results["data"],
        "gate": {},
        "operating_confusions": {},
        "distance_bands": {},
        "overall_bands": {},
        "subsets": {},
        "history_length": results["history_length"],
        "baselines": {
            "goal_only": results["goal_only_baseline"],
            "distance_only": results["distance_only_diagnostic"],
        },
    }

    for v in sorted(results["variants"]):
        output["gate"][v] = {}
        output["operating_confusions"][v] = {}
        output["distance_bands"][v] = {}
        output["overall_bands"][v] = {}
        output["subsets"][v] = {}
        for head in HEADS:
            external = results["variants"][v]["heads"][head]["external"]
            probabilities = np.load(
                args.probe_dir / f"predictions_v{v}_{head}_val.npz"
            )["probability"]
            if len(probabilities) != len(near):
                raise RuntimeError("Prediction/data length mismatch")
            score = score_for(probabilities, head)
            trigger = external.get("trigger_val", {})
            gate_entry = {}
            for op in ("f1_far_visible_5pct", "recall_0.5",
                       "precision_0.8", "precision_far_visible_5pct"):
                entry = trigger.get(op)
                if entry is None:
                    gate_entry[op] = None
                    continue
                flags = score >= entry["threshold"]
                gate_entry[op] = {
                    "threshold": entry["threshold"],
                    "precision": entry["near_target_invisible_precision"],
                    "recall": entry["near_target_invisible_recall"],
                    "far_visible_false_trigger": entry[
                        "far_visible_false_trigger_rate"
                    ],
                    "far_invisible_false_trigger": entry[
                        "far_invisible_false_trigger_rate"
                    ],
                    "near_visible_false_trigger": entry[
                        "near_visible_false_trigger_rate"
                    ],
                    "flags": entry["flags"],
                    "ci": entry.get("ci"),
                    "paired_at_trigger": entry.get("paired_at_trigger"),
                    "confusion": operating_confusion(near, flags),
                    "composition": entry.get("flag_composition"),
                }
            output["gate"][v][head] = gate_entry
            output["operating_confusions"][v][head] = {
                op: gate_entry[op]["confusion"]
                for op in gate_entry if gate_entry[op] is not None
            }
            # Distance-band behaviour at the f1 operating point.
            f1 = gate_entry.get("f1_far_visible_5pct")
            band_entry = {}
            if f1 is not None:
                flags = score >= f1["threshold"]
                band_entry["near_recall"] = {}
                for band in ("5-20", "20-50", "50-100"):
                    mask = near_supp & (bands == band)
                    band_entry["near_recall"][band] = {
                        "n": int(mask.sum()), "rate": rate(flags, mask)
                    }
                band_entry["near_visible_false_trigger"] = {}
                for band in ("5-20", "20-50", "50-100"):
                    mask = near_vis & (bands == band)
                    band_entry["near_visible_false_trigger"][band] = {
                        "n": int(mask.sum()), "rate": rate(flags, mask)
                    }
                band_entry["far_visible_false_trigger"] = {}
                for band in ("100-200", "200-400", "400+"):
                    mask = far_visible & (bands == band)
                    band_entry["far_visible_false_trigger"][band] = {
                        "n": int(mask.sum()), "rate": rate(flags, mask)
                    }
                band_entry["far_invisible_false_trigger"] = {}
                for band in ("100-200", "200-400", "400+"):
                    mask = far_invisible & (bands == band)
                    band_entry["far_invisible_false_trigger"][band] = {
                        "n": int(mask.sum()), "rate": rate(flags, mask)
                    }
            # Paired AUC by near distance band.
            rows = []
            for pid in np.unique(pair_id[source == 1]):
                visible_mask = (
                    (source == 1) & (pair_id == pid) & (variant == 0)
                )
                invisible_mask = (
                    (source == 1) & (pair_id == pid) & (variant == 1)
                )
                if not np.any(visible_mask) or not np.any(invisible_mask):
                    continue
                rows.append((
                    float(data["distance_m"][invisible_mask][0]),
                    float(score[invisible_mask][0]),
                    float(score[visible_mask][0]),
                ))
            rows = np.asarray(rows)
            band_entry["paired_auroc"] = {}
            if len(rows):
                for band, (low, high) in {
                    "5-20": (5, 20), "20-50": (20, 50),
                    "50-100": (50, 100),
                }.items():
                    mask = (rows[:, 0] >= low) & (rows[:, 0] < high)
                    if not np.any(mask):
                        continue
                    labels = np.concatenate([
                        np.ones(int(mask.sum()), dtype=int),
                        np.zeros(int(mask.sum()), dtype=int),
                    ])
                    values = np.concatenate([rows[mask, 1], rows[mask, 2]])
                    band_entry["paired_auroc"][band] = {
                        "pairs": int(mask.sum()),
                        "auroc": auc_rank(labels, values),
                    }
            output["distance_bands"][v][head] = band_entry
            output["overall_bands"][v][head] = {
                "overall": external["overall"],
                "bands": external["bands"],
                "near_invisible_vs_far_invisible_auroc": external[
                    "near_invisible_vs_far_invisible_auroc"
                ],
                "near_invisible_vs_hard_far_invisible_auroc": external.get(
                    "near_invisible_vs_hard_far_invisible_auroc"
                ),
                "near_invisible_vs_natural_far_invisible_auroc": external.get(
                    "near_invisible_vs_natural_far_invisible_auroc"
                ),
                "near_invisible_vs_far_visible_auroc": external[
                    "near_invisible_vs_far_visible_auroc"
                ],
                "paired": external["paired"],
            }
            output["subsets"][v][head] = external["subsets"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Markdown summary.
    lines = ["# Stage-2g.1 analysis", ""]
    lines += ["## Gate table (Dev Val, train-selected thresholds)", ""]
    lines += [
        "| Variant | Head | Op | Precision | Recall | Far-vis FT | Far-inv FT | Near-vis FT | Flags | Paired acc |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in sorted(output["gate"]):
        for head in HEADS:
            for op, entry in output["gate"][v][head].items():
                if entry is None:
                    continue
                pair = entry.get("paired_at_trigger") or {}
                lines.append(
                    f"| P{v} | {HEAD_SHORT[head]} | {op} | "
                    f"{entry['precision']:.3f} | {entry['recall']:.3f} | "
                    f"{entry['far_visible_false_trigger']:.4f} | "
                    f"{entry['far_invisible_false_trigger']:.3f} | "
                    f"{entry['near_visible_false_trigger']:.3f} | "
                    f"{entry['flags']} | "
                    f"{pair.get('paired_accuracy_at_threshold', float('nan')):.3f} |"
                )
    lines += ["", "## Operating-point confusion (near-target-invisible positive)", ""]
    lines += [
        "| Variant | Head | Op | TP | FP | FN | TN | Precision | Recall | Specificity |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in sorted(output["operating_confusions"]):
        for head in HEADS:
            for op, confusion in output["operating_confusions"][v][head].items():
                if confusion["flags"] == 0:
                    continue
                lines.append(
                    f"| P{v} | {HEAD_SHORT[head]} | {op} | "
                    f"{confusion['tp']} | {confusion['fp']} | "
                    f"{confusion['fn']} | {confusion['tn']} | "
                    f"{confusion['precision']:.3f} | "
                    f"{confusion['recall']:.3f} | "
                    f"{confusion['specificity']:.3f} |"
                )
    lines += ["", "## Distance bands at the f1 operating point", ""]
    lines += [
        "| Variant | Head | Near 5-20 | Near 20-50 | Near 50-100 | Far-vis 100-200 | Far-vis 200-400 | Far-vis 400+ | Far-inv 100-200 | Far-inv 200-400 | Far-inv 400+ | Paired 5-20 | Paired 20-50 | Paired 50-100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in sorted(output["distance_bands"]):
        for head in ("near_target_invisible", "target_visible"):
            entry = output["distance_bands"][v][head]
            if not entry:
                continue
            near_recall = entry.get("near_recall", {})
            far_vis = entry.get("far_visible_false_trigger", {})
            far_inv = entry.get("far_invisible_false_trigger", {})
            paired = entry.get("paired_auroc", {})
            def val(mapping, key):
                item = mapping.get(key)
                return f"{item['rate']:.3f}" if item else "—"
            def pav(key):
                item = paired.get(key)
                return f"{item['auroc']:.3f}" if item else "—"
            lines.append(
                f"| P{v} | {HEAD_SHORT[head]} | "
                f"{val(near_recall, '5-20')} | "
                f"{val(near_recall, '20-50')} | "
                f"{val(near_recall, '50-100')} | "
                f"{val(far_vis, '100-200')} | "
                f"{val(far_vis, '200-400')} | "
                f"{val(far_vis, '400+')} | "
                f"{val(far_inv, '100-200')} | "
                f"{val(far_inv, '200-400')} | "
                f"{val(far_inv, '400+')} | "
                f"{pav('5-20')} | {pav('20-50')} | {pav('50-100')} |"
            )
    lines += ["", "## Gate verdict at f1_far_visible_5pct", ""]
    for v in sorted(output["gate"]):
        for head in HEADS:
            entry = output["gate"][v][head].get("f1_far_visible_5pct")
            if entry is None:
                continue
            passed = (
                entry["precision"] >= 0.5
                and entry["far_visible_false_trigger"] <= 0.05
            )
            lines.append(
                f"- P{v} {HEAD_SHORT[head]}: precision "
                f"{entry['precision']:.3f}, far-visible FT "
                f"{entry['far_visible_false_trigger']:.4f}, recall "
                f"{entry['recall']:.3f} -> "
                f"{'PASS' if passed else 'FAIL'}"
            )
    lines += ["", "## History length (Dev Val)", ""]
    lines += [
        "| Variant | Head | AUROC | AUPRC | Near-vs-far-inv | Paired AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for v, heads in output["history_length"].items():
        for head, entry in heads.items():
            lines.append(
                f"| P{v} | {HEAD_SHORT[head]} | {entry['auroc']:.3f} | "
                f"{entry['auprc']:.3f} | "
                f"{entry['near_inv_vs_far_inv_auroc']:.3f} | "
                f"{entry['paired_auroc']:.3f} |"
            )
    lines += ["", "## Baseline controls", ""]
    goal = output["baselines"]["goal_only"]
    lines.append(
        f"- Goal-only near-inv AUROC: {goal['overall']['auroc']:.3f}; "
        f"near-vs-far-inv AUC: "
        f"{goal['near_invisible_vs_far_invisible_auroc']:.3f}; "
        f"paired AUC: {goal['paired'].get('paired_auroc')}"
    )
    distance = output["baselines"]["distance_only"]
    lines.append(
        f"- Distance-only diagnostic: near-vs-far-inv AUC "
        f"{distance['near_invisible_vs_far_invisible_auroc_val']:.3f}; "
        f"paired AUC 0.5 (same distance within a pair); not a model input."
    )
    (args.output.parent / "analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "gate_variants": list(output["gate"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
