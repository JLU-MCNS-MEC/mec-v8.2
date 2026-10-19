"""Stage-2g.2 reproduction check for the re-run P1-2 probe.

Compares the old and new external P1-2 near-inv Dev-Val predictions and the
existing Train-OOF-selected threshold. If max absolute difference <= 1e-4, the
old threshold may be reused. Otherwise the new probe must be treated as a new
probe and its own Train-OOF threshold must be used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2g_probe import metric_block  # noqa: E402
from v83_spatial_stage2.stage2g1_probe import (  # noqa: E402
    load_combined,
    trigger_metrics,
)

OLD_THRESHOLD = 0.13694876571430528
MAX_ABS_DIFF = 1e-4


def prediction_json(probe_dir):
    return json.loads(
        (Path(probe_dir) / "probe_results.json").read_text(encoding="utf-8")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-probe", type=Path, required=True)
    parser.add_argument("--new-probe", type=Path, required=True)
    parser.add_argument("--combined-val", type=Path, required=True)
    args = parser.parse_args()

    old = np.load(
        args.old_probe
        / "predictions_v2_near_target_invisible_val.npz"
    )["probability"]
    new = np.load(
        args.new_probe
        / "predictions_v2_near_target_invisible_val.npz"
    )["probability"]
    if len(old) != len(new):
        raise SystemExit(f"length mismatch {len(old)} vs {len(new)}")
    difference = np.abs(old - new)
    data = load_combined(args.combined_val)
    labels = data["near_target_invisible"]
    old_metrics = metric_block(labels, old)
    new_metrics = metric_block(labels, new)
    old_json = prediction_json(args.old_probe)
    new_json = prediction_json(args.new_probe)
    old_threshold = old_json["variants"]["2"]["heads"][
        "near_target_invisible"
    ]["external"]["thresholds"]["f1_far_visible_5pct"]["threshold"]
    new_threshold = new_json["variants"]["2"]["heads"][
        "near_target_invisible"
    ]["external"]["thresholds"]["f1_far_visible_5pct"]["threshold"]
    old_trigger = trigger_metrics(data, old, old_threshold)
    new_at_old = trigger_metrics(data, new, old_threshold)
    new_trigger = trigger_metrics(data, new, new_threshold)
    reproduced = bool(float(difference.max()) <= MAX_ABS_DIFF)
    # If the old prediction and new prediction are reproduced, use the old
    # Train-OOF threshold. Otherwise use the new probe's own Train-OOF
    # threshold; the Dev Val gate is checked below.
    selected_threshold = old_threshold if reproduced else new_threshold
    selected_trigger = old_trigger if reproduced else new_trigger
    gate_pass = bool(
        selected_trigger["near_target_invisible_precision"] >= 0.50
        and selected_trigger["far_visible_false_trigger_rate"] <= 0.05
    )
    output = {
        "old_probe": str(args.old_probe),
        "new_probe": str(args.new_probe),
        "states": int(len(old)),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "tolerance": MAX_ABS_DIFF,
        "reproduced": reproduced,
        "old_threshold_train_oof": float(old_threshold),
        "new_threshold_train_oof": float(new_threshold),
        "selected_threshold": float(selected_threshold),
        "threshold_source": (
            "old Train-OOF threshold (reproduced)"
            if reproduced
            else "new probe Train-OOF threshold"
        ),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "old_trigger_at_old_threshold": old_trigger,
        "new_trigger_at_old_threshold": new_at_old,
        "new_trigger_at_new_threshold": new_trigger,
        "selected_trigger": selected_trigger,
        "gate_pass": gate_pass,
        "gate_definition": {
            "near_target_invisible_precision_min": 0.50,
            "far_visible_false_trigger_max": 0.05,
            "far_invisible_checked": True,
        },
    }
    output_path = args.new_probe / "reproduction_check.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "reproduced": reproduced,
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "old_threshold": float(old_threshold),
        "new_threshold": float(new_threshold),
        "selected_threshold": float(selected_threshold),
        "gate_pass": gate_pass,
        "selected_trigger": {
            "precision": selected_trigger[
                "near_target_invisible_precision"
            ],
            "recall": selected_trigger["near_target_invisible_recall"],
            "far_visible_ft": selected_trigger[
                "far_visible_false_trigger_rate"
            ],
            "far_invisible_ft": selected_trigger[
                "far_invisible_false_trigger_rate"
            ],
        },
        "output": str(output_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
