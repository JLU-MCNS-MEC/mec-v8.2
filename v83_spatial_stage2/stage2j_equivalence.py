"""Verify that Stage-2j Transformer-M is the Stage-2i T5 model unchanged.

Checks, on CPU where possible:

1. state-dict keys, shapes and values are identical to
   ``Stage2ITransformer("T5")``;
2. a fixed Dev-Val batch produces identical move/arrival outputs;
3. the full Dev Val 180 closed-loop evaluation reproduces the Stage-2i T5
   per-episode outcomes, step actions, distances and arrival probabilities;
4. the aggregate metrics match.

Run after:
  python v83_spatial_stage2/stage2j_eval.py --model Transformer-M --split dev_val

Final Test is never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2i_data import load_shard_bundle, sorted_shards
from v83_spatial_stage2.stage2i_model import Stage2ITransformer
from v83_spatial_stage2.stage2j_common import (
    AUX_ROOT,
    T5_EXPECTED_SHA256,
    VAL_DIR,
    file_sha256,
    model_dir,
    reject_final_test,
)
from v83_spatial_stage2.stage2j_model import build_stage2j_model


AUX_FIELDS = (
    "history_vis", "history_pose", "history_action", "history_mask",
    "gab", "map_feat",
)
OUTCOME_FIELDS = (
    "success", "osr", "premature_stop", "max_steps_termination",
)
NUMERIC_EPISODE_FIELDS = (
    "spl", "ndtw", "min_distance_m", "final_distance_m",
)
STEP_FIELDS = (
    "seed", "band", "distance_m", "goal_visible", "arrival_probability",
    "move_action", "action", "gab_valid", "gab_confidence",
)


def read_csv_rows(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _state_dict_check():
    stage2i = Stage2ITransformer("T5")
    stage2j = build_stage2j_model("Transformer-M")
    left = stage2i.state_dict()
    right = stage2j.state_dict()
    result = {
        "keys_equal": set(left) == set(right),
        "shape_mismatches": [],
        "value_mismatches": [],
        "max_abs_diff": 0.0,
    }
    for key in sorted(set(left) & set(right)):
        if left[key].shape != right[key].shape:
            result["shape_mismatches"].append(key)
            continue
        diff = float((left[key] - right[key]).abs().max().item())
        result["max_abs_diff"] = max(result["max_abs_diff"], diff)
        if diff != 0.0:
            result["value_mismatches"].append(key)
    result["passed"] = (
        result["keys_equal"]
        and not result["shape_mismatches"]
        and not result["value_mismatches"]
    )
    return result, stage2i, stage2j


def _fixed_batch_check(stage2i, stage2j):
    shards = sorted_shards(VAL_DIR)
    bundle = load_shard_bundle(
        shards[0], AUX_ROOT / "val" / "aux_shard_0000.npz", AUX_FIELDS
    )
    indices = np.arange(8)
    current = bundle["tokens"][indices].float()
    goal = bundle["goal_tokens"][indices].float()
    kwargs = {
        "heatmap": None,
        "network_state": None,
        "history_vis": bundle["history_vis"][indices].float(),
        "history_pose": bundle["history_pose"][indices].float(),
        "history_action": bundle["history_action"][indices].float(),
        "history_mask": bundle["history_mask"][indices].float(),
        "gab": bundle["gab"][indices].float(),
        "map_feat": bundle["map_feat"][indices].float(),
    }
    # Load the Stage-2i T5 checkpoint into both modules.
    checkpoint = torch.load(
        model_dir("Transformer-M") / "best.pt",
        map_location="cpu", weights_only=False,
    )
    state = checkpoint["state_dict"]
    stage2i.load_state_dict(state, strict=True)
    stage2j.load_state_dict(state, strict=True)
    stage2i.eval()
    stage2j.eval()
    with torch.no_grad():
        left = stage2i.forward_components(current, goal, **kwargs)
        right = stage2j.forward_components(current, goal, **kwargs)
    result = {}
    for key in ("move_logits", "arrival_logit", "arrival_features"):
        diff = float((left[key] - right[key]).abs().max().item())
        result[key] = {"max_abs_diff": diff, "equal": diff == 0.0}
    result["passed"] = all(item["equal"] for item in result.values())
    return result


def _episode_check(stage2i_dir, stage2j_dir):
    left_rows = read_csv_rows(stage2i_dir / "T5_per_episode.csv")
    right_rows = read_csv_rows(
        stage2j_dir / "Transformer-M_per_episode.csv"
    )
    left = {int(row["episode_index"]): row for row in left_rows}
    right = {int(row["episode_index"]): row for row in right_rows}
    issues = []
    if set(left) != set(right):
        issues.append({
            "kind": "episode_set",
            "missing": sorted(set(left) - set(right))[:20],
            "extra": sorted(set(right) - set(left))[:20],
        })
    numeric_max = {key: 0.0 for key in NUMERIC_EPISODE_FIELDS}
    for episode in sorted(set(left) & set(right)):
        a, b = left[episode], right[episode]
        for key in OUTCOME_FIELDS:
            if int(a[key]) != int(b[key]):
                issues.append({
                    "kind": "outcome", "episode": episode, "field": key,
                    "stage2i": int(a[key]), "stage2j": int(b[key]),
                })
        if int(a["steps"]) != int(b["steps"]):
            issues.append({
                "kind": "steps", "episode": episode,
                "stage2i": int(a["steps"]), "stage2j": int(b["steps"]),
            })
        for key in NUMERIC_EPISODE_FIELDS:
            diff = abs(float(a[key]) - float(b[key]))
            numeric_max[key] = max(numeric_max[key], diff)
            if diff > 1e-5:
                issues.append({
                    "kind": "numeric", "episode": episode, "field": key,
                    "stage2i": float(a[key]), "stage2j": float(b[key]),
                    "diff": diff,
                })
    return {
        "episodes": len(right),
        "issues": issues,
        "numeric_max_abs_diff": numeric_max,
        "passed": not issues,
    }


def _step_check(stage2i_dir, stage2j_dir):
    left_rows = read_csv_rows(stage2i_dir / "T5_steps.csv")
    right_rows = read_csv_rows(
        stage2j_dir / "Transformer-M_steps.csv"
    )
    left = {
        (int(row["episode_index"]), int(row["step"])): row
        for row in left_rows
    }
    right = {
        (int(row["episode_index"]), int(row["step"])): row
        for row in right_rows
    }
    issues = []
    if set(left) != set(right):
        issues.append({
            "kind": "step_key_set",
            "missing": len(set(left) - set(right)),
            "extra": len(set(right) - set(left)),
        })
    numeric_max = {
        "distance_m": 0.0,
        "arrival_probability": 0.0,
    }
    action_mismatches = 0
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if int(a["action"]) != int(b["action"]):
            action_mismatches += 1
            if len(issues) < 50:
                issues.append({
                    "kind": "action", "episode": key[0], "step": key[1],
                    "stage2i": int(a["action"]), "stage2j": int(b["action"]),
                })
        if int(a["move_action"]) != int(b["move_action"]):
            if len(issues) < 50:
                issues.append({
                    "kind": "move_action", "episode": key[0], "step": key[1],
                    "stage2i": int(a["move_action"]),
                    "stage2j": int(b["move_action"]),
                })
        if int(a["gab_valid"]) != int(b["gab_valid"]):
            if len(issues) < 50:
                issues.append({
                    "kind": "gab_valid", "episode": key[0], "step": key[1],
                })
        diff = abs(float(a["distance_m"]) - float(b["distance_m"]))
        numeric_max["distance_m"] = max(numeric_max["distance_m"], diff)
        if diff > 1e-5:
            issues.append({
                "kind": "distance", "episode": key[0], "step": key[1],
                "diff": diff,
            })
        diff = abs(
            float(a["arrival_probability"])
            - float(b["arrival_probability"])
        )
        numeric_max["arrival_probability"] = max(
            numeric_max["arrival_probability"], diff
        )
        if diff > 1e-6:
            issues.append({
                "kind": "probability", "episode": key[0], "step": key[1],
                "diff": diff,
            })
    return {
        "steps": len(right),
        "issues": issues,
        "numeric_max_abs_diff": numeric_max,
        "action_mismatches": action_mismatches,
        "passed": not issues,
    }


def _metrics_check(stage2i_dir, stage2j_dir):
    left = json.loads((stage2i_dir / "metrics.json").read_text())
    right = json.loads((stage2j_dir / "metrics.json").read_text())
    issues = []
    for key in ("sr", "osr", "spl", "ndtw", "premature_stop", "timeout",
                "steps"):
        diff = abs(float(left["overall"][key]) - float(right["overall"][key]))
        if diff > 1e-9:
            issues.append({
                "kind": "metric", "field": key,
                "stage2i": float(left["overall"][key]),
                "stage2j": float(right["overall"][key]),
                "diff": diff,
            })
    for band in ("short", "medium", "long"):
        for key in ("sr", "osr", "spl", "ndtw"):
            diff = abs(
                float(left["bands"][band][key])
                - float(right["bands"][band][key])
            )
            if diff > 1e-9:
                issues.append({
                    "kind": "band_metric", "band": band, "field": key,
                    "diff": diff,
                })
    sha_left = left["meta"]["checkpoint_sha256"]
    sha_right = right["meta"]["checkpoint_sha256"]
    if sha_left != sha_right:
        issues.append({
            "kind": "checkpoint_sha",
            "stage2i": sha_left, "stage2j": sha_right,
        })
    if sha_right != T5_EXPECTED_SHA256:
        issues.append({
            "kind": "checkpoint_sha_expected",
            "actual": sha_right, "expected": T5_EXPECTED_SHA256,
        })
    return {
        "stage2i_metrics": left["overall"],
        "stage2j_metrics": right["overall"],
        "issues": issues,
        "passed": not issues,
    }


def run(args):
    stage2i_dir = Path(args.stage2i_eval_dir)
    stage2j_dir = Path(args.stage2j_eval_dir)
    reject_final_test(stage2i_dir)
    reject_final_test(stage2j_dir)
    state_result, stage2i, stage2j = _state_dict_check()
    fixed = _fixed_batch_check(stage2i, stage2j)
    episodes = _episode_check(stage2i_dir, stage2j_dir)
    steps = _step_check(stage2i_dir, stage2j_dir)
    metrics = _metrics_check(stage2i_dir, stage2j_dir)
    passed = all([
        state_result["passed"],
        fixed["passed"],
        episodes["passed"],
        steps["passed"],
        metrics["passed"],
    ])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "state_dict": state_result,
        "fixed_batch": fixed,
        "dev_val_episodes": episodes,
        "dev_val_steps": steps,
        "dev_val_metrics": metrics,
        "stage2i_checkpoint_sha256": file_sha256(
            model_dir("Transformer-M") / "best.pt"
        ),
        "final_test_read": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "equivalence.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "state_dict": state_result["passed"],
        "fixed_batch": fixed["passed"],
        "dev_val_episodes": episodes["passed"],
        "dev_val_steps": steps["passed"],
        "dev_val_metrics": metrics["passed"],
        "output": str(path),
    }, indent=2), flush=True)
    if not passed:
        print(json.dumps(payload, indent=2)[:8000], flush=True)
        raise SystemExit(1)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2i-eval-dir", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2i/T5_map_assisted/eval"),
    )
    parser.add_argument(
        "--stage2j-eval-dir", type=Path,
        default=model_dir("Transformer-M") / "eval_dev_val_180",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=model_dir("Transformer-M") / "dev_val_equivalence",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
