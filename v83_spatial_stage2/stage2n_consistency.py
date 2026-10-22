"""Stage-2n homogeneous-MEC consistency probe.

Runs a small set of Dev tasks with all non-privileged heuristic strategies and
checks that:

1. the same observation fed twice to the single frozen Transformer-M yields
   byte-identical actions/probabilities (node identity is not an input);
2. every strategy produces the same navigation action sequence and the same
   SR/SPL/nDTW on each task; only E2E may differ.

This is a real closed-loop audit, not a unit mock.  It runs before training so
a hidden model difference is caught early.

    python v83_spatial_stage2/stage2n_consistency.py --tasks 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now
from v83_spatial_stage2.stage2n_common import (
    EVIDENCE_DIR,
    MEC_CHECKPOINT_SHA256,
    MEC_MODEL,
    deployment_topology,
    dev_specs,
    load_deployment,
    prepare_runner_spec,
)
from v83_spatial_stage2.stage2n_runtime import (
    LatencyEvaluator,
    LatencyOnlyEpisodeRunner,
    load_stage2n_runtime,
    make_stage2n_scene_cache,
)
from v83_spatial_stage2.stage2n_train import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    bundle = load_bundle()
    specs = [prepare_runner_spec(item) for item in bundle["dev_specs"][
        :max(1, int(args.tasks))
    ]]
    device, encoder, backends = load_stage2n_runtime(args.device)
    cache = make_stage2n_scene_cache(encoder, device, capacity=16)
    evaluator = LatencyEvaluator(
        specs=specs,
        topology=bundle["topology"],
        encoder=encoder,
        backends=backends,
        quality_prior=bundle["quality_prior"],
        gab_config=bundle["gab_config"],
        device=device,
        scene_cache=cache,
    )

    first = evaluator._runner(specs[0])
    probe_inputs = {
        key: value.clone() if hasattr(value, "clone") else value
        for key, value in first.step_data["inputs"].items()
    }
    output_a = backends.get(MEC_MODEL, probe_inputs)
    output_b = backends.get(MEC_MODEL, probe_inputs)
    backend_equal = all(
        output_a[key] == output_b[key]
        for key in ("action", "arrival_probability", "move_action")
    )

    rows_by_strategy = {}
    for strategy in ("random", "nearest", "max_uplink_rate", "greedy_min_e2e"):
        print(f"[consistency] {strategy}", flush=True)
        rows_by_strategy[strategy] = evaluator.evaluate_strategy(
            strategy, rng_seed=20261022
        )

    reference = rows_by_strategy["greedy_min_e2e"]
    by_uid = {
        strategy: {row["episode_uid"]: row for row in rows}
        for strategy, rows in rows_by_strategy.items()
    }
    per_strategy = {}
    passed = bool(backend_equal)
    for strategy in rows_by_strategy:
        action_mismatch = 0
        metric_mismatch = 0
        max_spl = 0.0
        max_ndtw = 0.0
        for ref_row in reference:
            uid = ref_row["episode_uid"]
            row = by_uid[strategy][uid]
            if row["action_sha256"] != ref_row["action_sha256"]:
                action_mismatch += 1
            if (
                int(row["success"]) != int(ref_row["success"])
                or abs(row["spl"] - ref_row["spl"]) > 0.0
                or abs(row["ndtw"] - ref_row["ndtw"]) > 0.0
            ):
                metric_mismatch += 1
            max_spl = max(max_spl, abs(row["spl"] - ref_row["spl"]))
            max_ndtw = max(max_ndtw, abs(row["ndtw"] - ref_row["ndtw"]))
        item = {
            "tasks": len(rows_by_strategy[strategy]),
            "action_mismatches": action_mismatch,
            "navigation_metric_mismatches": metric_mismatch,
            "max_spl_abs_diff": float(max_spl),
            "max_ndtw_abs_diff": float(max_ndtw),
            "e2e_ms_mean": float(
                sum(row["mean_e2e_ms"]
                    for row in rows_by_strategy[strategy])
                / len(rows_by_strategy[strategy])
            ),
            "passed": bool(
                action_mismatch == 0 and metric_mismatch == 0
                and max_spl == 0.0 and max_ndtw == 0.0
            ),
        }
        passed = passed and item["passed"]
        per_strategy[strategy] = item

    deployment = load_deployment()
    payload = {
        "created_utc": utc_now(),
        "stage": "stage2n",
        "mec_model": MEC_MODEL,
        "mec_checkpoint_sha256": MEC_CHECKPOINT_SHA256,
        "deployment_sha256": deployment["deployment_sha256"],
        "deployment_nodes": len(deployment["nodes"]),
        "all_nodes_one_model": len({
            node["model_type"] for node in deployment["nodes"]
        }) == 1,
        "same_observation_repeated": {
            "identical": bool(backend_equal),
            "action": int(output_a["action"]),
            "arrival_probability": float(output_a["arrival_probability"]),
            "move_action": int(output_a["move_action"]),
        },
        "per_strategy": per_strategy,
        "task_uids": [row["episode_uid"] for row in reference],
        "passed": bool(passed),
        "interpretation": (
            "All nodes use the same frozen Transformer-M; node identity is "
            "not a MEC input.  Therefore action sequence, SR, SPL and nDTW "
            "must be identical across association strategies.  Only E2E "
            "latency may differ."
        ),
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(EVIDENCE_DIR / "consistency_probe.json", payload)
    print(
        f"[consistency] backend_identical={backend_equal} "
        f"strategy_identical={all(v['passed'] for v in per_strategy.values())} "
        f"passed={payload['passed']}",
        flush=True,
    )
    print("STAGE2N_CONSISTENCY_COMPLETE", flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
