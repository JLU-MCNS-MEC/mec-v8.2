"""Final Dev-60 evaluation for Stage-2r (no Test read).

For each deployment (10xSmall, 10xTransformer-M) this evaluates:

* the four fixed heuristics: Random, Nearest, Max-Uplink-Rate, Greedy-Min-E2E;
* the six learned best checkpoints (DQN/PPO x seeds 11/22/33).

All policies use the same Dev-60 task pool and the same frozen navigation
model.  Only the association choices (and therefore E2E/RewardD) may differ;
SR/OSR/nDTW are expected to be identical within a deployment and this is
audited explicitly.

Outputs are written to the tracked Stage-2r evidence directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2  # noqa: E402
from v83_spatial_stage2.stage2k_ppo_agent import PPOAgent  # noqa: E402
from v83_spatial_stage2.stage2n_common import (  # noqa: E402
    summarize_episode_rows,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    ACTION_DIM,
    ALGORITHMS,
    CHECKPOINT_SELECTION,
    DEV_EPISODES,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    MEC_MODELS,
    MODEL_LABELS,
    REWARD_SCHEME,
    RUNS_ROOT,
    STAGE,
    STATE_DIM,
    STATE_MODE,
    load_bundle,
    run_dir,
)
from v83_spatial_stage2.stage2r_10node_small_vs_tm.runtime import (  # noqa: E402
    LaneEvaluator,
    add_scheme_d,
    build_lanes,
    load_stage2r_runtime,
    make_scene_cache,
)

BASELINES = ("random", "nearest", "max_uplink_rate", "greedy_min_e2e")
BASELINE_TITLES = {
    "random": "Random",
    "nearest": "Nearest",
    "max_uplink_rate": "Max-Uplink-Rate",
    "greedy_min_e2e": "Greedy-Min-E2E",
}
RANDOM_RNG_SEED = 20261022
POLICY_METRICS = (
    "E2E_ms", "E2E_std_ms", "P95_E2E_ms", "RewardD", "LatencyReturn",
    "SR", "OSR", "SPL", "nDTW", "delay_violation_rate", "mean_steps",
)


def _load_learned(model, algorithm, seed, device):
    run_path = run_dir(model, algorithm, seed)
    result_path = run_path / "training_result.json"
    result = read_json(result_path)
    if result is None:
        raise FileNotFoundError(
            f"{result_path} missing; finish {model}/{algorithm}/s{seed}"
        )
    checkpoint = Path(result["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if result.get("final_test_read") is not False:
        raise ValueError(f"{result_path} claims a Final Test read")
    if result.get("model") != model or result.get("algorithm") != algorithm:
        raise ValueError(f"{result_path} condition mismatch")
    if algorithm == "dqn":
        agent = DQNAgentV2(
            device, seed=int(seed), gamma=1.0, state_dim=STATE_DIM,
            state_mode=STATE_MODE, action_dim=ACTION_DIM,
        )
    else:
        agent = PPOAgent(
            device, seed=int(seed), state_dim=STATE_DIM,
            state_mode=STATE_MODE, action_dim=ACTION_DIM, gamma=1.0,
        )
    payload = agent.load_model(checkpoint)
    if payload.get("stage") != STAGE:
        raise ValueError(f"{checkpoint} is not a Stage-2r checkpoint")
    if payload.get("model") != model:
        raise ValueError(f"{checkpoint} model mismatch")
    return agent, {
        "algorithm": algorithm,
        "seed": int(seed),
        "model": model,
        "run_dir": str(run_path),
        "training_result": str(result_path),
        "training_result_sha256": sha256_file(result_path),
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "best_dev_e2e_ms": float(result["best_dev_e2e_ms"]),
        "best_dev_episode": int(result["best_dev_episode"]),
        "convergence": result["convergence"],
        "training_counts": result["training_counts"],
        "final_test_read": False,
    }


def _strip_row(row):
    return {
        key: value for key, value in row.items()
        if key not in ("actions", "step_e2e_ms", "node_choices")
    }


def _overall(rows):
    item = dict(summarize_episode_rows(rows).get("overall", {}))
    item["RewardD"] = float(1.0 - item["E2E_ms"] / 100.0)
    return item


def _with_meta(rows, **meta):
    add_scheme_d(rows)
    for row in rows:
        row.update(meta)
        row["RewardD"] = float(1.0 - row["mean_e2e_ms"] / 100.0)
    return rows


def _summary_row(policy, condition, model, algorithm, seed, kind, title,
                 overall):
    row = {
        "policy": policy,
        "condition": condition,
        "model": model,
        "deployment": MODEL_LABELS[model],
        "algorithm": algorithm,
        "seed": seed,
        "kind": kind,
        "title": title,
        "episodes": overall["episodes"],
    }
    for key in POLICY_METRICS:
        row[key] = overall[key]
    return row


def _seed_aggregate(per_seed):
    rows = []
    for model in MEC_MODELS:
        for algorithm in ALGORITHMS:
            subset = [
                row for row in per_seed
                if row["model"] == model and row["algorithm"] == algorithm
            ]
            if not subset:
                continue
            row = {
                "model": model,
                "deployment": MODEL_LABELS[model],
                "algorithm": algorithm,
                "seeds": ",".join(str(item["seed"]) for item in subset),
                "n_seeds": len(subset),
            }
            for key in POLICY_METRICS:
                values = np.asarray(
                    [float(item[key]) for item in subset], dtype=np.float64
                )
                row[f"{key}_mean"] = float(values.mean())
                row[f"{key}_std"] = float(values.std(ddof=0))
                row[f"{key}_min"] = float(values.min())
                row[f"{key}_max"] = float(values.max())
            rows.append(row)
    return rows


def _consistency(rows_by_policy, model):
    policies = sorted(rows_by_policy)
    reference = "greedy_min_e2e" if "greedy_min_e2e" in policies else (
        policies[0]
    )
    ref = {row["episode_uid"]: row for row in rows_by_policy[reference]}
    audit = {
        "model": model,
        "reference_policy": reference,
        "policies": policies,
        "paired_tasks": len(ref),
        "per_policy": {},
        "passed": True,
        "interpretation": (
            "Navigation is produced by the frozen deployment MEC model; "
            "association only selects the serving node.  Therefore action "
            "sequences, success, steps, SPL and nDTW must be identical "
            "across all policies within a deployment.  E2E and RewardD may "
            "differ because node distance/channel conditions differ."
        ),
    }
    for policy in policies:
        rows = {row["episode_uid"]: row for row in rows_by_policy[policy]}
        item = {"task_set_equal": set(rows) == set(ref)}
        if not item["task_set_equal"]:
            audit["passed"] = False
            audit["per_policy"][policy] = item
            continue
        action_mismatch = sum(
            1 for uid in ref
            if rows[uid]["action_sha256"] != ref[uid]["action_sha256"]
        )
        success_mismatch = sum(
            1 for uid in ref
            if int(rows[uid]["success"]) != int(ref[uid]["success"])
        )
        step_mismatch = sum(
            1 for uid in ref
            if int(rows[uid]["steps"]) != int(ref[uid]["steps"])
        )
        max_spl = max(
            abs(float(rows[uid]["spl"]) - float(ref[uid]["spl"]))
            for uid in ref
        )
        max_ndtw = max(
            abs(float(rows[uid]["ndtw"]) - float(ref[uid]["ndtw"]))
            for uid in ref
        )
        node_mismatch = sum(
            1 for uid in ref
            if rows[uid].get("node_choices") != ref[uid].get("node_choices")
        )
        item.update({
            "action_mismatches": int(action_mismatch),
            "success_mismatches": int(success_mismatch),
            "step_mismatches": int(step_mismatch),
            "node_choice_mismatches": int(node_mismatch),
            "max_spl_abs_diff": float(max_spl),
            "max_ndtw_abs_diff": float(max_ndtw),
        })
        item["navigation_identical"] = bool(
            action_mismatch == 0 and success_mismatch == 0
            and step_mismatch == 0 and max_spl == 0.0 and max_ndtw == 0.0
        )
        if not item["navigation_identical"]:
            audit["passed"] = False
        audit["per_policy"][policy] = item
    return audit


def evaluate(args):
    if args.split != "dev":
        raise SystemExit("Stage-2r only evaluates Dev-60; Test is sealed")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    all_meta = {}
    lane_outcomes = []
    consistency = {}
    summaries = []
    for model in MEC_MODELS:
        print(f"[eval] {model}: loading bundle and building Dev-60 lanes",
              flush=True)
        bundle = load_bundle(model)
        device, encoder, backends = load_stage2r_runtime(
            args.device, model
        )
        scene_cache = make_scene_cache(
            encoder, device, capacity=max(64, len(bundle["dev_specs"]))
        )
        lanes = build_lanes(
            bundle["dev_specs"], bundle["topology"], encoder, backends,
            bundle["quality_prior"], bundle["gab_config"], device,
            scene_cache, verbose=True,
        )
        evaluator = LaneEvaluator(lanes)
        for lane in lanes:
            lane_outcomes.append({
                "model": model,
                "deployment": MODEL_LABELS[model],
                "episode_uid": lane.episode_uid,
                "band": lane.band,
                "success": int(lane.success),
                "osr": int(lane.osr),
                "spl": float(lane.spl),
                "ndtw": float(lane.ndtw),
                "steps": int(lane.episode_steps),
                "action_sha256": lane.action_sha256,
            })
        model_rows = {}
        for strategy in BASELINES:
            print(f"[eval] {model}: baseline {strategy}", flush=True)
            rows = evaluator.evaluate_strategy(
                strategy, rng_seed=RANDOM_RNG_SEED
            )
            _with_meta(
                rows, policy=strategy, model=model, deployment=MODEL_LABELS[model],
                condition=f"{model}|baseline", algorithm="heuristic",
                seed=-1, kind="heuristic",
            )
            model_rows[strategy] = rows
            all_meta[strategy] = {
                "type": "heuristic", "model": model,
                "privileged": False,
                "rng_seed": RANDOM_RNG_SEED if strategy == "random" else None,
            }
        for algorithm in ALGORITHMS:
            for seed in FORMAL_SEEDS:
                policy = f"{algorithm}_s{seed}"
                print(f"[eval] {model}: {policy}", flush=True)
                agent, meta = _load_learned(model, algorithm, seed, device)
                rows = (
                    evaluator.evaluate_dqn(agent)
                    if algorithm == "dqn" else evaluator.evaluate_ppo(agent)
                )
                _with_meta(
                    rows, policy=policy, model=model,
                    deployment=MODEL_LABELS[model],
                    condition=f"{model}|{algorithm}", algorithm=algorithm,
                    seed=int(seed), kind="learned",
                )
                model_rows[policy] = rows
                all_meta[f"{model}|{policy}"] = {
                    "type": "learned", **meta,
                }
        for policy, rows in model_rows.items():
            key = f"{model}|{policy}"
            all_rows[key] = rows
            overall = _overall(rows)
            summaries.append(_summary_row(
                policy, f"{model}|baseline" if policy in BASELINES
                else f"{model}|{policy.split('_s')[0]}",
                model,
                "heuristic" if policy in BASELINES else policy.split("_s")[0],
                -1 if policy in BASELINES else int(policy.split("_s")[1]),
                "heuristic" if policy in BASELINES else "learned",
                BASELINE_TITLES.get(policy, policy), overall,
            ))
        consistency[model] = _consistency(model_rows, model)

    learned_rows = [row for row in summaries if row["kind"] == "learned"]
    baseline_rows = [row for row in summaries if row["kind"] == "heuristic"]
    seed_aggregate = _seed_aggregate(learned_rows)
    baseline_comparison = []
    for model in MEC_MODELS:
        for algorithm in ALGORITHMS:
            condition = f"{model}|{algorithm}"
            subset = [row for row in learned_rows
                      if row["condition"] == condition]
            if not subset:
                continue
            mean_e2e = float(np.mean([row["E2E_ms"] for row in subset]))
            mean_reward = float(np.mean([row["RewardD"] for row in subset]))
            for baseline in ("greedy_min_e2e", "random", "nearest",
                             "max_uplink_rate"):
                base = next(
                    row for row in baseline_rows
                    if row["model"] == model and row["policy"] == baseline
                )
                baseline_comparison.append({
                    "model": model,
                    "deployment": MODEL_LABELS[model],
                    "algorithm": algorithm,
                    "condition": condition,
                    "baseline": baseline,
                    "learned_E2E_ms_mean": mean_e2e,
                    "baseline_E2E_ms": float(base["E2E_ms"]),
                    "delta_E2E_ms": mean_e2e - float(base["E2E_ms"]),
                    "delta_E2E_pct": 100.0 * (
                        mean_e2e - float(base["E2E_ms"])
                    ) / float(base["E2E_ms"]),
                    "learned_RewardD_mean": mean_reward,
                    "baseline_RewardD": float(base["RewardD"]),
                    "delta_RewardD": mean_reward - float(base["RewardD"]),
                    "SR_same_by_construction": float(base["SR"]),
                    "OSR_same_by_construction": float(base["OSR"]),
                    "nDTW_same_by_construction": float(base["nDTW"]),
                })

    write_csv(EVIDENCE_DIR / "validation_summary.csv", summaries)
    write_csv(EVIDENCE_DIR / "per_seed_metrics.csv", learned_rows)
    write_csv(EVIDENCE_DIR / "seed_aggregate.csv", seed_aggregate)
    write_csv(EVIDENCE_DIR / "baseline_comparison.csv", baseline_comparison)
    write_csv(EVIDENCE_DIR / "dev_lane_outcomes.csv", lane_outcomes)
    per_episode = []
    for key, rows in all_rows.items():
        model, policy = key.split("|", 1)
        for row in rows:
            item = _strip_row(row)
            item["model"] = model
            item["policy"] = policy
            item["condition"] = (
                f"{model}|baseline" if policy in BASELINES
                else f"{model}|{policy.split('_s')[0]}"
            )
            per_episode.append(item)
    write_csv(EVIDENCE_DIR / "per_episode_validation.csv", per_episode)
    summary = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "split": "dev",
        "episodes": DEV_EPISODES,
        "reward_scheme": REWARD_SCHEME,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "models": list(MEC_MODELS),
        "seeds": list(FORMAL_SEEDS),
        "baselines": list(BASELINES),
        "summary": summaries,
        "seed_aggregate": seed_aggregate,
        "baseline_comparison": baseline_comparison,
        "consistency_audit": consistency,
        "final_test_read": False,
        "test_sealed": True,
    }
    atomic_write_json(EVIDENCE_DIR / "validation_summary.json", summary)
    print(
        f"[eval] done: {len(summaries)} policies, "
        f"consistency={ {m: c['passed'] for m, c in consistency.items()} }",
        flush=True,
    )
    print("STAGE2R_EVAL_COMPLETE", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
