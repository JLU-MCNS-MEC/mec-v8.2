"""Final Dev-60 evaluation for Stage-2s (no Test read).

Evaluates on the same heterogeneous 10-node deployment, with the same real
Small / Transformer-M inference latency:

* Random, Nearest, Max-Uplink-Rate, Greedy-Min-E2E (closed loop);
* six learned best checkpoints (DQN/PPO x seeds 101/202/303);
* a forced-Small / forced-Transformer-M diagnostic for the best seed of each
  algorithm (clearly labelled diagnostic, not part of the pre-registered
  comparison).

Outputs are written to the tracked Stage-2s evidence directory.
"""

from __future__ import annotations

import argparse
import hashlib
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
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ACTION_DIM,
    ALGORITHMS,
    ALGORITHM_LABELS,
    CHECKPOINT_SELECTION,
    DEV_EPISODES,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    MEC_MODELS,
    REWARD_SCHEME,
    STAGE,
    STATE_DIM,
    STATE_MODE,
    artifact_hashes,
    condition_label,
    load_bundle,
    run_dir,
    source_hashes,
)
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.runtime import (  # noqa: E402
    HeterogeneousEvaluator,
    add_scheme_d,
    latency_stats,
    load_stage2s_runtime,
    make_scene_cache,
)

BASELINES = ("random", "nearest", "max_uplink_rate", "greedy_min_e2e")
BASELINE_TITLES = {
    "random": "Random",
    "nearest": "Nearest",
    "max_uplink_rate": "Max-Uplink-Rate",
    "greedy_min_e2e": "Greedy-Min-E2E",
}
RANDOM_RNG_SEED = 20261025
POLICY_METRICS = (
    "E2E_ms", "E2E_std_ms", "P95_E2E_ms", "RewardD", "LatencyReturn",
    "SR", "OSR", "SPL", "nDTW", "delay_violation_rate", "mean_steps",
    "small_ratio", "transformer_m_ratio", "mean_inference_ms",
    "mean_communication_ms",
)
CSV_EXCLUDE = {
    "actions", "step_e2e_ms", "model_choices", "steps_per_model",
    "node_choices",
}


def _strip(row):
    return {
        key: value for key, value in row.items()
        if key not in CSV_EXCLUDE
    }


def _overall(rows):
    item = dict(summarize_episode_rows(rows).get("overall", {}))
    item["RewardD"] = float(1.0 - item["E2E_ms"] / 100.0)
    total_steps = sum(int(row["steps"]) for row in rows)
    small_steps = sum(int(row.get("small_steps", 0)) for row in rows)
    transformer_steps = sum(
        int(row.get("transformer_m_steps", 0)) for row in rows
    )
    item["small_steps"] = int(small_steps)
    item["transformer_m_steps"] = int(transformer_steps)
    item["small_ratio"] = (
        float(small_steps) / total_steps if total_steps else 0.0
    )
    item["transformer_m_ratio"] = (
        float(transformer_steps) / total_steps if total_steps else 0.0
    )
    def _mean(key):
        values = [
            float(row[key]) for row in rows
            if row.get(key) is not None
            and np.isfinite(float(row[key]))
        ]
        return float(np.mean(values)) if values else None
    item["mean_inference_ms"] = _mean("mean_inference_ms")
    item["mean_communication_ms"] = _mean("mean_communication_ms")
    item["mean_extraction_ms"] = _mean("mean_extraction_ms")
    item["mean_inference_small_ms"] = _mean("mean_inference_small_ms")
    item["mean_inference_transformer_m_ms"] = _mean(
        "mean_inference_transformer_m_ms"
    )
    item["mean_communication_small_ms"] = _mean(
        "mean_communication_small_ms"
    )
    item["mean_communication_transformer_m_ms"] = _mean(
        "mean_communication_transformer_m_ms"
    )
    item["mean_e2e_small_ms"] = _mean("mean_e2e_small_ms")
    item["mean_e2e_transformer_m_ms"] = _mean(
        "mean_e2e_transformer_m_ms"
    )
    return item


def _with_meta(rows, **meta):
    add_scheme_d(rows)
    for row in rows:
        row.update(meta)
        row["RewardD"] = float(1.0 - row["mean_e2e_ms"] / 100.0)
    return rows


def _summary_row(policy, algorithm, seed, kind, title, overall):
    row = {
        "policy": policy,
        "condition": title,
        "algorithm": algorithm,
        "seed": int(seed),
        "kind": kind,
        "title": title,
        "episodes": overall["episodes"],
    }
    for key in POLICY_METRICS:
        row[key] = overall.get(key)
    return row


def _aggregate(per_seed, algorithm):
    subset = [row for row in per_seed if row["algorithm"] == algorithm]
    if not subset:
        return None
    row = {
        "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS[algorithm],
        "seeds": ",".join(str(item["seed"]) for item in subset),
        "n_seeds": len(subset),
    }
    for key in POLICY_METRICS:
        values = np.asarray(
            [float(item[key]) for item in subset
             if item.get(key) is not None],
            dtype=np.float64,
        )
        row[f"{key}_mean"] = float(values.mean())
        row[f"{key}_std"] = float(values.std(ddof=0))
        row[f"{key}_min"] = float(values.min())
        row[f"{key}_max"] = float(values.max())
    return row


def _load_learned(algorithm, seed, device):
    run_path = run_dir(algorithm, seed)
    result = read_json(run_path / "training_result.json")
    if result is None:
        raise FileNotFoundError(
            f"{run_path}/training_result.json missing"
        )
    if result.get("status") != "completed":
        raise ValueError(f"{run_path} not completed: {result.get('status')}")
    if int(result.get("episodes", 0)) != 4000:
        raise ValueError(f"{run_path} has {result.get('episodes')} episodes")
    checkpoint = Path(result["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if result.get("algorithm") != algorithm or int(
        result.get("seed", -1)
    ) != int(seed):
        raise ValueError(f"{run_path} condition mismatch")
    if result.get("training_mode") != "full_closed_loop_heterogeneous":
        raise ValueError(f"{run_path} is not the closed-loop mode")
    if result.get("lane_cache_used") is not False:
        raise ValueError(f"{run_path} used a lane cache")
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
        raise ValueError(f"{checkpoint} is not a Stage-2s checkpoint")
    if payload.get("algorithm") != algorithm or int(
        payload.get("seed", -1)
    ) != int(seed):
        raise ValueError(f"{checkpoint} condition mismatch")
    if payload.get("deployment_sha256") != result.get(
        "deployment_sha256"
    ):
        raise ValueError(f"{checkpoint} deployment mismatch")
    return agent, {
        "algorithm": algorithm,
        "seed": int(seed),
        "run_dir": str(run_path),
        "training_result": str(run_path / "training_result.json"),
        "training_result_sha256": sha256_file(
            run_path / "training_result.json"
        ),
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "best_dev_e2e_ms": float(result["best_dev_e2e_ms"]),
        "best_dev_episode": int(result["best_dev_episode"]),
        "convergence": result["convergence"],
        "training_counts": result["training_counts"],
        "training_small_ratio": result.get("training_small_ratio"),
        "final_test_read": False,
    }


def _success_e2e_diagnostic(rows_by_policy):
    """Check whether low E2E comes from short failed episodes.

    Scheme D has no success or stop term.  A policy can therefore obtain a
    high reward from short, low-E2E failures.  This table makes that
    mechanism visible instead of hiding it.
    """
    rows = []
    for policy, episodes in rows_by_policy.items():
        for success in (1, 0):
            subset = [
                row for row in episodes
                if int(row["success"]) == int(success)
            ]
            if not subset:
                continue
            rows.append({
                "policy": policy,
                "algorithm": episodes[0]["algorithm"],
                "kind": episodes[0]["kind"],
                "seed": episodes[0]["seed"],
                "success": int(success),
                "episodes": len(subset),
                "mean_e2e_ms": float(np.mean([
                    float(row["mean_e2e_ms"]) for row in subset
                ])),
                "mean_steps": float(np.mean([
                    int(row["steps"]) for row in subset
                ])),
                "mean_ndtw": float(np.mean([
                    float(row["ndtw"]) for row in subset
                ])),
                "mean_reward_D": float(np.mean([
                    float(row["reward_D"]) for row in subset
                ])),
            })
    return rows


def _policy_diversity(rows_by_policy):
    policies = sorted(rows_by_policy)
    rows = []
    for i, left in enumerate(policies):
        for right in policies[i + 1:]:
            left_rows = {row["episode_uid"]: row for row in rows_by_policy[
                left
            ]}
            right_rows = {row["episode_uid"]: row for row in rows_by_policy[
                right
            ]}
            uids = sorted(set(left_rows) & set(right_rows))
            if not uids:
                continue
            rows.append({
                "left": left,
                "right": right,
                "paired_tasks": len(uids),
                "action_mismatches": sum(
                    1 for uid in uids
                    if left_rows[uid]["action_sha256"]
                    != right_rows[uid]["action_sha256"]
                ),
                "node_mismatches": sum(
                    1 for uid in uids
                    if left_rows[uid].get("node_choices")
                    != right_rows[uid].get("node_choices")
                ),
                "success_mismatches": sum(
                    1 for uid in uids
                    if int(left_rows[uid]["success"])
                    != int(right_rows[uid]["success"])
                ),
                "mean_e2e_delta_ms": float(np.mean([
                    float(right_rows[uid]["mean_e2e_ms"])
                    - float(left_rows[uid]["mean_e2e_ms"])
                    for uid in uids
                ])),
                "note": (
                    "Navigation may differ because the selected node's model "
                    "produces the action; this is expected in Stage-2s."
                ),
            })
    return rows


def evaluate(args):
    if args.split != "dev":
        raise SystemExit("Stage-2s only evaluates Dev-60; Test is sealed")
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    device, encoder, backends, latency_model = load_stage2s_runtime(
        args.device
    )
    scene_cache = make_scene_cache(
        encoder, device, capacity=max(64, len(bundle["dev_specs"]))
    )
    evaluator = HeterogeneousEvaluator(
        bundle, encoder, backends, scene_cache, device, latency_model
    )
    rows_by_policy = {}
    meta_by_policy = {}
    summaries = []
    learned_per_seed = []
    learned_meta = {}

    for strategy in BASELINES:
        print(f"[eval] baseline {strategy}", flush=True)
        rows = evaluator.evaluate_strategy(
            strategy, rng_seed=RANDOM_RNG_SEED
        )
        _with_meta(
            rows,
            policy=strategy, condition=BASELINE_TITLES[strategy],
            algorithm="heuristic", seed=-1, kind="baseline",
        )
        rows_by_policy[strategy] = rows
        meta_by_policy[strategy] = {
            "type": "baseline", "privileged": False,
            "rng_seed": RANDOM_RNG_SEED if strategy == "random" else None,
        }
        overall = _overall(rows)
        summaries.append(_summary_row(
            strategy, "heuristic", -1, "baseline",
            BASELINE_TITLES[strategy], overall,
        ))

    for algorithm in ALGORITHMS:
        for seed in FORMAL_SEEDS:
            policy = f"{algorithm}_s{seed}"
            print(f"[eval] learned {policy}", flush=True)
            agent, meta = _load_learned(algorithm, seed, device)
            if algorithm == "dqn":
                rows = evaluator.evaluate_dqn(agent)
            else:
                rows = evaluator.evaluate_ppo(agent)
            title = f"{ALGORITHM_LABELS[algorithm]}-s{seed}"
            _with_meta(
                rows, policy=policy, condition=title,
                algorithm=algorithm, seed=int(seed), kind="learned",
            )
            rows_by_policy[policy] = rows
            meta_by_policy[policy] = {
                "type": "learned", "privileged": False, **meta,
            }
            learned_meta[policy] = meta
            overall = _overall(rows)
            summary = _summary_row(
                policy, algorithm, seed, "learned", title, overall,
            )
            summaries.append(summary)
            learned_per_seed.append(summary)

    seed_aggregate = []
    for algorithm in ALGORITHMS:
        row = _aggregate(learned_per_seed, algorithm)
        if row is not None:
            seed_aggregate.append(row)

    baseline_comparison = []
    for algorithm in ALGORITHMS:
        aggregate = next(
            (row for row in seed_aggregate if row["algorithm"] == algorithm),
            None,
        )
        if aggregate is None:
            continue
        for strategy in BASELINES:
            base_summary = next(
                (row for row in summaries if row["policy"] == strategy),
                None,
            )
            if base_summary is None:
                continue
            baseline_comparison.append({
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "baseline": strategy,
                "baseline_title": BASELINE_TITLES[strategy],
                "learned_E2E_ms_mean": aggregate["E2E_ms_mean"],
                "learned_E2E_ms_std": aggregate["E2E_ms_std"],
                "baseline_E2E_ms": base_summary["E2E_ms"],
                "delta_E2E_ms": (
                    aggregate["E2E_ms_mean"] - base_summary["E2E_ms"]
                ),
                "delta_E2E_pct": (
                    100.0
                    * (aggregate["E2E_ms_mean"] - base_summary["E2E_ms"])
                    / base_summary["E2E_ms"]
                ),
                "learned_RewardD_mean": aggregate["RewardD_mean"],
                "baseline_RewardD": base_summary["RewardD"],
                "delta_RewardD": (
                    aggregate["RewardD_mean"] - base_summary["RewardD"]
                ),
                "learned_SR_mean": aggregate["SR_mean"],
                "baseline_SR": base_summary["SR"],
                "delta_SR": aggregate["SR_mean"] - base_summary["SR"],
                "learned_nDTW_mean": aggregate["nDTW_mean"],
                "baseline_nDTW": base_summary["nDTW"],
                "delta_nDTW": (
                    aggregate["nDTW_mean"] - base_summary["nDTW"]
                ),
            })

    # Forced-model diagnostic for the best seed of each algorithm.
    forced_rows = []
    forced_summary = []
    for algorithm in ALGORITHMS:
        candidates = [
            row for row in learned_per_seed
            if row["algorithm"] == algorithm
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: row["E2E_ms"])
        seed = int(best["seed"])
        agent, _ = _load_learned(algorithm, seed, device)
        for model in MEC_MODELS:
            print(
                f"[eval] diagnostic {algorithm}-s{seed} forced {model}",
                flush=True,
            )
            rows = evaluator.evaluate_forced_model(
                agent, algorithm, model
            )
            _with_meta(
                rows, policy=f"{algorithm}_s{seed}_forced_{model}",
                condition=f"{ALGORITHM_LABELS[algorithm]}-s{seed}|"
                f"forced-{model}",
                algorithm=algorithm, seed=int(seed),
                kind="diagnostic", forced_model=model,
            )
            forced_rows.extend(rows)
            overall = _overall(rows)
            item = _summary_row(
                f"{algorithm}_s{seed}_forced_{model}", algorithm, seed,
                "diagnostic",
                f"{ALGORITHM_LABELS[algorithm]}-s{seed} 强制 {model}",
                overall,
            )
            item["forced_model"] = model
            item["forced_fallback_steps"] = int(sum(
                row.get("forced_fallback_steps", 0) for row in rows
            ))
            forced_summary.append(item)

    per_episode = []
    for policy, rows in rows_by_policy.items():
        for row in rows:
            per_episode.append(_strip(row))
    diversity = _policy_diversity(rows_by_policy)
    success_diagnostic = _success_e2e_diagnostic(rows_by_policy)
    latency_rows = []
    for policy, rows in rows_by_policy.items():
        overall = _overall(rows)
        item = {
            "policy": policy,
            "algorithm": rows[0]["algorithm"] if rows else "",
            "seed": rows[0]["seed"] if rows else -1,
            "kind": rows[0]["kind"] if rows else "",
            "steps": sum(int(row["steps"]) for row in rows),
            "small_steps": overall["small_steps"],
            "transformer_m_steps": overall["transformer_m_steps"],
            "small_ratio": overall["small_ratio"],
            "transformer_m_ratio": overall["transformer_m_ratio"],
            "mean_e2e_ms": overall["E2E_ms"],
            "mean_inference_ms": overall["mean_inference_ms"],
            "mean_communication_ms": overall["mean_communication_ms"],
            "mean_extraction_ms": overall["mean_extraction_ms"],
            "mean_inference_small_ms": overall[
                "mean_inference_small_ms"
            ],
            "mean_inference_transformer_m_ms": overall[
                "mean_inference_transformer_m_ms"
            ],
            "mean_communication_small_ms": overall[
                "mean_communication_small_ms"
            ],
            "mean_communication_transformer_m_ms": overall[
                "mean_communication_transformer_m_ms"
            ],
            "mean_e2e_small_ms": overall["mean_e2e_small_ms"],
            "mean_e2e_transformer_m_ms": overall[
                "mean_e2e_transformer_m_ms"
            ],
        }
        latency_rows.append(item)

    meta = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "split": "dev",
        "dev_episodes": DEV_EPISODES,
        "deployment_sha256": bundle["deployment"]["deployment_sha256"],
        "deployment": {
            "model_counts": bundle["deployment"]["mec"]["model_counts"],
            "nodes": bundle["deployment"]["nodes"],
        },
        "artifact_hashes": artifact_hashes(bundle),
        "source_hashes": source_hashes(),
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "reward_scheme": REWARD_SCHEME,
        "random_rng_seed": RANDOM_RNG_SEED,
        "policies": meta_by_policy,
        "learned": learned_meta,
        "final_test_read": False,
        "diagnostic_forcing": (
            "Forced Small/Transformer-M restricts the learned policy's action "
            "mask to nodes of one model type; if none is in coverage the step "
            "falls back to the unrestricted greedy action and is counted."
        ),
    }

    write_csv(
        EVIDENCE_DIR / "per_episode_validation.csv", per_episode
    )
    write_csv(
        EVIDENCE_DIR / "validation_summary.csv", summaries,
        fieldnames=[
            "policy", "condition", "algorithm", "seed", "kind", "title",
            "episodes", *POLICY_METRICS,
        ],
    )
    write_csv(
        EVIDENCE_DIR / "per_seed_metrics.csv", learned_per_seed
    )
    write_csv(
        EVIDENCE_DIR / "seed_aggregate.csv", seed_aggregate
    )
    write_csv(
        EVIDENCE_DIR / "baseline_comparison.csv", baseline_comparison
    )
    write_csv(
        EVIDENCE_DIR / "latency_decomposition.csv", latency_rows
    )
    write_csv(
        EVIDENCE_DIR / "policy_diversity.csv", diversity
    )
    write_csv(
        EVIDENCE_DIR / "success_e2e_diagnostic.csv", success_diagnostic
    )
    write_csv(
        EVIDENCE_DIR / "forced_model_diagnostic.csv", forced_summary
    )
    write_csv(
        EVIDENCE_DIR / "forced_model_episodes.csv",
        [_strip(row) for row in forced_rows],
    )
    atomic_write_json(EVIDENCE_DIR / "validation_summary.json", {
        "meta": meta,
        "summary": summaries,
        "seed_aggregate": seed_aggregate,
        "baseline_comparison": baseline_comparison,
        "latency_decomposition": latency_rows,
        "policy_diversity": diversity,
        "success_e2e_diagnostic": success_diagnostic,
        "forced_model_diagnostic": forced_summary,
    })
    print(
        f"[eval] wrote {len(summaries)} policy summaries, "
        f"{len(per_episode)} episode rows, "
        f"{len(diversity)} policy pairs, "
        f"{len(forced_summary)} forced-model rows",
        flush=True,
    )
    print("[eval] best Dev E2E per learned policy:")
    for row in learned_per_seed:
        print(
            f"  {row['policy']}: E2E={row['E2E_ms']:.2f} ms "
            f"RewardD={row['RewardD']:.4f} SR={100*row['SR']:.2f}% "
            f"Small={100*row['small_ratio']:.2f}%",
            flush=True,
        )
    print("STAGE2S_EVAL_COMPLETE", flush=True)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="dev", choices=("dev",))
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
