"""Stage-2n evaluator: DQN/PPO best checkpoints and four non-privileged baselines.

Dev is the default split and is used for checkpoint selection.  Test-400 is
sealed and is refused unless ``--allow-final-test`` is given; that flag does
not tune anything, it only allows the final report to read the sealed pool.

Strategies (none receives target position, future trajectory or oracle
actions):

* Random: uniform over the currently available MECs;
* Nearest: minimum current UAV-node distance;
* Max-Uplink-Rate: maximum current uplink rate;
* Greedy-Min-E2E: minimum current analytic E2E estimate (the same observable
  latency model that produces the candidate E2E feature in the RL state);
* DQN / PPO: greedy actions from the Dev-selected best checkpoint.

Usage:
    python v83_spatial_stage2/stage2n_eval.py --split dev
    python v83_spatial_stage2/stage2n_eval.py --split test --allow-final-test
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

from v83_spatial_stage2.stage2k_common import (
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_ppo_agent import PPOAgent
from v83_spatial_stage2.stage2n_common import (
    AGENT_SEEDS,
    CHECKPOINT_SELECTION,
    EVAL_DIR,
    MEC_CHECKPOINT_SHA256,
    PROTOCOL_PATH,
    RUNS_DIR,
    STAGE,
    STATE_DIM,
    STATE_MODE,
    TASK_SEED_BASE,
    canonical_sha256,
    load_task_pool,
    prepare_runner_spec,
    summarize_episode_rows,
    verify_mec_checkpoint,
)
from v83_spatial_stage2.stage2n_lanes import LaneEvaluator, build_lanes
from v83_spatial_stage2.stage2n_runtime import (
    load_stage2n_runtime,
    make_stage2n_scene_cache,
)
from v83_spatial_stage2.stage2n_train import (
    load_bundle,
    verify_frozen_protocol,
)

BASELINES = ("random", "nearest", "max_uplink_rate", "greedy_min_e2e")
BASELINE_TITLES = {
    "random": "Random",
    "nearest": "Nearest MEC",
    "max_uplink_rate": "Max-Uplink-Rate",
    "greedy_min_e2e": "Greedy-Min-E2E",
    "dqn": "DQN（best）",
    "ppo": "PPO（best）",
}
RANDOM_RNG_SEED = 20261022


def _strip_row(row):
    return {
        key: value for key, value in row.items()
        if key not in ("actions", "step_e2e_ms", "node_choices")
    }


def _policy_title(policy):
    if policy in BASELINE_TITLES:
        return BASELINE_TITLES[policy]
    if policy.startswith("dqn_s"):
        return f"DQN seed {policy.split('_s')[1]}"
    if policy.startswith("ppo_s"):
        return f"PPO seed {policy.split('_s')[1]}"
    return policy


def _policy_summary(rows, policy):
    overall = summarize_episode_rows(rows).get("overall", {})
    by_band = summarize_episode_rows(rows, key="band")
    return {
        "policy": policy,
        "title": _policy_title(policy),
        "episodes": len(rows),
        "E2E_ms": overall["E2E_ms"],
        "E2E_std_ms": overall["E2E_std_ms"],
        "P95_E2E_ms": overall["P95_E2E_ms"],
        "LatencyReturn": overall["LatencyReturn"],
        "SR": overall["SR"],
        "OSR": overall["OSR"],
        "SPL": overall["SPL"],
        "nDTW": overall["nDTW"],
        "delay_violation_rate": overall["delay_violation_rate"],
        "mean_steps": overall["mean_steps"],
        "by_band": by_band,
    }


def _consistency_audit(rows_by_policy):
    """Navigation metrics and MEC actions must be identical across policies."""
    policies = sorted(rows_by_policy)
    reference = policies[0]
    ref = {row["episode_uid"]: row for row in rows_by_policy[reference]}
    audit = {
        "reference_policy": reference,
        "policies": policies,
        "paired_tasks": len(ref),
        "per_policy": {},
        "passed": True,
    }
    for policy in policies:
        rows = {row["episode_uid"]: row for row in rows_by_policy[policy]}
        if set(rows) != set(ref):
            audit["passed"] = False
            audit["per_policy"][policy] = {"task_set_equal": False}
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
            1 for uid in ref if int(rows[uid]["steps"]) != int(ref[uid]["steps"])
        )
        max_spl = max(
            abs(float(rows[uid]["spl"]) - float(ref[uid]["spl"]))
            for uid in ref
        )
        max_ndtw = max(
            abs(float(rows[uid]["ndtw"]) - float(ref[uid]["ndtw"]))
            for uid in ref
        )
        item = {
            "task_set_equal": True,
            "action_mismatches": int(action_mismatch),
            "success_mismatches": int(success_mismatch),
            "step_mismatches": int(step_mismatch),
            "max_spl_abs_diff": float(max_spl),
            "max_ndtw_abs_diff": float(max_ndtw),
        }
        item["identical"] = bool(
            action_mismatch == 0 and success_mismatch == 0
            and step_mismatch == 0 and max_spl == 0.0 and max_ndtw == 0.0
        )
        if not item["identical"]:
            audit["passed"] = False
        audit["per_policy"][policy] = item
    audit["interpretation"] = (
        "All 20 nodes share one frozen Transformer-M and the MEC input never "
        "contains the node identity.  Therefore every association policy must "
        "produce the same navigation action sequence, SR, SPL and nDTW on the "
        "same task; only E2E latency may differ.  A mismatch is a hidden "
        "model/implementation problem, not an association effect."
    )
    return audit


def _load_learned_policy(algorithm, seed, device):
    run_dir = RUNS_DIR / f"{algorithm}_s{seed}"
    result_path = run_dir / "training_result.json"
    result = read_json(result_path)
    if result is None:
        raise FileNotFoundError(
            f"{result_path} missing; finish the {algorithm} seed {seed} run"
        )
    checkpoint = Path(result["best_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if result.get("final_test_read") is not False:
        raise ValueError(f"{result_path} claims a Final Test read")
    if algorithm == "dqn":
        agent = DQNAgentV2(
            device, seed=int(seed), gamma=1.0,
            state_dim=STATE_DIM, state_mode=STATE_MODE,
        )
    else:
        agent = PPOAgent(
            device, seed=int(seed), state_dim=STATE_DIM,
            state_mode=STATE_MODE, gamma=1.0,
        )
    payload = agent.load_model(checkpoint)
    if payload.get("stage2n_reward_mode") != "latency_only_v1":
        raise ValueError(
            f"{checkpoint} is not a Stage-2n latency-only checkpoint"
        )
    record = {
        "algorithm": algorithm,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "training_result": str(result_path),
        "training_result_sha256": sha256_file(result_path),
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": sha256_file(checkpoint),
        "best_dev_e2e_ms": float(result["best_dev_e2e_ms"]),
        "best_dev_episode": int(result["best_dev_episode"]),
        "convergence": result["convergence"],
        "selected_by": CHECKPOINT_SELECTION,
        "final_test_read_used_for_selection": False,
    }
    return agent, record


def _markdown(payload):
    lines = [
        f"# Stage-2n {payload['split']} 评测",
        "",
        f"- 时间：{payload['created_utc']}",
        f"- 任务数：{payload['episodes']}；固定城市 `map_seed={payload['map_seed']}`",
        f"- 同质部署 SHA256：`{payload['deployment_sha256']}`",
        f"- Transformer-M SHA256：`{MEC_CHECKPOINT_SHA256}`",
        f"- Final Test 读取：`{payload['final_test_read']}`",
        "- 奖励：LatencyReturn = −0.25 × mean(E2E ms) / 100；无导航奖励。",
        "",
        "## 总体结果",
        "",
        "| 策略 | Episodes | E2E (ms) | LatencyReturn | SR | OSR | SPL | nDTW | P95 E2E (ms) | 违约率 | 平均步数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in payload["policy_order"]:
        item = payload["overall"][policy]
        lines.append(
            f"| {item['title']} | {item['episodes']} | "
            f"{item['E2E_ms']:.2f} | {item['LatencyReturn']:.5f} | "
            f"{100*item['SR']:.2f}% | {100*item['OSR']:.2f}% | "
            f"{item['SPL']:.4f} | {item['nDTW']:.4f} | "
            f"{item['P95_E2E_ms']:.2f} | "
            f"{100*item['delay_violation_rate']:.2f}% | "
            f"{item['mean_steps']:.1f} |"
        )
    lines += [
        "",
        "## 分轨迹长度",
        "",
        "| 策略 | Band | E2E (ms) | LatencyReturn | SR | SPL | nDTW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy in payload["policy_order"]:
        for band in ("short", "medium", "long"):
            item = payload["by_band"][policy].get(band)
            if item is None:
                continue
            lines.append(
                f"| {payload['overall'][policy]['title']} | {band} | "
                f"{item['E2E_ms']:.2f} | {item['LatencyReturn']:.5f} | "
                f"{100*item['SR']:.2f}% | {item['SPL']:.4f} | "
                f"{item['nDTW']:.4f} |"
            )
    audit = payload["consistency_audit"]
    lines += [
        "",
        "## 同质 MEC 一致性审计",
        "",
        f"- 参考策略：`{audit['reference_policy']}`；配对任务：{audit['paired_tasks']}",
        f"- 总体通过：`{audit['passed']}`",
        "",
        "| 策略 | 动作序列不一致 | SR 不一致 | 步数不一致 | max ΔSPL | max ΔnDTW |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in payload["policy_order"]:
        item = audit["per_policy"].get(policy, {})
        lines.append(
            f"| {payload['overall'][policy]['title']} | "
            f"{item.get('action_mismatches', '-')} | "
            f"{item.get('success_mismatches', '-')} | "
            f"{item.get('step_mismatches', '-')} | "
            f"{item.get('max_spl_abs_diff', '-')} | "
            f"{item.get('max_ndtw_abs_diff', '-')} |"
        )
    lines += [
        "",
        "## 信息条件",
        "",
        "- 所有策略只使用当前 UAV 位置、当前可用 Mask 和当前可观测候选特征。",
        "- Greedy-Min-E2E 使用与 384-D 状态中候选 E2E 特征相同的解析时延模型；",
        "  不使用目标位置、未来轨迹或 Oracle 动作。",
        "- 同质部署下 20 个节点仅位置不同，因此 Nearest / Max-Uplink-Rate /",
        "  Greedy-Min-E2E 的排序相同；SR/SPL/nDTW 应与所有其他策略逐任务完全一致。",
        "",
        "## 检查点选择",
        "",
        f"- 规则：{CHECKPOINT_SELECTION}",
        "- Test 不参与训练、超参数选择、收敛判断或最佳检查点选择。",
    ]
    return "\n".join(lines) + "\n"


def evaluate(args):
    split = args.split
    protocol = verify_frozen_protocol(verify_test_file=(split == "test"))
    bundle = load_bundle()
    if split == "test":
        if not args.allow_final_test:
            raise SystemExit(
                "Test-400 is sealed; pass --allow-final-test only for the "
                "final evaluation"
            )
        test_path = Path(protocol["tasks"]["test"]["path"])
        if sha256_file(test_path) != protocol["tasks"]["test"]["sha256"]:
            raise ValueError("sealed Test-400 file SHA-256 mismatch")
        pool = load_task_pool("test", allow_test=True)
        specs = [prepare_runner_spec(item) for item in pool["tasks"]]
        final_test_read = True
    else:
        specs = bundle["dev_specs"]
        pool = load_task_pool("dev")
        final_test_read = False
    if len(specs) not in (60, 400):
        raise ValueError(f"unexpected {split} task count {len(specs)}")

    device, encoder, backends = load_stage2n_runtime(args.device)
    scene_cache = make_stage2n_scene_cache(
        encoder, device, capacity=max(64, len(specs))
    )
    print(
        f"[eval] building {len(specs)} frozen {split} MEC lanes",
        flush=True,
    )
    lanes = build_lanes(
        specs,
        bundle["topology"],
        encoder,
        backends,
        bundle["quality_prior"],
        bundle["gab_config"],
        device,
        scene_cache,
        verbose=True,
    )
    evaluator = LaneEvaluator(lanes)

    rows_by_policy = {}
    policy_meta = {}
    for strategy in BASELINES:
        print(f"[eval] {split}: {strategy}", flush=True)
        rows = evaluator.evaluate_strategy(
            strategy,
            rng_seed=RANDOM_RNG_SEED,
        )
        rows_by_policy[strategy] = rows
        policy_meta[strategy] = {
            "type": "heuristic",
            "privileged": False,
            "rng_seed": RANDOM_RNG_SEED if strategy == "random" else None,
        }

    learned = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            policy = f"{algorithm}_s{seed}"
            try:
                agent, record = _load_learned_policy(
                    algorithm, seed, device
                )
            except FileNotFoundError as exc:
                if split == "test":
                    raise
                print(f"[eval] skipping {policy}: {exc}", flush=True)
                continue
            print(f"[eval] {split}: {policy}", flush=True)
            rows = (
                evaluator.evaluate_dqn(agent)
                if algorithm == "dqn" else evaluator.evaluate_ppo(agent)
            )
            rows_by_policy[policy] = rows
            policy_meta[policy] = {
                "type": "learned",
                "privileged": False,
                **record,
            }
            learned.append(policy)

    if split == "test":
        expected = [
            f"{algorithm}_s{seed}"
            for algorithm in args.algorithms for seed in args.seeds
        ]
        missing = [item for item in expected if item not in rows_by_policy]
        if missing:
            raise RuntimeError(f"missing learned policies for Test: {missing}")

    policy_order = list(BASELINES) + learned
    overall = {
        policy: _policy_summary(rows_by_policy[policy], policy)
        for policy in policy_order
    }
    by_band = {
        policy: overall[policy]["by_band"] for policy in policy_order
    }
    for policy in policy_order:
        overall[policy].pop("by_band", None)
    audit = _consistency_audit(rows_by_policy)
    paired = len({
        row["episode_uid"] for rows in rows_by_policy.values() for row in rows
    })
    payload = {
        "stage": STAGE,
        "split": split,
        "created_utc": utc_now(),
        "episodes": len(specs),
        "episodes_per_band": {
            band: sum(1 for item in specs if item["band"] == band)
            for band in ("short", "medium", "long")
        },
        "map_seed": int(pool["map_seed"]),
        "task_pool_sha256": pool["tasks_sha256"],
        "task_pool_file_sha256": sha256_file(
            Path("runs/v83_spatial_stage2/stage2n_latency_only/specs")
            / f"{split}_tasks.json"
        ),
        "task_seed_base": int(TASK_SEED_BASE[split]),
        "deployment_sha256": bundle["deployment"]["deployment_sha256"],
        "mec": verify_mec_checkpoint(),
        "policy_order": policy_order,
        "policy_meta": policy_meta,
        "overall": overall,
        "by_band": by_band,
        "consistency_audit": audit,
        "paired_tasks": paired,
        "reward": "LatencyReturn = -0.25 * mean_E2E_ms / 100",
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "final_test_read": final_test_read,
        "test_sealed": True,
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    out_dir = EVAL_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "summary.json", payload)
    flat_rows = []
    for policy in policy_order:
        item = dict(overall[policy])
        item["policy"] = policy
        flat_rows.append(item)
    fields = [
        "policy", "title", "episodes", "E2E_ms", "E2E_std_ms",
        "P95_E2E_ms", "LatencyReturn", "SR", "OSR", "SPL", "nDTW",
        "delay_violation_rate", "mean_steps",
    ]
    write_csv(
        out_dir / "summary.csv",
        [{key: row.get(key) for key in fields} for row in flat_rows],
        fieldnames=fields,
    )
    band_rows = []
    for policy in policy_order:
        for band, item in by_band[policy].items():
            row = dict(item)
            row["policy"] = policy
            row["band"] = band
            band_rows.append(row)
    band_fields = [
        "policy", "band", "episodes", "E2E_ms", "E2E_std_ms", "P95_E2E_ms",
        "LatencyReturn", "SR", "OSR", "SPL", "nDTW",
        "delay_violation_rate", "mean_steps",
    ]
    write_csv(
        out_dir / "by_band.csv", band_rows, fieldnames=band_fields
    )
    per_episode = []
    for policy in policy_order:
        for row in rows_by_policy[policy]:
            item = _strip_row(row)
            item["policy"] = policy
            per_episode.append(item)
    per_fields = list(per_episode[0].keys())
    write_csv(
        out_dir / "per_episode.csv", per_episode, fieldnames=per_fields
    )
    (out_dir / "summary.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    if split == "test":
        atomic_write_json(out_dir / "test_read.json", {
            "created_utc": utc_now(),
            "split": "test",
            "episodes": len(specs),
            "task_pool_sha256": pool["tasks_sha256"],
            "sealed_at_build": bool(pool["sealed"]),
            "allow_final_test_flag": True,
            "final_test_read": True,
            "checkpoint_selection": CHECKPOINT_SELECTION,
            "selected_checkpoints": {
                policy: policy_meta[policy] for policy in learned
            },
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "note": (
                "Test-400 read exactly once, after Dev checkpoint selection "
                "and convergence classification; no tuning followed."
            ),
        })
    print(
        f"[eval] {split} done: {len(policy_order)} policies, "
        f"{len(specs)} tasks, consistency={audit['passed']}",
        flush=True,
    )
    print("STAGE2N_EVAL_COMPLETE", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--algorithms", nargs="+", choices=("dqn", "ppo"),
        default=["dqn", "ppo"],
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(AGENT_SEEDS)
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
