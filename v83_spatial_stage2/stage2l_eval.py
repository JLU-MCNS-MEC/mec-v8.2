"""Stage-2l evaluation: DQN, PPO and five baselines on fixed-city tasks.

The comparison always uses one navigation task list, one topology condition
per config, the same frozen MECs and the same JointR definition.  Dev is the
default split.  Final Test is refused unless ``--allow-final-test`` is given,
and the output records that the test was read.

Usage examples:
    python v83_spatial_stage2/stage2l_eval.py --config A --split dev
    python v83_spatial_stage2/stage2l_eval.py --config A --split test \
        --allow-final-test    # only after the protocol is frozen and converged
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
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_jointr_v2_agent import DQNAgentV2
from v83_spatial_stage2.stage2k_ppo_agent import PPOAgent
from v83_spatial_stage2.stage2k_runtime import load_gab_config, load_runtime
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior
from v83_spatial_stage2.stage2l_fixed_city import (
    CONFIG_DESCRIPTIONS,
    EVAL_DIR,
    STAGE2L_ROOT,
    config_specs,
    load_config_manifest,
    load_topology,
)
from v83_spatial_stage2.stage2l_protocol import (
    CAPABILITY_PRIOR_PATH,
    DevEvaluator,
    STATE_DIM,
    STATE_MODE,
    resolve_device,
    set_threads,
)
from v83_spatial_stage2.stage2l_runtime import FixedCitySceneCache


ALGORITHMS = (
    "dqn", "ppo", "random", "min_latency", "max_quality", "heuristic",
    "oracle",
)
ALGORITHM_TITLES = {
    "dqn": "DQN（best checkpoint）",
    "ppo": "PPO（best checkpoint）",
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Heuristic / Weighted Greedy",
    "oracle": "Oracle（特权上界，不参与训练）",
}
BASELINES = ("random", "min_latency", "max_quality", "heuristic", "oracle")


def _row(result, algorithm, split, config):
    return {
        "algorithm": algorithm,
        "episode_uid": result["episode_uid"],
        "split": split,
        "config": str(config),
        "scenario": result["scenario"],
        "band": result["band"],
        "task_seed": int(result["seed"]),
        "success": int(result["success"]),
        "osr": int(result["osr"]),
        "ndtw": float(result["ndtw"]),
        "spl": float(result["spl"]),
        "steps": int(result["steps"]),
        "mean_e2e_ms": float(result["mean_e2e_ms"]),
        "p95_e2e_ms": float(result["p95_e2e_ms"]),
        "delay_violation_rate": float(result["delay_violation_rate"]),
        "joint_reward": float(result["joint_reward"]),
        "model_usage": json.dumps(result["model_usage"], sort_keys=True),
        "privileged": int(algorithm == "oracle"),
    }


def _summary(rows, key=None):
    groups = {}
    for row in rows:
        value = row[key] if key else "overall"
        groups.setdefault(value, []).append(row)
    return {
        value: {
            "episodes": len(items),
            "SR": float(np.mean([item["success"] for item in items])),
            "OSR": float(np.mean([item["osr"] for item in items])),
            "SPL": float(np.mean([item["spl"] for item in items])),
            "nDTW": float(np.mean([item["ndtw"] for item in items])),
            "E2E_ms": float(np.mean([item["mean_e2e_ms"] for item in items])),
            "P95_ms": float(np.mean([item["p95_e2e_ms"] for item in items])),
            "JointR": float(
                np.mean([item["ndtw"] for item in items])
                - 0.25 * np.mean([item["mean_e2e_ms"] for item in items]) / 100.0
            ),
            "mean_steps": float(np.mean([item["steps"] for item in items])),
        }
        for value, items in groups.items()
    }


def _markdown(payload):
    lines = [
        "# Stage-2l 固定城市评测结果",
        "",
        f"- 时间：{payload['created_utc']}",
        f"- 配置：`{payload['config']}` — {payload['config_description']}",
        f"- 划分：`{payload['split']}`，{payload['episodes']} 条导航任务",
        f"- 拓扑："
        + "；".join(
            f"`{role}` (seed {seed}, {scenario})"
            for role, seed, scenario in payload["topologies"]
        ),
        f"- Final Test 读取：`{payload['final_test_read']}`",
        "- JointR = mean(nDTW) − 0.25 × mean(E2E ms) / 100。",
        "- Oracle 使用特权 `oracle_action()`，只作为上界，不参与训练。",
        "",
        "## 总体结果",
        "",
        "| 算法/策略 | Episodes | SR | SPL | nDTW | E2E (ms) | JointR | 平均步数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        item = payload["overall"].get(algorithm)
        if item is None:
            continue
        lines.append(
            f"| {ALGORITHM_TITLES[algorithm]} | {item['episodes']} | "
            f"{100*item['SR']:.2f}% | {item['SPL']:.4f} | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
            f"{item['JointR']:.4f} | {item['mean_steps']:.1f} |"
        )
    lines += [
        "",
        "## 分 Band",
        "",
        "| 算法/策略 | Short SR | Medium SR | Long SR | Short JointR | Medium JointR | Long JointR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm in ALGORITHMS:
        bands = payload["by_band"].get(algorithm)
        if bands is None:
            continue
        cells = []
        for band in ("short", "medium", "long"):
            item = bands.get(band)
            cells.append("-" if item is None else f"{100*item['SR']:.2f}%")
        for band in ("short", "medium", "long"):
            item = bands.get(band)
            cells.append("-" if item is None else f"{item['JointR']:.4f}")
        lines.append(f"| {ALGORITHM_TITLES[algorithm]} | " + " | ".join(cells) + " |")
    if payload["by_scenario"]:
        lines += [
            "",
            "## 分部署场景",
            "",
            "| 算法/策略 | 场景 | SR | nDTW | E2E (ms) | JointR |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for algorithm in ALGORITHMS:
            scenarios = payload["by_scenario"].get(algorithm, {})
            for scenario in sorted(scenarios):
                item = scenarios[scenario]
                lines.append(
                    f"| {ALGORITHM_TITLES[algorithm]} | {scenario} | "
                    f"{100*item['SR']:.2f}% | {item['nDTW']:.4f} | "
                    f"{item['E2E_ms']:.2f} | {item['JointR']:.4f} |"
                )
    lines += [
        "",
        "## 说明",
        "",
        "- 所有算法使用同一批导航任务、同一配置拓扑条件、同一五种冻结 MEC。",
        "- DQN/PPO 使用 best Dev checkpoint 的贪心策略（无 epsilon / 无采样）。",
        "- Random 每个任务使用固定种子 `1,000,003 + episode_index`。",
        "- 本表为固定环境内基线；新拓扑/新城市泛化必须另建任务池后单独运行。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("A", "B", "C"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS),
                        choices=ALGORITHMS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--limit-tasks", type=int, default=None,
        help="Smoke-only: evaluate the first N tasks instead of all 60.",
    )
    parser.add_argument("--run-root", type=Path, default=STAGE2L_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dqn-checkpoint", type=Path, default=None,
        help="Default: <run-root>/<config>/dqn/checkpoints/best.pt",
    )
    parser.add_argument(
        "--ppo-checkpoint", type=Path, default=None,
        help="Default: <run-root>/<config>/ppo/checkpoints/best.pt",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.allow_final_test:
        raise SystemExit(
            "Final Test is sealed; pass --allow-final-test explicitly after "
            "convergence and protocol freeze."
        )
    set_threads(args.threads)
    device = resolve_device(args.device)

    manifest = load_config_manifest(args.config)
    if args.split == "dev":
        specs, _, _ = config_specs(args.config, "dev")
        roles = manifest["topology_roles"]["dev"]
    else:
        specs, _, _ = config_specs(args.config, "test")
        roles = manifest["topology_roles"]["test"]
    if args.limit_tasks is not None:
        specs = specs[: int(args.limit_tasks)]
    topologies = {role: load_topology(role) for role in roles}
    bundle = {
        "config": args.config,
        "manifest": manifest,
        "dev_specs": specs,
        "eval_specs": specs,
        "topologies": topologies,
    }

    device, encoder, backends = load_runtime(str(device))
    gab_config = load_gab_config()
    _, quality = load_capability_prior(CAPABILITY_PRIOR_PATH)
    scene_cache = FixedCitySceneCache(encoder, device, max_entries=128)
    evaluator = DevEvaluator(
        bundle, encoder, backends, quality, gab_config, device,
        scene_cache=scene_cache, split=args.split,
    )

    output_dir = args.output_dir or (
        EVAL_DIR / args.config / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    algorithm_rows = {}
    for algorithm in args.algorithms:
        if algorithm == "dqn":
            checkpoint = args.dqn_checkpoint or (
                args.run_root / args.config / "dqn" / "checkpoints" / "best.pt"
            )
            if not checkpoint.is_file():
                print(f"[eval] 跳过 DQN：缺少 {checkpoint}", flush=True)
                continue
            agent = DQNAgentV2(
                device, state_dim=STATE_DIM, state_mode=STATE_MODE,
                epsilon_decay_steps=500_000,
            )
            agent.load_model(checkpoint)
            rows = evaluator.evaluate_dqn(agent)
            algorithm_rows[algorithm] = [
                _row(item, algorithm, args.split, args.config) for item in rows
            ]
        elif algorithm == "ppo":
            checkpoint = args.ppo_checkpoint or (
                args.run_root / args.config / "ppo" / "checkpoints" / "best.pt"
            )
            if not checkpoint.is_file():
                print(f"[eval] 跳过 PPO：缺少 {checkpoint}", flush=True)
                continue
            agent = PPOAgent(
                device, state_dim=STATE_DIM, state_mode=STATE_MODE,
            )
            agent.load_model(checkpoint)
            rows = evaluator.evaluate_ppo(agent)
            algorithm_rows[algorithm] = [
                _row(item, algorithm, args.split, args.config) for item in rows
            ]
        else:
            rows = evaluator.evaluate_strategy(algorithm)
            algorithm_rows[algorithm] = [
                _row(item, algorithm, args.split, args.config) for item in rows
            ]
        all_rows.extend(algorithm_rows[algorithm])
        summary = _summary(algorithm_rows[algorithm])
        print(
            f"[eval][{args.config}/{args.split}] {algorithm}: "
            f"SR={100*summary['overall']['SR']:.2f}% "
            f"nDTW={summary['overall']['nDTW']:.4f} "
            f"E2E={summary['overall']['E2E_ms']:.1f}ms "
            f"JointR={summary['overall']['JointR']:.4f}",
            flush=True,
        )

    fields = [
        "algorithm", "episode_uid", "split", "config", "scenario", "band",
        "task_seed", "success", "osr", "ndtw", "spl", "steps",
        "mean_e2e_ms", "p95_e2e_ms", "delay_violation_rate",
        "joint_reward", "model_usage", "privileged",
    ]
    write_csv(output_dir / "per_episode.csv",
              [{key: row.get(key) for key in fields} for row in all_rows],
              fieldnames=fields)
    overall = {
        algorithm: _summary(rows)["overall"]
        for algorithm, rows in algorithm_rows.items()
    }
    by_band = {
        algorithm: _summary(rows, "band")
        for algorithm, rows in algorithm_rows.items()
    }
    by_scenario = {}
    if args.split == "test" or args.config == "C":
        by_scenario = {
            algorithm: _summary(rows, "scenario")
            for algorithm, rows in algorithm_rows.items()
        }
    payload = {
        "version": "stage2l-eval-v1",
        "created_utc": utc_now(),
        "config": args.config,
        "config_description": CONFIG_DESCRIPTIONS[args.config],
        "split": args.split,
        "episodes": len(specs),
        "algorithms": list(algorithm_rows),
        "topologies": [
            (
                role,
                int(topologies[role]["topology_seed"]),
                topologies[role]["scenario"],
            )
            for role in roles
        ],
        "overall": overall,
        "by_band": by_band,
        "by_scenario": by_scenario,
        "final_test_read": bool(args.split == "test"),
        "episode_uids": [item["episode_uid"] for item in specs],
    }
    atomic_write_json(output_dir / "summary.json", payload)
    (output_dir / "EVAL_REPORT.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(f"[eval] 结果写入 {output_dir}", flush=True)
    print("STAGE2L_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
