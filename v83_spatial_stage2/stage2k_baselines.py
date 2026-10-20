"""Stage-2k Baseline evaluation on the fixed Dev Validation 180.

Strategies (exactly five, plus the privileged Oracle upper bound):
  Random, Min-Latency, Max-Quality, Heuristic / Weighted Greedy, Oracle.

Every strategy uses the same episodes, the same three deployments and the same
E2E latency model.  Oracle uses privileged oracle actions only for itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    BANDS,
    BASELINE_DIR,
    QUALITY_PRIOR_PATH,
    SCENARIO_TITLES,
    SCENARIOS,
    VISUAL_PCA_PATH,
    atomic_write_json,
    load_progress,
    load_split_specs,
    load_topology,
    read_json,
    summarize_episodes,
    update_progress,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import VisualProjector


STRATEGIES = ("random", "min_latency", "max_quality", "heuristic", "oracle")
STRATEGY_TITLES = {
    "random": "Random",
    "min_latency": "Min-Latency",
    "max_quality": "Max-Quality",
    "heuristic": "Heuristic / Weighted Greedy",
    "oracle": "Oracle (privileged upper bound)",
}
RANDOM_SEED_OFFSET = 1_000_003


def flat_quality(quality):
    return {
        name: float(quality["models"][name]["mean_ndtw"])
        for name in quality["models"]
    }


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def result_row(result):
    return {
        "strategy": result["strategy"],
        "episode_uid": result["episode_uid"],
        "split": result["split"],
        "scenario": result["scenario"],
        "seed": result["seed"],
        "band": result["band"],
        "success": int(result["success"]),
        "osr": int(result["osr"]),
        "ndtw": result["ndtw"],
        "spl": result["spl"],
        "steps": result["steps"],
        "mean_e2e_ms": result["mean_e2e_ms"],
        "p95_e2e_ms": result["p95_e2e_ms"],
        "delay_violation_rate": result["delay_violation_rate"],
        "joint_reward": result["joint_reward"],
        "mean_navigation_reward": result["mean_navigation_reward"],
        "mean_latency_penalty": result["mean_latency_penalty"],
        "fallback_steps": result["fallback_steps"],
        "model_usage": json.dumps(result["model_usage"], sort_keys=True),
        "oracle_privileged": int(result["strategy"] == "oracle"),
    }


def step_rows(result):
    rows = []
    for row in result["step_records"]:
        rows.append({
            "strategy": result["strategy"],
            "episode_uid": result["episode_uid"],
            "scenario": result["scenario"],
            "band": result["band"],
            **row,
        })
    return rows


def summarize_by(rows, key):
    output = {}
    for value in sorted({row[key] for row in rows}):
        output[value] = summarize_episodes(
            [row for row in rows if row[key] == value]
        )
    return output


def write_markdown(path, summary):
    lines = [
        "# Stage-2k Baseline Results — Dev Validation 180",
        "",
        f"- 日期：{utc_now()}",
        "- 数据：Dev Validation 180（三个部署条件各 60；每个条件 18 Short / 30 Medium / 12 Long）",
        "- 质量先验：Stage-2j Dev Val 180 closed-loop mean nDTW，仅用于 "
        "Max-Quality / Heuristic / DQN 状态；未使用 Stage-2j Final Test 300。",
        "- E2E 时延：Stage-2j 八项输入实际字节数 470,200 B 每步上行 + "
        "解析式模型 FLOPs / 节点算力 + 链路传输。",
        "- Oracle 为特权信息上界，只用于比较，不参与 DQN 训练或状态构造。",
        "",
        "## 总体结果",
        "",
        "| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | 违约率 | 平均步数 | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        item = summary["overall"][strategy]
        lines.append(
            f"| {STRATEGY_TITLES[strategy]} | {100*item['SR']:.2f}% | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
            f"{item['P95_ms']:.2f} | "
            f"{100*item['delay_violation_rate']:.2f}% | "
            f"{item['mean_steps']:.1f} | {item['JointR']:.4f} |"
        )
    lines += [
        "",
        "JointR = mean(nDTW) - 0.25 × mean(E2E_ms) / 100。",
        "",
        "## 分部署条件",
        "",
    ]
    for scenario in SCENARIOS:
        lines += [
            f"### {SCENARIO_TITLES[scenario]}",
            "",
            "| 策略 | SR | nDTW | E2E (ms) | P95 (ms) | JointR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for strategy in STRATEGIES:
            item = summary["by_scenario"][scenario][strategy]
            lines.append(
                f"| {STRATEGY_TITLES[strategy]} | {100*item['SR']:.2f}% | "
                f"{item['nDTW']:.4f} | {item['E2E_ms']:.2f} | "
                f"{item['P95_ms']:.2f} | {item['JointR']:.4f} |"
            )
        lines.append("")
    lines += [
        "## 说明",
        "",
        "- Random 使用每个 episode 固定种子（episode seed + 1,000,003），可复现。",
        "- Min-Latency 选择预测 E2E 最低的可用节点；Max-Quality 选择质量先验最高者，"
        "同质量时选时延更低者。",
        "- Heuristic 使用 0.5 × quality + 0.5 × (1 - clip(E2E/100, 0, 1))；"
        "同分时选时延更低者。",
        "- Oracle 使用 `env.oracle_action()` 选择模型输出与 oracle 一致的节点；"
        "若没有精确一致，则选对 oracle 动作概率最高者，再以时延打破平局。",
        "- 所有策略共享同一 Episode 清单、同一拓扑、同一时延模型与同一 JointR。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev",))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    progress = load_progress()
    if (
        progress["baseline"]["status"] == "completed"
        and not args.force
        and args.max_episodes is None
    ):
        print("Baseline already completed; skipping. Use --force to rerun.")
        return

    quality = read_json(QUALITY_PRIOR_PATH)
    if quality is None:
        raise SystemExit("Missing quality prior")
    flat = flat_quality(quality)
    specs_payload = load_split_specs(args.split)
    topologies = {
        scenario: load_topology(scenario, args.split)
        for scenario in SCENARIOS
    }
    projector = VisualProjector.load(VISUAL_PCA_PATH)
    device, encoder, backends = load_runtime(args.device)
    gab_config = load_gab_config()

    smoke = args.max_episodes is not None
    output_dir = args.output_dir or (
        BASELINE_DIR / ("smoke" if smoke else args.split)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_episode_path = output_dir / "per_episode.csv"
    steps_path = output_dir / "steps.csv"
    if args.force:
        for path in (per_episode_path, steps_path,
                     output_dir / "summary.json", output_dir / "summary.csv"):
            path.unlink(missing_ok=True)
    existing_rows = [] if smoke else read_csv_rows(per_episode_path)
    done_pairs = {
        (row["episode_uid"], row["strategy"]) for row in existing_rows
    }
    step_fieldnames = None
    if steps_path.is_file() and steps_path.stat().st_size > 0:
        with steps_path.open(encoding="utf-8") as handle:
            step_fieldnames = next(csv.reader(handle), None)
    rows = list(existing_rows)
    step_buffer = []
    started = time.time()
    total_planned = sum(
        len(specs_payload["scenarios"][scenario]["episodes"])
        for scenario in SCENARIOS
    )
    done = len({row["episode_uid"] for row in existing_rows})
    for scenario in SCENARIOS:
        topology = topologies[scenario]
        episodes = specs_payload["scenarios"][scenario]["episodes"]
        if smoke and done >= args.max_episodes:
            break
        for spec in episodes:
            if smoke and done >= args.max_episodes:
                break
            needed = any(
                (spec["episode_uid"], strategy) not in done_pairs
                for strategy in STRATEGIES
            )
            if not needed:
                continue
            scene = build_scene(spec, encoder, device)
            new_step_rows = []
            newly_completed = False
            for strategy in STRATEGIES:
                if (spec["episode_uid"], strategy) in done_pairs:
                    continue
                rng = (
                    np.random.default_rng(
                        int(spec["seed"]) + RANDOM_SEED_OFFSET
                    )
                    if strategy == "random" else None
                )
                runner = EpisodeRunner(
                    spec=spec,
                    topology=topology,
                    encoder=encoder,
                    backends=backends,
                    projector=projector,
                    quality_prior=flat,
                    gab_config=gab_config,
                    scene=scene,
                    device=device,
                )
                result = runner.run(strategy, rng=rng)
                rows.append(result_row(result))
                new_step_rows.extend(step_rows(result))
                done_pairs.add((spec["episode_uid"], strategy))
                newly_completed = True
            if newly_completed:
                write_csv(per_episode_path, rows)
                if new_step_rows:
                    step_buffer.extend(new_step_rows)
                    append_csv(
                        steps_path, new_step_rows,
                        step_fieldnames or list(new_step_rows[0]),
                    )
                    step_fieldnames = step_fieldnames or list(
                        new_step_rows[0]
                    )
            if (spec["episode_uid"], STRATEGIES[-1]) in done_pairs:
                done = len({row["episode_uid"] for row in rows})
            if done % 10 == 0 or done == total_planned:
                elapsed = time.time() - started
                print(
                    f"[Baseline] 已完成 {done}/{total_planned} 个 Episode，"
                    f"{elapsed:.0f}s",
                    flush=True,
                )
    if not smoke:
        # Completed pairs may start from a previous partial run; rewrite the
        # canonical CSVs once at the end.
        write_csv(per_episode_path, rows)
    elapsed = time.time() - started
    if smoke:
        write_csv(per_episode_path, rows)
        write_csv(steps_path, step_buffer)
    overall = {
        strategy: summarize_episodes(
            [row for row in rows if row["strategy"] == strategy]
        )
        for strategy in STRATEGIES
    }
    by_band = {
        band: {
            strategy: summarize_episodes([
                row for row in rows
                if row["strategy"] == strategy and row["band"] == band
            ])
            for strategy in STRATEGIES
        }
        for band in BANDS
    }
    by_scenario = {
        scenario: {
            strategy: summarize_episodes([
                row for row in rows
                if row["strategy"] == strategy
                and row["scenario"] == scenario
            ])
            for strategy in STRATEGIES
        }
        for scenario in SCENARIOS
    }
    summary = {
        "version": "stage2k-baseline-v1",
        "created_utc": utc_now(),
        "split": args.split,
        "episodes": len({row["episode_uid"] for row in rows}),
        "strategies": list(STRATEGIES),
        "random_seed_offset": RANDOM_SEED_OFFSET,
        "quality_prior_path": str(QUALITY_PRIOR_PATH),
        "quality_prior": flat,
        "oracle_note": (
            "Oracle is a privileged upper bound and is not used for DQN "
            "training or state construction."
        ),
        "overall": overall,
        "by_band": by_band,
        "by_scenario": by_scenario,
        "elapsed_s": elapsed,
        "final_test_read": False,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    summary_csv = []
    for strategy in STRATEGIES:
        item = overall[strategy]
        summary_csv.append({
            "strategy": strategy,
            "episodes": item["episodes"],
            "SR": item["SR"],
            "nDTW": item["nDTW"],
            "E2E_ms": item["E2E_ms"],
            "P95_ms": item["P95_ms"],
            "delay_violation_rate": item["delay_violation_rate"],
            "mean_steps": item["mean_steps"],
            "JointR": item["JointR"],
        })
    write_csv(output_dir / "summary.csv", summary_csv)
    if not smoke:
        write_markdown(
            BASELINE_DIR / "STAGE2K_BASELINE_RESULTS.md", summary
        )
        update_progress(
            "baseline",
            status="completed",
            episodes=int(summary["episodes"]),
            summary_path=str(output_dir / "summary.json"),
            completed_utc=utc_now(),
        )
    print(json.dumps({
        "output_dir": str(output_dir),
        "smoke": smoke,
        "episodes": summary["episodes"],
        "overall": overall,
        "elapsed_s": elapsed,
    }, indent=2), flush=True)
    print("STAGE2K_BASELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
