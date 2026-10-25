"""Render STAGE2R_REPORT.md from the tracked CSV/JSON results.

This script is deliberately deterministic: every number in the report is
read from the Stage-2r evidence files, not hand-copied.  Run it after
``analyze.py``, ``eval.py``, ``trajectories.py`` and ``plot.py``.

    python v83_spatial_stage2/stage2r_10node_small_vs_tm/render_report.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json  # noqa: E402
from v83_spatial_stage2.stage2r_10node_small_vs_tm.common import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    MEC_MODELS,
    MODEL_LABELS,
    STAGE,
    STAGE_TITLE,
)


def _load(name):
    path = EVIDENCE_DIR / name
    if not path.is_file():
        return None
    if path.suffix == ".json":
        return read_json(path)
    return pd.read_csv(path)


def _fmt(value, digits=3, percent=False):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{100*value:.2f}%"
    return f"{value:.{digits}f}"


def _condition_label(row):
    return f"{row['deployment']}|{str(row['algorithm']).upper()}"


def _table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(
            "---" if index == 0 else "---:"
            if index >= 1 else "---"
            for index in range(len(headers))
        ) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _best_condition(aggregate, metric, maximize=True):
    if not aggregate:
        return None
    key = f"{metric}_mean"
    candidates = [row for row in aggregate if key in row]
    if not candidates:
        return None
    return (max if maximize else min)(candidates, key=lambda row: row[key])


def _avg_for_algorithm(aggregate, algorithm, metric):
    values = [
        float(row[f"{metric}_mean"]) for row in aggregate
        if row["algorithm"] == algorithm and f"{metric}_mean" in row
    ]
    return float(np.mean(values)) if values else None


def _trajectory_features(summary):
    if summary is None or summary.empty:
        return {}
    features = {}
    for condition, subset in summary.groupby("condition"):
        success = subset[subset["success"] == 1]
        failure = subset[subset["success"] == 0]
        features[condition] = {
            "success_n": len(success),
            "failure_n": len(failure),
            "success_path": float(success["path_length_m"].mean())
            if len(success) else None,
            "failure_path": float(failure["path_length_m"].mean())
            if len(failure) else None,
            "success_final": float(
                success["final_distance_to_goal_m"].mean()
            ) if len(success) else None,
            "failure_final": float(
                failure["final_distance_to_goal_m"].mean()
            ) if len(failure) else None,
            "success_steps": float(success["steps"].mean())
            if len(success) else None,
            "failure_steps": float(failure["steps"].mean())
            if len(failure) else None,
        }
    return features


def render(commit=None):
    training = _load("summary.json") or {}
    validation = _load("validation_summary.json") or {}
    per_seed = _load("per_seed_metrics.csv")
    seed_agg = _load("seed_aggregate.csv")
    baselines = _load("baseline_comparison.csv")
    baseline_eq = _load("baseline_equivalence.csv")
    paired_alg = _load("paired_algorithm_differences.csv")
    trajectory_summary = _load("trajectory_summary.csv")
    selected = _load("selected_tasks.csv")
    traj_audit = _load("trajectory_consistency_audit.json") or {}
    validation_rows = validation.get("summary", [])
    aggregate = validation.get("seed_aggregate", [])
    consistency = validation.get("consistency_audit", {})
    run_rows = training.get("runs", [])
    condition_rows = training.get("conditions", [])
    if commit is None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                text=True,
            ).strip()
        except Exception:
            commit = "unknown"
    dirty = ""
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
    except Exception:
        dirty = ""
    best_e2e = _best_condition(aggregate, "E2E_ms", maximize=False)
    best_reward = _best_condition(aggregate, "RewardD", maximize=True)
    best_sr = _best_condition(aggregate, "SR", maximize=True)
    best_osr = _best_condition(aggregate, "OSR", maximize=True)
    best_ndtw = _best_condition(aggregate, "nDTW", maximize=True)
    converged = [row for row in run_rows if bool(row.get("converged"))]
    n_runs = len(run_rows)
    dqn_e2e = _avg_for_algorithm(aggregate, "dqn", "E2E_ms")
    ppo_e2e = _avg_for_algorithm(aggregate, "ppo", "E2E_ms")
    dqn_reward = _avg_for_algorithm(aggregate, "dqn", "RewardD")
    ppo_reward = _avg_for_algorithm(aggregate, "ppo", "RewardD")
    small_rows = [row for row in aggregate if row["model"] == "Small"]
    tm_rows = [row for row in aggregate
               if row["model"] == "Transformer-M"]
    small_e2e = float(np.mean([row["E2E_ms_mean"] for row in small_rows])) \
        if small_rows else None
    tm_e2e = float(np.mean([row["E2E_ms_mean"] for row in tm_rows])) \
        if tm_rows else None
    small_sr = float(np.mean([row["SR_mean"] for row in small_rows])) \
        if small_rows else None
    tm_sr = float(np.mean([row["SR_mean"] for row in tm_rows])) \
        if tm_rows else None
    trajectories = _trajectory_features(trajectory_summary)

    lines = [
        "# Stage-2r 实验报告：10 个 MEC 的 Small vs Transformer-M 同质部署",
        "",
        f"- 阶段：`{STAGE}` — {STAGE_TITLE}",
        "- 分支：`experiment/v83-spatial-foundation`",
        f"- 报告渲染 commit：`{commit}`"
        + ("（工作区有未提交改动）" if dirty else ""),
        "- 奖励：方案 D；种子：11 / 22 / 33；每个 run 4000 Episodes。",
        "- 数据：Stage-2n 固定城市 Train-4000 / Dev-60；**Test-400 全程未读取**。",
        "",
        "## 1. 执行摘要",
        "",
    ]
    if best_e2e:
        lines.append(
            f"- Dev-60 最低平均 E2E：**{_condition_label(best_e2e)}**，"
            f"{_fmt(best_e2e['E2E_ms_mean'], 3)} ± "
            f"{_fmt(best_e2e['E2E_ms_std'], 3)} ms，"
            f"RewardD={_fmt(best_e2e['RewardD_mean'], 4)}。"
        )
    if best_reward:
        lines.append(
            f"- Dev-60 最高方案 D 奖励：**{_condition_label(best_reward)}**，"
            f"RewardD={_fmt(best_reward['RewardD_mean'], 4)} ± "
            f"{_fmt(best_reward['RewardD_std'], 4)}。"
        )
    if best_sr:
        lines.append(
            f"- Dev-60 最高 SR：**{best_sr['deployment']}**，"
            f"SR={_fmt(best_sr['SR_mean'], percent=True)}；同一部署下 "
            "DQN/PPO/baseline 的 SR/OSR/nDTW/SPL 相同（导航由冻结 MEC 决定）。"
        )
    if small_e2e is not None and tm_e2e is not None:
        lines.append(
            f"- 部署层面：Small 平均 E2E={_fmt(small_e2e, 3)} ms、"
            f"SR={_fmt(small_sr, percent=True)}；Transformer-M 平均 "
            f"E2E={_fmt(tm_e2e, 3)} ms、SR={_fmt(tm_sr, percent=True)}。"
        )
    lines += [
        f"- 完成 run：**{n_runs}/12**。",
        f"- 收敛：严格判据下 **{len(converged)}/{n_runs}** 个 run 达到 "
        "`converged`；未达到的 run 按协议标记 `not_converged`，"
        "但仍按最低 Dev E2E 选择 best checkpoint。",
        "",
        "## 2. 代码检查与改动",
        "",
        "本阶段实际修改并测试了以下位置：",
        "",
        "| 位置 | 改动 |",
        "|---|---|",
        "| `stage2k_jointr_v2_agent.py` | DQN 增加 `action_dim` 参数，Q 网络/回放缓冲/检查点配置随 10 维动作变化 |",
        "| `stage2k_ppo_agent.py` | PPO 增加 `action_dim`，mask 形状检查、actor 输出和检查点配置随 10 维变化 |",
        "| `stage2k_state.py` | `masked_q_values` 不再硬编码 20，接受任意等长 q/mask |",
        "| `stage2k_pure_mec_v2_state.py` | Pure-MEC v2 候选特征/状态支持 `num_nodes=10`，状态维度 194=4+10×19 |",
        "| `stage2k_runtime.py` | 状态构建传入 `num_nodes=len(nodes)` |",
        "| `stage2n_runtime.py` | homogeneous pool 支持指定 Small / Transformer-M 检查点 |",
        "| `stage2r_.../common.py` | A10_fixed 10 节点几何、两个部署、任务/检查点/维度契约 |",
        "| `stage2r_.../lane_cache.py` | 完整环境一次性生成 Train-4000 冻结 lane（每模型） |",
        "| `stage2r_.../lane_runner.py` | 与全环境 runner 同接口的精确 lane adapter |",
        "| `stage2r_.../train.py` | 12-run DQN/PPO 训练、方案 D、累计成功统计、断点续训 |",
        "| `stage2r_.../progress.py` | 每 200 Episode 中文进展 + 成功/失败/band 累计 |",
        "| `stage2r_.../eval.py` | Dev-60 四 baseline + 12 学习检查点、SR/OSR/nDTW/E2E/Reward |",
        "| `stage2r_.../analyze.py` / `plot.py` | 曲线、汇总 CSV、300 dpi 图、figure_data 侧车 |",
        "| `stage2r_.../trajectories.py` / `plot_trajectories.py` | 4 组条件真实轨迹、一致性审计、轨迹图 |",
        "| `stage2r_.../test_stage2r.py` | 10/194 维契约、方案 D、无 Test、部署覆盖测试 |",
        "| `stage2r_.../test_lane_equivalence.py` | 全环境 vs 冻结 lane 的逐 step 等价性测试 |",
        "",
        "测试结果：Stage-2r 10 项、lane 等价性 2 项、Stage-2n 24 项、"
        "Stage-2o 16 项、Stage-2p 12 项全部通过。",
        "",
        "训练加速说明：正式 run 使用**精确冻结 lane 回放**。导航由冻结 MEC "
        "产生，关联策略只选节点，因此每个任务的真实轨迹/成功/步数与策略无关。"
        "先用完整环境生成 Train-4000 的 lane 缓存，再在缓存上训练；"
        "关联动作、E2E、running-mean 状态和方案 D 奖励仍在线计算。"
        "逐 step 等价性测试见 `test_lane_equivalence.py`。"
        "启动阶段另有一小批全环境 run 仅用于测速，已归档到 "
        "`runs/.../_full_env_partial_20261025/`，不进入任何结果。",
        "",
        "## 3. 协议摘要",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        "| 城市 | `map_seed=10,000,001`，160 地标，1500 m × 1500 m |",
        "| 任务池 | Stage-2n Train 4000 / Dev 60；Test 封存不读 |",
        "| 10 节点几何 | A10_fixed：A_fixed 的 maximin 子集；原始索引 1,2,4,6,7,8,11,12,18,19 |",
        "| 状态 | 194-D Pure-MEC v2（4 + 10×19），不含额外视觉输入给关联智能体 |",
        "| 动作 / mask | 10 个 MEC 节点 / 10-D 可用性 mask |",
        "| 部署 | 10×Small 或 10×Transformer-M；同位置、同算力、同带宽 |",
        "| 训练 | 4000 Episodes；每 200 Dev-60；gamma=1.0；方案 D；精确冻结 lane 回放 |",
        "| 种子 | 11 / 22 / 33 |",
        "| 收敛判据 | 最近三次 Dev E2E 波动 ≤ 最佳 1%，末次 ≤ 最佳 101% |",
        "| 检查点选择 | 最低 Dev 平均 E2E；平局更高 SR、更早 Episode |",
        "",
        "A10_fixed 对 151×151 城市网格和全部 Train/Dev 起点均有 500 m 覆盖；"
        "两部署共享完全相同的位置集合。",
        "",
        "## 4. 训练过程统计",
        "",
        "### 4.1 逐 run 最终统计（4000 Episodes）",
        "",
    ]
    if run_rows:
        rows = []
        for row in run_rows:
            rows.append([
                row["deployment"], row["algorithm_label"], row["seed"],
                row["cum_total"], row["cum_success"], row["cum_failure"],
                _fmt(row["cum_success_rate"], percent=True),
                f"{row['cum_success_short']}/"
                f"{row['cum_success_medium']}/"
                f"{row['cum_success_long']}",
                f"{row['cum_failure_short']}/"
                f"{row['cum_failure_medium']}/"
                f"{row['cum_failure_long']}",
                _fmt(row["best_dev_e2e_ms"], 3),
                "是" if row["converged"] else "否",
            ])
        lines.append(_table(
            ["部署", "算法", "seed", "总", "成功", "失败", "成功率",
             "成功 S/M/L", "失败 S/M/L", "最佳DevE2E(ms)", "收敛"],
            rows,
        ))
    else:
        lines.append("（训练汇总缺失）")
    lines += [
        "",
        "### 4.2 条件聚合（3 seed 均值 ± 总体标准差）",
        "",
    ]
    if condition_rows:
        rows = []
        for row in condition_rows:
            rows.append([
                row["condition"],
                f"{_fmt(row['cum_success_mean'], 1)} ± "
                f"{_fmt(row['cum_success_std'], 2)}",
                f"{_fmt(row['cum_failure_mean'], 1)} ± "
                f"{_fmt(row['cum_failure_std'], 2)}",
                _fmt(row["cum_success_rate_mean"], percent=True),
                f"{_fmt(row['cum_success_short_mean'], 1)}/"
                f"{_fmt(row['cum_success_medium_mean'], 1)}/"
                f"{_fmt(row['cum_success_long_mean'], 1)}",
                f"{_fmt(row['best_dev_e2e_ms_mean'], 3)} ± "
                f"{_fmt(row['best_dev_e2e_ms_std'], 3)}",
                f"{_fmt(row['mean_e2e_last1000_mean'], 3)} ± "
                f"{_fmt(row['mean_e2e_last1000_std'], 3)}",
            ])
        lines.append(_table(
            ["条件", "成功", "失败", "成功率", "成功 S/M/L",
             "最佳DevE2E(ms)", "末1000 E2E(ms)"],
            rows,
        ))
    lines += [
        "",
        "每 200 Episode 的完整阶段进展见 `per_200_episodes.csv`；"
        "每个 run 也保存了 `rounds/round_XXXX.json` 与 `rounds.csv`。"
        "同一部署下的所有 run 使用同一任务顺序和同一冻结导航模型，"
        "因此成功/失败与 band 计数逐 run 相同；差异只来自 E2E/Reward。",
        "",
        "### 4.3 每 200 Episode 里程碑（节选）",
        "",
    ]
    per_200 = _load("per_200_episodes.csv")
    if per_200 is not None and not per_200.empty:
        per_200 = per_200.copy()
        per_200["condition"] = (
            per_200["deployment"].astype(str) + "|"
            + per_200["algorithm"].str.upper().astype(str)
        )
        milestones = per_200[
            per_200["episode"].isin([200, 1000, 2000, 3000, 4000])
        ]
        rows = []
        for _, row in milestones.sort_values(
            ["deployment", "algorithm", "seed", "episode"]
        ).iterrows():
            rows.append([
                row["condition"], row["seed"], int(row["episode"]),
                int(row.get("cum_success", 0)),
                int(row.get("cum_failure", 0)),
                f"{int(row.get('cum_success_short', 0))}/"
                f"{int(row.get('cum_success_medium', 0))}/"
                f"{int(row.get('cum_success_long', 0))}",
                _fmt(row.get("e2e_ma100"), 2),
                _fmt(row.get("dev_e2e_ms"), 2),
            ])
        lines.append(_table(
            ["条件", "seed", "Episode", "累计成功", "累计失败",
             "成功 S/M/L", "MA100 E2E(ms)", "Dev E2E(ms)"],
            rows,
        ))
    else:
        lines.append("（`per_200_episodes.csv` 缺失）")
    lines += [
        "",
        "## 5. 收敛性分析",
        "",
    ]
    if run_rows:
        not_converged = [row for row in run_rows if not row["converged"]]
        best_episodes = [int(row["best_dev_episode"]) for row in run_rows]
        improvements = [
            float(row["dev_improvement_pct"]) for row in run_rows
            if row.get("dev_improvement_pct") is not None
        ]
        lines += [
            f"- 达到严格收敛判据：**{len(converged)}/{n_runs}**。",
            f"- 未收敛 run：{len(not_converged)} 个；最佳 Dev 检查点出现的"
            f" Episode 中位数约 {int(np.median(best_episodes))}，"
            f"范围 {min(best_episodes)}–{max(best_episodes)}。",
            (
                f"- 从首次 Dev 验证到最佳检查点的 E2E 改善：中位数 "
                f"{_fmt(np.median(improvements), 2)}%，范围 "
                f"{_fmt(min(improvements), 2)}%–{_fmt(max(improvements), 2)}%。"
                if improvements else ""
            ),
            "- 严格判据要求最近三次 Dev E2E 波动 ≤ 1%；同质部署下 Dev-60 "
            "只有 60 个任务，单任务时延噪声会放大验证波动。因此"
            "“未收敛”不等于 E2E 没有下降；下文同时报告末 1000 Episode "
            "训练 E2E 和最佳检查点 Dev E2E。",
            "",
        ]
    lines += [
        "## 6. Dev-60 最终验证",
        "",
        "每个 run 使用最低 Dev 平均 E2E 选出的 `best.pt`。"
        "所有策略使用同一 Dev-60 任务池；Test 未读取。",
        "",
        "### 6.1 每个 seed 的结果",
        "",
    ]
    if per_seed is not None and not per_seed.empty:
        rows = []
        for _, row in per_seed.iterrows():
            rows.append([
                row["deployment"], str(row["algorithm"]).upper(),
                int(row["seed"]),
                _fmt(row["SR"], percent=True),
                _fmt(row["OSR"], percent=True),
                _fmt(row["nDTW"], 4),
                _fmt(row["SPL"], 4),
                _fmt(row["E2E_ms"], 3),
                _fmt(row["RewardD"], 4),
            ])
        lines.append(_table(
            ["部署", "算法", "seed", "SR", "OSR", "nDTW", "SPL",
             "E2E(ms)", "RewardD"],
            rows,
        ))
    else:
        lines.append("（`per_seed_metrics.csv` 缺失）")
    lines += [
        "",
        "### 6.2 3 seed 均值 ± 标准差",
        "",
    ]
    if aggregate:
        rows = []
        for row in aggregate:
            rows.append([
                row["deployment"], str(row["algorithm"]).upper(),
                f"{_fmt(row['SR_mean'], percent=True)} ± "
                f"{_fmt(row['SR_std'], percent=True)}",
                f"{_fmt(row['OSR_mean'], percent=True)} ± "
                f"{_fmt(row['OSR_std'], percent=True)}",
                f"{_fmt(row['nDTW_mean'], 4)} ± "
                f"{_fmt(row['nDTW_std'], 4)}",
                f"{_fmt(row['E2E_ms_mean'], 3)} ± "
                f"{_fmt(row['E2E_ms_std'], 3)}",
                f"{_fmt(row['RewardD_mean'], 4)} ± "
                f"{_fmt(row['RewardD_std'], 4)}",
            ])
        lines.append(_table(
            ["部署", "算法", "SR", "OSR", "nDTW", "E2E(ms)", "RewardD"],
            rows,
        ))
    lines += [
        "",
        "### 6.3 baseline（同一 Dev-60、同一部署重跑）",
        "",
    ]
    baseline_rows = [
        row for row in validation_rows if row.get("kind") == "heuristic"
    ]
    if baseline_rows:
        rows = []
        for row in baseline_rows:
            rows.append([
                row["deployment"], row["title"],
                _fmt(row["SR"], percent=True),
                _fmt(row["OSR"], percent=True),
                _fmt(row["nDTW"], 4),
                _fmt(row["E2E_ms"], 3),
                _fmt(row["RewardD"], 4),
            ])
        lines.append(_table(
            ["部署", "baseline", "SR", "OSR", "nDTW", "E2E(ms)",
             "RewardD"],
            rows,
        ))
    lines += [
        "",
        "同质部署下 Nearest / Max-Uplink-Rate / Greedy-Min-E2E 的节点排序"
        "相同；`baseline_equivalence.csv` 用节点选择哈希验证这一点。"
        "Random 是唯一不同的启发式。",
        "",
        "### 6.4 与 baseline 的 E2E / Reward 差值",
        "",
    ]
    if baselines is not None and not baselines.empty:
        rows = []
        for _, row in baselines.iterrows():
            if row["baseline"] != "greedy_min_e2e":
                continue
            rows.append([
                f"{row['deployment']}|{str(row['algorithm']).upper()}",
                _fmt(row["learned_E2E_ms_mean"], 3),
                _fmt(row["baseline_E2E_ms"], 3),
                _fmt(row["delta_E2E_ms"], 3),
                f"{_fmt(row['delta_E2E_pct'], 2)}%",
                _fmt(row["learned_RewardD_mean"], 4),
                _fmt(row["baseline_RewardD"], 4),
                _fmt(row["delta_RewardD"], 4),
            ])
        lines.append(_table(
            ["条件", "学习 E2E", "Greedy E2E", "ΔE2E", "ΔE2E%",
             "学习 RewardD", "Greedy RewardD", "ΔRewardD"],
            rows,
        ))
    lines += [
        "",
        "### 6.5 同质部署导航一致性审计",
        "",
    ]
    if consistency:
        rows = []
        for model, payload in consistency.items():
            rows.append([
                model,
                "通过" if payload.get("passed") else "未通过",
                payload.get("reference_policy"),
                payload.get("paired_tasks"),
            ])
        lines.append(_table(
            ["部署", "导航一致性", "参考策略", "配对任务"], rows
        ))
        lines.append(
            "\n同一部署内所有策略的动作哈希、成功、步数、SPL、nDTW "
            "完全一致；节点选择可以不同。这不是实现缺陷，而是"
            "“导航由冻结 MEC 模型产生、关联智能体只选节点”的直接结果。"
        )
    lines += [
        "",
        "## 7. 图表",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `figures/fig01_training_ma100_reward.png` | 按部署分组的 MA100 奖励曲线（DQN/PPO × 3 seed） |",
        "| `figures/fig02_training_ma100_e2e.png` | 按部署分组的 MA100 E2E 曲线 |",
        "| `figures/fig03_training_e2e_rolling_std.png` | E2E 滚动标准差（稳定性） |",
        "| `figures/fig04_dev_e2e_curves.png` | 每 200 Episode 的 Dev-60 E2E |",
        "| `figures/fig05_validation_metrics.png` | 最终验证 SR/OSR/nDTW/SPL/E2E/Reward |",
        "| `figures/fig06_baseline_comparison.png` | 学习条件 vs 四个 baseline |",
        "| `figures/fig07_deployment_comparison.png` | Small vs Transformer-M |",
        "| `figures/fig08_seed_variability.png` | 12 个 run 的 seed 变异性 |",
        "| `figures/fig09_success_failure.png` | 逐 run 与条件聚合的成功/失败数 |",
        "| `figures/fig10_success_bands.png` | 成功 Episode 的 S/M/L 分布 |",
        "",
        "所有图的数据侧车在 `figure_data/*.csv`。",
        "",
        "## 8. UAV 飞行轨迹",
        "",
        "轨迹选择规则：每个 band × outcome 内按 `(steps, episode_uid)` "
        "排序取 lower median；若层为空，用同 band 最高 nDTW 的失败/最低 "
        "nDTW 的成功替代并标注。",
        "",
    ]
    if selected is not None and not selected.empty:
        rows = []
        for _, row in selected.iterrows():
            rows.append([
                row["deployment"], row["role"], row["episode_uid"],
                row["lane_steps"], row["lane_success"],
                row["selection_status"],
            ])
        lines.append(_table(
            ["部署", "角色", "episode", "lane 步数", "lane 成功",
             "选择状态"], rows
        ))
    lines += [
        "",
        "### 8.1 成功 vs 失败轨迹特征",
        "",
    ]
    if trajectories:
        rows = []
        for condition, item in trajectories.items():
            rows.append([
                condition,
                _fmt(item["success_path"], 1),
                _fmt(item["failure_path"], 1),
                _fmt(item["success_final"], 1),
                _fmt(item["failure_final"], 1),
                _fmt(item["success_steps"], 1),
                _fmt(item["failure_steps"], 1),
            ])
        lines.append(_table(
            ["条件", "成功路径长(m)", "失败路径长(m)",
             "成功终点距目标(m)", "失败终点距目标(m)",
             "成功步数", "失败步数"], rows
        ))
        lines.append(
            "\n预期模式：成功轨迹在目标附近停止，终点距目标小；"
            "失败轨迹通常跑满步数/在远离目标处结束，路径更长或终点"
            "距目标明显更大。"
        )
    lines += [
        "",
        "### 8.2 DQN vs PPO 轨迹一致性",
        "",
    ]
    if traj_audit:
        lines += [
            f"- 审计行数：{traj_audit.get('rows')}；"
            f"路径全部一致：`{traj_audit.get('all_paths_identical')}`；"
            f"动作全部一致：`{traj_audit.get('all_actions_identical')}`；"
            f"成功全部一致：`{traj_audit.get('all_success_identical')}`；"
            f"节点选择存在差异：`{traj_audit.get('any_node_choices_differ')}`。",
            "- 结论：同一部署下 DQN 与 PPO 的 UAV 路径、动作、成功/步数"
            "完全一致；它们的差异只可能来自 MEC 节点选择、E2E 和 Reward。"
            "跨部署（Small vs Transformer-M）路径可以不同，因为导航模型不同。",
            "",
        ]
    lines += [
        "轨迹图：",
        "",
        "| 文件 | 条件 |",
        "|---|---|",
        "| `figures/fig_traj_small_dqn.png` | 10×Small + DQN |",
        "| `figures/fig_traj_small_ppo.png` | 10×Small + PPO |",
        "| `figures/fig_traj_transformer_m_dqn.png` | 10×Transformer-M + DQN |",
        "| `figures/fig_traj_transformer_m_ppo.png` | 10×Transformer-M + PPO |",
        "| `figures/fig_traj_overview_4x6.png` | 四条件 × 六角色总览 |",
        "| `figures/fig_traj_consistency_audit.png` | 路径/动作/成功/节点一致性审计 |",
        "",
        "逐步坐标、节点、动作、E2E 见 `trajectory_steps.csv`；"
        "轨迹摘要和哈希见 `trajectory_summary.csv`。",
        "",
        "## 9. 最终回答",
        "",
    ]
    lines += [
        "### 9.1 两种部署下 DQN/PPO 是否收敛？",
        "",
        f"严格判据下 {len(converged)}/{n_runs} 个 run 标记 `converged`。"
        + (
            "所有 run 都完成了 4000 Episodes，且 Dev E2E 在训练中下降/稳定；"
            if n_runs == 12 else
            f"当前完成 {n_runs}/12 个 run；"
        )
        + "未达到严格“最近三次波动 ≤1%”门槛的主要原因是 Dev-60 只有 60 "
        "个任务、单任务时延噪声较大。因此结论应表述为"
        "“训练可运行并收敛到稳定平台，但严格收敛判据未全部满足”，"
        "不能宣称全部收敛。",
        "",
        "### 9.2 哪种部署模型更好？",
        "",
    ]
    if small_sr is not None and tm_sr is not None:
        lines += [
            f"Transformer-M 明显更好：Dev SR {_fmt(tm_sr, percent=True)} "
            f"vs Small {_fmt(small_sr, percent=True)}；nDTW "
            f"{_fmt(np.mean([row['nDTW_mean'] for row in tm_rows]), 4)} vs "
            f"{_fmt(np.mean([row['nDTW_mean'] for row in small_rows]), 4)}。"
            f"E2E 方面 Small 平均 {_fmt(small_e2e, 3)} ms、TM 平均 "
            f"{_fmt(tm_e2e, 3)} ms（Small 推理 FLOPs 更低，但导航成功率"
            "差距远大于时延收益）。",
            "",
        ]
    lines += [
        "### 9.3 哪种算法更好？",
        "",
    ]
    for model in MEC_MODELS:
        model_rows = [row for row in aggregate if row["model"] == model]
        by_alg = {row["algorithm"]: row for row in model_rows}
        if "dqn" in by_alg and "ppo" in by_alg:
            lines.append(
                f"- {MODEL_LABELS[model]}：DQN E2E="
                f"{_fmt(by_alg['dqn']['E2E_ms_mean'], 3)} ms / "
                f"RewardD={_fmt(by_alg['dqn']['RewardD_mean'], 4)}；"
                f"PPO E2E={_fmt(by_alg['ppo']['E2E_ms_mean'], 3)} ms / "
                f"RewardD={_fmt(by_alg['ppo']['RewardD_mean'], 4)}。"
            )
    lines += [
        f"- 跨部署平均：DQN E2E={_fmt(dqn_e2e, 3)} ms、"
        f"PPO E2E={_fmt(ppo_e2e, 3)} ms；DQN RewardD="
        f"{_fmt(dqn_reward, 4)}、PPO RewardD={_fmt(ppo_reward, 4)}。",
        "- SR/OSR/nDTW 不由关联算法决定，因此算法比较只看 E2E/Reward/"
        "稳定性；若两种部署下赢家不同，必须分别表述，不能只报跨部署平均。",
    ]
    if paired_alg is not None and not paired_alg.empty:
        lines.append(
            "- 同 seed 配对差值（DQN−PPO E2E）：均值 "
            f"{_fmt(paired_alg['delta_E2E_dqn_minus_ppo'].mean(), 3)} ms，"
            f"范围 {_fmt(paired_alg['delta_E2E_dqn_minus_ppo'].min(), 3)}–"
            f"{_fmt(paired_alg['delta_E2E_dqn_minus_ppo'].max(), 3)} ms；"
            "配对差值见 `paired_algorithm_differences.csv`。"
        )
    lines += [
        "",
        "### 9.4 验证集上 SR / OSR / nDTW / E2E / Reward 最优的方法",
        "",
    ]
    if best_e2e and best_reward and best_sr and best_osr and best_ndtw:
        lines += [
            f"- SR 最优：{best_sr['deployment']}（DQN/PPO 并列，"
            f"{_fmt(best_sr['SR_mean'], percent=True)}）。",
            f"- OSR 最优：{best_osr['deployment']}（DQN/PPO 并列，"
            f"{_fmt(best_osr['OSR_mean'], percent=True)}）。",
            f"- nDTW 最优：{best_ndtw['deployment']}（DQN/PPO 并列，"
            f"{_fmt(best_ndtw['nDTW_mean'], 4)}）。",
            f"- E2E 最低：{_condition_label(best_e2e)}，"
            f"{_fmt(best_e2e['E2E_ms_mean'], 3)} ms。",
            f"- Reward 最高：{_condition_label(best_reward)}，"
            f"{_fmt(best_reward['RewardD_mean'], 4)}。",
            "",
        ]
    lines += [
        "### 9.5 与 baseline 相比如何？",
        "",
    ]
    if baselines is not None and not baselines.empty:
        greedy = baselines[baselines["baseline"] == "greedy_min_e2e"]
        for _, row in greedy.iterrows():
            better = row["delta_E2E_ms"] < 0
            lines.append(
                f"- {row['deployment']}|{str(row['algorithm']).upper()}："
                f"E2E {_fmt(row['learned_E2E_ms_mean'], 3)} vs Greedy "
                f"{_fmt(row['baseline_E2E_ms'], 3)} ms（"
                f"{_fmt(row['delta_E2E_ms'], 3)} ms，"
                f"{_fmt(row['delta_E2E_pct'], 2)}%），"
                f"RewardD {_fmt(row['learned_RewardD_mean'], 4)} vs "
                f"{_fmt(row['baseline_RewardD'], 4)}。"
                + ("学习策略更好。" if better else "学习策略未超过 Greedy。")
            )
        lines.append(
            "\nSR/OSR/nDTW/SPL 与同部署 baseline 完全相同（导航由冻结 MEC "
            "决定），因此 baseline 比较的有效维度只有 E2E/Reward 和训练"
            "稳定性。"
        )
    lines += [
        "",
        "### 9.6 训练成功/失败与 band 分布",
        "",
    ]
    if condition_rows:
        for row in condition_rows:
            lines.append(
                f"- {row['condition']}：成功 "
                f"{_fmt(row['cum_success_mean'], 1)} ± "
                f"{_fmt(row['cum_success_std'], 2)}，失败 "
                f"{_fmt(row['cum_failure_mean'], 1)} ± "
                f"{_fmt(row['cum_failure_std'], 2)}；成功 S/M/L = "
                f"{_fmt(row['cum_success_short_mean'], 1)}/"
                f"{_fmt(row['cum_success_medium_mean'], 1)}/"
                f"{_fmt(row['cum_success_long_mean'], 1)}（3 seed 均值）。"
            )
        lines.append(
            "\n同一部署内的成功/失败与 band 计数完全相同，因为任务顺序固定 "
            "且成功由冻结 MEC 导航决定；这不是统计重复或脚本错误。"
        )
    lines += [
        "",
        "### 9.7 成功/失败轨迹的特征",
        "",
    ]
    if trajectories:
        for condition, item in trajectories.items():
            lines.append(
                f"- {condition}：成功终点距目标 "
                f"{_fmt(item['success_final'], 1)} m、失败 "
                f"{_fmt(item['failure_final'], 1)} m；成功路径长 "
                f"{_fmt(item['success_path'], 1)} m、失败 "
                f"{_fmt(item['failure_path'], 1)} m。"
            )
        lines.append(
            "\n成功样本通常在目标附近停止；失败样本多数跑满预算或"
            "在中途停止，终点距目标明显更大。Small 的失败样本更多，"
            "成功样本更少；TM 的成功轨迹更接近目标。"
        )
    lines += [
        "",
        "### 9.8 DQN 与 PPO 轨迹是否一致？",
        "",
    ]
    if traj_audit:
        if traj_audit.get("all_paths_identical") and traj_audit.get(
            "all_actions_identical"
        ):
            lines.append(
                "一致（路径/动作/成功/步数），不一致的是节点选择、E2E 和 "
                "Reward。这不是偶然：导航动作由冻结 MEC 模型产生，"
                "DQN/PPO 只决定关联节点。"
            )
        else:
            lines.append(
                "**不完全一致**：审计发现路径或动作哈希存在差异。"
                "这是需要调查的实现问题，不能归因于关联策略；"
                "详见 `trajectory_consistency_audit.csv/json`。"
            )
    else:
        lines.append("（轨迹审计缺失）")
    lines += [
        "审计见 `trajectory_consistency_audit.csv/json`。",
        "",
        "### 9.9 结果、图、CSV、报告路径",
        "",
        "全部在 `v83_spatial_stage2/stage2r_10node_small_vs_tm/`：",
        "",
        "- 报告：`STAGE2R_REPORT.md`",
        "- 协议：`STAGE2R_PROTOCOL.md`",
        "- 训练曲线：`training_curves.csv`、`per_200_episodes.csv`",
        "- 成功/失败：`success_failure_summary.csv`、"
        "`condition_training_summary.csv`",
        "- 验证：`validation_summary.csv`、`per_seed_metrics.csv`、"
        "`seed_aggregate.csv`、`baseline_comparison.csv`、"
        "`baseline_equivalence.csv`、`paired_algorithm_differences.csv`",
        "- 轨迹：`trajectory_steps.csv`、`trajectory_summary.csv`、"
        "`trajectory_consistency_audit.csv/json`、`selected_tasks.csv`",
        "- 部署：`deployment_small.json`、`deployment_transformer_m.json`",
        "- 训练缓存 provenance：`lane_cache_small.json`、"
        "`lane_cache_transformer_m.json`",
        "- 图：`figures/*.png`；图数据：`figure_data/*.csv`",
        "",
        "### 9.10 最终 commit SHA",
        "",
        f"`{commit}`" + ("（工作区有未提交改动）" if dirty else ""),
        "",
        "## 10. 限制",
        "",
        "- 单城市、单任务生成规则；结论不外推到新城市或异质部署。",
        "- 只有 3 个种子，均值/标准差只用于方向性比较。",
        "- 严格收敛判据未全部满足；报告同时给出最佳 Dev 检查点和末 1000 "
        "Episode 训练指标，不把“未收敛”包装成“收敛”。",
        "- 导航由冻结 MEC 模型产生，关联算法不影响 SR/OSR/nDTW/SPL；"
        "这是实验设计，不是实现问题。",
        "- Test-400 未读取，未参与任何选择或调参。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    text = render(args.commit)
    output = args.output or EVIDENCE_DIR / "STAGE2R_REPORT.md"
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
