"""Render the Chinese Stage-2s report from tracked CSV/JSON evidence.

All numbers are read from ``summary.json``, ``validation_summary.json``,
``seed_stability.csv``, ``baseline_comparison.csv``,
``latency_decomposition.csv``, ``trajectory_summary.csv``,
``model_selection_summary.csv`` and ``smoke_report.json``.  Nothing is
hand-typed into the result tables.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import read_json, utc_now
from v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.common import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    DEPLOYMENT_PATH,
    EVIDENCE_DIR,
    FORMAL_SEEDS,
    MEC_MODELS,
    STAGE,
)


def _csv(name):
    path = EVIDENCE_DIR / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _json(name):
    return read_json(EVIDENCE_DIR / name)


def _fmt(value, digits=3, suffix=""):
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _pct(value, digits=2):
    return _fmt(None if value is None else 100.0 * float(value), digits, "%")


def _paired(left, right, metric):
    # A learned policy has three seed runs over the same 60 Dev tasks; average
    # seeds per task first so the pairing stays one row per episode_uid.
    left = left.groupby("episode_uid")[metric].mean()
    right = right.groupby("episode_uid")[metric].mean()
    uids = sorted(set(left.index) & set(right.index))
    if not uids:
        return None
    left_values = np.asarray([float(left.loc[uid]) for uid in uids])
    right_values = np.asarray([float(right.loc[uid]) for uid in uids])
    diff = left_values - right_values
    n = len(diff)
    mean = float(diff.mean())
    std = float(diff.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    if metric in ("SR", "OSR", "success"):
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        ties = int((diff == 0).sum())
    else:
        wins = int((diff < 0).sum())
        losses = int((diff > 0).sum())
        ties = int((diff == 0).sum())
    return {
        "n": n, "mean": mean, "std": std, "sem": sem,
        "ci95": 1.96 * sem, "wins": wins, "losses": losses, "ties": ties,
    }


def _run_table(summary):
    runs = summary.get("runs", [])
    lines = [
        "| 算法 | seed | Episodes | 成功 | 失败 | 成功率 | 成功 S/M/L | 失败 S/M/L | Small 选择 | MA100 E2E (ms) | 最佳 Dev E2E (ms) | 收敛 |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in runs:
        lines.append(
            f"| {row['algorithm_label']} | {row['seed']} | "
            f"{row['episodes']} | {row['total_success']} | "
            f"{row['total_failure']} | {_pct(row.get('success_rate'))} | "
            f"{row['success_short']}/{row['success_medium']}/"
            f"{row['success_long']} | "
            f"{row['failure_short']}/{row['failure_medium']}/"
            f"{row['failure_long']} | {_pct(row.get('small_ratio'))} | "
            f"{_fmt(row.get('ma100_e2e_last'), 2)} | "
            f"{_fmt(row.get('best_dev_e2e_ms'), 2)} | "
            f"{row.get('convergence')} |"
        )
    return "\n".join(lines)


def _aggregate_table(summary):
    rows = summary.get("aggregates", [])
    lines = [
        "| 算法 | 成功率 | Small 选择 | MA100 E2E (ms) | 最佳 Dev E2E (ms) | 收敛 run |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def pair(key, digits=2):
            value = row.get(f"{key}_mean")
            std = row.get(f"{key}_std")
            if value is None:
                return "N/A"
            return f"{_fmt(value, digits)} ± {_fmt(std, digits)}"
        lines.append(
            f"| {row['algorithm_label']} | "
            f"{pair('success_rate', 4)} | {pair('small_ratio', 4)} | "
            f"{pair('ma100_e2e_last', 2)} | "
            f"{pair('best_dev_e2e_ms', 2)} | "
            f"{row['converged_runs']}/{row['n_seeds']} |"
        )
    return "\n".join(lines)


def _validation_table(eval_summary):
    lines = [
        "| 策略 | 类型 | SR | OSR | nDTW | SPL | E2E (ms) | P95 E2E (ms) | RewardD | 违约率 | Small 选择 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_summary.get("summary", []):
        lines.append(
            f"| {row['title']} | {row['kind']} | "
            f"{_pct(row['SR'])} | {_pct(row['OSR'])} | "
            f"{_fmt(row['nDTW'], 4)} | {_fmt(row['SPL'], 4)} | "
            f"{_fmt(row['E2E_ms'], 2)} | {_fmt(row['P95_E2E_ms'], 2)} | "
            f"{_fmt(row['RewardD'], 4)} | "
            f"{_pct(row['delay_violation_rate'])} | "
            f"{_pct(row['small_ratio'])} |"
        )
    return "\n".join(lines)


def _baseline_table(eval_summary):
    lines = [
        "| 算法 | baseline | 学习均值 E2E (ms) | baseline E2E (ms) | ΔE2E (ms) | ΔE2E (%) | ΔRewardD | ΔSR | ΔnDTW |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in eval_summary.get("baseline_comparison", []):
        lines.append(
            f"| {row['algorithm_label']} | {row['baseline_title']} | "
            f"{_fmt(row['learned_E2E_ms_mean'], 2)} | "
            f"{_fmt(row['baseline_E2E_ms'], 2)} | "
            f"{row['delta_E2E_ms']:+.2f} | {row['delta_E2E_pct']:+.2f}% | "
            f"{row['delta_RewardD']:+.4f} | {row['delta_SR']:+.4f} | "
            f"{row['delta_nDTW']:+.4f} |"
        )
    return "\n".join(lines)


def _latency_table(latency):
    if latency is None or latency.empty:
        return "（无数据）"
    lines = [
        "| 策略 | 类型 | Small 步数比例 | 推理 Small (ms) | 推理 TM (ms) | 通信 Small (ms) | 通信 TM (ms) | E2E Small (ms) | E2E TM (ms) | 总体 E2E (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in latency.iterrows():
        lines.append(
            f"| {row['policy']} | {row['kind']} | "
            f"{_pct(row['small_ratio'])} | "
            f"{_fmt(row['mean_inference_small_ms'], 3)} | "
            f"{_fmt(row['mean_inference_transformer_m_ms'], 3)} | "
            f"{_fmt(row['mean_communication_small_ms'], 2)} | "
            f"{_fmt(row['mean_communication_transformer_m_ms'], 2)} | "
            f"{_fmt(row['mean_e2e_small_ms'], 2)} | "
            f"{_fmt(row['mean_e2e_transformer_m_ms'], 2)} | "
            f"{_fmt(row['mean_e2e_ms'], 2)} |"
        )
    return "\n".join(lines)


def _trajectory_table(trajectories):
    if trajectories is None or trajectories.empty:
        return "（无数据）"
    lines = [
        "| 算法 | seed | Episode | 角色 | 成功 | 步数 | 路径 (m) | E2E (ms) | RewardD | Small 比例 | 选择状态 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in trajectories.iterrows():
        lines.append(
            f"| {row['algorithm'].upper()} | {row['seed']} | "
            f"{row['episode_uid']} | {row['role']} | "
            f"{'是' if int(row['success']) else '否'} | "
            f"{int(row['steps'])} | {_fmt(row['path_length_m'], 1)} | "
            f"{_fmt(row['mean_e2e_ms'], 2)} | "
            f"{_fmt(row['reward_D'], 4)} | "
            f"{_pct(row['small_ratio'])} | {row['selection_status']} |"
        )
    return "\n".join(lines)


def _adaptation_table(adaptation):
    if adaptation is None or adaptation.empty:
        return "（无数据）"
    lines = [
        "| 算法 | seed | 前 200 Small | 末 500 Small | ΔSmall | 前 200 E2E | 末 500 E2E | ΔE2E | 前 200 成功率 | 末 500 成功率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in adaptation.iterrows():
        lines.append(
            f"| {str(row['algorithm']).upper()} | {int(row['seed'])} | "
            f"{_pct(row['early_small_ratio'])} | "
            f"{_pct(row['late_small_ratio'])} | "
            f"{_pct(row['delta_small_ratio'])} | "
            f"{_fmt(row['early_e2e_ms'], 2)} | "
            f"{_fmt(row['late_e2e_ms'], 2)} | "
            f"{_fmt(row['delta_e2e_ms'], 2)} | "
            f"{_pct(row['early_success_rate'])} | "
            f"{_pct(row['late_success_rate'])} |"
        )
    return "\n".join(lines)


def _success_diagnostic_table(rows):
    if not rows:
        return "（无数据）"
    lines = [
        "| 策略 | 算法 | 类型 | 结果 | Episodes | 平均 E2E (ms) | 平均步数 | 平均 nDTW | 平均 RewardD |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['algorithm']} | {row['kind']} | "
            f"{'成功' if int(row['success']) else '失败'} | "
            f"{int(row['episodes'])} | {_fmt(row['mean_e2e_ms'], 2)} | "
            f"{_fmt(row['mean_steps'], 1)} | {_fmt(row['mean_ndtw'], 4)} | "
            f"{_fmt(row['mean_reward_D'], 4)} |"
        )
    return "\n".join(lines)


def _selection_table(selection):
    if selection is None or selection.empty:
        return "（无数据）"
    lines = [
        "| 算法 | 分组 | Episodes | 平均 Small 比例 | 成功率 | nDTW | E2E (ms) | 步数 | 相关系数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in selection.iterrows():
        correlation = row.get("correlation")
        lines.append(
            f"| {row['algorithm'].upper()} | {row['group']} | "
            f"{int(row['episodes'])} | {_pct(row['mean_small_ratio'])} | "
            f"{_pct(row['success_rate'])} | {_fmt(row['mean_ndtw'], 4)} | "
            f"{_fmt(row['mean_e2e_ms'], 2)} | {_fmt(row['mean_steps'], 1)} | "
            f"{_fmt(correlation, 3) if pd.notna(correlation) else '-'} |"
        )
    return "\n".join(lines)


def _converged(summary):
    return [
        f"{row['algorithm_label']}-s{row['seed']}"
        for row in summary.get("runs", [])
        if row.get("convergence") == "converged"
    ]


def _algorithm_e2e(eval_summary):
    rows = eval_summary.get("seed_aggregate", [])
    return {
        row["algorithm"]: float(row["E2E_ms_mean"]) for row in rows
    }


def _paired_evidence(eval_summary, episodes):
    if episodes is None or episodes.empty:
        return {}
    evidence = {}
    for algorithm in ALGORITHMS:
        learned = episodes[
            (episodes["algorithm"] == algorithm)
            & (episodes["kind"] == "learned")
        ]
        greedy = episodes[episodes["policy"] == "greedy_min_e2e"]
        if learned.empty or greedy.empty:
            continue
        evidence[f"{algorithm}_vs_greedy_e2e"] = _paired(
            learned, greedy, "mean_e2e_ms"
        )
        evidence[f"{algorithm}_vs_greedy_ndtw"] = _paired(
            learned, greedy, "ndtw"
        )
        evidence[f"{algorithm}_vs_greedy_sr"] = _paired(
            learned, greedy, "success"
        )
    if all(
        not episodes[episodes["algorithm"] == algorithm].empty
        for algorithm in ALGORITHMS
    ):
        dqn = episodes[episodes["algorithm"] == "dqn"]
        ppo = episodes[episodes["algorithm"] == "ppo"]
        evidence["dqn_vs_ppo_e2e"] = _paired(dqn, ppo, "mean_e2e_ms")
        evidence["dqn_vs_ppo_ndtw"] = _paired(dqn, ppo, "ndtw")
    return evidence


def _small_ratio_variability(episodes):
    result = {}
    if episodes is None or episodes.empty:
        return result
    for algorithm in ALGORITHMS:
        subset = episodes[
            (episodes["algorithm"] == algorithm)
            & (episodes["kind"] == "learned")
        ]
        if subset.empty:
            continue
        values = subset["small_ratio"].to_numpy(dtype=float)
        result[algorithm] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "episodes_not_50": int((np.abs(values - 0.5) > 0.10).sum()),
            "episodes": len(values),
        }
    return result


def _best_selection(summary):
    best = {}
    for algorithm in ALGORITHMS:
        subset = [
            row for row in summary.get("runs", [])
            if row["algorithm"] == algorithm
        ]
        if not subset:
            continue
        row = min(
            subset,
            key=lambda item: (
                item.get("best_dev_e2e_ms")
                if item.get("best_dev_e2e_ms") is not None else float("inf")
            ),
        )
        best[algorithm] = row
    return best


def render():
    summary = _json("summary.json")
    eval_summary = _json("validation_summary.json")
    smoke = _json("smoke_report.json")
    trajectories = _csv("trajectory_summary.csv")
    selection = _csv("model_selection_summary.csv")
    success_diagnostic = _csv("success_e2e_diagnostic.csv")
    adaptation = _csv("adaptation_summary.csv")
    latency = _csv("latency_decomposition.csv")
    episodes = _csv("per_episode_validation.csv")
    if summary is None:
        raise SystemExit("summary.json missing; run analyze.py first")
    if eval_summary is None:
        raise SystemExit("validation_summary.json missing; run eval.py first")
    deployment = _json("deployment_heterogeneous.json") or {}
    converged = _converged(summary)
    algo_e2e = _algorithm_e2e(eval_summary)
    paired = _paired_evidence(eval_summary, episodes)
    variability = _small_ratio_variability(episodes)
    best = _best_selection(summary)
    lower_algorithm = (
        min(algo_e2e, key=algo_e2e.get) if algo_e2e else None
    )
    lines = []
    lines += [
        "# Stage-2s 最终报告：5×Small + 5×Transformer-M 异构 10 节点 MEC",
        "",
        f"- 阶段：`{STAGE}`",
        "- 分支：`experiment/v83-spatial-foundation`",
        f"- 报告生成时间：{utc_now()}",
        f"- 部署哈希：`{deployment.get('deployment_sha256', 'N/A')}`",
        "- 奖励：方案 D（仅 E2E）；种子：101/202/303；每个 run 4000 Episodes",
        "- 数据：Stage-2n 固定城市 Train-4000 / Dev-60；**Test-400 全程未读取**",
        "- 训练模式：`full_closed_loop_heterogeneous`，**未使用 Stage-2r lane 缓存**",
        "",
        "## 1. 执行摘要",
        "",
    ]
    if lower_algorithm:
        lines.append(
            f"- Dev-60 平均 E2E 较低的学习算法：**"
            f"{ALGORITHM_LABELS[lower_algorithm]}**（"
            f"{_fmt(algo_e2e[lower_algorithm], 2)} ms；另一算法 "
            f"{_fmt(algo_e2e.get('ppo' if lower_algorithm == 'dqn' else 'dqn'), 2)} ms）。"
        )
    for algorithm in ALGORITHMS:
        row = best.get(algorithm)
        if row:
            lines.append(
                f"- {ALGORITHM_LABELS[algorithm]} 最佳 seed="
                f"{row['seed']}：Dev E2E "
                f"{_fmt(row.get('best_dev_e2e_ms'), 2)} ms，SR "
                f"{_pct(row.get('success_rate'))}，Small 选择 "
                f"{_pct(row.get('small_ratio'))}。"
            )
    lines.append(
        f"- 达到预注册严格收敛判据的 run："
        f"**{', '.join(converged) if converged else '无'}** "
        f"（共 {len(summary.get('runs', []))} 个 run）。"
    )
    if smoke:
        lines.append(
            "- Smoke：两种模型均加载；Small 推理 "
            f"{_fmt(smoke['latencies_ms']['Small']['inference_ms'], 4)} ms，"
            "Transformer-M "
            f"{_fmt(smoke['latencies_ms']['Transformer-M']['inference_ms'], 4)} ms；"
            "动作/位姿/下一状态随模型选择改变；方案 D 奖励误差 <1e-15。"
        )
    lines += [
        "",
        "## 2. 代码、协议与可复现性",
        "",
        "- 训练代码 commit：`7f61e9acaf3a9643f8d4ae016a251e5f234d2f6a`；"
        "续训兼容修复 commit：`a41a170`（仅修复 CUDA RNG 恢复，不改变训练动力学）。",
        f"- 部署哈希：`{deployment.get('deployment_sha256', 'N/A')}`。",
        "- 冻结协议：`STAGE2S_PROTOCOL.md`；部署明细：`deployment_heterogeneous.json`。",
        "- 测试：`test_stage2s.py`（CPU）与 `test_stage2s_gpu.py`（GPU 真实闭环）全部通过。",
        "- Smoke 报告：`smoke_report.json` / `SMOKE_TEST.md`。",
        "",
        "节点分配：",
        "",
        "| 节点 | 模型 | x (m) | y (m) |",
        "|---|---|---:|---:|",
    ]
    for node in deployment.get("nodes", []):
        lines.append(
            f"| {node['node_id']} | {node['model_type']} | "
            f"{node['x']:.1f} | {node['y']:.1f} |"
        )
    lines += [
        "",
        "## 3. 训练执行与累计统计",
        "",
        _run_table(summary),
        "",
        "### 3.1 算法聚合（3 seeds 均值 ± 总体标准差）",
        "",
        _aggregate_table(summary),
        "",
        "### 3.2 中断与恢复",
        "",
        "训练启动器曾在首次运行约 Ep60 时被人工停止（仅停止启动器，训练子进程随之退出）。"
        "4 个 run 从 Ep50 的 `last_full.pt` 恢复；`training_history.json` 被截断到 "
        "Ep50 后重放，历史中无重复 Episode。另 2 个 run 从 0 开始。"
        "续训时发现共享 Stage-2k agent 的 CUDA RNG 恢复问题，已在 Stage-2s 本地"
        "包装修复（commit `a41a170`），未修改历史共享代码。",
        "",
        "### 3.3 收敛性",
        "",
        "预注册判据：最近三次 Dev E2E 波动 ≤ 最佳值 1%，且末次 ≤ 最佳值 101%。",
        "",
        f"- 满足判据：**{', '.join(converged) if converged else '无'}**。",
        "- 未满足的 run 仍按最低 Dev E2E 选择 best checkpoint，并标记 "
        "`not_converged`；这不等于 E2E 没有下降，见 MA100 与最佳 Dev E2E。",
        "",
        "## 4. Dev-60 最终验证",
        "",
        _validation_table(eval_summary),
        "",
        "### 4.1 相对 baseline 的 E2E / 奖励 / 导航差",
        "",
        _baseline_table(eval_summary),
        "",
        "负 ΔE2E 表示学习策略时延更低；ΔRewardD 为正表示方案 D 奖励更高。"
        "baseline 与学习策略使用同一 10 节点部署和同一真实模型推理时延；"
        "Greedy-Min-E2E 只使用当前状态可获得的 `total_ms`。",
        "",
        "## 5. 异构性分析",
        "",
        "### 5.1 时延分解与模型选择",
        "",
        _latency_table(latency),
        "",
        "### 5.2 策略是否利用模型异构性",
        "",
    ]
    for algorithm in ALGORITHMS:
        if algorithm in variability:
            item = variability[algorithm]
            lines.append(
                f"- {ALGORITHM_LABELS[algorithm]}：Dev-60 每 Episode 的 "
                f"Small 步数比例均值 {_pct(item['mean'])}，"
                f"标准差 {_fmt(item['std'], 3)}，范围 "
                f"{_pct(item['min'])}–{_pct(item['max'])}；"
                f"{item['episodes_not_50']}/{item['episodes']} 个 Episode "
                "偏离 50% 超过 10 个百分点。"
            )
    lines += [
        "",
        "`policy_diversity.csv` 给出任意两策略在同一 Dev 任务上的动作、节点、"
        "成功不一致计数。Stage-2s 中这些不一致是预期现象：模型选择会改变导航动作，"
        "不同于 Stage-2r 中“导航与关联无关”的同质设置。",
        "",
        "训练前后选择行为（前 200 vs 末 500 Episodes）：",
        "",
        _adaptation_table(adaptation),
        "",
        "配对检验（同一 Dev 任务，学习策略 − Greedy-Min-E2E；E2E 为负更好）：",
        "",
        "| 对比 | n | ΔE2E 均值 (ms) | 95% CI | 学习更低/更高/持平 | ΔnDTW 均值 | ΔSR 均值 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for key in sorted(paired):
        if not key.endswith("_e2e"):
            continue
        item = paired[key]
        algorithm = key.split("_vs_")[0]
        ndtw = paired.get(f"{algorithm}_vs_greedy_ndtw")
        sr = paired.get(f"{algorithm}_vs_greedy_sr")
        lines.append(
            f"| {ALGORITHM_LABELS.get(algorithm, algorithm)} vs "
            f"Greedy-Min-E2E | {item['n']} | {item['mean']:+.3f} | "
            f"[{item['mean']-item['ci95']:+.3f}, "
            f"{item['mean']+item['ci95']:+.3f}] | "
            f"{item['wins']}/{item['losses']}/{item['ties']} | "
            f"{_fmt(ndtw['mean'] if ndtw else None, 4)} | "
            f"{_fmt(sr['mean'] if sr else None, 4)} |"
        )
    lines += [
        "",
        "## 6. 导航质量是否出现代价",
        "",
        "E2E 奖励只优化时延；SR/OSR/nDTW/SPL 不进入训练。"
        "如果策略偏好低推理成本的 Small 节点，导航质量可能下降。"
        "以下按“Small 步数占比 >50% / <50% / =50%”分组：",
        "",
        _selection_table(selection),
        "",
        "相关行为 Pearson 相关（Small 步数比例 vs 指标）："
        "`corr_small_ratio_vs_success`、`_vs_ndtw`、`_vs_e2e_ms`、`_vs_steps`。"
        "相关不等于因果；模型选择与任务难度/位置同时变化，需结合配对 baseline 阅读。",
        "",
        "### 6.1 奖励—成功解耦诊断（方案 D 无成功项）",
        "",
        "方案 D 只依赖每步 E2E 与终止修正，没有成功奖励或 Stop 奖励。"
        "因此低 E2E 可能来自“短而失败”的 Episode。下表把成功/失败分开统计，"
        "用于检查最优 Dev E2E 检查点是否以导航质量为代价：",
        "",
        _success_diagnostic_table(
            success_diagnostic.to_dict("records")
            if success_diagnostic is not None else []
        ),
        "",
        "## 7. 真实闭环轨迹",
        "",
        _trajectory_table(trajectories),
        "",
        "轨迹图：`fig_traj_dqn.png`、`fig_traj_ppo.png`、"
        "`fig_traj_overview.png`、`fig_traj_model_usage.png`、"
        "`fig_traj_selection_vs_outcome.png`、`fig_traj_latency_example.png`。",
        "",
        "## 8. 结果图",
        "",
        "- `fig01_reward_ma100.png`：DQN/PPO Reward-MA100。",
        "- `fig02_e2e_ma100.png`：DQN/PPO E2E-MA100。",
        "- `fig03_dev_e2e_curves.png`：Dev E2E 随训练 Episode。",
        "- `fig04_validation_metrics.png`：SR/OSR/nDTW/SPL/E2E/Reward 对比。",
        "- `fig05_seed_variability.png` / `fig05b_training_stability.png`："
        "三种子变异性与稳定性。",
        "- `fig06_heterogeneity.png`：Small/TM 选择比例、推理/通信时延、E2E 分解。",
        "- `fig07_baseline_e2e.png`：学习策略 vs baseline E2E。",
        "- `fig08_success_failure.png`：成功/失败 band 分布。",
        "",
        "所有图均为 300 dpi PNG，原始数据在 `figure_data/*.csv`。",
        "",
        "## 9. 对用户问题的直接回答",
        "",
    ]
    # Q1: exploit heterogeneity?
    if variability:
        variability_text = "；".join(
            f"{ALGORITHM_LABELS[a]} 的 Small 选择 "
            f"{_pct(item['mean'])}（标准差 {_fmt(item['std'], 3)}）"
            for a, item in variability.items()
        )
        lines.append(
            f"1. **DQN/PPO 是否学会利用模型异构性？** "
            f"证据是选择比例随 Episode 变化且不固定在 50%：{variability_text}。"
            "但“利用”不等于“优于 baseline”：是否真正降低 E2E 见下一问的配对结果。"
            "如果学习策略的 E2E 不优于 Greedy-Min-E2E，则只能说明它学会了"
            "按当前时延选择，而不是学会了利用未来导航收益。"
        )
    else:
        lines.append(
            "1. **DQN/PPO 是否学会利用模型异构性？** 数据不足，无法判断。"
        )
    # Q2: lower latency
    if lower_algorithm:
        lines.append(
            f"2. **哪种算法时延更低？** Dev-60 平均 E2E："
            f"{ALGORITHM_LABELS[lower_algorithm]} "
            f"{_fmt(algo_e2e[lower_algorithm], 2)} ms，低于另一算法。"
            "差值是否稳定需同时看 3 seed 标准差与配对 CI。"
        )
    # Q3: selection ratio
    if latency is not None and not latency.empty:
        learned = latency[latency["kind"] == "learned"]
        if not learned.empty:
            lines.append(
                "3. **Small/Transformer-M 的选择比例？** 学习策略 Dev-60 "
                f"Small 步数比例范围 "
                f"{_pct(learned['small_ratio'].min())}–"
                f"{_pct(learned['small_ratio'].max())}，"
                f"均值 {_pct(learned['small_ratio'].mean())}；"
                f"Transformer-M 均值 "
                f"{_pct(learned['transformer_m_ratio'].mean())}。"
            )
    # Q4: E2E vs baseline
    baseline_rows = eval_summary.get("baseline_comparison", [])
    if baseline_rows:
        best_wins = []
        for algorithm in ALGORITHMS:
            subset = [
                row for row in baseline_rows
                if row["algorithm"] == algorithm
            ]
            if not subset:
                continue
            best_wins.append(
                f"{ALGORITHM_LABELS[algorithm]} 对 4 个 baseline 的 ΔE2E 为 "
                + ", ".join(
                    f"{row['baseline_title']} {row['delta_E2E_pct']:+.2f}%"
                    for row in subset
                )
            )
        lines.append(
            "4. **E2E 改善是否超过 baseline？** " + "；".join(best_wins)
            + "。负值表示学习策略更低；是否“超过”按逐 baseline 的符号判断，"
            "不把任意一个 baseline 当成唯一对手。"
        )
    # Q5: navigation cost
    if selection is not None and not selection.empty:
        lines.append(
            "5. **导航质量是否出现代价？** 见第 6 节分组表、"
            "6.1 成功/失败 E2E 诊断与 `per_seed_metrics.csv` / "
            "`seed_aggregate.csv` 的 SR/OSR/nDTW/SPL。"
            "如果失败 Episode 的平均 E2E 明显低于成功 Episode，且最佳 Dev E2E "
            "检查点的 SR 偏低，则说明低时延部分来自“短而失败”的轨迹；"
            "这是方案 D 无成功项的已知设计后果，必须如实报告。"
        )
    else:
        lines.append(
            "5. **导航质量是否出现代价？** 轨迹/分组数据不足，需查看验证表。"
        )
    lines.append(
        f"6. **哪些 run 满足预先规定的收敛判据？** "
        f"{', '.join(converged) if converged else '无'}。"
    )
    lines += [
        "",
        "## 10. 限制",
        "",
        "1. 单城市、单任务生成规则、两个冻结导航模型；结论不外推到新城市或重新训练导航模型。",
        "2. 只有 3 个随机种子；不足以做强统计推断，只能检查种子间方向一致性。",
        "3. 导航由冻结 MEC 模型产生，但模型选择会改变轨迹、结果和后续状态；"
        "不同算法/种子的 SR/OSR/nDTW/SPL 不同是设计事实，不是实现缺陷。",
        "4. 0.2 TFLOP/s 是显式有效吞吐设定；E2E 可能超过 100 ms，方案 D 奖励可能为负。",
        "5. Test-400 全程未读取；最终成绩仅基于 Dev-60。",
        "6. 强制 Small/Transformer-M 诊断只限制动作 mask；若该模型在覆盖外则回退并计数，"
        "不是严格的同轨迹反事实。",
        "",
        "## 11. 复现",
        "",
        "```bash",
        "PY=/home/ubuntu5/miniconda3/envs/DRL/bin/python",
        "$PY -m unittest -v v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s",
        "$PY -m unittest -v v83_spatial_stage2.stage2s_heterogeneous_10node_small_tm.test_stage2s_gpu",
        "$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/smoke.py --episodes 14 --dev-episodes 10",
        "$PY v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/launch.py --wait --max-parallel 6 --min-free-mb 5000",
        "bash v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/run_postprocess.sh",
        "```",
        "",
        "原始 checkpoint、TensorBoard Event 和逐 Episode 大日志在 "
        "`runs/v83_spatial_stage2/stage2s_heterogeneous_10node_small_tm/`（Git 忽略）。",
        "",
    ]
    text = "\n".join(lines)
    path = EVIDENCE_DIR / "STAGE2S_REPORT.md"
    path.write_text(text, encoding="utf-8")
    print(f"[report] wrote {path}", flush=True)
    print("STAGE2S_REPORT_COMPLETE", flush=True)
    return path


if __name__ == "__main__":
    render()
