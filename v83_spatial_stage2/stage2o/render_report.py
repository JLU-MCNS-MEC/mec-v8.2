"""Render the data-driven tables used by STAGE2O_REPORT.md.

    python v83_spatial_stage2/stage2o/render_report.py

Reads ``evidence/summary.json`` and ``evidence/protocol_freeze.json`` and
writes ``evidence/REPORT_TABLES.md``.  It does not modify any run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2o.plot import (  # noqa: E402
    ALGO_LABELS,
    SCHEME_LABELS,
    SCHEME_ORDER,
)

EVIDENCE = Path(__file__).resolve().parent / "evidence"
ALGORITHMS = ("dqn", "ppo")


def _fmt(value, digits=2, suffix=""):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _table(rows, columns, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            _fmt(row.get(column)) for column in columns
        ) + " |")
    return "\n".join(lines)


def _value(row, key):
    value = row.get(key)
    return None if value is None else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    summary_payload = json.loads(
        (evidence / "summary.json").read_text(encoding="utf-8")
    )
    summary = summary_payload["runs"]
    comparison = summary_payload.get("comparison", {})
    freeze = json.loads(
        (evidence / "protocol_freeze.json").read_text(encoding="utf-8")
    )
    rows = sorted(summary, key=lambda row: (
        ALGORITHMS.index(row["algorithm"]), SCHEME_ORDER.index(row["scheme"])
    ))

    overview = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "episodes": row["episodes"],
        "elapsed_h": row["elapsed_h"],
        "best": row["best_dev_e2e_ms"],
        "best_ep": row["best_dev_episode"],
        "final": row["final_dev_e2e_ms"],
        "converged": row["converged"],
    } for row in rows]

    trajectory = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "m1000": row["ma100_e2e_at_1000"],
        "m2000": row["ma100_e2e_at_2000"],
        "m3000": row["ma100_e2e_at_3000"],
        "m4000": row["ma100_e2e_at_4000"],
        "auc": row["auc_ma100_e2e"],
        "min": row["min_ma100_e2e"],
        "std": row["mean_std100_e2e_after400"],
        "t5": row["time_to_self_best_5pct"],
        "slope": row["slope_last2000_ms_per_1000ep"],
    } for row in rows]

    stability = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "dev_range": row["dev_last5_range_pct"],
        "final_gap": (
            100.0 * (row["final_dev_e2e_ms"] - row["best_dev_e2e_ms"])
            / row["best_dev_e2e_ms"]
            if row["final_dev_e2e_ms"] and row["best_dev_e2e_ms"] else None
        ),
        "reward_cv": row["reward_cv_last500"],
        "e2e_std": row["mean_std100_e2e_after400"],
        "q_mean": row.get("q_mean_final"),
        "q_target": row.get("target_mean_final"),
        "td": row.get("td_loss_final"),
        "value_loss": row.get("value_loss_final"),
        "ev": row.get("explained_variance_final"),
    } for row in rows]

    convergence = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "best": row["best_dev_e2e_ms"],
        "best_ep": row["best_dev_episode"],
        "spread": (
            100.0 * row["last3_spread_fraction"]
            if row.get("last3_spread_fraction") is not None else None
        ),
        "gap": (
            100.0 * row["final_gap_fraction"]
            if row.get("final_gap_fraction") is not None else None
        ),
        "status": row["convergence_status"],
        "converged": row["converged"],
    } for row in rows]

    terminal = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "corr_mean": row["terminal_correction_mean"],
        "corr_median": row["terminal_correction_median"],
        "corr_max": row["terminal_correction_max"],
        "share": row["terminal_share_mean"],
        "ratio": row["terminal_to_nonterminal_ratio"],
        "T_over_H": row["mean_T_over_H"],
    } for row in rows]

    consistency = [{
        "scheme": row["scheme_label"],
        "algorithm": row["algorithm_label"],
        "corr": row.get("return_corr_last1000"),
        "ratio": row.get("return_ratio_last1000"),
        "value": (
            row.get("q_mean_final") if row["algorithm"] == "dqn"
            else row.get("value_mean_final")
        ),
        "return": row.get("episode_return_final"),
    } for row in rows]

    ranking = []
    for algorithm in ALGORITHMS:
        payload = comparison.get(algorithm, {})
        ranking.append({
            "algorithm": ALGO_LABELS[algorithm],
            "final": " > ".join(
                SCHEME_LABELS[item]
                for item in payload.get("rank_final_ma100_e2e", [])
            ),
            "auc": " > ".join(
                SCHEME_LABELS[item]
                for item in payload.get("rank_auc_ma100_e2e", [])
            ),
            "best_dev": " > ".join(
                SCHEME_LABELS[item]
                for item in payload.get("rank_best_dev_e2e", [])
            ),
            "converged": ", ".join(
                SCHEME_LABELS[item]
                for item in payload.get("converged", [])
            ) or "无",
        })

    text = "\n\n".join([
        "# Stage-2o 数据表（由 render_report.py 生成）\n",
        f"生成时间：{summary_payload.get('generated_utc')}；"
        f"seed={summary_payload.get('seed')}；"
        f"Test 读取：{summary_payload.get('test_read')}。",
        "## 表 1 训练概况\n\n" + _table(
            overview,
            ["scheme", "algorithm", "episodes", "elapsed_h", "best",
             "best_ep", "final", "converged"],
            ["方案", "算法", "Episodes", "耗时/h", "最佳 Dev E2E/ms",
             "@Ep", "最终 Dev E2E/ms", "收敛"],
        ),
        "## 表 2 MA100 E2E 轨迹\n\n" + _table(
            trajectory,
            ["scheme", "algorithm", "m1000", "m2000", "m3000", "m4000",
             "auc", "min", "std", "t5", "slope"],
            ["方案", "算法", "Ep1000", "Ep2000", "Ep3000", "Ep4000",
             "AUC/ms", "最小 MA100/ms", "滚动σ/ms", "达到自身最佳5%/Ep",
             "末2000斜率 ms/1000Ep"],
        ),
        "## 表 3 稳定性与诊断\n\n" + _table(
            stability,
            ["scheme", "algorithm", "dev_range", "final_gap", "reward_cv",
             "e2e_std", "q_mean", "q_target", "td", "value_loss", "ev"],
            ["方案", "算法", "Dev近5次极差/%", "最终比最佳差/%",
             "奖励CV(末500)", "E2E滚动σ/ms", "Q mean", "Target",
             "TD Loss", "Value Loss", "EV"],
        ),
        "## 表 4 收敛判据\n\n" + _table(
            convergence,
            ["scheme", "algorithm", "best", "best_ep", "spread", "gap",
             "status", "converged"],
            ["方案", "算法", "最佳 Dev/ms", "@Ep", "最近三次极差/%",
             "最终差距/%", "状态", "收敛"],
        ),
        "## 表 5 D 终止修正与信用分配\n\n" + _table(
            terminal,
            ["scheme", "algorithm", "corr_mean", "corr_median", "corr_max",
             "share", "ratio", "T_over_H"],
            ["方案", "算法", "修正均值", "修正中位数", "修正最大",
             "终止占比均值", "修正/非终止步尺度", "平均 T/H"],
        ),
        "## 表 6 Q/Value 与 Episode 回报（方案一为精确比较；B/D 的 Q/Value 是剩余回报，仅作诊断）\n\n"
        + _table(
            consistency,
            ["scheme", "algorithm", "corr", "ratio", "value", "return"],
            ["方案", "算法", "相关系数", "均值比",
             "末 200 Q/Value 均值", "末 200 Episode 回报均值"],
        ),
        "## 表 7 排名（越靠前越好）\n\n" + _table(
            ranking,
            ["algorithm", "final", "auc", "best_dev", "converged"],
            ["算法", "最终 MA100 E2E", "MA100 AUC", "最佳 Dev E2E", "收敛"],
        ),
        "## 审计记录\n\n"
        f"- 单元测试：{freeze['unit_tests']['summary']}；"
        f"通过={freeze['unit_tests']['passed']}\n"
        f"- 冒烟最大奖励恒等式误差："
        f"{freeze['smoke']['max_reward_abs_error']:.3e}\n"
        f"- 冻结时间：{freeze['created_utc']}\n"
        f"- Test 读取：{freeze['final_test_read']}\n"
        "- 注意：D/方案一的累计回报相同，B 比它们低常数 1；"
        "奖励均值比只能用于检查价值目标，不用于比较方案优劣。\n"
        "- 注意：方案一的 V(s)=J 严格成立；B/D 的 Q/Value 是剩余回报，"
        "与完整 Episode 回报的相关系数/均值比只能作诊断，不能直接解释为"
        "校准误差。",
    ]) + "\n"
    path = evidence / "REPORT_TABLES.md"
    path.write_text(text, encoding="utf-8")
    print(f"[render] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
