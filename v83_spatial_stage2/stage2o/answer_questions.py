"""Compute the five Stage-2o answers from summary.json.

    python v83_spatial_stage2/stage2o/answer_questions.py

Writes ``evidence/ANSWERS.md`` with data-driven Chinese text.  The text is a
first draft: the final report manually reviews it for causal language and
single-seed caveats.
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
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _row(rows, scheme, algorithm):
    for row in rows:
        if row["scheme"] == scheme and row["algorithm"] == algorithm:
            return row
    return None


def _rank(rows, algorithm, key, reverse=False):
    subset = [row for row in rows if row["algorithm"] == algorithm]
    subset = [
        row for row in subset
        if row.get(key) is not None
        and (not isinstance(row[key], float) or math.isfinite(row[key]))
    ]
    return sorted(subset, key=lambda row: row[key], reverse=reverse)


def _best_dev_gap(rows):
    """Largest best-Dev separation between schemes, per algorithm."""
    output = {}
    for algorithm in ALGORITHMS:
        ranked = _rank(rows, algorithm, "best_dev_e2e_ms")
        if len(ranked) >= 2:
            output[algorithm] = {
                "best": ranked[0]["scheme"],
                "gap_ms": (
                    ranked[1]["best_dev_e2e_ms"]
                    - ranked[0]["best_dev_e2e_ms"]
                ),
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    payload = json.loads(
        (evidence / "summary.json").read_text(encoding="utf-8")
    )
    rows = payload["runs"]
    freeze = json.loads(
        (evidence / "protocol_freeze.json").read_text(encoding="utf-8")
    )
    lines = ["# Stage-2o 五问数据回答（草稿）\n"]

    # Q1 -----------------------------------------------------------------
    lines.append("## Q1 哪个方案使 MA100 E2E 下降更快、更稳定？\n")
    for algorithm in ALGORITHMS:
        lines.append(f"### {ALGO_LABELS[algorithm]}\n")
        lines.append(
            "| 方案 | 最终 MA100 E2E | AUC | 达到自身最佳5% | 滚动σ | "
            "最佳 Dev | @Ep |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for scheme in SCHEME_ORDER:
            row = _row(rows, scheme, algorithm)
            if row is None:
                continue
            lines.append(
                f"| {SCHEME_LABELS[scheme]} | "
                f"{_fmt(row['final_ma100_e2e'])} | "
                f"{_fmt(row['auc_ma100_e2e'])} | "
                f"{_fmt(row['time_to_self_best_5pct'], 0)} | "
                f"{_fmt(row['mean_std100_e2e_after400'])} | "
                f"{_fmt(row['best_dev_e2e_ms'])} | "
                f"{row['best_dev_episode']} |"
            )
        auc_rank = _rank(rows, algorithm, "auc_ma100_e2e")
        std_rank = _rank(rows, algorithm, "mean_std100_e2e_after400")
        final_rank = _rank(rows, algorithm, "final_ma100_e2e")
        if auc_rank and std_rank and final_rank:
            lines.append(
                f"\n- AUC 最低（整体下降最快）：**{SCHEME_LABELS[auc_rank[0]['scheme']]}**"
                f"（{_fmt(auc_rank[0]['auc_ma100_e2e'])} ms）。"
            )
            lines.append(
                f"- 最终 MA100 E2E 最低：**{SCHEME_LABELS[final_rank[0]['scheme']]}**"
                f"（{_fmt(final_rank[0]['final_ma100_e2e'])} ms）。"
            )
            lines.append(
                f"- E2E 滚动标准差最低（最稳定）：**{SCHEME_LABELS[std_rank[0]['scheme']]}**"
                f"（{_fmt(std_rank[0]['mean_std100_e2e_after400'])} ms）。"
            )
    lines.append("")

    # Q2 -----------------------------------------------------------------
    lines.append("## Q2 D 的终止修正是否造成过大的末步反馈或信用分配问题？\n")
    for algorithm in ALGORITHMS:
        row = _row(rows, "D", algorithm)
        if row is None:
            continue
        lines.append(
            f"- {ALGO_LABELS[algorithm]}：终止修正均值 "
            f"{_fmt(row['terminal_correction_mean'], 4)}，中位数 "
            f"{_fmt(row['terminal_correction_median'], 4)}，最大 "
            f"{_fmt(row['terminal_correction_max'], 4)}；"
            f"终止转移占 Episode 回报均值 "
            f"{_fmt(row['terminal_share_mean'], 3)}；"
            f"修正 / 非终止步绝对尺度 = "
            f"{_fmt(row['terminal_to_nonterminal_ratio'], 1)}；"
            f"平均 T/H = {_fmt(row['mean_T_over_H'], 3)}。"
        )
    for algorithm in ALGORITHMS:
        d = _row(rows, "D", algorithm)
        b = _row(rows, "B", algorithm)
        one = _row(rows, "one", algorithm)
        if not (d and b and one):
            continue
        lines.append(
            f"- {ALGO_LABELS[algorithm]} 最终 MA100 E2E：D "
            f"{_fmt(d['final_ma100_e2e'])}，B {_fmt(b['final_ma100_e2e'])}，"
            f"方案一 {_fmt(one['final_ma100_e2e'])} ms；"
            f"最佳 Dev E2E：D {_fmt(d['best_dev_e2e_ms'])}，B "
            f"{_fmt(b['best_dev_e2e_ms'])}，方案一 "
            f"{_fmt(one['best_dev_e2e_ms'])} ms。"
        )
    lines.append(
        "\n终止修正的占比由 `1 - T/H` 决定：T 越短，末步占比越大。"
        "若 D 与方案一（100% 末步回报）表现接近，说明 D 的稠密基底没有显著"
        "改善信用分配；若 D 明显优于方案一，说明逐步反馈有帮助；若 D 的 "
        "Q/Value 诊断更差，则终止修正可能带来不稳定。结论以第 4 节数值为准。\n"
    )

    # Q3 -----------------------------------------------------------------
    lines.append("## Q3 方案一的目标是否正确实现？DQN/PPO 是否改善？\n")
    lines.append(
        f"- 单元测试：{freeze['unit_tests']['passed']}，"
        f"{freeze['unit_tests']['summary']}；"
        f"冒烟最大奖励恒等式误差 "
        f"{freeze['smoke']['max_reward_abs_error']:.3e}。"
    )
    for scheme in ("B", "D", "one"):
        for algorithm in ALGORITHMS:
            row = _row(rows, scheme, algorithm)
            if row is None:
                continue
            lines.append(
                f"- {SCHEME_LABELS[scheme]}/{ALGO_LABELS[algorithm]}："
                f"末 1000 Episode 的 Q/Value 与 Episode 回报相关系数 "
                f"{_fmt(row.get('return_corr_last1000'), 3)}，均值比 "
                f"{_fmt(row.get('return_ratio_last1000'), 3)}。"
            )
    lines.append(
        "\n注意：方案一的 `V(s)=J` 严格成立，其 Value/回报比较是精确的；"
        "B/D 的 Q/Value 是剩余回报，与完整 Episode 回报的相关系数/均值比"
        "只作诊断，不能直接解释为校准误差。"
    )
    for algorithm in ALGORITHMS:
        one = _row(rows, "one", algorithm)
        b = _row(rows, "B", algorithm)
        d = _row(rows, "D", algorithm)
        if not (one and b and d):
            continue
        lines.append(
            f"- {ALGO_LABELS[algorithm]} 方案一 vs B：最终 MA100 E2E "
            f"{_fmt(one['final_ma100_e2e'])} vs "
            f"{_fmt(b['final_ma100_e2e'])} ms；最佳 Dev "
            f"{_fmt(one['best_dev_e2e_ms'])} vs "
            f"{_fmt(b['best_dev_e2e_ms'])} ms。"
        )
    lines.append(
        "\n实现正确性由奖励恒等式、DQN 终止 target、PPO GAE 终止重置与"
        "无未来泄漏测试共同保证；性能是否改善则看上面的 E2E 数值，"
        "两者不能混为一谈。\n"
    )

    # Q4 -----------------------------------------------------------------
    lines.append("## Q4 哪些方案达到收敛判据？\n")
    lines.append("| 方案 | 算法 | 最佳 Dev | @Ep | 最近三次极差 | 最终差距 | 状态 |")
    lines.append("|---|---|---|---|---|---|---|")
    converged = []
    for scheme in SCHEME_ORDER:
        for algorithm in ALGORITHMS:
            row = _row(rows, scheme, algorithm)
            if row is None:
                continue
            if row["converged"]:
                converged.append(f"{SCHEME_LABELS[scheme]}/{ALGO_LABELS[algorithm]}")
            lines.append(
                f"| {SCHEME_LABELS[scheme]} | {ALGO_LABELS[algorithm]} | "
                f"{_fmt(row['best_dev_e2e_ms'])} | "
                f"{row['best_dev_episode']} | "
                f"{_fmt(100*row['last3_spread_fraction'] if row.get('last3_spread_fraction') is not None else None, 2)}% | "
                f"{_fmt(100*row['final_gap_fraction'] if row.get('final_gap_fraction') is not None else None, 2)}% | "
                f"{row['convergence_status']} |"
            )
    lines.append(
        "\n达到收敛判据："
        + ("、".join(converged) if converged else "无")
        + "。未达到的 run 按协议标记为 `not_converged`。\n"
    )

    # Q5 -----------------------------------------------------------------
    lines.append("## Q5 是否有充分证据扩到三个随机种子？\n")
    dev_gap = _best_dev_gap(rows)
    for algorithm in ALGORITHMS:
        final_rank = _rank(rows, algorithm, "final_ma100_e2e")
        if len(final_rank) >= 2:
            margin = (
                final_rank[1]["final_ma100_e2e"]
                - final_rank[0]["final_ma100_e2e"]
            )
            noise = max(
                row["dev_last5_std"] for row in rows
                if row["algorithm"] == algorithm
                and row.get("dev_last5_std") is not None
            )
            ratio = margin / noise if noise and noise > 0 else None
            lines.append(
                f"- {ALGO_LABELS[algorithm]}：最终 MA100 E2E 第一名 "
                f"{SCHEME_LABELS[final_rank[0]['scheme']]}，领先第二名 "
                f"{_fmt(margin)} ms；该算法下最大 Dev 近 5 次标准差 "
                f"{_fmt(noise)} ms；领先/噪声 = {_fmt(ratio, 2)}。"
            )
        if algorithm in dev_gap:
            lines.append(
                f"- {ALGO_LABELS[algorithm]}：最佳 Dev E2E 第一名 "
                f"{SCHEME_LABELS[dev_gap[algorithm]['best']]}，领先 "
                f"{_fmt(dev_gap[algorithm]['gap_ms'])} ms。"
            )
    lines.append(
        "\nStage-2n 六次 run 的末 5 次 Dev E2E 标准差均值为 1.26 ms"
        "（范围 0.89–1.67 ms），可作为 Dev 验证噪声的参照。"
    )
    lines.append(
        "\n判据（预先声明）：只有当同一方案在 DQN 与 PPO 中都领先，且领先幅度"
        "明显大于 Dev 验证噪声、并且没有系统性 Q 漂移/价值异常时，才值得"
        "直接扩到 3 个随机种子；否则单种子证据不足，应先复跑最接近的 1–2 个"
        "方案或增加验证任务。\n"
    )

    path = evidence / "ANSWERS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[answers] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
