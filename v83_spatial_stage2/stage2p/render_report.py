"""Render a data-driven Stage-2p answer draft and report tables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVIDENCE = Path(__file__).resolve().parent / "evidence"
SCHEME_LABELS = {"baseline": "原始奖励", "D": "方案D"}


def _fmt(value, digits=2):
    if value is None:
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _signed(value, digits=2):
    if value is None:
        return "N/A"
    return f"{float(value):+.{digits}f}"


def _mean(rows, key):
    values = [
        float(row[key]) for row in rows
        if row.get(key) is not None
        and math.isfinite(float(row[key]))
    ]
    return float(sum(values) / len(values)) if values else None


def _std(rows, key):
    values = [
        float(row[key]) for row in rows
        if row.get(key) is not None
        and math.isfinite(float(row[key]))
    ]
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return float((sum((value - mean) ** 2 for value in values)
                  / (len(values) - 1)) ** 0.5)


def _count_wins(paired, key):
    values = [
        float(row[key]) for row in paired
        if row.get(key) is not None
    ]
    return sum(1 for value in values if value < 0), len(values), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    payload = json.loads(
        (evidence / "summary.json").read_text(encoding="utf-8")
    )
    per_seed = payload["runs"]
    paired = payload["paired_differences"]
    aggregate = payload["aggregate"]
    lines = ["# Stage-2p 五问数据回答（草稿）\n"]

    # Q1/Q2 --------------------------------------------------------------
    lines.append("## Q1/Q2 下降速度与稳定性\n")
    lines.append("| 方案 | 最佳 Dev (mean±std) | 最终 Dev | MA100 AUC | "
                 "末500 E2E σ | 退化/% | 距 Greedy |")
    lines.append("|---|---|---|---|---|---|---|")
    for scheme in ("baseline", "D"):
        rows = [row for row in per_seed if row["scheme"] == scheme]
        if not rows:
            continue
        lines.append(
            f"| {SCHEME_LABELS[scheme]} | "
            f"{_fmt(_mean(rows, 'best_dev_e2e_ms'))} ± "
            f"{_fmt(_std(rows, 'best_dev_e2e_ms'))} | "
            f"{_fmt(_mean(rows, 'final_dev_e2e_ms'))} | "
            f"{_fmt(_mean(rows, 'auc_ma100_e2e'))} | "
            f"{_fmt(_mean(rows, 'last500_e2e_std'))} | "
            f"{_fmt(_mean(rows, 'degradation_pct'))} | "
            f"{_fmt(_mean(rows, 'gap_to_greedy_ms'))} |"
        )
    baseline = [row for row in per_seed if row["scheme"] == "baseline"]
    scheme_d = [row for row in per_seed if row["scheme"] == "D"]
    if baseline and scheme_d:
        fastest = min(
            ("baseline", "D"),
            key=lambda scheme: _mean(
                [row for row in per_seed if row["scheme"] == scheme],
                "auc_ma100_e2e",
            ) or float("inf"),
        )
        most_stable = min(
            ("baseline", "D"),
            key=lambda scheme: _mean(
                [row for row in per_seed if row["scheme"] == scheme],
                "degradation_pct",
            ) or float("inf"),
        )
        lines.append(
            f"\n- MA100 E2E AUC 均值更低（下降更快）：**"
            f"{SCHEME_LABELS[fastest]}**。"
        )
        lines.append(
            f"- 最终相对最佳的退化均值更低（末段更稳）：**"
            f"{SCHEME_LABELS[most_stable]}**。"
        )
    lines.append("")

    # Q3/Q4 --------------------------------------------------------------
    lines.append("## Q3/Q4 D 的收益是否可重复\n")
    metrics = (
        ("delta_best_dev_e2e_ms", "最佳 Dev E2E"),
        ("delta_final_dev_e2e_ms", "最终 Dev E2E"),
        ("delta_auc_ma100_e2e", "MA100 E2E AUC"),
        ("delta_last500_e2e_std", "末500 E2E σ"),
        ("delta_degradation_pct", "最终退化/%"),
        ("delta_gap_to_greedy_ms", "距 Greedy/ms"),
    )
    lines.append("| 指标 | D−Baseline 逐 seed | 均值 | D 更好的 seed 数 |")
    lines.append("|---|---|---|---|")
    for key, title in metrics:
        wins, total, values = _count_wins(paired, key)
        values_text = ", ".join(_signed(value) for value in values)
        mean = sum(values) / len(values) if values else None
        lines.append(
            f"| {title} | {values_text} | {_signed(mean)} | "
            f"{wins}/{total} |"
        )
    lines.append(
        "\n判定：若某指标在 3/3 个 seed 上同向、且均值差明显大于 Dev 噪声"
        "（Stage-2n 参照 1.26 ms；本轮逐 run 末 5 次 Dev 标准差见 "
        "per_seed_metrics.csv），才可称为可重复收益。"
    )
    lines.append("")

    # Q5 -----------------------------------------------------------------
    lines.append("## Q5 与 Greedy 的差距\n")
    for scheme in ("baseline", "D"):
        rows = [row for row in per_seed if row["scheme"] == scheme]
        if not rows:
            continue
        lines.append(
            f"- {SCHEME_LABELS[scheme]}：最佳 Dev 均值 "
            f"{_fmt(_mean(rows, 'best_dev_e2e_ms'))} ms，距 Greedy "
            f"49.18 ms 为 "
            f"{_signed(_mean(rows, 'gap_to_greedy_ms'))} ms "
            f"（{_signed(_mean(rows, 'gap_to_greedy_pct'))}%）。"
        )
    lines.append(
        "\n若 D−Baseline 的最佳 Dev 改善小于验证噪声，则即使方向一致，"
        "也不能说 D 显著更接近 Greedy；应看三种子配对差值与置信范围。"
    )
    path = evidence / "ANSWERS_DRAFT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[render] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
