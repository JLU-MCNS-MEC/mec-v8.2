"""Generate a data-driven draft of the five Stage-2p answers."""

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
LABELS = {"baseline": "原始奖励", "D": "方案D"}
NOISE_MS = 1.26  # Stage-2n six-run last-5 Dev E2E std mean


def _fmt(value, digits=2):
    if value is None:
        return "N/A"
    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


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
    return sum(values) / len(values) if values else None


def _std(rows, key):
    values = [
        float(row[key]) for row in rows
        if row.get(key) is not None
        and math.isfinite(float(row[key]))
    ]
    if len(values) < 2:
        return 0.0 if values else None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values)
            / (len(values) - 1)) ** 0.5


def _scheme(rows, scheme):
    return [row for row in rows if row["scheme"] == scheme]


def _wins(paired, key):
    values = [float(row[key]) for row in paired
              if row.get(key) is not None]
    return sum(1 for value in values if value < 0), len(values), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    payload = json.loads(
        (evidence / "summary.json").read_text(encoding="utf-8")
    )
    runs = payload["runs"]
    paired = payload["paired_differences"]
    baseline = _scheme(runs, "baseline")
    scheme_d = _scheme(runs, "D")
    lines = ["# Stage-2p 五问数据回答（草稿）\n"]

    auc_b = _mean(baseline, "auc_ma100_e2e")
    auc_d = _mean(scheme_d, "auc_ma100_e2e")
    faster = "方案D" if auc_d < auc_b else "原始奖励"
    lines.append("## Q1 哪个使 E2E 下降更快？\n")
    lines.append(
        f"- MA100 E2E AUC 均值：原始奖励 {_fmt(auc_b)} ms，"
        f"方案D {_fmt(auc_d)} ms。**{faster}** 的整体 AUC 更低。"
    )
    lines.append(
        f"- 最佳 Dev E2E 均值：原始奖励 "
        f"{_fmt(_mean(baseline, 'best_dev_e2e_ms'))} ms，方案D "
        f"{_fmt(_mean(scheme_d, 'best_dev_e2e_ms'))} ms。"
    )
    lines.append("")

    deg_b = _mean(baseline, "degradation_pct")
    deg_d = _mean(scheme_d, "degradation_pct")
    std_b = _mean(baseline, "last500_e2e_std")
    std_d = _mean(scheme_d, "last500_e2e_std")
    stable = "方案D" if deg_d < deg_b else "原始奖励"
    lines.append("## Q2 哪个末段更稳定、退化更少？\n")
    lines.append(
        f"- 最终相对最佳的退化均值：原始奖励 {_fmt(deg_b)}%，"
        f"方案D {_fmt(deg_d)}%；末 500 Episode E2E σ 均值："
        f"原始奖励 {_fmt(std_b)} ms，方案D {_fmt(std_d)} ms。"
        f"按退化幅度，**{stable}** 更稳；需与末 500 σ 一起看。"
    )
    stable_std = "方案D" if std_d < std_b else "原始奖励"
    lines.append(f"- 按末 500 E2E σ，**{stable_std}** 的波动更小。")
    lines.append("")

    lines.append("## Q3/Q4 D 的收益是否可重复？\n")
    for key, title in (
        ("delta_best_dev_e2e_ms", "最佳 Dev E2E"),
        ("delta_final_dev_e2e_ms", "最终 Dev E2E"),
        ("delta_auc_ma100_e2e", "MA100 E2E AUC"),
        ("delta_degradation_pct", "最终退化/%"),
        ("delta_last500_e2e_std", "末 500 E2E σ"),
    ):
        wins, total, values = _wins(paired, key)
        mean = sum(values) / len(values) if values else None
        std = (
            (sum((value - mean) ** 2 for value in values)
             / (len(values) - 1)) ** 0.5
            if len(values) > 1 else 0.0
        )
        lines.append(
            f"- {title}: D−Baseline = "
            + ", ".join(_signed(value) for value in values)
            + f"；均值 {_signed(mean)}，标准差 {_fmt(std)}，"
            f"D 更好的 seed 数 {wins}/{total}。"
        )
    lines.append(
        "\n判定：3/3 同向且均值差明显大于 Dev 噪声（Stage-2n 参照 "
        f"{NOISE_MS:.2f} ms）才算可重复收益；若只有 2/3 或均值差小于噪声，"
        "应视为种子依赖/不显著。"
    )
    lines.append("")

    gap_b = _mean(baseline, "gap_to_greedy_ms")
    gap_d = _mean(scheme_d, "gap_to_greedy_ms")
    improvement = gap_b - gap_d if gap_b is not None and gap_d is not None else None
    lines.append("## Q5 哪个更接近 Greedy？\n")
    lines.append(
        f"- 最佳 Dev 距 Greedy 49.18 ms 的均值：原始奖励 "
        f"{_signed(gap_b)} ms，方案D {_signed(gap_d)} ms；"
        f"D 相对 Baseline 的改善为 {_signed(improvement)} ms。"
    )
    if improvement is not None:
        lines.append(
            f"- 该改善 {'大于' if abs(improvement) > NOISE_MS else '不大于'} "
            f"Stage-2n 的 Dev 噪声参照 {NOISE_MS:.2f} ms。"
        )
    lines.append("")
    path = evidence / "ANSWERS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[answers] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
