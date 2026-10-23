"""Fill the data-table placeholders in STAGE2O_REPORT.md.

Run after ``analyze.py``, ``render_report.py``, ``answer_questions.py`` and
``baseline_compare.py``:

    python v83_spatial_stage2/stage2o/fill_report.py

The narrative placeholders ``RESULTS_SUMMARY`` and ``RESULTS_REWARD`` are left
for manual writing; everything else is data-driven.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE_DIR = Path(__file__).resolve().parent
REPORT = STAGE_DIR / "STAGE2O_REPORT.md"
EVIDENCE = STAGE_DIR / "evidence"


def _section(markdown: str, prefix: str) -> str:
    lines = markdown.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            start = index
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _replace(text: str, placeholder: str, replacement: str) -> str:
    token = f"<!-- {placeholder} -->"
    if token not in text:
        raise SystemExit(f"placeholder {token} not found")
    return text.replace(token, replacement.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    report = args.report.read_text(encoding="utf-8")
    evidence = Path(args.evidence_dir)
    tables = (evidence / "REPORT_TABLES.md").read_text(encoding="utf-8")
    answers = (evidence / "ANSWERS.md").read_text(encoding="utf-8")
    baseline = (evidence / "BASELINE_COMPARE.md").read_text(encoding="utf-8")

    table1 = _section(tables, "## 表 1")
    table2 = _section(tables, "## 表 2")
    table3 = _section(tables, "## 表 3")
    table4 = _section(tables, "## 表 4")
    table5 = _section(tables, "## 表 5")
    table6 = _section(tables, "## 表 6")

    replacements = {
        "RESULTS_OVERVIEW": "\n\n".join([
            table1,
            "六个 run 的收敛判据结果：",
            table4,
            "收敛判据为最近三次 Dev E2E 极差 ≤ 最佳值的 1% 且最终值 ≤ "
            "最佳值的 101%；未满足的 run 按协议标记为 `not_converged`。",
        ]),
        "RESULTS_E2E": "\n\n".join([
            table2,
            "AUC 为 Episode 200–4000 的 MA100 E2E 时间平均，越低表示整体"
            "下降越快；`达到自身最佳5%` 需与最小 MA100 一起阅读，"
            "避免把“过早停在较差平台”误读为收敛快。",
        ]),
        "RESULTS_DEV": "\n\n".join([
            table1,
            "Dev-60 的 SR/SPL/nDTW 在所有 run 中应与 Stage-2n 完全一致"
            "（60.00% / 0.5993 / 0.5569），只允许 E2E 不同。",
        ]),
        "RESULTS_STABILITY": "\n\n".join([
            table3,
            "E2E 滚动标准差与 Dev 近 5 次极差可直接比较；奖励标准差的"
            "绝对尺度受方案常数平移影响，不能跨方案直接比较。",
        ]),
        "RESULTS_DQN": "\n\n".join([
            table3,
            table6,
            "Q mean 与 Target 的持续分离、Q max 的持续增长、以及 Q 与 "
            "Episode 回报的相关系数/均值比用于判断 Q 漂移与目标一致性。",
        ]),
        "RESULTS_PPO": "\n\n".join([
            table3,
            table6,
            "Value Loss、Explained Variance、Entropy、KL 用于判断价值估计"
            "质量与策略更新幅度；表 6 的相关系数/均值比用于检查 Value 是否"
            "跟踪 Episode 回报。",
        ]),
        "RESULTS_DEV_SUMMARY": baseline,
        "OUR_VALUES": baseline,
        "RESULTS_TERMINAL": "\n\n".join([
            table5,
            "终止修正的占比由 `1 - T/H` 决定；方案一的终止转移占比恒为 1。"
            "该表用于判断末步反馈是否过大以及 D 的稠密基底是否值得。",
        ]),
        "RESULTS_ANSWERS": answers,
    }
    for placeholder, replacement in replacements.items():
        report = _replace(report, placeholder, replacement)
    args.report.write_text(report, encoding="utf-8")
    remaining = [
        line for line in report.splitlines() if line.strip().startswith("<!--")
    ]
    print(
        f"[fill] wrote {args.report}; remaining placeholders: "
        f"{remaining if remaining else 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
