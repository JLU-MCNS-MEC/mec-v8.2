"""Fill Stage-2p report table placeholders from the generated evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE_DIR = Path(__file__).resolve().parent
REPORT = STAGE_DIR / "STAGE2P_REPORT.md"
EVIDENCE = STAGE_DIR / "evidence"
LABELS = {"baseline": "原始奖励", "D": "方案D"}


def _section(markdown, prefix):
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


def _replace(text, placeholder, replacement):
    token = f"<!-- {placeholder} -->"
    if token not in text:
        raise SystemExit(f"placeholder {token} not found")
    return text.replace(token, replacement.strip())


def _fmt(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    report = args.report.read_text(encoding="utf-8")
    evidence = Path(args.evidence_dir)
    tables = (evidence / "SUMMARY_TABLES.md").read_text(encoding="utf-8")
    payload = json.loads(
        (evidence / "summary.json").read_text(encoding="utf-8")
    )
    per_seed = payload["runs"]
    table1 = _section(tables, "## 表 1")
    table2 = _section(tables, "## 表 2")
    table3 = _section(tables, "## 表 3")
    convergence = [
        "| 方案 | seed | 最佳 Dev/ms | @Ep | 最终 Dev/ms | "
        "最近三次极差/% | 最终差距/% | Dev 末5次 σ/ms | 状态 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(
        per_seed, key=lambda item: (item["scheme"], item["seed"])
    ):
        spread = (
            100.0 * row["last3_spread_fraction"]
            if row.get("last3_spread_fraction") is not None else None
        )
        gap = (
            100.0 * row["final_gap_fraction"]
            if row.get("final_gap_fraction") is not None else None
        )
        convergence.append(
            f"| {LABELS[row['scheme']]} | {row['seed']} | "
            f"{_fmt(row['best_dev_e2e_ms'])} | {row['best_dev_episode']} | "
            f"{_fmt(row['final_dev_e2e_ms'])} | "
            f"{_fmt(spread)} | {_fmt(gap)} | "
            f"{_fmt(row.get('dev_last5_std'))} | "
            f"{row['convergence_status']} |"
        )
    converged = [
        f"{LABELS[row['scheme']]}/s{row['seed']}"
        for row in per_seed if row["converged"]
    ]
    convergence_text = "\n".join(convergence) + "\n\n" + (
        "达到收敛判据："
        + ("、".join(converged) if converged else "无")
        + "。判据仍为最近三次 Dev E2E 极差 ≤ 最佳值的 1% 且最终值 "
        "≤ 最佳值的 101%；Dev 末 5 次标准差作为验证噪声参照。"
    )
    for placeholder, replacement in (
        ("TABLE1", table1),
        ("TABLE2", table2),
        ("TABLE3", table3),
        ("CONVERGENCE", convergence_text),
    ):
        report = _replace(report, placeholder, replacement)
    args.report.write_text(report, encoding="utf-8")
    remaining = [
        line for line in report.splitlines() if line.strip().startswith("<!--")
    ]
    print(
        f"[fill] wrote {args.report}; remaining: "
        f"{remaining if remaining else 'none'}", flush=True,
    )


if __name__ == "__main__":
    main()
