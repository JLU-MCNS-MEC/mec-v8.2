"""Compare Stage-2o best Dev E2E with Stage-2n baselines.

    python v83_spatial_stage2/stage2o/baseline_compare.py

Writes ``evidence/BASELINE_COMPARE.md``.  The Stage-2n values come from
``stage2n_evidence/dev_summary_snapshot.json`` (same Dev-60, same deployment).
These are reference points, not paired comparisons.
"""

from __future__ import annotations

import json
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
GREEDY_DEV_E2E = 49.18036816866046
STAGE2N = {
    "dqn": {"best": 75.76, "values": (75.76, 80.11, 78.53)},
    "ppo": {"best": 58.39, "values": (62.63, 58.70, 58.39)},
}


def _fmt(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _signed(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):+.{digits}f}"


def main():
    payload = json.loads(
        (EVIDENCE / "summary.json").read_text(encoding="utf-8")
    )
    rows = payload["runs"]
    lines = [
        "# Stage-2o Dev-60 与基线参照\n",
        "| 方案 | 算法 | 最佳 Dev E2E/ms | @Ep | 相对 Greedy 49.18ms | "
        "相对 Stage-2n 同算法最佳 |",
        "|---|---|---|---|---|---|",
    ]
    for scheme in SCHEME_ORDER:
        for algorithm in ("dqn", "ppo"):
            row = next(
                (item for item in rows
                 if item["scheme"] == scheme
                 and item["algorithm"] == algorithm),
                None,
            )
            if row is None:
                continue
            best = row["best_dev_e2e_ms"]
            gap_greedy = best - GREEDY_DEV_E2E
            gap_greedy_pct = 100.0 * gap_greedy / GREEDY_DEV_E2E
            gap_stage2n = best - STAGE2N[algorithm]["best"]
            lines.append(
                f"| {SCHEME_LABELS[scheme]} | {ALGO_LABELS[algorithm]} | "
                f"{_fmt(best)} | {row['best_dev_episode']} | "
                f"{_signed(gap_greedy)} ms "
                f"({_signed(gap_greedy_pct)}%) | "
                f"{_signed(gap_stage2n)} ms |"
            )
    lines.extend([
        "",
        "Stage-2n 原奖励（0.25 系数）的 Dev-60 最佳值：",
        f"- DQN 3 seeds：{', '.join(_fmt(v) for v in STAGE2N['dqn']['values'])} ms",
        f"- PPO 3 seeds：{', '.join(_fmt(v) for v in STAGE2N['ppo']['values'])} ms",
        "",
        "说明：方案 B 的奖励是 Stage-2n 原奖励的 4 倍，因此 B 与 Stage-2n"
        "的差异主要反映奖励尺度；D/方案一与 Stage-2n 不是同一目标。",
        "所有策略的 Dev SR/SPL/nDTW 均为 60.00% / 0.5993 / 0.5569。",
    ])
    path = EVIDENCE / "BASELINE_COMPARE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[baseline] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
