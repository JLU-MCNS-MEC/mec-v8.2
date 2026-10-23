"""Print a compact Chinese table for one Stage-2p 200-episode report point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2p.plot import (  # noqa: E402
    SCHEME_LABELS,
    SCHEME_ORDER,
    SEEDS,
)
from v83_spatial_stage2.stage2p.train import RUNS_ROOT  # noqa: E402


def _read(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits=2):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--episode", type=int, default=200)
    args = parser.parse_args()
    rows = []
    for scheme in SCHEME_ORDER:
        for seed in SEEDS:
            run_dir = Path(args.runs_root) / scheme / f"ppo_s{seed}"
            payload = _read(
                run_dir / "rounds" / f"round_{args.episode:04d}.json"
            )
            metrics = (payload or {}).get("metrics", {})
            rows.append({
                "scheme": SCHEME_LABELS[scheme],
                "seed": seed,
                "elapsed_h": metrics.get("elapsed_h"),
                "dev_e2e": metrics.get("dev_e2e_ms"),
                "p95": metrics.get("dev_p95_ms"),
                "violation": (
                    100.0 * metrics["dev_delay_violation_rate"]
                    if metrics.get("dev_delay_violation_rate") is not None
                    else None
                ),
                "sr": (
                    100.0 * metrics["dev_sr"]
                    if metrics.get("dev_sr") is not None else None
                ),
                "spl": metrics.get("dev_spl"),
                "ndtw": metrics.get("dev_ndtw"),
                "reward": metrics.get("reward_ma100"),
                "e2e": metrics.get("e2e_ma100"),
                "diag": (
                    f"VL {_fmt(metrics.get('ppo_value_loss'), 4)} / "
                    f"EV {_fmt(metrics.get('ppo_explained_variance'), 3)} / "
                    f"H {_fmt(metrics.get('ppo_entropy'), 3)} / "
                    f"KL {_fmt(metrics.get('ppo_kl'), 4)}"
                ),
                "flags": metrics.get("flags") or "",
            })
    headers = [
        "方案", "seed", "耗时/h", "Dev E2E", "P95", "违约率%", "SR%",
        "SPL", "nDTW", "MA100 奖励", "MA100 E2E", "PPO 诊断", "异常",
    ]
    columns = [
        "scheme", "seed", "elapsed_h", "dev_e2e", "p95", "violation",
        "sr", "spl", "ndtw", "reward", "e2e", "diag", "flags",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(
            str(row[column]) if column in ("scheme", "seed", "diag", "flags")
            else _fmt(row[column])
            for column in columns
        ) + " |")


if __name__ == "__main__":
    main()
