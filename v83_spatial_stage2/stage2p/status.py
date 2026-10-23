"""Live Stage-2p status table (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2o.progress import gpu_status  # noqa: E402
from v83_spatial_stage2.stage2p.plot import (  # noqa: E402
    SCHEME_LABELS,
    SCHEME_ORDER,
    SEEDS,
)
from v83_spatial_stage2.stage2p.train import RUNS_ROOT  # noqa: E402


def _read(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    rows = []
    for scheme in SCHEME_ORDER:
        for seed in SEEDS:
            run_dir = runs_root / scheme / f"ppo_s{seed}"
            progress = _read(run_dir / "progress.json") or {}
            rounds = sorted((run_dir / "rounds").glob("round_*.json"))
            latest = _read(rounds[-1]) if rounds else None
            metrics = (latest or {}).get("metrics", {})
            rows.append({
                "scheme": scheme,
                "scheme_label": SCHEME_LABELS[scheme],
                "seed": seed,
                "episode": progress.get("current_episode", 0),
                "max_episodes": progress.get("max_episodes", 4000),
                "status": progress.get("status", "starting"),
                "best_dev_e2e_ms": progress.get("best_dev_e2e_ms"),
                "best_dev_episode": progress.get("best_dev_episode"),
                "elapsed_h": (
                    float(progress.get("elapsed_seconds", 0.0)) / 3600.0
                ),
                "dev_e2e_ms": metrics.get("dev_e2e_ms"),
                "ma100_e2e": metrics.get("e2e_ma100"),
                "ma100_reward": metrics.get("reward_ma100"),
                "flags": metrics.get("flags", ""),
            })
    if args.json:
        print(json.dumps({"gpu": gpu_status(), "runs": rows},
                         ensure_ascii=False, indent=2))
        return
    gpu = gpu_status()
    print(
        f"GPU: util={gpu['gpu_util_pct']}% "
        f"used={gpu['gpu_used_mb']}/{gpu['gpu_total_mb']} MiB"
    )
    header = (
        f"{'方案':<8} {'seed':>4} {'进度':>10} {'状态':<14} "
        f"{'最佳Dev':>9} {'@Ep':>5} {'最新Dev':>9} {'MA100 E2E':>10} "
        f"{'耗时h':>6}  异常"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        best = (f"{row['best_dev_e2e_ms']:.2f}"
                if row["best_dev_e2e_ms"] is not None else "—")
        dev = (f"{row['dev_e2e_ms']:.2f}"
               if row["dev_e2e_ms"] is not None else "—")
        e2e = (f"{row['ma100_e2e']:.2f}"
               if row["ma100_e2e"] is not None else "—")
        print(
            f"{row['scheme_label']:<8} {row['seed']:>4} "
            f"{row['episode']:>4}/{row['max_episodes']:<5} "
            f"{row['status']:<14} {best:>9} "
            f"{str(row['best_dev_episode'] or '—'):>5} {dev:>9} "
            f"{e2e:>10} {row['elapsed_h']:>6.2f}  {row['flags']}"
        )


if __name__ == "__main__":
    main()
