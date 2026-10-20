"""Read-only degeneration analysis for the Stage-2k DQN.

Combines the recorded training/validation history, the two Dev180 checkpoint
evaluations, the fixed-state Q-value audit and the scenario coverage audit.
No training and no checkpoint modification.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import DQN_DIR, SCENARIOS

OUTPUT_DIR = DQN_DIR / "convergence_audit"
BEST_SUMMARY = DQN_DIR / "dev180_eval" / "summary.json"
LAST_SUMMARY = DQN_DIR / "dev180_eval_last" / "summary.json"
BEST_EPISODES = DQN_DIR / "dev180_eval" / "per_episode.csv"
LAST_EPISODES = DQN_DIR / "dev180_eval_last" / "per_episode.csv"
Q_AUDIT = DQN_DIR / "q_value_audit" / "q_value_audit.json"
COVERAGE = DQN_DIR / "scenario_coverage_audit" / "scenario_coverage.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def window(rows, low, high):
    selected = [row for row in rows if low <= row["episode"] <= high]
    if not selected:
        return None
    output = {"n": len(selected)}
    for key in (
        "return", "navigation_reward", "latency_penalty", "mean_e2e_ms",
        "epsilon", "loss",
    ):
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    output["success"] = float(np.mean([
        float(row["success"]) for row in selected
    ]))
    output["steps"] = float(np.mean([
        float(row["steps"]) for row in selected
    ]))
    return output


def per_episode_aux(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    return {
        "episodes": len(rows),
        "OSR": float(np.mean([float(row["osr"]) for row in rows])),
        "SPL": float(np.mean([float(row["spl"]) for row in rows])),
        "SR": float(np.mean([float(row["success"]) for row in rows])),
        "nDTW": float(np.mean([float(row["ndtw"]) for row in rows])),
        "E2E_ms": float(np.mean([float(row["mean_e2e_ms"]) for row in rows])),
        "P95_ms": float(np.mean([float(row["p95_e2e_ms"]) for row in rows])),
        "violation": float(np.mean([
            float(row["delay_violation_rate"]) for row in rows
        ])),
        "JointR": float(np.mean([float(row["joint_reward"]) for row in rows])),
    }


def scenario_table(summary, aux):
    output = {}
    for scenario in SCENARIOS:
        item = summary["by_scenario"][scenario]
        output[scenario] = {
            "SR": item["SR"],
            "nDTW": item["nDTW"],
            "E2E_ms": item["E2E_ms"],
            "P95_ms": item["P95_ms"],
            "JointR": item["JointR"],
            "mean_steps": item["mean_steps"],
            "episodes": item["episodes"],
        }
    output["overall"] = aux
    return output


def model_usage_ratio(summary):
    return summary["overall"]["model_usage_ratio"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = read_json(DQN_DIR / "training_history.json")
    best_summary = read_json(BEST_SUMMARY)
    last_summary = read_json(LAST_SUMMARY)
    q_audit = read_json(Q_AUDIT)
    coverage = read_json(COVERAGE)

    train = history["train"]
    validation = history["validation"]
    episodes = np.asarray([row["episode"] for row in train], dtype=np.int64)
    returns = np.asarray([row["return"] for row in train], dtype=np.float64)
    ma = np.convolve(returns, np.ones(100) / 100, mode="valid")
    ma_episodes = np.arange(100, len(returns) + 1)
    val_episodes = np.asarray(
        [row["episode"] for row in validation], dtype=np.int64
    )
    val_joint = np.asarray(
        [row["joint_reward"] for row in validation], dtype=np.float64
    )
    trailing = []
    for episode in val_episodes:
        mask = (episodes > int(episode) - 200) & (episodes <= int(episode))
        trailing.append(float(returns[mask].mean()))

    best_aux = per_episode_aux(BEST_EPISODES)
    last_aux = per_episode_aux(LAST_EPISODES)

    output = {
        "version": "stage2k-convergence-audit-v1",
        "training": {
            "episodes": len(train),
            "return": {
                "mean": float(returns.mean()),
                "std": float(returns.std()),
                "ma100_at_800": float(ma[800 - 100]),
                "ma100_at_2000": float(ma[-1]),
                "ma100_peak": float(ma.max()),
                "ma100_peak_episode": int(ma_episodes[int(np.argmax(ma))]),
            },
            "blocks": {
                f"{low}-{high}": window(train, low, high)
                for low, high in (
                    (1, 500), (501, 1000), (1001, 1500), (1501, 2000),
                )
            },
            "windows": {
                "700-900": window(train, 700, 900),
                "1801-2000": window(train, 1801, 2000),
            },
        },
        "validation": {
            "points": [
                {
                    key: row[key]
                    for key in (
                        "episode", "joint_reward", "SR", "nDTW", "E2E_ms",
                        "p95_ms", "mean_steps",
                    )
                }
                for row in validation
            ],
            "model_usage": {
                str(row["episode"]): row["model_usage"] for row in validation
            },
            "trailing_200_return": trailing,
            "correlation_trailing_return_vs_joint": float(
                np.corrcoef(trailing, val_joint)[0, 1]
            ),
            "note": (
                "Recorded validation used 60 train episodes with dev "
                "topologies; it is not Dev Val."
            ),
        },
        "dev180_best800": {
            "overall": best_aux,
            "by_scenario": scenario_table(best_summary, best_aux),
            "model_usage_ratio": model_usage_ratio(best_summary),
        },
        "dev180_last2000": {
            "overall": last_aux,
            "by_scenario": scenario_table(last_summary, last_aux),
            "model_usage_ratio": model_usage_ratio(last_summary),
        },
        "dev180_delta_last_minus_best": {
            key: last_aux[key] - best_aux[key]
            for key in ("SR", "nDTW", "E2E_ms", "P95_ms", "violation", "JointR")
        },
        "q_value_audit": q_audit,
        "scenario_coverage": coverage,
    }
    path = OUTPUT_DIR / "convergence_audit.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(json.dumps({
        "training_return": output["training"]["return"],
        "windows": output["training"]["windows"],
        "validation_best": {
            "episode": 800,
            "joint": val_joint[list(val_episodes).index(800)],
        },
        "validation_last": {
            "episode": 2000,
            "joint": val_joint[-1],
        },
        "dev180_best800": best_aux,
        "dev180_last2000": last_aux,
        "dev180_delta": output["dev180_delta_last_minus_best"],
        "model_usage_best": best_summary["overall"]["model_usage_ratio"],
        "model_usage_last": last_summary["overall"]["model_usage_ratio"],
        "q_comparison": q_audit["comparison"],
    }, indent=2, ensure_ascii=False))
    print("WROTE", path)


if __name__ == "__main__":
    main()
