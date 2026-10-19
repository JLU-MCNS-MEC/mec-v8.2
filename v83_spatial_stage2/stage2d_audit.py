"""Independent Stage-2d closed-loop metric audit.

Recomputes every overall and per-band metric directly from the saved
per-episode CSVs, then compares it with the stored metrics JSON. This checks
that:
  * all 180 episodes are present (60 per band);
  * Overall is the direct mean over all 180 episodes, not a 30/50/20 weight;
  * success counts, OSR, SPL, nDTW, premature-Stop and Timeout are consistent;
  * reach-radius-without-success equals OSR - SR;
  * R4/calibration B0 replay outcome mismatches are zero;
  * calibration movement / original Arrival-Head hashes are unchanged.

No model is loaded and no episode is re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2d_common import BANDS


R4_METHODS = (
    ("R0_M0_R4", "threshold_0.5"),
    ("R4_threshold_0.6", "threshold_0.6"),
    ("R4_threshold_0.7", "threshold_0.7"),
    ("R4_2of2_t0.5", "persist_2of2_t0.5"),
    ("R4_2of2_t0.6", "persist_2of2_t0.6"),
    ("R4_3of3_t0.5", "persist_3of3_t0.5"),
)
CAL_METHODS = (
    ("R0_M0_Cal", "B0"),
    ("R1_Platt", "R1"),
    ("R2_HardResidual", "R2"),
    ("R3_RandomResidual", "R3"),
)
TRAIN_MIX_WEIGHTS = {"short": 0.30, "medium": 0.50, "long": 0.20}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metrics(rows):
    n = len(rows)
    success = np.asarray([int(row["success"]) for row in rows])
    osr = np.asarray([int(row["osr"]) for row in rows])
    premature = np.asarray([int(row["premature_stop"]) for row in rows])
    timeout = np.asarray([int(row["max_steps_termination"]) for row in rows])
    reach_no_success = np.asarray(
        [int(row["reached_radius_not_success"]) for row in rows]
    )
    spl = np.asarray([float(row["spl"]) for row in rows])
    ndtw = np.asarray([float(row["ndtw"]) for row in rows])
    steps = np.asarray([int(row["steps"]) for row in rows])
    min_distance = np.asarray([float(row["min_distance_m"]) for row in rows])
    return {
        "n": int(n),
        "success_count": int(success.sum()),
        "osr_count": int(osr.sum()),
        "premature_count": int(premature.sum()),
        "timeout_count": int(timeout.sum()),
        "reach_no_success_count": int(reach_no_success.sum()),
        "sr": float(success.mean()),
        "osr": float(osr.mean()),
        "spl": float(spl.mean()),
        "ndtw": float(ndtw.mean()),
        "premature_stop": float(premature.mean()),
        "timeout": float(timeout.mean()),
        "mean_steps": float(steps.mean()),
        "mean_min_distance_m": float(min_distance.mean()),
    }


def by_band(rows):
    return {
        band: metrics([row for row in rows if str(row["band"]) == band])
        for band in BANDS
    }


def compare(stored, computed, label, checks):
    if stored is None:
        checks.append({"check": f"{label}: stored metrics present", "pass": False})
        return
    for key in ("sr", "osr", "spl", "ndtw", "premature_stop", "timeout"):
        if key not in stored:
            checks.append({
                "check": f"{label}: stored {key} present", "pass": False,
            })
            continue
        delta = abs(float(stored[key]) - float(computed[key]))
        checks.append({
            "check": f"{label}: {key} matches per-episode mean",
            "stored": float(stored[key]),
            "recomputed": float(computed[key]),
            "abs_delta": delta,
            "pass": bool(delta <= 1e-6),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r4-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks = []
    results = {}
    r4_metrics = json.loads(
        (args.r4_dir / "metrics_recomputed.json").read_text(encoding="utf-8")
    )
    cal_metrics = json.loads(
        (args.calibration_dir / "closedloop_metrics_recomputed.json").read_text(
            encoding="utf-8"
        )
    )

    for label, rule in R4_METHODS:
        rows = read_rows(args.r4_dir / f"{rule}_per_episode.csv")
        computed = metrics(rows)
        bands = by_band(rows)
        results[label] = {
            "source": "R4", "rule": rule, "overall": computed, "by_band": bands,
        }
        stored = r4_metrics["metrics"][rule]["overall"]
        compare(stored, computed, label, checks)
        for band in BANDS:
            compare(
                r4_metrics["metrics"][rule]["by_band"][band],
                bands[band], f"{label}/{band}", checks,
            )

    for label, name in CAL_METHODS:
        rows = read_rows(args.calibration_dir / f"{name}_per_episode.csv")
        computed = metrics(rows)
        bands = by_band(rows)
        results[label] = {
            "source": "calibration", "rule": name,
            "overall": computed, "by_band": bands,
        }
        stored = cal_metrics["metrics"][name]["overall"]
        compare(stored, computed, label, checks)
        for band in BANDS:
            compare(
                cal_metrics["metrics"][name]["by_band"][band],
                bands[band], f"{label}/{band}", checks,
            )

    # Structural checks.
    for label, item in results.items():
        overall = item["overall"]
        bands = item["by_band"]
        checks.append({
            "check": f"{label}: 180 episodes",
            "value": overall["n"], "pass": overall["n"] == 180,
        })
        checks.append({
            "check": f"{label}: 60 episodes per band",
            "value": {band: bands[band]["n"] for band in BANDS},
            "pass": all(bands[band]["n"] == 60 for band in BANDS),
        })
        total_success = sum(bands[band]["success_count"] for band in BANDS)
        checks.append({
            "check": f"{label}: overall success count = sum of bands",
            "overall": overall["success_count"],
            "sum_bands": total_success,
            "pass": overall["success_count"] == total_success,
        })
        checks.append({
            "check": f"{label}: OSR - SR = reach-no-success",
            "osr_minus_sr": overall["osr_count"] - overall["success_count"],
            "reach_no_success": overall["reach_no_success_count"],
            "pass": (
                overall["osr_count"] - overall["success_count"]
                == overall["reach_no_success_count"]
            ),
        })
        weighted = sum(
            TRAIN_MIX_WEIGHTS[band] * bands[band]["sr"] for band in BANDS
        )
        checks.append({
            "check": f"{label}: Overall is not the 30/50/20 training-mix weight",
            "direct_overall_sr": overall["sr"],
            "train_mix_weighted_sr": weighted,
            "pass": abs(overall["sr"] - weighted) > 1e-9,
        })
        checks.append({
            "check": f"{label}: nDTW finite",
            "value": overall["ndtw"],
            "pass": bool(np.isfinite(overall["ndtw"])),
        })

    # Replay checks.
    r4_replay = json.loads(
        (args.r4_dir / "replay_check.json").read_text(encoding="utf-8")
    )
    cal_replay = json.loads(
        (args.calibration_dir / "replay_check.json").read_text(encoding="utf-8")
    )
    checks.append({
        "check": "R4 B0 replay outcome mismatches",
        "value": r4_replay.get("outcome_mismatch_count"),
        "pass": r4_replay.get("outcome_mismatch_count") == 0,
    })
    checks.append({
        "check": "Calibration B0 replay outcome mismatches",
        "value": cal_replay.get("outcome_mismatch_count"),
        "pass": cal_replay.get("outcome_mismatch_count") == 0,
    })

    # Hash checks.
    provenance = json.loads(
        (args.calibration_dir / "provenance.json").read_text(encoding="utf-8")
    )
    m0_movement = provenance["movement_hash_m0"]
    m0_head = provenance["arrival_head_hash_m0"]
    hashes_ok = all(
        item["movement_hash"] == m0_movement
        and item["arrival_head_hash"] == m0_head
        for item in provenance["variant_hashes"].values()
    )
    checks.append({
        "check": "Calibration movement/Arrival-head hashes equal M0",
        "pass": bool(hashes_ok),
    })

    # Paired comparison for the best method vs R0.
    paired = r4_metrics["paired_vs_b0"]["persist_2of2_t0.5"]
    best_label = "R4_2of2_t0.5"
    best = results[best_label]
    r0 = results["R0_M0_R4"]
    table2 = {"method": best_label, "base": "R0_M0_R4", "bands": {}}
    for band in ("all",) + BANDS:
        if band == "all":
            b = best["overall"]
            a = r0["overall"]
            pair = paired["all"]
        else:
            b = best["by_band"][band]
            a = r0["by_band"][band]
            pair = paired[band]
        table2["bands"][band] = {
            "base_success_count": a["success_count"],
            "base_n": a["n"],
            "base_sr": a["sr"],
            "method_success_count": b["success_count"],
            "method_n": b["n"],
            "method_sr": b["sr"],
            "sr_delta": pair["success"]["delta_rate"],
            "sr_wins": pair["success"]["other_better"],
            "sr_losses": pair["success"]["base_better"],
            "sr_ci95": pair["success"]["delta_ci95"],
            "osr_delta": pair["osr"]["delta_rate"],
            "osr_wins": pair["osr"]["other_better"],
            "osr_losses": pair["osr"]["base_better"],
            "osr_ci95": pair["osr"]["delta_ci95"],
            "premature_delta": pair["premature_stop"]["delta_rate"],
            "premature_wins": pair["premature_stop"]["other_better"],
            "premature_losses": pair["premature_stop"]["base_better"],
            "premature_ci95": pair["premature_stop"]["delta_ci95"],
            "spl_delta": pair["spl"]["delta"],
            "spl_ci95": pair["spl_delta_ci95"],
            "ndtw_delta": pair["ndtw_delta"],
            "ndtw_ci95": pair["ndtw_delta_ci95"],
        }

    payload = {
        "methods": results,
        "checks": checks,
        "all_checks_pass": all(item["pass"] for item in checks),
        "best_method_table2": table2,
        "train_mix_weights_note": (
            "Train-600 uses 30/50/20 (Short/Medium/Long); Dev Val 180 is "
            "60/60/60. Overall metrics are direct means over all 180 "
            "episodes; the 30/50/20 weighted band SR is reported only as a "
            "negative check and is not used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Print a compact audit summary and Table 1.
    print("=== Audit checks ===")
    for item in checks:
        status = "PASS" if item["pass"] else "FAIL"
        print(f"[{status}] {item['check']}")
    print("\n=== Table 1: Stage-2d closed-loop metrics (N=180) ===")
    print(
        "method,SR,success/180,OSR,SPL,nDTW,Premature,Timeout,mean_steps"
    )
    for label, item in results.items():
        overall = item["overall"]
        print(
            f"{label},{overall['sr']:.4f},{overall['success_count']}/180,"
            f"{overall['osr']:.4f},{overall['spl']:.4f},{overall['ndtw']:.4f},"
            f"{overall['premature_stop']:.4f},{overall['timeout']:.4f},"
            f"{overall['mean_steps']:.1f}"
        )
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    main()
