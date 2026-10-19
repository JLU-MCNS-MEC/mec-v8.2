"""Stage-2d R4: Stop-decision rule comparison with full closed-loop re-runs.

Every rule re-runs the Dev Val 180 closed loop from scratch with the frozen
Stage-2a M0 model. The Stop-confirmation state is created fresh per episode.
The original checkpoint is never modified.

Rules:
  threshold_0.5 / 0.6 / 0.7      fixed probability thresholds
  persist_2of2_t0.5              2 consecutive Stop decisions at 0.5
  persist_3of3_t0.5              3 consecutive Stop decisions at 0.5
  persist_2of2_t0.6              supplementary threshold + confirmation

Outputs are development evidence only. Final Test is sealed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2d_common import (
    BANDS,
    VAL_SPECS,
    build_oracle_cache,
    device_summary,
    evaluate_policy,
    file_sha256,
    hazard_from_steps,
    load_dev_specs,
    load_encoder,
    load_m0,
    paired_by_band,
    risk_rows,
    rule_from_name,
    stop_event_summary,
    summarize_by_band,
    verify_b0_replay,
    write_csv,
    write_json,
)


DEFAULT_RULES = (
    "threshold_0.5",
    "threshold_0.6",
    "threshold_0.7",
    "persist_2of2_t0.5",
    "persist_3of3_t0.5",
    "persist_2of2_t0.6",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--max-episodes", type=int, default=180)
    parser.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES))
    parser.add_argument("--progress-every", type=int, default=30)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "steps").mkdir()
    (args.output_dir / "stops").mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    specs = load_dev_specs(args.val_specs, args.max_episodes)
    encoder = load_encoder(device)
    oracle_cache = build_oracle_cache(specs)
    model, m0_state, _ = load_m0(device)
    m0_sha256 = file_sha256(
        Path("runs/v83_spatial_stage2/stage2a_tm_20261018T044255Z")
        / "Transformer-M/M0/best.pt"
    )
    print(
        f"Loaded M0; {len(specs)} Dev Val episodes; rules={list(args.rules)}",
        flush=True,
    )

    episodes_by_rule = {}
    steps_by_rule = {}
    stops_by_rule = {}
    for name in args.rules:
        rule = rule_from_name(name)
        print(f"Closed-loop rule {name}", flush=True)
        episodes, steps, stops = evaluate_policy(
            model, specs, oracle_cache, encoder, device, rule,
            progress_every=args.progress_every,
        )
        episodes_by_rule[name] = episodes
        steps_by_rule[name] = steps
        stops_by_rule[name] = stops
        write_csv(args.output_dir / "steps" / f"{name}.csv", steps)
        write_csv(args.output_dir / "stops" / f"{name}.csv", stops)
        write_csv(args.output_dir / f"{name}_per_episode.csv", episodes)
        print(
            f"  {name}: SR={np.mean([r['success'] for r in episodes]):.4f} "
            f"OSR={np.mean([r['osr'] for r in episodes]):.4f} "
            f"premature={np.mean([r['premature_stop'] for r in episodes]):.4f}",
            flush=True,
        )

    b0_name = "threshold_0.5"
    if b0_name not in episodes_by_rule:
        raise SystemExit("threshold_0.5 (B0 reference) must be in --rules")
    replay_check = verify_b0_replay(episodes_by_rule[b0_name])
    if replay_check["outcome_mismatch_count"] != 0:
        write_json(args.output_dir / "replay_check.json", replay_check)
        raise SystemExit(
            "B0 replay outcome mismatch: "
            f"{replay_check['outcome_mismatch_count']} fields; "
            "see replay_check.json"
        )
    write_json(args.output_dir / "replay_check.json", replay_check)

    all_episodes = [
        row for name in args.rules for row in episodes_by_rule[name]
    ]
    write_csv(args.output_dir / "per_episode.csv", all_episodes)

    metrics = {}
    paired = {}
    for name in args.rules:
        metrics[name] = {
            "overall": summarize_by_band(episodes_by_rule[name])["all"],
            "by_band": summarize_by_band(episodes_by_rule[name]),
            "stop_events": {
                band: stop_event_summary([
                    row for row in stops_by_rule[name]
                    if band == "all" or str(row["band"]) == band
                ])
                for band in ("all",) + BANDS
            },
            "hazard": {
                band: hazard_from_steps(
                    steps_by_rule[name] if band == "all" else [
                        row for row in steps_by_rule[name]
                        if str(row["band"]) == band
                    ],
                    threshold=rule_from_name(name).threshold,
                )
                for band in ("all",) + BANDS
            },
        }
        if name != b0_name:
            paired[name] = paired_by_band(
                {b0_name: episodes_by_rule[b0_name], name: episodes_by_rule[name]},
                base=b0_name, other=name,
            )
    risk = risk_rows(
        steps_by_rule,
        threshold_map={name: rule_from_name(name).threshold for name in args.rules},
    )
    write_csv(args.output_dir / "risk.csv", risk)
    write_json(
        args.output_dir / "metrics.json",
        {
            "rules": list(args.rules),
            "b0_reference": b0_name,
            "metrics": metrics,
            "paired_vs_b0": paired,
            "risk": risk,
        },
    )
    write_json(
        args.output_dir / "provenance.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "V8.3 Stage-2d R4 Stop-decision rules",
            "command": sys.argv,
            "m0_checkpoint_sha256": m0_sha256,
            "val_specs": str(args.val_specs),
            "val_specs_sha256": file_sha256(args.val_specs),
            "episodes": int(len(specs)),
            "rules": [
                {
                    "name": name,
                    "threshold": rule_from_name(name).threshold,
                    "confirm_k": rule_from_name(name).confirm_k,
                    "confirm_n": rule_from_name(name).confirm_n,
                }
                for name in args.rules
            ],
            "device": device_summary(device),
            "note": (
                "Frozen M0; every rule re-runs the full closed loop; per-episode "
                "decision state reset; Final Test not read."
            ),
        },
    )
    print(json.dumps(
        {
            name: metrics[name]["overall"] for name in args.rules
        }, indent=2,
    ), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
