"""Stage-2e: M1@0.7 closed loop with the SpatialFusionHead residual disabled.

This isolates the contribution of the learned fusion residual *inside the
current M1 checkpoint*. The rest of the M1 network (base movement trunk,
arrival head, jointly trained weights) is unchanged. It is therefore NOT an
independently trained M0 control: M1 base weights differ from M0 weights even
when the residual is zero.

No training, no checkpoint modification, Final Test 300 never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2d_common import (
    BANDS,
    build_oracle_cache,
    device_summary,
    file_sha256,
    hazard_from_steps,
    load_dev_specs,
    load_encoder,
    stop_event_summary,
    write_csv,
    write_json,
)
from v83_spatial_stage2.stage2d_m1_threshold_eval import (
    M1_CKPT,
    evaluate,
    summarize_with_counts,
)
from v83_spatial_training.spatial_mec import SpatialAwareMEC
from gbs_model_v3 import build_gbs_model


class ZeroFusion(nn.Module):
    """Drop-in replacement that emits an exact zero 3-way residual."""

    def forward(self, spatial_maps):
        return spatial_maps.new_zeros(spatial_maps.shape[0], 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_episodes = min(args.max_episodes, 6)
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "steps").mkdir()
    (args.output_dir / "stops").mkdir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = load_dev_specs(max_episodes=args.max_episodes)
    encoder = load_encoder(device)
    oracle_cache = build_oracle_cache(specs)
    model = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M1").to(device)
    state = torch.load(M1_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    fusion_parameter_count = int(sum(p.numel() for p in model.fusion.parameters()))
    model.fusion = ZeroFusion().to(device)
    model.eval()
    print(
        f"M1 checkpoint {M1_CKPT}; fusion residual disabled "
        f"({fusion_parameter_count} parameters ignored); {len(specs)} episodes",
        flush=True,
    )

    episodes, steps, stops = evaluate(
        model, specs, oracle_cache, encoder, device, 0.7
    )
    write_csv(args.output_dir / "M1-no-fusion@0.7_per_episode.csv", episodes)
    write_csv(args.output_dir / "steps/M1-no-fusion@0.7.csv", steps)
    write_csv(args.output_dir / "stops/M1-no-fusion@0.7.csv", stops)

    metrics = {
        "rule": "M1-no-fusion@0.7",
        "summaries": summarize_with_counts(episodes),
        "hazard": {
            band: hazard_from_steps(
                steps if band == "all" else [
                    row for row in steps if str(row["band"]) == band
                ],
                threshold=0.7,
            )
            for band in ("all",) + BANDS
        },
        "stop_events": {
            band: stop_event_summary([
                row for row in stops
                if band == "all" or str(row["band"]) == band
            ])
            for band in ("all",) + BANDS
        },
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "V8.3 Stage-2e M1 fusion residual OFF (eval only)",
        "command": sys.argv,
        "m1_checkpoint": str(M1_CKPT),
        "m1_checkpoint_sha256": file_sha256(M1_CKPT),
        "fusion_parameters_disabled": fusion_parameter_count,
        "fusion_replacement": "exact zero residual; base movement logits used",
        "threshold": 0.7,
        "episodes": int(len(specs)),
        "training": "none; frozen Stage-2a M1 checkpoint only",
        "interpretation": (
            "Isolates the fusion residual inside the M1 checkpoint. The base "
            "weights and arrival head are M1's, not M0's; this is not an "
            "independent M0 control."
        ),
        "final_test": "sealed; not read",
        "device": device_summary(device),
    })
    print(json.dumps({
        "overall": metrics["summaries"]["all"],
        "by_band": {
            band: metrics["summaries"][band] for band in BANDS
        },
    }, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
