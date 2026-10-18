"""Run paired V8.2 stop-threshold experiments on held-out V8.3 validation.

No training, no checkpoint writes, no final-test tuning. Use --max-episodes 2
and --models Transformer-M --budgets 50 for a cheap first run.
The threshold=never option forces Movement and measures potential OSR only.
"""
from __future__ import annotations

import argparse
import csv
import json
import types
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

import eval_fixed_mec_epoch_compare as evaluation

ARCHIVE = Path("runs/archive_pycharm_v82_20261018")
VALIDATION_STATS = Path("runs/v83_validation/closedloop_val_specs.json")
OUTPUT_ROOT = Path("runs/v83_validation/threshold_diagnostics")


def parse_threshold(s: str):
    if s == "never":
        return None
    value = float(s)
    if not (0.0 < value < 1.0):
        raise ValueError(f"Threshold outside (0,1): {value}")
    return value


def attach_stop_policy(model, threshold):
    """Only change the predicted action; original model computations unchanged."""
    original = model.predict_action

    def predict_action(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["stop_threshold"] = 0.5 if threshold is None else threshold
        decision = original(*args, **kwargs)
        if threshold is None:
            decision["action"] = decision["move_probabilities"].argmax(dim=-1)
        return decision

    model.predict_action = types.MethodType(predict_action, model)


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", type=Path, default=ARCHIVE)
    p.add_argument("--val-specs", type=Path, default=VALIDATION_STATS)
    p.add_argument("--models", nargs="+", default=["Transformer-M", "Small"])
    p.add_argument("--budgets", nargs="+", type=int, default=[50])
    p.add_argument("--thresholds", nargs="+", default=["0.5", "0.7", "0.9", "never"])
    p.add_argument("--max-episodes", type=int, default=10)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args=p.parse_args()
    if args.max_episodes < 1:
        p.error("--max-episodes must be positive")
    if not args.val_specs.is_file():
        raise SystemExit(f"Validation specs missing: {args.val_specs}; run prepare_closedloop_validation_v83.py")
    document=json.loads(args.val_specs.read_text(encoding="utf-8"))
    if document.get("purpose") != "closedloop-validation-v83" or document.get("split_is_final_test") is not False:
        raise SystemExit("Refusing to tune thresholds on non-validation specs")
    if not (args.archive_root/"shared_encoder.pt").is_file():
        raise SystemExit("shared_encoder.pt absent in archive")
    for name in args.models:
        if name not in evaluation.GBS_MODEL_CONFIGS:
            raise SystemExit(f"Unknown model: {name}")
    if any(b not in (50,100) for b in args.budgets):
        raise SystemExit("Only saved budget 50/100 checkpoints supported")
    thresholds = [parse_threshold(s) for s in args.thresholds]
    for name in args.models:
        for budget in args.budgets:
            ckpt=args.archive_root/"gbs_v82_epoch_compare_results"/f"gbs_{evaluation.safe_name(name)}_best{budget}.pt"
            if not ckpt.is_file():
                raise SystemExit(f"Missing checkpoint: {ckpt}")

    # Existing evaluator looks up its paths at module scope; override only within
    # this process. The archived files are read-only.
    evaluation.TEST_STATS_PATH = args.val_specs
    evaluation.CHECKPOINT_DIR = args.archive_root/"gbs_v82_epoch_compare_results"
    evaluation.NUM_EPISODES = args.max_episodes
    # load_frozen_encoder expects shared_encoder.pt relative to cwd.
    import token_data
    old_loader=token_data.load_frozen_encoder
    def from_archive(device):
        from encoder import VisualTokenEncoder
        encoder=VisualTokenEncoder().to(device)
        encoder.load_state_dict(torch.load(args.archive_root/"shared_encoder.pt",map_location=device,weights_only=True))
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad=False
        return encoder

    from pathlib import Path as _Path
    from config import TRAIN
    import os
    # The evaluator uses a function imported earlier from token_data; replace
    # only its module-level reference, not the global encoder implementation.
    evaluation.load_frozen_encoder=from_archive

    specs = evaluation.load_test_episode_specs()
    oracle_cache={s["episode_index"]:evaluation.build_oracle_reference(s) for s in specs}
    encoder = from_archive(evaluation.DEVICE)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out=args.output_root / stamp
    if out.exists():
        raise SystemExit(f"Output exists: {out}")
    out.mkdir(parents=True,exist_ok=False)
    detail=[]
    summary=[]
    try:
        for budget in args.budgets:
            for name in args.models:
                for threshold in thresholds:
                    label="never" if threshold is None else f"{threshold:.3f}"
                    print(f"Evaluating {name} best{budget}, stop threshold {label}, {len(specs)} validation episodes",flush=True)
                    model = evaluation.load_model(name,budget)
                    attach_stop_policy(model,threshold)
                    episodes=[]
                    for i,spec in enumerate(specs):
                        with torch.inference_mode():
                            result=evaluation.rollout_fixed_model(spec,oracle_cache[spec["episode_index"]],encoder,model)
                        row={"budget":budget,"model":name,"threshold":label,
                             "episode_index":spec["episode_index"],"seed":spec["seed"],
                             "success":int(result["success"]),"osr":int(result["osr"]),
                             "premature_stop":int(result["premature_stop"]),
                             "max_steps_termination":int(result["max_steps_termination"]),
                             "steps":result["steps"],"spl":result["spl"],"ndtw":result["ndtw"],
                             "min_distance_m":result["min_distance_m"]}
                        detail.append(row)
                        episodes.append(row)
                    sr=float(np.mean([x["success"] for x in episodes]))
                    osr=float(np.mean([x["osr"] for x in episodes]))
                    prem=float(np.mean([x["premature_stop"] for x in episodes]))
                    rate=float(np.mean([x["max_steps_termination"] for x in episodes]))
                    item={"budget":budget,"model":name,"threshold":label,"episodes":len(episodes),
                          "sr":sr,"osr":osr,"premature_stop":prem,
                          "timeout":rate,"spl":float(np.mean([x["spl"] for x in episodes])),
                          "ndtw":float(np.mean([x["ndtw"] for x in episodes]))}
                    summary.append(item)
                    print(f"  SR={sr:.2%} OSR={osr:.2%} PremStop={prem:.2%} Timeout={rate:.2%}",flush=True)
                    del model
        _write_csv(out/"per_episode.csv", detail)
        _write_csv(out/"summary.csv", summary)
        (out/"metadata.json").write_text(json.dumps({
            "archive":str(args.archive_root),"validation":str(args.val_specs),
            "thresholds":args.thresholds,"models":args.models,"budgets":args.budgets,
            "requested_max_episodes":args.max_episodes,"actual_episodes":len(specs)
        },indent=2),encoding="utf-8")
        print(f"Results: {out}")
        print("NOTE: threshold=never is an oracle-free Movement counterfactual, NOT a deployable Stop policy.")
    finally:
        pass


if __name__=="__main__":
    main()
