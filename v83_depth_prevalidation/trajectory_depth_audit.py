"""Offline trajectory Depth-coverage audit for V8.3 pre-validation.

This script rolls out the frozen Transformer-M Best-50 with the frozen shared
RGB encoder on the held-out V8.3 validation tasks.  It records, at every step:

  * true Goal-region Depth coverage (simulator projection; OFFLINE DIAGNOSTIC
    ONLY -- never passed to the model or to a probe feature vector);
  * visual-match-region Depth coverage (Current/Goal token similarity; legal
    MEC-side signal);
  * mismatch flags between the visual-match region and the true Goal region;
  * horizontal distance milestones 60 m / 40 m / 20 m.

The model input remains exactly the existing RGB-only interface: current
tokens, goal tokens, spatial heatmap, network state.  True Goal projection or
mask is computed only after the model decision, for CSV logging.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Allow running as "python v83_depth_prevalidation/trajectory_depth_audit.py".
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
import eval_stop_threshold_v83 as stop_eval
from config import AUX, ENCODER, TRAIN
from encoder import VisualTokenEncoder
from env import ToyUAVEnv
from gbs_model_v3 import goal_current_similarity
from gbs_network import build_random_gbs_layout, global_network_state
from token_data import (
    encode_multiscale_goal_tokens,
    goal_conditioned_spatial_heatmap,
)
from utils import image_to_tensor
from depth_evidence_v83 import goal_conditioned_depth_evidence


ARCHIVE = Path("runs/archive_pycharm_v82_20261018")
VAL_SPECS = Path("runs/v83_validation/closedloop_val_specs.json")
OUTPUT_ROOT = Path("runs/v83_depth_prevalidation")
DISTANCE_RANGES = (60.0, 40.0, 20.0)
TOPK = 12
MATCH_TEMPERATURE = 0.10
GOAL_DEPTH_TOLERANCE_M = 1.0


def load_frozen_encoder(archive_root: Path, device: torch.device):
    encoder = VisualTokenEncoder().to(device)
    state = torch.load(
        archive_root / "shared_encoder.pt",
        map_location=device,
        weights_only=True,
    )
    encoder.load_state_dict(state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def goal_region_mask(env: ToyUAVEnv, projection):
    """Return a bool [H,W] silhouette mask for the true Goal landmark.

    OFFLINE DIAGNOSTIC ONLY.  This function is not called before the model
    decision and its output is never passed into any model or feature vector.
    """
    if projection is None:
        return None, None
    px, py, slant, half_size = projection
    hs = max(float(half_size), 1.0)
    H = W = env.img_size
    x0 = max(0, int(math.floor(px - hs - 1)))
    x1 = min(W - 1, int(math.ceil(px + hs + 1)))
    y0 = max(0, int(math.floor(py - hs - 1)))
    y1 = min(H - 1, int(math.ceil(py + hs + 1)))
    if x1 < x0 or y1 < y0:
        return None, None
    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    local_x = (xx - float(px)) / hs
    local_y = (yy - float(py)) / hs
    local = ToyUAVEnv._shape_mask_local(env.goal_landmark.shape, local_x, local_y)
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = local
    return mask, (float(px), float(py), float(slant), float(hs))


def visual_region_masks(current_tokens, goal_tokens):
    """Return top-k and top-1 patch masks plus the softmax match weights.

    Uses only Current/Goal token similarity: no Goal coordinates or mask.
    """
    similarity = goal_current_similarity(current_tokens, goal_tokens).amax(dim=1)
    k = min(TOPK, int(similarity.shape[1]))
    scores, indices = torch.topk(similarity, k=k, dim=1)
    weights = torch.softmax(scores / MATCH_TEMPERATURE, dim=1)
    grid = ENCODER.grid_size
    patch_mask = torch.zeros_like(similarity, dtype=torch.bool)
    patch_mask.scatter_(1, indices, True)
    top1_patch = torch.zeros_like(similarity, dtype=torch.bool)
    top1_patch.scatter_(1, indices[:, :1], True)
    topk_pixels = (
        patch_mask.view(1, grid, grid)
        .repeat_interleave(ENCODER.patch_size, dim=1)
        .repeat_interleave(ENCODER.patch_size, dim=2)
        .squeeze(0)
        .cpu()
        .numpy()
    )
    top1_pixels = (
        top1_patch.view(1, grid, grid)
        .repeat_interleave(ENCODER.patch_size, dim=1)
        .repeat_interleave(ENCODER.patch_size, dim=2)
        .squeeze(0)
        .cpu()
        .numpy()
    )
    return topk_pixels, top1_pixels, indices, weights


def _fraction(mask_bool):
    if mask_bool is None or mask_bool.size == 0:
        return float("nan")
    return float(mask_bool.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", type=Path, default=ARCHIVE)
    p.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    p.add_argument("--models", nargs="+", default=["Transformer-M"])
    p.add_argument("--budgets", nargs="+", type=int, default=[50])
    p.add_argument("--policies", nargs="+", default=["0.6", "0.9"])
    p.add_argument("--max-episodes", type=int, default=60)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = p.parse_args()

    if not args.val_specs.is_file():
        raise SystemExit(f"Validation specs missing: {args.val_specs}")
    document = json.loads(args.val_specs.read_text(encoding="utf-8"))
    if document.get("purpose") != "closedloop-validation-v83" or document.get(
        "split_is_final_test"
    ) is not False:
        raise SystemExit("Refusing non-validation specs")
    if not (args.archive_root / "shared_encoder.pt").is_file():
        raise SystemExit("shared_encoder.pt absent in archive")

    torch.set_grad_enabled(False)
    evaluation.TEST_STATS_PATH = args.val_specs
    evaluation.CHECKPOINT_DIR = args.archive_root / "gbs_v82_epoch_compare_results"
    evaluation.NUM_EPISODES = args.max_episodes
    specs = evaluation.load_test_episode_specs()
    device = evaluation.DEVICE
    encoder = load_frozen_encoder(args.archive_root, device)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output_root / stamp
    out.mkdir(parents=True, exist_ok=False)
    step_rows = []
    milestone_rows = []
    episode_rows = []

    for model_name in args.models:
        for budget in args.budgets:
            ckpt = (
                args.archive_root
                / "gbs_v82_epoch_compare_results"
                / f"gbs_{evaluation.safe_name(model_name)}_best{budget}.pt"
            )
            if not ckpt.is_file():
                raise SystemExit(f"Missing checkpoint: {ckpt}")
            for policy_label in args.policies:
                threshold = stop_eval.parse_threshold(policy_label)
                model = evaluation.load_model(model_name, budget)
                stop_eval.attach_stop_policy(model, threshold)
                print(
                    f"Auditing {model_name} best{budget}, policy {policy_label}, "
                    f"{len(specs)} episodes",
                    flush=True,
                )
                for spec in specs:
                    env = ToyUAVEnv()
                    obs = env.reset(
                        seed=spec["seed"],
                        goal_distance_range_m=spec["distance_range"],
                        render_observation=True,
                    )
                    network_nodes = build_random_gbs_layout(
                        seed=spec["network_layout_seed"], verbose=False
                    )
                    goal_tokens = encode_multiscale_goal_tokens(
                        env.get_goal_reference_multiscale_rgbs(),
                        encoder,
                        device,
                    ).unsqueeze(0).to(device, dtype=torch.float32)
                    episode_steps = []
                    milestone = {d: None for d in DISTANCE_RANGES}
                    min_distance = float(env.distance_to_goal)
                    ended_by_stop = False
                    success = False
                    done = False
                    for step in range(evaluation.MAX_STEPS):
                        current_tokens = encoder(image_to_tensor(obs, device))
                        heatmap = goal_conditioned_spatial_heatmap(
                            current_tokens, goal_tokens
                        )
                        network_state = torch.from_numpy(
                            global_network_state(
                                network_nodes,
                                env.position,
                                num_tokens=ENCODER.num_tokens,
                                token_dim=ENCODER.token_dim,
                            )
                        ).unsqueeze(0).to(device, dtype=torch.float32)
                        # Model decision uses RGB-only inputs.  No Depth, Goal
                        # coordinates, true distance, or oracle mask is passed.
                        decision = model.predict_action(
                            current_tokens, goal_tokens, heatmap, network_state
                        )
                        action = int(decision["action"].item())

                        # ---------- offline diagnostics from this state ----------
                        horizontal = float(env.distance_to_goal)
                        depth_np = env.render_depth()
                        valid = np.isfinite(depth_np) & (depth_np > 0) & (
                            depth_np <= float(AUX.depth_max_m)
                        )
                        projection = env.goal_projection()
                        target_mask, goal_geom = goal_region_mask(env, projection)
                        topk_mask, top1_mask, indices, weights = visual_region_masks(
                            current_tokens, goal_tokens
                        )
                        # Legal MEC-side depth features (visual-match region only).
                        depth_features = goal_conditioned_depth_evidence(
                            current_tokens,
                            goal_tokens,
                            depth_np,
                            topk=TOPK,
                            match_temperature=MATCH_TEMPERATURE,
                        )[0].detach().cpu().numpy()

                        if target_mask is not None:
                            target_pixels = int(target_mask.sum())
                            target_valid = bool(valid[target_mask].any())
                            target_valid_fraction = float(
                                valid[target_mask].mean()
                            ) if target_pixels else float("nan")
                            target_goal_match_fraction = float(
                                np.mean(
                                    np.abs(depth_np[target_mask] - goal_geom[2])
                                    <= GOAL_DEPTH_TOLERANCE_M
                                )
                            ) if target_pixels else float("nan")
                            target_mean_depth = float(
                                depth_np[target_mask][valid[target_mask]].mean()
                            ) if (target_mask & valid).any() else float("nan")
                        else:
                            target_pixels = 0
                            target_valid = False
                            target_valid_fraction = float("nan")
                            target_goal_match_fraction = float("nan")
                            target_mean_depth = float("nan")

                        visual_valid = bool((valid & topk_mask).any())
                        visual_valid_fraction = float(
                            valid[topk_mask].mean()
                        ) if topk_mask.any() else float("nan")
                        visual_goal_match_fraction = float(
                            np.mean(
                                np.abs(depth_np[topk_mask] - goal_geom[2])
                                <= GOAL_DEPTH_TOLERANCE_M
                            )
                        ) if (topk_mask.any() and goal_geom is not None) else float("nan")
                        if goal_geom is not None:
                            inter = int((topk_mask & target_mask).sum())
                            union = int((topk_mask | target_mask).sum())
                            visual_goal_iou = inter / union if union else float("nan")
                            top1_overlap = float(
                                (top1_mask & target_mask).sum()
                                / max(int(top1_mask.sum()), 1)
                            )
                            ix, iy = int(round(goal_geom[0])), int(round(goal_geom[1]))
                            top1_contains_center = bool(
                                0 <= ix < env.img_size
                                and 0 <= iy < env.img_size
                                and top1_mask[iy, ix]
                            )
                        else:
                            visual_goal_iou = float("nan")
                            top1_overlap = float("nan")
                            top1_contains_center = False

                        visual_depth_from_non_target = bool(
                            visual_valid and goal_geom is not None
                            and visual_goal_iou == 0.0
                        )
                        target_missed_by_visual = bool(
                            target_valid and goal_geom is not None
                            and visual_goal_iou == 0.0
                        )
                        visual_depth_not_goal = bool(
                            visual_valid and goal_geom is not None
                            and visual_goal_match_fraction < 0.5
                        )
                        target_occluded = bool(
                            target_valid and target_goal_match_fraction < 0.5
                        )

                        row = {
                            "model": model_name,
                            "budget": budget,
                            "policy": policy_label,
                            "episode_index": spec["episode_index"],
                            "seed": spec["seed"],
                            "band": spec["band"],
                            "step": step,
                            "action": action,
                            "horizontal_distance_m": horizontal,
                            "goal_visible": projection is not None,
                            "goal_slant_m": goal_geom[2] if goal_geom else float("nan"),
                            "target_region_pixels": target_pixels,
                            "target_depth_available": target_valid,
                            "target_valid_fraction": target_valid_fraction,
                            "target_goal_match_fraction": target_goal_match_fraction,
                            "target_mean_depth_m": target_mean_depth,
                            "visual_depth_available": visual_valid,
                            "visual_valid_fraction": visual_valid_fraction,
                            "visual_matched_validity": float(depth_features[3]),
                            "visual_goal_match_fraction": visual_goal_match_fraction,
                            "visual_goal_iou": visual_goal_iou,
                            "top1_overlap_goal": top1_overlap,
                            "top1_contains_goal_center": top1_contains_center,
                            "visual_depth_from_non_target": visual_depth_from_non_target,
                            "target_missed_by_visual": target_missed_by_visual,
                            "visual_depth_not_goal": visual_depth_not_goal,
                            "target_occluded": target_occluded,
                            "scene_valid_depth_fraction": float(depth_features[7]),
                            "visual_similarity_peak": float(depth_features[5]),
                            "visual_similarity_margin": float(depth_features[6]),
                        }
                        step_rows.append(row)
                        episode_steps.append(row)
                        min_distance = min(min_distance, horizontal)
                        for d in DISTANCE_RANGES:
                            if milestone[d] is None and horizontal <= d:
                                milestone[d] = row

                        ended_by_stop = bool(action == 3)
                        obs, success, done = env.step(
                            action, render_observation=(action != 3)
                        )
                        if done:
                            break

                    final_row = episode_steps[-1]
                    episode_rows.append({
                        "model": model_name,
                        "budget": budget,
                        "policy": policy_label,
                        "episode_index": spec["episode_index"],
                        "seed": spec["seed"],
                        "band": spec["band"],
                        "steps": len(episode_steps),
                        "success": int(success),
                        "ended_by_stop": int(ended_by_stop),
                        "min_distance_m": min_distance,
                        "final_distance_m": final_row["horizontal_distance_m"],
                        "reached_60m": int(milestone[60.0] is not None),
                        "reached_40m": int(milestone[40.0] is not None),
                        "reached_20m": int(milestone[20.0] is not None),
                    })
                    for d in DISTANCE_RANGES:
                        row = milestone[d]
                        if row is None:
                            continue
                        milestone_rows.append({
                            "model": model_name,
                            "budget": budget,
                            "policy": policy_label,
                            "episode_index": spec["episode_index"],
                            "seed": spec["seed"],
                            "band": spec["band"],
                            "range_m": d,
                            "step": row["step"],
                            "horizontal_distance_m": row["horizontal_distance_m"],
                            "goal_visible": row["goal_visible"],
                            "target_depth_available": row["target_depth_available"],
                            "target_valid_fraction": row["target_valid_fraction"],
                            "target_goal_match_fraction": row["target_goal_match_fraction"],
                            "visual_depth_available": row["visual_depth_available"],
                            "visual_valid_fraction": row["visual_valid_fraction"],
                            "visual_matched_validity": row["visual_matched_validity"],
                            "visual_goal_iou": row["visual_goal_iou"],
                            "top1_overlap_goal": row["top1_overlap_goal"],
                            "top1_contains_goal_center": row["top1_contains_goal_center"],
                            "visual_depth_from_non_target": row["visual_depth_from_non_target"],
                            "target_missed_by_visual": row["target_missed_by_visual"],
                            "visual_depth_not_goal": row["visual_depth_not_goal"],
                            "target_occluded": row["target_occluded"],
                        })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def write_csv(path, rows):
        if not rows:
            return
        with path.open("x", encoding="utf-8", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    write_csv(out / "trajectory_steps.csv", step_rows)
    write_csv(out / "milestones.csv", milestone_rows)
    write_csv(out / "episode_summary.csv", episode_rows)
    (out / "metadata.json").write_text(json.dumps({
        "archive": str(args.archive_root),
        "validation": str(args.val_specs),
        "models": args.models,
        "budgets": args.budgets,
        "policies": args.policies,
        "requested_max_episodes": args.max_episodes,
        "actual_episodes": len(specs),
        "distance_ranges_m": list(DISTANCE_RANGES),
        "note": "True Goal-region masks are offline diagnostics only; model input is RGB-only.",
    }, indent=2), encoding="utf-8")
    print(f"Results: {out}")
    print("NOTE: true Goal-region Depth is diagnostic only and was never passed to the model.")


if __name__ == "__main__":
    main()
