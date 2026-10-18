"""V8.3 visual-spatial diagnostics: localization, yaw temporal alignment, MEC leakage.

No model training.  Frozen shared encoder + Transformer-M Best-50 generate
development-validation trajectories.  The true Goal projection/silhouette is
used ONLY for offline localization metrics and the MEC leakage probe targets;
it is never passed to any model or grounding feature.

Maps compared:
  * old8_up      : the existing [3,8,8] RGB similarity heatmap, max over scales,
                   nearest-upsampled to 24x24 (the resolution the current model
                   actually sees is 8x8; this is a fair localization baseline).
  * baseline24   : old per-patch 3-scale MAX cosine on the 24x24 token grid.
  * coherent24   : new multiscale local-consistency map from visual_grounding_v83.
  * temporal24   : coherent + yaw-registered previous coherent map (turn-only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
import eval_stop_threshold_v83 as stop_eval
from config import AUX, ENCODER, ENV
from encoder import VisualTokenEncoder
from env import ACTION_NAMES, FORWARD, LEFT, RIGHT, STOP, ToyUAVEnv
from gbs_network import build_random_gbs_layout, global_network_state
from mec_geometry_v83 import mec_relative_geometry
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from visual_grounding_v83 import VisualBeliefTracker, yaw_warp_previous


ARCHIVE = Path("runs/archive_pycharm_v82_20261018")
VAL_SPECS = Path("runs/v83_validation/closedloop_val_specs.json")
OUTPUT_ROOT = Path("runs/v83_spatial_prevalidation")
DIST_BINS = (0.0, 20.0, 40.0, 60.0, 100.0, 200.0, float("inf"))
TOPK = 4
LEAK_STRIDE = 5
LEAK_ALPHA = 1.0
LEAK_PERMUTATIONS = 50


def load_frozen_encoder(device):
    encoder = VisualTokenEncoder().to(device)
    encoder.load_state_dict(
        torch.load(ARCHIVE / "shared_encoder.pt", map_location=device, weights_only=True)
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def old_heatmap_24(current_tokens, goal_tokens):
    """Existing 3x8x8 heatmap -> max over scales -> nearest 24x24."""
    heat = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
    old8 = heat.amax(dim=1, keepdim=True)  # [B,1,8,8]
    old24 = F.interpolate(old8.float(), size=(24, 24), mode="nearest")
    return old24[:, 0]


def goal_mask_24(env, projection):
    """Offline Goal silhouette mask at 24x24. Diagnostic only."""
    if projection is None:
        return None
    px, py, _slant, half_size = projection
    hs = max(float(half_size), 1.0)
    H = W = env.img_size
    x0 = max(0, int(math.floor(px - hs - 1)))
    x1 = min(W - 1, int(math.ceil(px + hs + 1)))
    y0 = max(0, int(math.floor(py - hs - 1)))
    y1 = min(H - 1, int(math.ceil(py + hs + 1)))
    if x1 < x0 or y1 < y0:
        return None
    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    local = ToyUAVEnv._shape_mask_local(
        env.goal_landmark.shape,
        (xx - float(px)) / hs,
        (yy - float(py)) / hs,
    )
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y0:y1 + 1, x0:x1 + 1] = local.astype(np.float32)
    pooled = F.adaptive_avg_pool2d(
        torch.from_numpy(mask)[None, None], (ENCODER.grid_size, ENCODER.grid_size)
    )[0, 0].numpy()
    if not np.any(pooled > 0.05):
        return None
    return pooled > 0.05


def center_patch(projection, image_size=ENCODER.image_size, grid=ENCODER.grid_size):
    if projection is None:
        return None
    px, py = float(projection[0]), float(projection[1])
    if not (0.0 <= px < image_size and 0.0 <= py < image_size):
        return None
    cell = image_size / float(grid)
    row = min(max(int(py // cell), 0), grid - 1)
    col = min(max(int(px // cell), 0), grid - 1)
    return row * grid + col


def localization_metrics(map24, center, goal_mask, k=TOPK):
    flat = np.asarray(map24, dtype=np.float64).reshape(-1)
    top = np.argsort(-flat)[:k]
    r1 = float(top[0] == center) if center is not None else float("nan")
    r4 = float(center in set(int(i) for i in top)) if center is not None else float("nan")
    iou = float("nan")
    if goal_mask is not None and np.any(goal_mask):
        candidate = np.zeros(flat.size, dtype=bool)
        candidate[top] = True
        candidate = candidate.reshape(ENCODER.grid_size, ENCODER.grid_size)
        inter = np.logical_and(candidate, goal_mask).sum()
        union = np.logical_or(candidate, goal_mask).sum()
        iou = float(inter / union) if union else float("nan")
    return r1, r4, iou


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def _topk_jaccard(a, b, k=10):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    sa = set(np.argsort(-a)[:k].tolist())
    sb = set(np.argsort(-b)[:k].tolist())
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else float("nan")


def consistency_metrics(a, b):
    return {
        "pearson": _pearson(a, b),
        "cosine": _cosine(a, b),
        "mse": float(np.mean((np.asarray(a) - np.asarray(b)) ** 2)),
        "top10_jaccard": _topk_jaccard(a, b),
    }


def distance_bin(d):
    d = float(d)
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        if lo <= d < hi:
            return f"{int(lo)}-{int(hi)}" if math.isfinite(hi) else f"{int(lo)}+"
    return "unknown"


def _ridge_fit_predict(Xtr, ytr, Xte, alpha=LEAK_ALPHA):
    Xtr = np.asarray(Xtr, np.float64); ytr = np.asarray(ytr, np.float64)
    Xte = np.asarray(Xte, np.float64)
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ym = ytr.mean()
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    b = Xtr.T @ (ytr - ym)
    beta = np.linalg.solve(A, b)
    return (Xte @ beta + ym), ytr, beta


def _r2(y, pred):
    y = np.asarray(y, np.float64); pred = np.asarray(pred, np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def leakage_probe(rows, outdir):
    """Ridge probe on held-out episodes: do MEC features predict Goal bearing/distance?"""
    if not rows:
        return {"note": "no leakage rows"}
    episodes = sorted({int(r["episode_index"]) for r in rows})
    train_eps = set(episodes[::2])
    test_eps = set(episodes[1::2])
    train = [r for r in rows if int(r["episode_index"]) in train_eps]
    test = [r for r in rows if int(r["episode_index"]) in test_eps]
    if not train or not test:
        return {"note": "insufficient episodes for split"}
    feature_sets = {
        "network5": np.asarray([r["network5"] for r in rows], np.float64),
        "bev192": np.asarray([r["bev192"] for r in rows], np.float64),
    }
    idx_tr = np.asarray([i for i, r in enumerate(rows) if int(r["episode_index"]) in train_eps])
    idx_te = np.asarray([i for i, r in enumerate(rows) if int(r["episode_index"]) in test_eps])
    targets = {
        "bearing_sin": np.asarray([r["bearing_sin"] for r in rows], np.float64),
        "bearing_cos": np.asarray([r["bearing_cos"] for r in rows], np.float64),
        "goal_distance": np.asarray([r["goal_distance"] for r in rows], np.float64),
    }
    result = {"train_episodes": sorted(train_eps), "test_episodes": sorted(test_eps),
              "n_train": len(train), "n_test": len(test), "feature_sets": {}}
    rng = np.random.default_rng(20261018)
    for fs, X in feature_sets.items():
        result["feature_sets"][fs] = {}
        for tn, y in targets.items():
            pred, _, _ = _ridge_fit_predict(X[idx_tr], y[idx_tr], X[idx_te])
            actual = _r2(y[idx_te], pred)
            mae = float(np.mean(np.abs(y[idx_te] - pred)))
            perm = []
            for _ in range(LEAK_PERMUTATIONS):
                yp = y[idx_te].copy(); rng.shuffle(yp)
                # Use the same trained mapping; permuting held-out targets tests
                # whether the learned feature mapping is actually aligned.
                perm.append(_r2(yp, pred))
            perm = np.asarray(perm)
            p = float((1 + np.sum(perm >= actual)) / (LEAK_PERMUTATIONS + 1))
            result["feature_sets"][fs][tn] = {
                "r2": actual, "mae": mae,
                "permutation_r2_mean": float(perm.mean()),
                "permutation_r2_p95": float(np.percentile(perm, 95)),
                "permutation_p": p,
            }
    (outdir / "leakage_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def summarize_localization(rows, outdir):
    maps = ("old8_up", "baseline24", "coherent24", "temporal24")
    overall = []
    by_band = []
    by_dist = []
    for name in maps:
        valid = [r for r in rows if r["center_inside"] and math.isfinite(float(r[f"{name}_r1"]))]
        if not valid:
            continue
        overall.append({
            "map": name, "n": len(valid),
            "r1": float(np.mean([r[f"{name}_r1"] for r in valid])),
            "r4": float(np.mean([r[f"{name}_r4"] for r in valid])),
            "iou": float(np.nanmean([r[f"{name}_iou"] for r in valid])),
            "visible_rate": float(np.mean([r["goal_visible"] for r in rows])),
            "center_inside_rate": float(np.mean([r["center_inside"] for r in rows])),
        })
        for group_key, bucket in (("band", by_band), ("dist_bin", by_dist)):
            for key in sorted({r[group_key] for r in valid}):
                sub = [r for r in valid if r[group_key] == key]
                bucket.append({
                    "map": name, group_key: key, "n": len(sub),
                    "r1": float(np.mean([r[f"{name}_r1"] for r in sub])),
                    "r4": float(np.mean([r[f"{name}_r4"] for r in sub])),
                    "iou": float(np.nanmean([r[f"{name}_iou"] for r in sub])),
                })
    _write_csv(outdir / "localization_overall.csv", overall)
    _write_csv(outdir / "localization_by_band.csv", by_band)
    _write_csv(outdir / "localization_by_distance.csv", by_dist)
    return overall, by_band, by_dist


def summarize_temporal(rows, outdir):
    groups = {}
    for r in rows:
        key = (r["policy"], r["turn_type"])
        groups.setdefault(key, []).append(r)
    summary = []
    for (policy, turn_type), sub in sorted(groups.items()):
        warped = [r for r in sub if r["warp_valid"]]
        row = {
            "policy": policy, "turn_type": turn_type, "n": len(sub),
            "unwarped_pearson": float(np.mean([r["unwarped_pearson"] for r in sub])),
            "unwarped_cosine": float(np.mean([r["unwarped_cosine"] for r in sub])),
            "unwarped_mse": float(np.mean([r["unwarped_mse"] for r in sub])),
            "unwarped_top10": float(np.mean([r["unwarped_top10_jaccard"] for r in sub])),
        }
        if warped:
            row.update({
                "warped_pearson": float(np.mean([r["warped_pearson"] for r in warped])),
                "warped_cosine": float(np.mean([r["warped_cosine"] for r in warped])),
                "warped_mse": float(np.mean([r["warped_mse"] for r in warped])),
                "warped_top10": float(np.mean([r["warped_top10_jaccard"] for r in warped])),
                "delta_pearson": float(np.mean([r["warped_pearson"] - r["unwarped_pearson"] for r in warped])),
                "delta_cosine": float(np.mean([r["warped_cosine"] - r["unwarped_cosine"] for r in warped])),
                "win_rate_pearson": float(np.mean([r["warped_pearson"] > r["unwarped_pearson"] for r in warped])),
                "warp_valid_n": len(warped),
            })
        else:
            row.update({"warped_pearson": float("nan"), "warped_cosine": float("nan"),
                        "warped_mse": float("nan"), "warped_top10": float("nan"),
                        "delta_pearson": float("nan"), "delta_cosine": float("nan"),
                        "win_rate_pearson": float("nan"), "warp_valid_n": 0})
        summary.append(row)
    _write_csv(outdir / "temporal_summary.csv", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policies", nargs="+", default=["0.6", "0.9"])
    p.add_argument("--max-episodes", type=int, default=60)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    started = time.time()
    torch.set_grad_enabled(False)
    device = evaluation.DEVICE
    evaluation.CHECKPOINT_DIR = ARCHIVE / "gbs_v82_epoch_compare_results"
    evaluation.TEST_STATS_PATH = VAL_SPECS
    evaluation.NUM_EPISODES = args.max_episodes
    specs = evaluation.load_test_episode_specs()
    encoder = load_frozen_encoder(device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    outdir = args.output_root / ("smoke_" + stamp if args.smoke else stamp)
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"Output: {outdir}", flush=True)

    step_rows = []
    transition_rows = []
    leak_rows = []
    for policy_label in args.policies:
        threshold = stop_eval.parse_threshold(policy_label)
        model = evaluation.load_model("Transformer-M", 50)
        model.eval()
        for p_ in model.parameters():
            p_.requires_grad = False
        stop_eval.attach_stop_policy(model, threshold)
        print(f"Policy {policy_label}: {len(specs)} episodes", flush=True)
        for spec in specs:
            env = ToyUAVEnv()
            obs = env.reset(
                seed=spec["seed"],
                goal_distance_range_m=spec["distance_range"],
                render_observation=True,
            )
            nodes = build_random_gbs_layout(seed=spec["network_layout_seed"], verbose=False)
            goal_tokens = encode_multiscale_goal_tokens(
                env.get_goal_reference_multiscale_rgbs(), encoder, device
            ).unsqueeze(0).to(device, dtype=torch.float32)
            tracker = VisualBeliefTracker()
            prev_coherent = None
            prev_action = None
            prev_xy = None
            prev_network = None
            for step in range(evaluation.MAX_STEPS):
                current_tokens = encoder(image_to_tensor(obs, device))
                heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
                network_np = global_network_state(
                    nodes, env.position,
                    num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
                ).astype(np.float32)
                network_t = torch.from_numpy(network_np).unsqueeze(0).to(device)
                gm = tracker.track(
                    current_tokens, goal_tokens,
                    previous_action=None if prev_action is None else prev_action,
                )
                baseline = gm.baseline[0].detach().cpu().numpy()
                coherent = gm.coherent[0].detach().cpu().numpy()
                temporal = gm.temporal[0].detach().cpu().numpy()
                temporal_valid = bool(gm.temporal_valid[0])
                old24 = old_heatmap_24(current_tokens, goal_tokens)[0].detach().cpu().numpy()
                projection = env.goal_projection()
                center = center_patch(projection)
                gmask = goal_mask_24(env, projection)
                distance = float(env.distance_to_goal)
                dist_bin = distance_bin(distance)
                row = {
                    "policy": policy_label, "episode_index": spec["episode_index"],
                    "seed": spec["seed"], "band": spec["band"], "step": step,
                    "distance_m": distance, "dist_bin": dist_bin,
                    "goal_visible": 1 if projection is not None else 0,
                    "center_inside": 1 if center is not None else 0,
                    "temporal_valid": 1 if temporal_valid else 0,
                }
                for name, m in (("old8_up", old24), ("baseline24", baseline),
                                ("coherent24", coherent), ("temporal24", temporal)):
                    r1, r4, iou = localization_metrics(m, center, gmask)
                    row[f"{name}_r1"] = r1; row[f"{name}_r4"] = r4; row[f"{name}_iou"] = iou
                step_rows.append(row)
                if prev_coherent is not None and prev_action is not None:
                    un = consistency_metrics(prev_coherent, coherent)
                    trans = {
                        "policy": policy_label, "episode_index": spec["episode_index"],
                        "seed": spec["seed"], "band": spec["band"], "step": step,
                        "prev_action": int(prev_action),
                        "action_name": ACTION_NAMES.get(int(prev_action), str(prev_action)),
                        "turn_type": ("left" if int(prev_action) == LEFT else
                                      "right" if int(prev_action) == RIGHT else
                                      "forward" if int(prev_action) == FORWARD else "other"),
                        "distance_m": distance,
                        "unwarped_pearson": un["pearson"],
                        "unwarped_cosine": un["cosine"],
                        "unwarped_mse": un["mse"],
                        "unwarped_top10_jaccard": un["top10_jaccard"],
                        "warp_valid": 0,
                        "warped_pearson": float("nan"),
                        "warped_cosine": float("nan"),
                        "warped_mse": float("nan"),
                        "warped_top10_jaccard": float("nan"),
                    }
                    if int(prev_action) in (LEFT, RIGHT):
                        warped, valid = yaw_warp_previous(
                            torch.from_numpy(prev_coherent)[None].to(device),
                            torch.tensor([int(prev_action)], device=device),
                        )
                        if bool(valid[0]):
                            w = consistency_metrics(warped[0].detach().cpu().numpy(), coherent)
                            trans.update({
                                "warp_valid": 1,
                                "warped_pearson": w["pearson"],
                                "warped_cosine": w["cosine"],
                                "warped_mse": w["mse"],
                                "warped_top10_jaccard": w["top10_jaccard"],
                            })
                    transition_rows.append(trans)
                if step % LEAK_STRIDE == 0:
                    geom = mec_relative_geometry(
                        nodes, env.position, env.yaw, network_np,
                        previous_xy=prev_xy, previous_network_state=prev_network,
                    )
                    dx = float(env.goal[0] - env.position[0])
                    dy = float(env.goal[1] - env.position[1])
                    bearing = math.atan2(dy, dx) - float(env.yaw)
                    leak_rows.append({
                        "policy": policy_label, "episode_index": spec["episode_index"],
                        "seed": spec["seed"], "band": spec["band"], "step": step,
                        "network5": network_np.astype(float).tolist(),
                        "bev192": geom.heatmaps.reshape(-1).astype(float).tolist(),
                        "visible_nodes": int(geom.visible_nodes),
                        "bearing_sin": float(math.sin(bearing)),
                        "bearing_cos": float(math.cos(bearing)),
                        "goal_distance": float(math.hypot(dx, dy)),
                    })
                decision = model.predict_action(current_tokens, goal_tokens, heatmap, network_t)
                action = int(decision["action"].item())
                prev_coherent = coherent
                prev_action = action
                prev_xy = env.position.copy()
                prev_network = network_np
                obs, success, done = env.step(action, render_observation=(action != STOP))
                if done:
                    break
            if (spec["episode_index"] + 1) % 10 == 0:
                print(f"  processed {spec['episode_index'] + 1}/{len(specs)} episodes", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(outdir / "step_metrics.csv", step_rows)
    _write_csv(outdir / "transition_metrics.csv", transition_rows)
    overall, by_band, by_dist = summarize_localization(step_rows, outdir)
    temporal = summarize_temporal(transition_rows, outdir)
    leak = leakage_probe(leak_rows, outdir)
    metadata = {
        "branch": __import__("subprocess").check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "policies": args.policies,
        "episodes": len(specs),
        "steps": len(step_rows),
        "transitions": len(transition_rows),
        "elapsed_s": time.time() - started,
        "note": "Goal projection/silhouette used only for offline metrics; never a model input. No Depth used.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Localization overall:", flush=True)
    for r in overall:
        print(r, flush=True)
    print("Temporal:", flush=True)
    for r in temporal:
        print(r, flush=True)
    print("Leakage probe:", json.dumps(leak, indent=2), flush=True)
    print(f"Total time: {metadata['elapsed_s']:.1f}s", flush=True)
    print(f"Results: {outdir}", flush=True)


if __name__ == "__main__":
    main()
