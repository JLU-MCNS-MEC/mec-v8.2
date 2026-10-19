"""Stage-2c Arrival calibration diagnostics for B0/B1/B2.

B0 = original Stage-2a M0 Arrival Head (no training).
B1 = M0 head retrained on the original 3% far-negative mask with plain BCE.
B2 = M0 head retrained on the 10% far-negative mask with plain BCE.

This script is diagnostic only. It does not train or modify any weight. It
answers two questions left open by the B0/B1/B2 closed-loop result:

1. How are B0/B1/B2 calibrated on the frozen Dev Val 180 Arrival samples,
   overall, by band, and by distance / goal visibility?
2. Do the far-distance states visited by the *actual* closed-loop policy look
   like the far negatives used for head training? Per-step arrival probability,
   goal visibility (offline simulator diagnostic only, never an input), and the
   eight interpretable Arrival match features are compared between the offline
   Dev Val distribution and on-policy rollout states.

No Final Test data is read. Goal visibility / true distance are used only to
label diagnostic rows after the model decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_fixed_mec_epoch_compare as evaluation
from config import ARRIVAL_STOP_THRESHOLD, ENCODER
from env import STOP, ToyUAVEnv
from gbs_model_v3 import build_gbs_model
from gbs_network import build_random_gbs_layout, global_network_state
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from v83_spatial_training.eval_closedloop_spatial import (
    load_encoder,
    paired_bootstrap_ci,
    paired_comparison,
)
from v83_spatial_training.spatial_mec import SpatialAwareMEC
from v83_spatial_stage2.stage2b_arrival_ablation import (
    M0_CKPT,
    _load_m0_and_head,
    _hash_state,
)


STAGE2_ROOT = Path("runs/v83_spatial_stage2")
STAGE2A_EVAL = STAGE2_ROOT / "stage2a_eval_20261018T070139Z"
STAGE2B_FEATURES = STAGE2_ROOT / "stage2b/arrival_20261018T150031Z/features"
STAGE2C = STAGE2_ROOT / "stage2c/20261018T164111Z"
DATASETS = STAGE2_ROOT / "datasets"
VAL_SPECS = STAGE2_ROOT / "prep_audit_20261018T043420Z/dev_val_180_spec.json"
BANDS = ("short", "medium", "long")
BAND_ID = {"short": 0, "medium": 1, "long": 2}
DISTANCE_BINS = (
    (0.0, 20.0),
    (20.0, 40.0),
    (40.0, 60.0),
    (60.0, 100.0),
    (100.0, 200.0),
    (200.0, float("inf")),
)
MATCH_FEATURE_NAMES = (
    "max_match", "topk_mean", "mean_match", "entropy",
    "weighted_x", "weighted_y", "best_x", "best_y",
)


def _bin_name(lo, hi):
    return f"{int(lo)}_{int(hi) if np.isfinite(hi) else 'inf'}m"


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def _median(values):
    return float(np.median(values)) if values else float("nan")


def load_val_features():
    """Load cached Dev Val features plus metadata including offline visibility."""
    cached = np.load(STAGE2B_FEATURES / "dev_val.npz")
    data = {k: cached[k] for k in cached.files}
    goal_visible, band_id, episode_id, distance = [], [], [], []
    for path in sorted((DATASETS / "dev_val_180").glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=True)
        goal_visible.append(shard["goal_visible"].numpy())
        band_id.append(shard["trajectory_band_ids"].numpy())
        episode_id.append(shard["episode_ids"].numpy())
        distance.append(shard["distance_to_goal_m"].numpy())
    goal_visible = np.concatenate(goal_visible).astype(np.int64)
    band_id = np.concatenate(band_id).astype(np.int64)
    episode_id = np.concatenate(episode_id).astype(np.int64)
    distance = np.concatenate(distance).astype(np.float32)
    if len(goal_visible) != len(data["features"]):
        raise RuntimeError(
            f"Dev Val metadata length {len(goal_visible)} != features {len(data['features'])}"
        )
    if not np.array_equal(episode_id, data["episode_id"]):
        raise RuntimeError("Dev Val episode ids from shards do not align with cached features")
    if not np.allclose(distance, data["distance"], atol=1e-4):
        raise RuntimeError("Dev Val distances from shards do not align with cached features")
    data["goal_visible"] = goal_visible
    data["band_id"] = band_id
    return data


def load_variant_heads(device):
    """Load B0/B1/B2 heads and full checkpoints; verify shared movement hash."""
    _, m0_state, b0_head = _load_m0_and_head(device)
    variants = {
        "B0": {"head": b0_head, "checkpoint": M0_CKPT, "state": m0_state},
    }
    movement_hash = _hash_state(
        {k: v for k, v in m0_state.items() if not k.startswith("base.arrival_head.")}
    )
    for name in ("B1", "B2"):
        ckpt = STAGE2C / name / "best.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        other_hash = _hash_state(
            {k: v for k, v in state.items() if not k.startswith("base.arrival_head.")}
        )
        if other_hash != movement_hash:
            raise RuntimeError(f"{name} movement weights differ from B0")
        model = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0").to(device)
        model.load_state_dict(state)
        model.eval()
        variants[name] = {"head": model.base.arrival_head, "checkpoint": ckpt, "state": state}
    variants["B0"]["head"].eval()
    return variants, movement_hash


def _probabilities(features, head, device, batch_size=512):
    out = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).float().to(device)
            out.append(torch.sigmoid(head.net(batch).squeeze(-1)).cpu().numpy())
    return np.concatenate(out)


def _binary_metrics(y, p, threshold=0.5):
    pred = p >= threshold
    y = y.astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = float(np.mean((p - y.astype(float)) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < 9 else p <= hi)
        if m.any():
            ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return {
        "n": int(len(y)), "positives": int(y.sum()), "negatives": int((~y).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "brier": brier, "ece": float(ece),
    }


def _distance_bin_rows(y, p, distance, goal_visible, band_id, source, threshold):
    rows = []
    for band in ("all",) + BANDS:
        for lo, hi in DISTANCE_BINS:
            mask = (distance >= lo) & (distance < hi)
            if band != "all":
                mask = mask & (band_id == BAND_ID[band])
            for visible in ("all", "visible", "occluded"):
                m = mask.copy()
                if visible == "visible":
                    m &= goal_visible.astype(bool)
                elif visible == "occluded":
                    m &= ~goal_visible.astype(bool)
                if not m.any():
                    continue
                pred = p[m] >= threshold
                rows.append({
                    "source": source, "band": band, "distance_bin": _bin_name(lo, hi),
                    "visibility": visible, "n": int(m.sum()),
                    "positive_rate": float(y[m].mean()),
                    "mean_prob": float(p[m].mean()),
                    "median_prob": float(np.median(p[m])),
                    "pred_positive_rate": float(pred.mean()),
                    "brier": float(np.mean((p[m] - y[m].astype(float)) ** 2)),
                })
    return rows


def offline_analysis(data, variants, device, threshold):
    labels = data["arrival_label"].astype(np.float32)
    sample_type = data["sample_type"]
    distance = data["distance"]
    band_id = data["band_id"]
    goal_visible = data["goal_visible"]
    natural = sample_type == 0
    hard = ~natural
    result = {}
    bin_rows = []
    feature_stats = []
    for name, variant in variants.items():
        p = _probabilities(data["features"], variant["head"], device)
        natural_metrics = _binary_metrics(labels[natural], p[natural], threshold)
        hard_metrics = _binary_metrics(labels[hard], p[hard], threshold)
        far40 = natural & (distance >= 40.0)
        far100 = natural & (distance >= 100.0)
        near_pos = natural & (labels >= 0.5)
        per_band = {}
        for band in BANDS:
            m = band_id == BAND_ID[band]
            per_band[band] = {
                "natural": _binary_metrics(labels[m & natural], p[m & natural], threshold),
                "hard": _binary_metrics(labels[m & hard], p[m & hard], threshold),
            }
        hard_by_type = {}
        for type_id, type_name in ((1, "hard_positive"), (2, "boundary_negative"),
                                    (3, "distractor_negative")):
            m = sample_type == type_id
            hard_by_type[type_name] = _binary_metrics(labels[m], p[m], threshold)
        result[name] = {
            "checkpoint": str(variant["checkpoint"]),
            "natural": natural_metrics,
            "hard": hard_metrics,
            "hard_by_type": hard_by_type,
            "per_band": per_band,
            "far40_false_stop_rate": float((p[far40] >= threshold).mean()) if far40.any() else 0.0,
            "far100_false_stop_rate": float((p[far100] >= threshold).mean()) if far100.any() else 0.0,
            "near20_miss_stop_rate": float((p[near_pos] < threshold).mean()) if near_pos.any() else 0.0,
            "natural_positive_prob_mean": float(p[natural & (labels >= 0.5)].mean()),
            "natural_negative_prob_mean": float(p[natural & (labels < 0.5)].mean()),
            "far_negative_prob_mean": float(p[natural & (labels < 0.5) & (distance >= 40.0)].mean()),
            "hard_positive_prob_mean": float(p[hard & (labels >= 0.5)].mean()),
            "hard_negative_prob_mean": float(p[hard & (labels < 0.5)].mean()),
        }
        rows = _distance_bin_rows(
            labels[natural], p[natural], distance[natural], goal_visible[natural],
            band_id[natural], source=f"offline_{name}", threshold=threshold,
        )
        for row in rows:
            row["variant"] = name
        bin_rows.extend(rows)
        # Interpretable feature means for natural negatives, by distance bin.
        feats = data["features"]
        for lo, hi in DISTANCE_BINS:
            m = natural & (labels < 0.5) & (distance >= lo) & (distance < hi)
            if not m.any():
                continue
            sub = feats[m]
            feature_stats.append({
                "source": f"offline_{name}", "variant": name,
                "distance_bin": _bin_name(lo, hi), "n": int(m.sum()),
                **{f"mean_{n}": float(sub[:, i].mean()) for i, n in enumerate(MATCH_FEATURE_NAMES)},
                "mean_heat_l2": float(np.linalg.norm(sub[:, 8:72], axis=1).mean()),
                "mean_net_l2": float(np.linalg.norm(sub[:, 72:104], axis=1).mean()),
                "mean_prob": float(p[m].mean()),
                "pred_positive_rate": float((p[m] >= threshold).mean()),
            })
    return result, bin_rows, feature_stats


def run_episode(model, variant, spec, encoder, device, threshold):
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
    prev_action = None
    steps = []
    stop_events = []
    min_distance = float(env.distance_to_goal)
    osr = bool(env.distance_to_goal < env.stop_radius)
    success = False
    done = False
    ended_by_stop = False
    for step in range(evaluation.MAX_STEPS):
        current_tokens = encoder(image_to_tensor(obs, device))
        heatmap = goal_conditioned_spatial_heatmap(current_tokens, goal_tokens)
        network_np = global_network_state(
            nodes, env.position,
            num_tokens=ENCODER.num_tokens, token_dim=ENCODER.token_dim,
        ).astype(np.float32)
        network = torch.from_numpy(network_np).unsqueeze(0).to(device)
        with torch.inference_mode():
            decision = model.predict_action(
                current_tokens, goal_tokens, heatmap, network,
                stop_threshold=threshold,
            )
        distance = float(env.distance_to_goal)
        projection = env.goal_projection()
        goal_visible = int(projection is not None)
        arrival_p = float(decision["arrival_probability"].item())
        action = int(decision["action"].item())
        features = decision["arrival_features"][0].detach().cpu().numpy().astype(np.float32)
        steps.append({
            "variant": variant, "episode_index": spec["episode_index"],
            "seed": spec["seed"], "band": spec["band"], "step": step,
            "distance_m": distance, "arrival_probability": arrival_p,
            "goal_visible": goal_visible, "stop": int(action == STOP),
            "move_action": int(action),
        })
        if action == STOP:
            stop_events.append({
                "variant": variant, "episode_index": spec["episode_index"],
                "seed": spec["seed"], "band": spec["band"], "step": step,
                "distance_m": distance, "arrival_probability": arrival_p,
                "goal_visible": goal_visible,
            })
        obs, step_success, done = env.step(
            action, render_observation=(action != STOP)
        )
        success = bool(step_success)
        ended_by_stop = bool(action == STOP)
        min_distance = min(min_distance, float(env.distance_to_goal))
        osr = bool(osr or env.distance_to_goal < env.stop_radius)
        prev_action = action
        if done:
            break
    episode = {
        "variant": variant, "episode_index": spec["episode_index"],
        "seed": spec["seed"], "band": spec["band"],
        "success": int(success), "osr": int(osr),
        "premature_stop": int(ended_by_stop and not success),
        "max_steps_termination": int((not ended_by_stop) and (not success) and done),
        "steps": len(steps), "min_distance_m": min_distance,
        "final_distance_m": float(env.distance_to_goal),
        "last_arrival_probability": steps[-1]["arrival_probability"],
    }
    return episode, steps, stop_events


def rollout_analysis(variants, specs, encoder, device, threshold):
    episodes, steps, stops = [], [], []
    for name, variant in variants.items():
        model = SpatialAwareMEC(build_gbs_model("Transformer-M"), "M0").to(device)
        model.load_state_dict(variant["state"])
        model.eval()
        print(f"Rollout {name} from {variant['checkpoint']}", flush=True)
        for spec in specs:
            episode, ep_steps, ep_stops = run_episode(
                model, name, spec, encoder, device, threshold
            )
            episodes.append(episode)
            steps.extend(ep_steps)
            stops.extend(ep_stops)
            if (spec["episode_index"] + 1) % 30 == 0:
                print(f"  {name} {spec['episode_index'] + 1}/{len(specs)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return episodes, steps, stops


def summarize_stops(stops):
    out = []
    for variant in ("B0", "B1", "B2"):
        for band in ("all",) + BANDS:
            sub = [e for e in stops if e["variant"] == variant and
                   (band == "all" or e["band"] == band)]
            if not sub:
                continue
            distances = [e["distance_m"] for e in sub]
            probs = [e["arrival_probability"] for e in sub]
            out.append({
                "variant": variant, "band": band, "stops": len(sub),
                "premature_gt20": sum(d > 20.0 for d in distances),
                "far_gt40": sum(d > 40.0 for d in distances),
                "far_gt100": sum(d > 100.0 for d in distances),
                "distance_mean_m": _mean(distances),
                "distance_median_m": _median(distances),
                "distance_min_m": min(distances),
                "distance_max_m": max(distances),
                "arrival_p_mean": _mean(probs),
                "arrival_p_median": _median(probs),
                "goal_visible_rate": _mean([e["goal_visible"] for e in sub]),
            })
    return out


def summarize_rollout_bins(steps, threshold):
    rows = []
    for variant in ("B0", "B1", "B2"):
        for band in ("all",) + BANDS:
            for lo, hi in DISTANCE_BINS:
                m = [
                    s for s in steps
                    if s["variant"] == variant
                    and (band == "all" or s["band"] == band)
                    and lo <= s["distance_m"] < hi
                ]
                if not m:
                    continue
                p = np.asarray([s["arrival_probability"] for s in m])
                vis = np.asarray([s["goal_visible"] for s in m], dtype=bool)
                for visibility in ("all", "visible", "occluded"):
                    mm = np.ones(len(m), dtype=bool) if visibility == "all" else (
                        vis if visibility == "visible" else ~vis
                    )
                    if not mm.any():
                        continue
                    rows.append({
                        "source": f"rollout_{variant}", "variant": variant,
                        "band": band, "distance_bin": _bin_name(lo, hi),
                        "visibility": visibility, "n": int(mm.sum()),
                        "mean_prob": float(p[mm].mean()),
                        "median_prob": float(np.median(p[mm])),
                        "pred_stop_rate": float((p[mm] >= threshold).mean()),
                    })
    return rows


def summarize_episodes(episodes):
    out = []
    for variant in ("B0", "B1", "B2"):
        for band in ("all",) + BANDS:
            sub = [e for e in episodes if e["variant"] == variant and
                   (band == "all" or e["band"] == band)]
            if not sub:
                continue
            out.append({
                "variant": variant, "band": band, "episodes": len(sub),
                "sr": _mean([e["success"] for e in sub]),
                "osr": _mean([e["osr"] for e in sub]),
                "premature_stop": _mean([e["premature_stop"] for e in sub]),
                "timeout": _mean([e["max_steps_termination"] for e in sub]),
                "mean_steps": _mean([e["steps"] for e in sub]),
                "mean_min_distance_m": _mean([e["min_distance_m"] for e in sub]),
            })
    return out


def _load_episode_rows(path, mode=None):
    rows = list(csv.DictReader(open(path)))
    if mode is not None:
        rows = [r for r in rows if r.get("mode") == mode]
    return rows


def _paired_table(rows_by_variant):
    out = []
    metric_keys = ("success", "osr", "premature_stop", "max_steps_termination")
    continuous = ("spl", "ndtw", "steps", "min_distance_m")
    for band in ("all",) + BANDS:
        subsets = {
            name: [r for r in rows if band == "all" or r["band"] == band]
            for name, rows in rows_by_variant.items()
        }
        for other in ("B1", "B2"):
            paired = paired_comparison(subsets, base="B0", other=other)
            for metric in metric_keys:
                item = paired[metric]
                out.append({
                    "band": band, "comparison": f"{other}_vs_B0", "metric": metric,
                    "other_better": item["other_better"], "base_better": item["base_better"],
                    "ties": item["ties"], "delta": item["delta_rate"],
                    "ci_low": item["delta_ci95"][0], "ci_high": item["delta_ci95"][1],
                })
            for metric in continuous:
                if metric == "steps":
                    a = {int(r["episode_index"]): r for r in subsets["B0"]}
                    b = {int(r["episode_index"]): r for r in subsets[other]}
                    diffs = [
                        int(b[i]["steps"]) - int(a[i]["steps"]) for i in sorted(set(a) & set(b))
                    ]
                    delta = float(np.mean(diffs)); ci = paired_bootstrap_ci(diffs)
                else:
                    delta = paired[f"{metric}_delta"]
                    ci = paired[f"{metric}_delta_ci95"]
                out.append({
                    "band": band, "comparison": f"{other}_vs_B0", "metric": metric,
                    "other_better": None, "base_better": None, "ties": None,
                    "delta": delta, "ci_low": ci[0], "ci_high": ci[1],
                })
    return out


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    p.add_argument("--max-episodes", type=int, default=180)
    p.add_argument("--threshold", type=float, default=ARRIVAL_STOP_THRESHOLD)
    p.add_argument("--skip-rollout", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.max_episodes > 180:
        raise SystemExit("Dev Val 180 is the maximum; Final Test is sealed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    data = load_val_features()
    variants, movement_hash = load_variant_heads(device)
    offline, offline_bins, feature_stats = offline_analysis(
        data, variants, device, args.threshold
    )
    (args.output_dir / "offline_arrival_metrics.json").write_text(
        json.dumps({
            "threshold": args.threshold,
            "movement_hash": movement_hash,
            "variants": offline,
        }, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_dir / "offline_distance_bins.csv", offline_bins)
    _write_csv(args.output_dir / "offline_feature_stats.csv", feature_stats)

    if not args.skip_rollout:
        evaluation.TEST_STATS_PATH = args.val_specs
        evaluation.NUM_EPISODES = args.max_episodes
        specs = evaluation.load_test_episode_specs()
        encoder = load_encoder(device)
        episodes, steps, stops = rollout_analysis(
            variants, specs, encoder, device, args.threshold
        )
        rollout_bins = summarize_rollout_bins(steps, args.threshold)
        episode_summary = summarize_episodes(episodes)
        stop_summary = summarize_stops(stops)
        _write_csv(args.output_dir / "rollout_per_episode.csv", episodes)
        _write_csv(args.output_dir / "rollout_steps.csv", steps)
        _write_csv(args.output_dir / "rollout_stop_events.csv", stops)
        _write_csv(args.output_dir / "rollout_distance_bins.csv", rollout_bins)
        _write_csv(args.output_dir / "rollout_episode_summary.csv", episode_summary)
        _write_csv(args.output_dir / "rollout_stop_summary.csv", stop_summary)

        # Reuse the archived closed-loop rows for the official paired table and
        # verify the replay reproduces B0 exactly.
        archived = {
            "B0": _load_episode_rows(STAGE2A_EVAL / "per_episode.csv", mode="M0"),
            "B1": _load_episode_rows(STAGE2C / "B1_per_episode.csv"),
            "B2": _load_episode_rows(STAGE2C / "B2_per_episode.csv"),
        }
        replay = {name: [e for e in episodes if e["variant"] == name] for name in archived}
        replay_check = {}
        for name in archived:
            a = {int(r["episode_index"]): r for r in archived[name]}
            b = {int(r["episode_index"]): r for r in replay[name]}
            diffs = {
                key: sum(int(a[i][key]) != int(b[i][key]) for i in sorted(a))
                for key in ("success", "osr", "premature_stop", "max_steps_termination")
            }
            replay_check[name] = {"episodes": len(a), "mismatches": diffs}
        paired_table = _paired_table(archived)
        _write_csv(args.output_dir / "paired_vs_B0.csv", paired_table)
        result = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "threshold": args.threshold,
            "episodes": len(specs),
            "movement_hash": movement_hash,
            "replay_check": replay_check,
            "offline_arrival_metrics": offline,
            "rollout_episode_summary": episode_summary,
            "rollout_stop_summary": stop_summary,
            "rollout_distance_bins": rollout_bins,
            "note": (
                "Goal visibility and true distance are offline diagnostic labels only; "
                "they are never model inputs. Dev Val 180 only; Final Test sealed."
            ),
        }
        (args.output_dir / "stage2c_diagnostics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps({"replay_check": replay_check}, indent=2), flush=True)
        print(json.dumps({"rollout_episode_summary": episode_summary}, indent=2), flush=True)
        print(json.dumps({"rollout_stop_summary": stop_summary}, indent=2), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
