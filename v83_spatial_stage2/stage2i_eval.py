"""Stage-2i closed-loop Dev-Val-180 evaluation for one variant.

Uses the best checkpoint from ``stage2i_train.py``.  The movement head selects
Forward/Left/Right; the binary Arrival head gates Stop at a fixed 0.7.  Only
observable inputs are used: frozen RGB tokens, Goal tokens, optional heatmap /
network state, causal history/GAB, and (T5 only) the anonymous global landmark
retrieval graph.

No Final Test file is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV
from env import STOP, ToyUAVEnv, wrap_angle
from gbs_network import build_random_gbs_layout, global_network_state
from navigation_reward import ndtw_score
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from v83_spatial_stage2.stage2d_common import (
    build_oracle_cache, load_dev_specs, VAL_SPECS,
)
from v83_spatial_stage2.stage2f_collect import oracle_move_action, projection_vector
from v83_spatial_stage2.stage2h_gab import GABConfig, GABState
from v83_spatial_stage2.stage2i_common import (
    MODEL_DIRS, MODEL_TITLES, cumulative_table, format_result_block,
    load_progress, model_dir, read_json, reject_final_test, update_progress,
    utc_now,
)
from v83_spatial_stage2.stage2i_data import (
    build_landmark_map, map_retrieval_features, online_history_arrays,
    retrieval_diagnostics, visual_descriptor,
)
from v83_spatial_stage2.stage2i_model import build_stage2i_model
from v83_spatial_training.eval_closedloop_spatial import (
    load_encoder, paired_comparison,
)


BANDS = ("short", "medium", "long")
STOP_THRESHOLD = 0.7
NOISY_RETRIEVAL_LEVELS = (0.05, 0.15)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_checkpoint(model, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return payload


def summarize(episodes):
    if not episodes:
        return {"episodes": 0}
    return {
        "episodes": len(episodes),
        "sr": float(np.mean([row["success"] for row in episodes])),
        "osr": float(np.mean([row["osr"] for row in episodes])),
        "spl": float(np.mean([row["spl"] for row in episodes])),
        "ndtw": float(np.mean([row["ndtw"] for row in episodes])),
        "premature_stop": float(np.mean(
            [row["premature_stop"] for row in episodes])),
        "timeout": float(np.mean(
            [row["max_steps_termination"] for row in episodes])),
        "steps": float(np.mean([row["steps"] for row in episodes])),
        "min_distance_m": float(np.mean(
            [row["min_distance_m"] for row in episodes])),
    }


def noisy_retrieval(goal_index, embeddings, goal_embedding, seed, level):
    rng = np.random.default_rng(int(seed) ^ 0xBEEF)
    scale = float(np.mean(np.linalg.norm(embeddings, axis=1))) * float(level)
    noisy = np.asarray(embeddings) + rng.normal(
        0.0, scale, size=np.asarray(embeddings).shape
    )
    return retrieval_diagnostics(goal_index, noisy, goal_embedding)


def run_variant(args):
    reject_final_test(args.val_specs)
    variant = args.variant
    if variant not in MODEL_DIRS:
        raise SystemExit(f"Unknown variant {variant}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    training_dir = Path(args.model_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        training_dir / "best.pt"
    )
    training_result = read_json(training_dir / "training_result.json", {})
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing checkpoint {checkpoint_path}")
    aux_manifest = read_json(args.aux_root / "manifest.json", {})
    gab_threshold = aux_manifest.get("gab_reliability_threshold")
    if variant in ("T4", "T5") and gab_threshold is None:
        raise SystemExit("Missing GAB reliability threshold in aux manifest")

    model = build_stage2i_model(variant).to(device)
    load_checkpoint(model, checkpoint_path, device)
    encoder = load_encoder(device)

    specs = load_dev_specs(args.val_specs, int(args.max_episodes))
    print(f"{variant}: {len(specs)} Dev Val episodes; "
          f"checkpoint={checkpoint_path}", flush=True)
    print("building oracle references", flush=True)
    oracle_cache = build_oracle_cache(specs)

    output_dir = Path(args.output_dir) if args.output_dir else (
        training_dir / "eval"
    )
    if output_dir.exists() and not (args.resume or args.smoke):
        if (output_dir / "metrics.json").is_file():
            raise SystemExit(
                f"Refusing to overwrite {output_dir}; use --resume"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / f"{variant}_per_episode.csv"
    steps_path = output_dir / f"{variant}_steps.csv"
    existing = {int(row["episode_index"]): row
                for row in read_csv_rows(episodes_path)}
    existing_steps = read_csv_rows(steps_path)
    step_rows = list(existing_steps)

    gab_config = GABConfig(reliability_threshold=float(gab_threshold))
    retrieval_records = []
    retrieval_errors = []
    run_started = datetime.now(timezone.utc)

    for position, spec in enumerate(specs):
        episode_index = int(spec["episode_index"])
        if episode_index in existing and args.resume:
            continue
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
            env.get_goal_reference_multiscale_rgbs(), encoder, device
        ).unsqueeze(0).to(device, dtype=torch.float32)
        goal_token_np = goal_tokens[0, 2].detach().cpu().float().numpy()

        landmark_map = None
        if variant == "T5":
            landmark_map = build_landmark_map(
                env, spec["seed"], encoder, device
            )
            clean = retrieval_diagnostics(
                landmark_map["goal_index"], landmark_map["embeddings"],
                goal_token_np,
            )
            record = {
                "episode_index": episode_index,
                "seed": int(spec["seed"]),
                "band": spec["band"],
                **{f"clean_{key}": value for key, value in clean.items()},
            }
            for level in NOISY_RETRIEVAL_LEVELS:
                noisy = noisy_retrieval(
                    landmark_map["goal_index"], landmark_map["embeddings"],
                    goal_token_np, spec["seed"], level,
                )
                record.update({
                    f"noise{level}_{key}": value
                    for key, value in noisy.items()
                })
                if not noisy["top1"] and len(retrieval_errors) < 50:
                    retrieval_errors.append({
                        "episode_index": episode_index,
                        "level": level,
                        **noisy,
                    })
            retrieval_records.append(record)

        gab_state = GABState(gab_config) if variant in ("T4", "T5") else None
        past = []
        previous_action = -1
        trajectory = [env.position.copy()]
        min_distance = float(env.distance_to_goal)
        osr = bool(env.distance_to_goal < env.stop_radius)
        success = False
        done = False
        ended_by_stop = False
        episode_steps = []

        for step in range(int(ENV.max_steps)):
            pose = env.pose.copy()
            current_tokens = encoder(image_to_tensor(obs, device))
            heatmap = None
            if variant in ("T0", "T2", "T4", "T5"):
                heatmap = goal_conditioned_spatial_heatmap(
                    current_tokens, goal_tokens
                )
            network_state = None
            if variant in ("T0", "T1"):
                network_np = global_network_state(
                    network_nodes, env.position,
                    num_tokens=ENCODER.num_tokens,
                    token_dim=ENCODER.token_dim,
                ).astype(np.float32)
                network_state = torch.from_numpy(network_np).unsqueeze(0).to(device)

            history = (None, None, None, None)
            if variant in ("T4", "T5"):
                history = online_history_arrays(past)
                gab_state.update(
                    heatmap[0].detach().cpu().float().numpy(), previous_action
                )
            map_feature = None
            if variant == "T5":
                map_feature = torch.from_numpy(map_retrieval_features(
                    pose, landmark_map["positions"],
                    landmark_map["embeddings"], goal_token_np,
                )).unsqueeze(0).to(device)

            with torch.inference_mode():
                components = model.forward_components(
                    current_tokens, goal_tokens,
                    heatmap=heatmap if variant in ("T0", "T2") else None,
                    network_state=network_state,
                    history_vis=(
                        torch.from_numpy(history[0]).unsqueeze(0).to(device)
                        if variant in ("T4", "T5") else None
                    ),
                    history_pose=(
                        torch.from_numpy(history[1]).unsqueeze(0).to(device)
                        if variant in ("T4", "T5") else None
                    ),
                    history_action=(
                        torch.from_numpy(history[2]).unsqueeze(0).to(device)
                        if variant in ("T4", "T5") else None
                    ),
                    history_mask=(
                        torch.from_numpy(history[3]).unsqueeze(0).to(device)
                        if variant in ("T4", "T5") else None
                    ),
                    gab=(
                        torch.from_numpy(gab_state.features()).unsqueeze(0).to(device)
                        if variant in ("T4", "T5") else None
                    ),
                    map_feat=map_feature,
                )
                arrival_probability = float(
                    torch.sigmoid(components["arrival_logit"]).item()
                )
                move_action = int(
                    torch.argmax(components["move_logits"], -1).item()
                )

            action = STOP if arrival_probability >= STOP_THRESHOLD else move_action
            oracle_diagnostic = int(oracle_move_action(env))
            projection = projection_vector(env)
            distance = float(env.distance_to_goal)
            episode_steps.append({
                "variant": variant,
                "episode_index": episode_index,
                "seed": int(spec["seed"]),
                "band": spec["band"],
                "step": step,
                "distance_m": distance,
                "goal_visible": int(projection is not None),
                "arrival_probability": arrival_probability,
                "move_action": move_action,
                "action": action,
                "oracle_move_action_diagnostic": oracle_diagnostic,
                "gab_valid": int(gab_state.valid) if gab_state else 0,
                "gab_confidence": float(gab_state.confidence) if gab_state else 0.0,
            })

            visual = visual_descriptor(current_tokens)[0].detach().cpu().float()
            obs, step_success, done = env.step(
                action, render_observation=(action != STOP)
            )
            past.append({
                "pose": pose,
                "next_pose": env.pose.copy(),
                "visual": visual.numpy(),
                "action": int(action),
            })
            previous_action = int(action)
            success = bool(step_success)
            ended_by_stop = bool(action == STOP)
            trajectory.append(env.position.copy())
            min_distance = min(min_distance, float(env.distance_to_goal))
            osr = bool(osr or env.distance_to_goal < env.stop_radius)
            if done:
                break

        trajectory = np.asarray(trajectory, dtype=np.float32)
        path_length = (
            float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
            if len(trajectory) > 1 else 0.0
        )
        oracle_info = oracle_cache[episode_index]
        shortest = float(oracle_info["shortest_distance_m"])
        spl = shortest / max(shortest, path_length, 1e-8) if success else 0.0
        ndtw = ndtw_score(
            trajectory=trajectory[:, :2],
            reference=oracle_info["reference"],
            distance_threshold=env.stop_radius,
        )
        row = {
            "variant": variant,
            "episode_index": episode_index,
            "seed": int(spec["seed"]),
            "band": spec["band"],
            "success": int(success),
            "osr": int(osr),
            "premature_stop": int(bool(ended_by_stop and not success)),
            "max_steps_termination": int(
                bool((not ended_by_stop) and (not success) and done)
            ),
            "steps": int(len(trajectory) - 1),
            "spl": float(spl),
            "ndtw": float(ndtw),
            "min_distance_m": float(min_distance),
            "final_distance_m": float(env.distance_to_goal),
        }
        existing[episode_index] = row
        step_rows.extend(episode_steps)
        ordered = [existing[key] for key in sorted(existing)]
        write_csv(episodes_path, ordered)
        if (position + 1) % 20 == 0 or position + 1 == len(specs):
            write_csv(steps_path, step_rows)
            summary = summarize(ordered)
            print(f"  {variant}: {position + 1}/{len(specs)} episodes | "
                  f"SR={100*summary['sr']:.2f}% | "
                  f"SPL={summary['spl']:.4f}", flush=True)

    write_csv(steps_path, step_rows)
    episodes = [existing[key] for key in sorted(existing)]
    overall = summarize(episodes)
    bands = {
        band: summarize([row for row in episodes if row["band"] == band])
        for band in BANDS
    }

    paired = None
    if variant != "T0":
        t0_rows = read_csv_rows(args.t0_eval_dir / "T0_per_episode.csv") \
            if args.t0_eval_dir else []
        if t0_rows:
            paired = paired_comparison(
                {"T0": t0_rows, variant: episodes}, base="T0", other=variant
            )

    best_epoch = training_result.get("best_epoch", -1) \
        if isinstance(training_result, dict) else -1
    training_time_min = (
        float(training_result.get("training_time_s", 0.0)) / 60.0
        if isinstance(training_result, dict) else 0.0
    )
    map_assisted = variant == "T5"
    if paired:
        conclusion = (
            f"{variant} reaches SR {100*overall['sr']:.2f}% / SPL "
            f"{overall['spl']:.4f} / nDTW {overall['ndtw']:.4f}; paired "
            f"Delta SR vs T0 {100*paired['success']['delta_rate']:+.2f} pp "
            f"(95% CI {100*paired['success']['delta_ci95'][0]:+.2f} to "
            f"{100*paired['success']['delta_ci95'][1]:+.2f}). Single-seed "
            f"Dev Val result; not a final generalisation claim."
        )
    else:
        conclusion = (
            f"{variant}{' (map-assisted)' if map_assisted else ''} reaches "
            f"SR {100*overall['sr']:.2f}% / SPL {overall['spl']:.4f} / nDTW "
            f"{overall['ndtw']:.4f}. Actual single-seed Dev Val result; the "
            f"paired comparison against T0 is pending until T0 finishes."
        )
    overall_with_meta = dict(overall)
    overall_with_meta.update({
        "best_epoch": best_epoch,
        "training_time_min": training_time_min,
        "conclusion": conclusion,
        "map_assisted": map_assisted,
    })

    retrieval_summary = None
    if variant == "T5" and retrieval_records:
        retrieval_summary = {
            "episodes": len(retrieval_records),
            "clean_top1": float(np.mean(
                [row["clean_top1"] for row in retrieval_records])),
            "clean_top5": float(np.mean(
                [row["clean_top5"] for row in retrieval_records])),
            "clean_mean_goal_similarity": float(np.mean(
                [row["clean_goal_similarity"] for row in retrieval_records])),
        }
        for level in NOISY_RETRIEVAL_LEVELS:
            retrieval_summary[f"noise{level}_top1"] = float(np.mean(
                [row[f"noise{level}_top1"] for row in retrieval_records]))
            retrieval_summary[f"noise{level}_top5"] = float(np.mean(
                [row[f"noise{level}_top5"] for row in retrieval_records]))
        write_csv(
            output_dir / "retrieval_episodes.csv", retrieval_records
        )
        write_csv(
            output_dir / "retrieval_error_cases.csv", retrieval_errors
        )

    metrics = {
        "meta": {
            "variant": variant,
            "model": MODEL_TITLES[variant],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "episodes": len(episodes),
            "stop_threshold": STOP_THRESHOLD,
            "band_episodes": {band: bands[band].get("episodes", 0)
                              for band in BANDS},
            "final_test_read": False,
            "device": str(device),
            "eval_seconds": (
                datetime.now(timezone.utc) - run_started
            ).total_seconds(),
        },
        "overall": overall,
        "bands": bands,
        "paired_vs_T0": paired,
        "retrieval": retrieval_summary,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if args.smoke:
        print("SMOKE_ONLY: progress.json and RESULTS.md were not updated",
              flush=True)
        print(format_result_block(
            variant, overall_with_meta, bands,
            {name: False for name in MODEL_DIRS},
        ), flush=True)
        return metrics
    update_progress(
        variant,
        eval_status="evaluated",
        eval_finished_utc=utc_now(),
        eval_output_dir=str(output_dir),
        metrics=overall_with_meta,
        bands=bands,
        paired_vs_T0=paired,
        retrieval=retrieval_summary,
    )
    progress = load_progress()
    table = cumulative_table(progress)
    (Path("runs/v83_spatial_stage2/stage2i") / "RESULTS.md").write_text(
        "# Stage-2i cumulative results\n\n" + table + "\n",
        encoding="utf-8",
    )
    completed = {
        name: bool(progress["models"].get(name, {}).get("metrics"))
        for name in MODEL_DIRS
    }
    print(format_result_block(variant, overall_with_meta, bands, completed),
          flush=True)
    if paired:
        print("\nPaired vs T0 (per-episode, 180):", flush=True)
        for key in ("success", "osr", "premature_stop",
                    "max_steps_termination"):
            value = paired[key]
            print(f"  {key}: {value['other_better']}W/"
                  f"{value['base_better']}L/{value['ties']}T | "
                  f"delta={100*value['delta_rate']:+.2f} pp | "
                  f"CI {100*value['delta_ci95'][0]:+.2f} to "
                  f"{100*value['delta_ci95'][1]:+.2f}", flush=True)
        for key in ("spl", "ndtw"):
            print(f"  {key}: delta={paired[key]['delta']:+.4f} | "
                  f"CI {paired[f'{key}_delta_ci95'][0]:+.4f} to "
                  f"{paired[f'{key}_delta_ci95'][1]:+.4f}", flush=True)
    if retrieval_summary:
        print("\nT5 anonymous-map retrieval:", flush=True)
        print(json.dumps(retrieval_summary, indent=2), flush=True)
    print("\n" + table, flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=list(MODEL_DIRS))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--aux-root", type=Path,
                        default=Path("runs/v83_spatial_stage2/stage2i/aux"))
    parser.add_argument("--t0-eval-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = model_dir(args.variant)
    if args.t0_eval_dir is None:
        args.t0_eval_dir = model_dir("T0") / "eval"
    run_variant(args)


if __name__ == "__main__":
    main()
