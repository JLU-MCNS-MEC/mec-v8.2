"""Stage-2j closed-loop evaluation.

Two modes:

* ``--split dev_val``: Dev Val 180, used only for the Transformer-M
  checkpoint-equivalence reproduction and optional diagnostics.  Final Test
  paths remain rejected.
* ``--split final_test --allow-final-test``: the one formal Final Test 300
  evaluation per model.  Requires a frozen protocol file, verifies every
  SHA256 in ``SEALED.json``, derives the task list from the sealed token
  shards, and never updates weights or the 0.7 Stop threshold.

The movement head selects Forward/Left/Right; the binary Arrival head gates
Stop at 0.7.  All five MECs receive the identical T5 input set: Current/Goal
tokens, 16-step visual history, relative pose + action history, 20-D GAB,
and the 22-D anonymous landmark graph feature.  True distance, target
coordinates, labels and oracle actions appear only in evaluation records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENCODER, ENV, TRAIN
from env import STOP, ToyUAVEnv
from navigation_reward import ndtw_score
from token_data import encode_multiscale_goal_tokens, goal_conditioned_spatial_heatmap
from utils import image_to_tensor
from v83_spatial_stage2.stage2d_common import (
    build_oracle_cache,
    load_dev_specs,
    VAL_SPECS,
)
from v83_spatial_stage2.stage2f_collect import (
    oracle_move_action,
    projection_vector,
)
from v83_spatial_stage2.stage2h_gab import GABConfig, GABState
from v83_spatial_stage2.stage2i_data import (
    build_landmark_map,
    map_retrieval_features,
    online_history_arrays,
    retrieval_diagnostics,
    visual_descriptor,
)
from v83_spatial_stage2.stage2j_common import (
    AUX_ROOT,
    BAND_EPISODES,
    BANDS,
    EVAL_PROTOCOL,
    FINAL_TEST_DIR,
    FREEZE_PATH,
    MODEL_DIRS,
    MODEL_TITLES,
    SPECS_DIR,
    STOP_THRESHOLD,
    TEST_EPISODES,
    append_results_snapshot,
    atomic_write_json,
    cumulative_table,
    file_sha256,
    format_result_block,
    latency_summary,
    load_progress,
    model_dir,
    read_json,
    reject_final_test,
    update_progress,
    utc_now,
)
from v83_spatial_stage2.stage2j_model import (
    STAGE2J_CONFIGS,
    build_stage2j_model,
    parameter_count,
)
from v83_spatial_training.eval_closedloop_spatial import load_encoder


BAND_ID_TO_NAME = {0: "short", 1: "medium", 2: "long"}
LATENCY_WARMUP_STEPS = 5


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
    state = (
        payload["state_dict"]
        if isinstance(payload, dict) and "state_dict" in payload
        else payload
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return payload


def summarize(episodes):
    if not episodes:
        return {"episodes": 0}
    return {
        "episodes": len(episodes),
        "sr": float(np.mean([float(row["success"]) for row in episodes])),
        "osr": float(np.mean([float(row["osr"]) for row in episodes])),
        "spl": float(np.mean([float(row["spl"]) for row in episodes])),
        "ndtw": float(np.mean([float(row["ndtw"]) for row in episodes])),
        "premature_stop": float(np.mean(
            [float(row["premature_stop"]) for row in episodes])),
        "timeout": float(np.mean(
            [float(row["max_steps_termination"]) for row in episodes])),
        "steps": float(np.mean([float(row["steps"]) for row in episodes])),
        "min_distance_m": float(np.mean(
            [float(row["min_distance_m"]) for row in episodes])),
    }


def band_distance_range(band):
    for name, _weight, dmin, dmax, _amin, _amax in TRAIN.trajectory_mix:
        if str(name) == str(band):
            return (float(dmin), float(dmax))
    raise KeyError(f"Unknown band {band}")


def verify_sealed(test_dir):
    """Verify every SHA256 in SEALED.json and the shard set itself."""
    test_dir = Path(test_dir)
    sealed_path = test_dir / "SEALED.json"
    if not sealed_path.is_file():
        raise FileNotFoundError(f"Missing {sealed_path}")
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    if sealed.get("split") != "final_test":
        raise ValueError("SEALED.json is not a final_test manifest")
    files = sealed.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SEALED.json has no file hash map")

    verified = {}
    for name, expected in files.items():
        path = test_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Sealed file missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Sealed file SHA256 mismatch for {name}: "
                f"{actual} != {expected}"
            )
        verified[name] = actual

    sealed_shards = {name for name in files if name.endswith(".pt")}
    actual_shards = {path.name for path in test_dir.glob("*.pt")}
    if sealed_shards != actual_shards:
        raise ValueError(
            "Final Test shard set differs from SEALED.json: "
            f"missing={sorted(sealed_shards - actual_shards)}, "
            f"extra={sorted(actual_shards - sealed_shards)}"
        )
    return {
        "manifest_path": str(sealed_path),
        "manifest_sha256": file_sha256(sealed_path),
        "files_verified": len(verified),
        "shards": len(sealed_shards),
        "verified_utc": utc_now(),
        "hashes": verified,
    }


def extract_final_test_specs(test_dir, seal=None, expected_episodes=None,
                             expected_band_counts=None):
    """Derive the exact task list from the sealed token shards.

    No episode is regenerated and no seed is invented.  Every row must agree
    on ``(episode_seed, network_layout_seed, trajectory_band_id)`` for its
    ``episode_id``; step ids must be contiguous from zero.
    """
    test_dir = Path(test_dir)
    expected_episodes = (
        TEST_EPISODES if expected_episodes is None else int(expected_episodes)
    )
    if expected_band_counts is None:
        if expected_episodes == TEST_EPISODES:
            expected_band_counts = {band: BAND_EPISODES for band in BANDS}
        else:
            raise ValueError(
                "Synthetic episode counts require explicit band counts"
            )
    if seal is None:
        seal = verify_sealed(test_dir)
    shard_paths = sorted(test_dir.glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No token shards in {test_dir}")

    metadata = {}
    steps = {}
    for path in shard_paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        required = (
            "episode_ids", "episode_seeds", "network_layout_seeds",
            "trajectory_band_ids", "step_ids",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path.name} missing metadata {missing}")
        episode_ids = data["episode_ids"].cpu().numpy().astype(np.int64)
        episode_seeds = data["episode_seeds"].cpu().numpy().astype(np.int64)
        network_seeds = data["network_layout_seeds"].cpu().numpy().astype(
            np.int64
        )
        band_ids = data["trajectory_band_ids"].cpu().numpy().astype(np.int64)
        step_ids = data["step_ids"].cpu().numpy().astype(np.int64)
        for episode_id, seed, network_seed, band_id, step_id in zip(
            episode_ids.tolist(), episode_seeds.tolist(),
            network_seeds.tolist(), band_ids.tolist(), step_ids.tolist(),
        ):
            key = int(episode_id)
            value = {
                "seed": int(seed),
                "network_layout_seed": int(network_seed),
                "band_id": int(band_id),
            }
            if key in metadata:
                if metadata[key] != value:
                    raise ValueError(
                        f"Episode {key} has inconsistent metadata across "
                        f"shards: {metadata[key]} vs {value}"
                    )
            else:
                metadata[key] = value
            steps.setdefault(key, []).append(int(step_id))

    episode_ids = sorted(metadata)
    if len(episode_ids) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episodes, found "
            f"{len(episode_ids)}"
        )
    if episode_ids != list(range(expected_episodes)):
        raise ValueError(
            f"Episode ids are not 0..{expected_episodes - 1}: "
            f"{episode_ids[:5]}...{episode_ids[-5:]}"
        )
    for episode_id in episode_ids:
        values = steps[episode_id]
        if values != list(range(len(values))):
            raise ValueError(
                f"Episode {episode_id} step ids are not contiguous from 0"
            )

    specs = []
    for episode_id in episode_ids:
        meta = metadata[episode_id]
        if meta["band_id"] not in BAND_ID_TO_NAME:
            raise ValueError(
                f"Episode {episode_id} has unknown band id {meta['band_id']}"
            )
        band = BAND_ID_TO_NAME[meta["band_id"]]
        specs.append({
            "episode_index": int(episode_id),
            "seed": int(meta["seed"]),
            "network_layout_seed": int(meta["network_layout_seed"]),
            "band": band,
            "distance_range": band_distance_range(band),
            "dataset_steps": int(len(steps[episode_id])),
        })

    counts = {band: sum(1 for spec in specs if spec["band"] == band)
              for band in BANDS}
    if counts != dict(expected_band_counts):
        raise ValueError(
            f"Band counts {counts} != expected {expected_band_counts}"
        )

    # Cross-check against the sealed generation stats.  This only validates
    # that the shard-derived task list reproduces the generated tasks; it is
    # not a source of task selection and cannot change seeds or bands.
    stats_path = test_dir / "dataset_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing {stats_path}")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    records = list(stats.get("trajectories", []))
    if len(records) != expected_episodes:
        raise ValueError(
            f"dataset_stats has {len(records)} trajectories, expected "
            f"{expected_episodes}"
        )
    by_seed = {}
    for record in records:
        seed = int(record["seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate trajectory seed {seed} in stats")
        by_seed[seed] = record
    for spec in specs:
        record = by_seed.get(spec["seed"])
        if record is None:
            raise ValueError(
                f"Episode {spec['episode_index']} seed {spec['seed']} "
                "is absent from dataset_stats"
            )
        if str(record["band"]) != spec["band"]:
            raise ValueError(
                f"Episode {spec['episode_index']} band mismatch: "
                f"{record['band']} vs {spec['band']}"
            )
        if int(record["network_layout_seed"]) != spec["network_layout_seed"]:
            raise ValueError(
                f"Episode {spec['episode_index']} network seed mismatch"
            )
        start = float(record["start_goal_distance_m"])
        low, high = spec["distance_range"]
        if not (low <= start <= high):
            raise ValueError(
                f"Episode {spec['episode_index']} start distance {start} "
                f"outside band range [{low}, {high}]"
            )
        spec["dataset_start_goal_distance_m"] = start

    return {
        "specs": specs,
        "seal": seal,
        "band_counts": counts,
        "episodes": len(specs),
        "seeds_min": min(spec["seed"] for spec in specs),
        "seeds_max": max(spec["seed"] for spec in specs),
        "network_seeds_min": min(
            spec["network_layout_seed"] for spec in specs
        ),
        "network_seeds_max": max(
            spec["network_layout_seed"] for spec in specs
        ),
        "created_utc": utc_now(),
        "final_test_read": True,
    }


def verify_protocol_freeze(freeze_path, model_name, model):
    """Refuse the formal test if the frozen code/model contract drifted."""
    freeze_path = Path(freeze_path)
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Missing protocol freeze {freeze_path}")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("Protocol freeze is not marked frozen")
    critical = payload.get("critical_files") or {}
    for relative, expected in critical.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen source missing: {path}")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen source changed: {relative} ({actual} != {expected})"
            )
    expected_params = (
        payload.get("models", {}).get(model_name, {}).get("parameters")
    )
    actual_params = parameter_count(model)
    if expected_params is not None and int(expected_params) != actual_params:
        raise ValueError(
            f"{model_name} parameter count {actual_params} != frozen "
            f"{expected_params}"
        )
    if float(payload.get("stop_threshold", -1.0)) != STOP_THRESHOLD:
        raise ValueError("Stop threshold drifted from the frozen protocol")
    return payload


def load_final_test_specs(test_dir=FINAL_TEST_DIR):
    """Verify the seal, extract the task list, and freeze it to disk."""
    result = extract_final_test_specs(test_dir)
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPECS_DIR / "final_test_300_spec.json"
    payload = {
        "version": "stage2j-final-test-300",
        "created_utc": result["created_utc"],
        "sealed_manifest_sha256": result["seal"]["manifest_sha256"],
        "episodes": result["episodes"],
        "band_counts": result["band_counts"],
        "seeds_min": result["seeds_min"],
        "seeds_max": result["seeds_max"],
        "network_seeds_min": result["network_seeds_min"],
        "network_seeds_max": result["network_seeds_max"],
        "specs": result["specs"],
        "note": "Derived only from sealed token shard metadata; no episode "
                "was regenerated or selected by model performance.",
    }
    atomic_write_json(path, payload)
    result["spec_path"] = str(path)
    result["spec_sha256"] = file_sha256(path)
    return result


def _output_dir(args, final_test):
    if args.output_dir is not None:
        return Path(args.output_dir)
    suffix = "eval_final_test_300" if final_test else "eval_dev_val_180"
    return model_dir(args.model) / suffix


def run_eval(args):
    model_name = args.model
    if model_name not in MODEL_DIRS:
        raise SystemExit(f"Unknown model {model_name}")
    final_test = args.split == "final_test"
    if final_test and not args.allow_final_test:
        raise SystemExit(
            "Formal Final Test requires --allow-final-test and --freeze-json"
        )
    if final_test and args.smoke:
        raise SystemExit("Smoke evaluation on Final Test is forbidden")
    if not final_test:
        # Dev mode never reads the test directory at all.
        reject_final_test(args.val_specs)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    training_dir = Path(args.model_dir) if args.model_dir else model_dir(
        model_name
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        training_dir / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Missing checkpoint {checkpoint_path}")
    training_result = read_json(training_dir / "training_result.json", {})
    if training_result.get("model") not in (None, model_name):
        raise SystemExit(
            f"Checkpoint directory model {training_result.get('model')} != "
            f"{model_name}"
        )
    if training_result.get("final_test_read"):
        raise SystemExit("Training result unexpectedly claims a test read")

    model = build_stage2j_model(model_name).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    params = parameter_count(model)
    checkpoint_sha = file_sha256(checkpoint_path)
    architecture = STAGE2J_CONFIGS[model_name]["architecture"]

    freeze = None
    seal = None
    spec_sha = None
    if final_test:
        if args.freeze_json is None:
            args.freeze_json = FREEZE_PATH
        freeze = verify_protocol_freeze(args.freeze_json, model_name, model)
        specs_payload = load_final_test_specs(args.test_dir)
        specs = specs_payload["specs"]
        seal = specs_payload["seal"]
        spec_sha = specs_payload["spec_sha256"]
        frozen_seal = freeze.get("final_test", {}).get(
            "sealed_manifest_sha256"
        )
        if frozen_seal is not None and frozen_seal != seal["manifest_sha256"]:
            raise RuntimeError(
                "SEALED.json changed after the protocol freeze: "
                f"{seal['manifest_sha256']} != {frozen_seal}"
            )
        if len(specs) != TEST_EPISODES:
            raise RuntimeError(f"Expected {TEST_EPISODES} test specs")
        print(
            f"{model_name}: FINAL TEST {len(specs)} episodes; "
            f"seal verified ({seal['shards']} shards, "
            f"{seal['files_verified']} files); checkpoint={checkpoint_path}",
            flush=True,
        )
    else:
        specs = load_dev_specs(args.val_specs, int(args.max_episodes))
        print(
            f"{model_name}: {len(specs)} Dev Val episodes; "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )

    aux_manifest = read_json(AUX_ROOT / "manifest.json", {})
    gab_threshold = aux_manifest.get("gab_reliability_threshold")
    if gab_threshold is None:
        raise SystemExit("Missing GAB reliability threshold in aux manifest")
    gab_config = GABConfig(reliability_threshold=float(gab_threshold))

    encoder = load_encoder(device)
    print("building oracle references", flush=True)
    oracle_cache = build_oracle_cache(specs)

    output_dir = _output_dir(args, final_test)
    if output_dir.exists() and not (args.resume or args.smoke):
        if (output_dir / "metrics.json").is_file():
            raise SystemExit(
                f"Refusing to overwrite {output_dir}; use --resume"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / f"{model_name}_per_episode.csv"
    steps_path = output_dir / f"{model_name}_steps.csv"
    retrieval_path = output_dir / "retrieval_episodes.csv"

    existing = {
        int(row["episode_index"]): row
        for row in read_csv_rows(episodes_path)
    }
    existing_steps = read_csv_rows(steps_path)
    step_rows = [
        row for row in existing_steps
        if int(row["episode_index"]) in existing
    ]
    retrieval_records = read_csv_rows(retrieval_path)
    retrieval_by_episode = {
        int(row["episode_index"]): row for row in retrieval_records
    }

    run_started = datetime.now(timezone.utc)
    forward_times = [
        float(row["mec_forward_ms"]) for row in step_rows
        if row.get("mec_forward_ms") not in (None, "")
    ]
    feature_times = [
        float(row["feature_build_ms"]) for row in step_rows
        if row.get("feature_build_ms") not in (None, "")
    ]
    decision_times = [
        float(row["decision_ms"]) for row in step_rows
        if row.get("decision_ms") not in (None, "")
    ]
    decisions_seen = len(step_rows)

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
        reproduced = float(np.linalg.norm(env.position - env.goal))
        expected_start = spec.get("dataset_start_goal_distance_m")
        if expected_start is not None and abs(
            reproduced - float(expected_start)
        ) > 1e-3:
            raise RuntimeError(
                f"Episode {episode_index} start distance mismatch: "
                f"{reproduced} vs {expected_start}"
            )
        goal_tokens = encode_multiscale_goal_tokens(
            env.get_goal_reference_multiscale_rgbs(), encoder, device
        ).unsqueeze(0).to(device, dtype=torch.float32)
        goal_token_np = goal_tokens[0, 2].detach().cpu().float().numpy()

        landmark_map = build_landmark_map(env, spec["seed"], encoder, device)
        retrieval = retrieval_diagnostics(
            landmark_map["goal_index"], landmark_map["embeddings"],
            goal_token_np,
        )
        retrieval_row = {
            "model": model_name,
            "episode_index": episode_index,
            "seed": int(spec["seed"]),
            "band": spec["band"],
            **retrieval,
        }
        retrieval_by_episode[episode_index] = retrieval_row

        gab_state = GABState(gab_config)
        past = []
        previous_action = -1
        trajectory = [env.position.copy()]
        min_distance = float(env.distance_to_goal)
        osr = bool(env.distance_to_goal < env.stop_radius)
        success = False
        done = False
        ended_by_stop = False
        stop_distance = float("nan")
        stop_probability = float("nan")
        episode_steps = []
        episode_forward = []
        episode_feature = []
        episode_decision = []

        for step in range(int(ENV.max_steps)):
            pose = env.pose.copy()

            if device.type == "cuda":
                torch.cuda.synchronize()
            feature_started = time.perf_counter()
            current_tokens = encoder(image_to_tensor(obs, device))
            heatmap = goal_conditioned_spatial_heatmap(
                current_tokens, goal_tokens
            )
            history = online_history_arrays(past)
            gab_state.update(
                heatmap[0].detach().cpu().float().numpy(), previous_action
            )
            map_feature = torch.from_numpy(map_retrieval_features(
                pose, landmark_map["positions"],
                landmark_map["embeddings"], goal_token_np,
            )).unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            feature_ended = time.perf_counter()

            forward_started = time.perf_counter()
            with torch.inference_mode():
                components = model.forward_components(
                    current_tokens,
                    goal_tokens,
                    heatmap=None,
                    network_state=None,
                    history_vis=torch.from_numpy(
                        history[0]).unsqueeze(0).to(device),
                    history_pose=torch.from_numpy(
                        history[1]).unsqueeze(0).to(device),
                    history_action=torch.from_numpy(
                        history[2]).unsqueeze(0).to(device),
                    history_mask=torch.from_numpy(
                        history[3]).unsqueeze(0).to(device),
                    gab=torch.from_numpy(
                        gab_state.features()).unsqueeze(0).to(device),
                    map_feat=map_feature,
                )
                arrival_probability = float(
                    torch.sigmoid(components["arrival_logit"]).item()
                )
                move_action = int(
                    torch.argmax(components["move_logits"], -1).item()
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_ended = time.perf_counter()

            mec_forward_ms = (forward_ended - forward_started) * 1000.0
            feature_build_ms = (feature_ended - feature_started) * 1000.0
            decision_ms = (forward_ended - feature_started) * 1000.0
            decisions_seen += 1
            if decisions_seen > LATENCY_WARMUP_STEPS:
                episode_forward.append(mec_forward_ms)
                episode_feature.append(feature_build_ms)
                episode_decision.append(decision_ms)

            action = (
                STOP if arrival_probability >= STOP_THRESHOLD else move_action
            )
            oracle_diagnostic = int(oracle_move_action(env))
            projection = projection_vector(env)
            distance = float(env.distance_to_goal)
            episode_steps.append({
                "model": model_name,
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
                "gab_valid": int(gab_state.valid),
                "gab_confidence": float(gab_state.confidence),
                "mec_forward_ms": mec_forward_ms,
                "feature_build_ms": feature_build_ms,
                "decision_ms": decision_ms,
            })
            if action == STOP:
                stop_distance = distance
                stop_probability = arrival_probability

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
            float(np.linalg.norm(
                np.diff(trajectory[:, :2], axis=0), axis=1).sum())
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
            "model": model_name,
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
            "stop_distance_m": float(stop_distance),
            "stop_arrival_probability": float(stop_probability),
            "retrieval_goal_rank": int(retrieval["goal_rank"]),
            "retrieval_top1": int(retrieval["top1"]),
            "retrieval_top5": int(retrieval["top5"]),
            "mec_forward_ms_mean": (
                float(np.mean(episode_forward)) if episode_forward else float("nan")
            ),
            "feature_build_ms_mean": (
                float(np.mean(episode_feature)) if episode_feature else float("nan")
            ),
            "decision_ms_mean": (
                float(np.mean(episode_decision)) if episode_decision else float("nan")
            ),
        }
        existing[episode_index] = row
        step_rows.extend(episode_steps)
        forward_times.extend(episode_forward)
        feature_times.extend(episode_feature)
        decision_times.extend(episode_decision)
        ordered = [existing[key] for key in sorted(existing)]
        write_csv(episodes_path, ordered)
        write_csv(
            retrieval_path,
            [retrieval_by_episode[key] for key in sorted(retrieval_by_episode)],
        )
        if (position + 1) % 20 == 0 or position + 1 == len(specs):
            write_csv(steps_path, step_rows)
            summary = summarize(ordered)
            print(
                f"  {model_name}: {position + 1}/{len(specs)} episodes | "
                f"SR={100*summary['sr']:.2f}% | SPL={summary['spl']:.4f}",
                flush=True,
            )

    write_csv(steps_path, step_rows)
    episodes = [existing[key] for key in sorted(existing)]
    overall = summarize(episodes)
    bands = {
        band: summarize([row for row in episodes if row["band"] == band])
        for band in BANDS
    }
    retrieval_rows = [
        retrieval_by_episode[key] for key in sorted(retrieval_by_episode)
    ]
    retrieval_summary = {
        "episodes": len(retrieval_rows),
        "clean_top1": float(np.mean(
            [float(row["top1"]) for row in retrieval_rows
             ])) if retrieval_rows else 0.0,
        "clean_top5": float(np.mean(
            [float(row["top5"]) for row in retrieval_rows
             ])) if retrieval_rows else 0.0,
        "clean_mean_goal_similarity": float(np.mean(
            [float(row["goal_similarity"]) for row in retrieval_rows]
        )) if retrieval_rows else 0.0,
    }
    latency = {
        "mec_forward": latency_summary(forward_times),
        "feature_build": latency_summary(feature_times),
        "decision_pipeline": latency_summary(decision_times),
        "batch_size": 1,
        "device": str(device),
        "warmup_steps_excluded": LATENCY_WARMUP_STEPS,
    }

    eval_seconds = (
        datetime.now(timezone.utc) - run_started
    ).total_seconds()
    metrics = {
        "meta": {
            "model": model_name,
            "title": MODEL_TITLES[model_name],
            "architecture": architecture,
            "native_config": STAGE2J_CONFIGS[model_name],
            "parameters": params,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch": checkpoint.get("epoch")
            if isinstance(checkpoint, dict) else None,
            "episodes": len(episodes),
            "stop_threshold": STOP_THRESHOLD,
            "band_episodes": {
                band: bands[band].get("episodes", 0) for band in BANDS
            },
            "split": "final_test" if final_test else "dev_val",
            "final_test_read": bool(final_test),
            "device": str(device),
            "eval_seconds": eval_seconds,
            "freeze_json": str(args.freeze_json) if final_test else None,
            "freeze_sha256": (
                file_sha256(args.freeze_json)
                if final_test and args.freeze_json else None
            ),
            "sealed_manifest_sha256": (
                seal["manifest_sha256"] if seal else None
            ),
            "sealed_files_verified": (
                seal["files_verified"] if seal else 0
            ),
            "test_spec_sha256": spec_sha,
            "protocol": EVAL_PROTOCOL,
        },
        "overall": overall,
        "bands": bands,
        "retrieval": retrieval_summary,
        "latency": latency,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if args.smoke:
        print("SMOKE_ONLY: progress.json and RESULTS.md were not updated",
              flush=True)
        print(json.dumps(metrics, indent=2)[:4000], flush=True)
        return metrics

    if final_test:
        update_progress(
            model_name,
            architecture=architecture,
            parameters=params,
            test_status="tested",
            metrics=overall,
            bands=bands,
            retrieval=retrieval_summary,
            latency=latency,
            eval_output_dir=str(output_dir),
            eval_started_utc=run_started.isoformat(),
            eval_finished_utc=utc_now(),
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            final_test_read=True,
            final_test_meta=metrics["meta"],
        )
    else:
        update_progress(
            model_name,
            dev_eval_metrics=overall,
            dev_eval_bands=bands,
            dev_eval_output_dir=str(output_dir),
            dev_eval_finished_utc=utc_now(),
            dev_eval_checkpoint_sha256=checkpoint_sha,
        )

    progress = load_progress()
    completed = {
        name: bool(progress["models"][name].get("metrics"))
        and progress["models"][name].get("test_status") in (
            "tested", "reused_tested"
        )
        for name in MODEL_DIRS
    }
    entry = progress["models"][model_name]
    if final_test:
        print(format_result_block(model_name, entry, completed), flush=True)
        print("\n" + cumulative_table(progress), flush=True)
        append_results_snapshot(progress, f"{model_name} completed")
    else:
        print(
            f"{model_name} Dev Val: SR={100*overall['sr']:.2f}% | "
            f"OSR={100*overall['osr']:.2f}% | SPL={overall['spl']:.4f} | "
            f"nDTW={overall['ndtw']:.4f}",
            flush=True,
        )
        for band in BANDS:
            item = bands[band]
            print(
                f"  {band:6s}: SR={100*item['sr']:.2f}% | "
                f"SPL={item['spl']:.4f} | nDTW={item['ndtw']:.4f}",
                flush=True,
            )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODEL_DIRS))
    parser.add_argument("--split", choices=("dev_val", "final_test"),
                        default="dev_val")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--val-specs", type=Path, default=VAL_SPECS)
    parser.add_argument("--test-dir", type=Path, default=FINAL_TEST_DIR)
    parser.add_argument("--freeze-json", type=Path, default=None)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
