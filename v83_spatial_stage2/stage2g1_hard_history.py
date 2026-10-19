"""Stage-2g.1 compact short history for the isolated hard states.

The hard arrival/distractor samples in Train 600 / Dev Val 180 are isolated
poses with no logged trajectory. To avoid P1 using "history missing" as a
shortcut, this script reconstructs each hard pose exactly (same RNG stream as
token_data.py) and renders a real three-step RIGHT-turn history ending at that
pose: t-3 = final yaw + 45 deg, t-2 = +30 deg, t-1 = +15 deg, t = final.

Only compact observable features (74-d) and actions/time are saved. The
current hard-state tokens/features already exist in the Stage-2g matrices; no
raw tokens are written and no existing data is modified.

This history is synthetic in the sense that the original hard sample was a
single isolated pose, not a logged trajectory. It is used only for P1; P0 uses
the current frame. The generation is documented in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import TRAIN  # noqa: E402
from env import ToyUAVEnv, wrap_angle  # noqa: E402
from token_data import (  # noqa: E402
    _set_pose,
    encode_multiscale_goal_tokens,
    goal_conditioned_spatial_heatmap,
)
from utils import image_to_tensor  # noqa: E402
from visual_grounding_v83 import static_grounding  # noqa: E402
from v83_spatial_stage2.stage2g_features import (  # noqa: E402
    compute_feature_vector,
    feature_names,
)
from v83_spatial_stage2.stage2g_labels import (  # noqa: E402
    HARD_PLAN,
    band_ranges,
    sample_hard_state,
    split_seed_start,
)
from v83_spatial_stage2.stage2g1_common import (  # noqa: E402
    COMPACT_FEATURE_NAMES,
    COMPACT_DIM,
    HISTORY_FRAMES,
    compact_indices,
)
from v83_spatial_training.eval_closedloop_spatial import (  # noqa: E402
    ARCHIVE,
    load_encoder,
)

RIGHT = 2
OFFSETS = (15.0, 30.0, 45.0)  # t-1, t-2, t-3 relative to final yaw


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("runs/v83_spatial_stage2/datasets"),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    data_dir = args.data_root / (
        "train_600" if args.split == "train" else "dev_val_180"
    )
    stats = json.loads(
        (data_dir / "dataset_stats.json").read_text(encoding="utf-8")
    )
    trajectories = stats["trajectories"]
    if args.max_episodes is not None:
        trajectories = trajectories[: args.max_episodes]
    ranges = band_ranges()
    seed_start = split_seed_start(args.data_root, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    encoder = load_encoder(device)
    names = feature_names()
    compact_idx = compact_indices(names)
    env = ToyUAVEnv()
    arrival_rng = np.random.default_rng(int(seed_start) + 99173)
    goal_cache = {}
    episode_buf = []
    step_buf = []
    features_buf = []
    actions_buf = []
    dt_buf = []
    started = time.time()
    for episode_id, record in enumerate(trajectories):
        seed = int(record["seed"])
        band = str(record["band"])
        dmin, dmax = ranges[band]
        env.reset(
            seed=seed,
            goal_distance_range_m=(dmin, dmax),
            render_observation=False,
        )
        if episode_id not in goal_cache:
            goal_cache[episode_id] = (
                encode_multiscale_goal_tokens(
                    env.get_goal_reference_multiscale_rgbs(),
                    encoder,
                    device,
                ).detach().cpu().half()
            )
        goal_tokens = goal_cache[episode_id].to(
            device=device, dtype=torch.float32
        )
        # Replay the natural trajectory to advance arrival_rng exactly as
        # token_data.py did before the hard samples.
        done = False
        while not done:
            distance = float(env.distance_to_goal)
            arrival = float(distance < env.stop_radius)
            # Match token_data's arrival_train_mask RNG consumption exactly.
            if not (
                arrival > 0.5
                or distance <= float(TRAIN.arrival_natural_near_radius_m)
            ):
                arrival_rng.random()
            _, _, done = env.step(
                int(env.oracle_action()), render_observation=False
            )
        for hard_index, sample_type in enumerate(HARD_PLAN):
            sampled = sample_hard_state(env, sample_type, arrival_rng)
            if sampled is None:
                raise RuntimeError(
                    f"episode {episode_id} hard {hard_index} failed"
                )
            position, final_yaw, _distance, _bearing = sampled
            history = []
            for offset in OFFSETS:
                _set_pose(
                    env, position,
                    wrap_angle(final_yaw + math.radians(offset)),
                )
                observation = env.render()
                tokens = encoder(image_to_tensor(observation, device))
                goal = goal_tokens.unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                heatmap = goal_conditioned_spatial_heatmap(tokens, goal)
                baseline, coherent = static_grounding(tokens, goal)
                features = compute_feature_vector(
                    tokens, goal, heatmap, baseline, coherent
                )
                history.append(
                    features.squeeze(0).cpu().numpy()[compact_idx]
                )
            episode_buf.append(int(episode_id))
            step_buf.append(-(hard_index + 1))
            features_buf.append(np.stack(history, axis=0))
            actions_buf.append(np.full(HISTORY_FRAMES, RIGHT, np.int8))
            dt_buf.append(np.ones(HISTORY_FRAMES, np.float32))
        if (episode_id + 1) % 50 == 0 or episode_id + 1 == len(trajectories):
            print(
                f"  hard history {episode_id + 1}/{len(trajectories)} "
                f"episodes ({time.time() - started:.1f}s)",
                flush=True,
            )

    arrays = {
        "episode_id": np.asarray(episode_buf, dtype=np.int64),
        "step_id": np.asarray(step_buf, dtype=np.int64),
        "history_features": np.stack(features_buf, axis=0).astype(np.float32),
        "history_actions": np.stack(actions_buf, axis=0),
        "history_dt": np.stack(dt_buf, axis=0),
        "compact_feature_names": np.asarray(COMPACT_FEATURE_NAMES),
    }
    out_path = output_dir / f"hard_history_{args.split}.npz"
    np.savez(out_path, **arrays)
    summary = {
        "split": args.split,
        "hard_states": int(len(episode_buf)),
        "episodes": int(len(trajectories)),
        "history_frames": HISTORY_FRAMES,
        "actions": "RIGHT x3 to the final hard pose",
        "offsets_deg": list(OFFSETS),
        "order": "t-1, t-2, t-3",
        "compact_dim": COMPACT_DIM,
        "elapsed_s": float(time.time() - started),
        "size_bytes": int(out_path.stat().st_size),
        "encoder": {
            "path": str(ARCHIVE / "shared_encoder.pt"),
            "sha256": sha256(ARCHIVE / "shared_encoder.pt"),
        },
    }
    (output_dir / f"hard_history_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "split": args.split,
        "hard_states": summary["hard_states"],
        "elapsed_s": round(summary["elapsed_s"], 1),
        "size_bytes": summary["size_bytes"],
        "output": str(out_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
