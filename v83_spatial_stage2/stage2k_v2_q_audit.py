"""Read-only Q-value stability audit for Stage-2k v2 best/last checkpoints.

Collects a fixed set of Dev states with a checkpoint-independent Min-Latency
behavior policy, then forwards both checkpoints on exactly those states.  No
training, no Final Test read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MAX_STEPS,
    MODEL_NAMES,
    SCENARIOS,
    atomic_write_json,
    load_topology,
    sha256_file,
    utc_now,
)
from v83_spatial_stage2.stage2k_dqn import QNetwork, dev_validation_specs
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    STATE_DIM_PURE_MEC_V2,
    STATE_DIM_VISUAL_ABLATION_V2,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    SceneCache,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import VisualProjector, masked_argmax
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_PRIOR_PATH,
    load_capability_prior,
)
from v83_spatial_stage2.stage2k_common import VISUAL_PCA_PATH


def state_dim_for(mode):
    if mode == "pure_mec_v2":
        return STATE_DIM_PURE_MEC_V2
    if mode == "visual_ablation_v2":
        return STATE_DIM_VISUAL_ABLATION_V2
    raise ValueError(f"Unsupported v2 state mode {mode}")


def collect_states(args):
    _, capability = load_capability_prior(args.capability_prior)
    all_specs = dev_validation_specs()
    specs = []
    for scenario in SCENARIOS:
        scenario_specs = [
            item for item in all_specs if item["scenario"] == scenario
        ]
        specs.extend(scenario_specs[: int(args.episodes)])
    per_episode = max(
        1, int(np.ceil(float(args.max_states) / max(len(specs), 1)))
    )
    stride = max(1, MAX_STEPS // per_episode)
    topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    projector = (
        VisualProjector.load(VISUAL_PCA_PATH)
        if args.state_mode == "visual_ablation_v2" else None
    )
    device, encoder, backends = load_runtime(args.device)
    gab_config = load_gab_config()
    cache = SceneCache(encoder, device, max_entries=len(specs))
    states = []
    masks = []
    node_models = []
    episodes = []
    for spec in specs:
        runner = EpisodeRunner(
            spec=spec,
            topology=topologies[spec["scenario"]],
            encoder=encoder,
            backends=backends,
            projector=projector,
            quality_prior=capability,
            gab_config=gab_config,
            scene_cache=cache,
            device=device,
            jointr_aligned=True,
            pure_mec_v2_state=args.state_mode == "pure_mec_v2",
            visual_ablation_v2_state=(
                args.state_mode == "visual_ablation_v2"
            ),
        )
        step_index = 0
        states_this_episode = 0
        while True:
            if len(states) >= int(args.max_states):
                break
            if (
                states_this_episode < per_episode
                and step_index % stride == 0
            ):
                state = np.asarray(
                    runner.step_data["state"], dtype=np.float32
                )
                mask = np.asarray(runner.step_data["mask"], dtype=bool)
                states.append(state.copy())
                masks.append(mask.copy())
                node_models.append([
                    node["model_type"] for node in runner.nodes
                ])
                episodes.append(spec["episode_uid"])
                states_this_episode += 1
            action = runner.select_node("min_latency")
            transition = runner.apply(action)
            step_index += 1
            if transition["done"]:
                break
        if len(states) >= int(args.max_states):
            break
    if not states:
        raise RuntimeError("No states collected")
    return np.stack(states), np.stack(masks), node_models, episodes, device


def checkpoint_stats(net, states, masks):
    q_means = []
    q_maxs = []
    q_stds = []
    q_top2 = []
    greedy = []
    q_abs_max = 0.0
    finite = True
    device = next(net.parameters()).device
    with torch.inference_mode():
        for state, mask in zip(states, masks):
            tensor = torch.from_numpy(
                np.asarray(state, dtype=np.float32)
            ).unsqueeze(0).to(device)
            q = net(tensor)[0].detach().cpu().numpy()
            available = np.where(mask)[0]
            if len(available) == 0:
                raise RuntimeError("empty action mask")
            values = q[available]
            finite = finite and bool(np.isfinite(values).all())
            q_means.append(float(values.mean()))
            q_maxs.append(float(values.max()))
            q_stds.append(float(values.std()))
            ordered = np.sort(values)[::-1]
            gap = (
                float(ordered[0] - ordered[1]) if len(ordered) > 1
                else 0.0
            )
            q_top2.append(gap)
            greedy.append(int(masked_argmax(q, mask)))
            q_abs_max = max(q_abs_max, float(np.abs(values).max()))
    return {
        "states": len(q_means),
        "q_mean": float(np.mean(q_means)),
        "q_max": float(np.max(q_maxs)),
        "q_max_mean": float(np.mean(q_maxs)),
        "q_std_mean": float(np.mean(q_stds)),
        "q_top2_gap_mean": float(np.mean(q_top2)),
        "q_abs_max": q_abs_max,
        "finite": finite,
        "greedy_actions": greedy,
    }


def load_network(checkpoint, state_dim, device):
    payload = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    config = payload.get("config", {})
    if int(config.get("state_dim", -1)) != state_dim:
        raise RuntimeError(
            f"{checkpoint} state_dim {config.get('state_dim')} != "
            f"{state_dim}"
        )
    net = QNetwork(state_dim=state_dim).to(device)
    net.load_state_dict(payload["state_dict"], strict=True)
    net.eval()
    return net, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--state-mode", choices=("pure_mec_v2", "visual_ablation_v2"),
        required=True,
    )
    parser.add_argument("--capability-prior", type=Path,
                        default=CAPABILITY_PRIOR_PATH)
    parser.add_argument(
        "--episodes", type=int, default=6,
        help="Validation episodes per scenario used for state collection.",
    )
    parser.add_argument("--max-states", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.run_dir / "q_audit" / "q_audit.json"
    if output.is_file():
        raise SystemExit(f"{output} exists; refusing to overwrite")

    state_dim = state_dim_for(args.state_mode)
    states, masks, node_models, episodes, device = collect_states(args)
    checkpoints = {
        "best": args.run_dir / "checkpoints" / "best.pt",
        "last": args.run_dir / "checkpoints" / "last.pt",
    }
    stats = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        net, payload = load_network(path, state_dim, device)
        item = checkpoint_stats(net, states, masks)
        item.update({
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "episode": int(payload.get("episode", 0)),
            "best_joint": payload.get("best_joint"),
        })
        stats[name] = item
    agreement = float(np.mean([
        int(a == b) for a, b in zip(
            stats["best"]["greedy_actions"],
            stats["last"]["greedy_actions"],
        )
    ]))
    model_usage = {}
    for name in ("best", "last"):
        counter = {model: 0 for model in MODEL_NAMES}
        for state_index, action in enumerate(
            stats[name]["greedy_actions"]
        ):
            model = node_models[state_index][int(action)]
            counter[model] += 1
        total = max(sum(counter.values()), 1)
        model_usage[name] = {
            model: counter[model] for model in MODEL_NAMES
        }
        model_usage[name + "_ratio"] = {
            model: counter[model] / total for model in MODEL_NAMES
        }
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.parent / "q_audit_states.npz",
        states=states,
        masks=masks,
        episode_uids=np.asarray(episodes),
    )
    payload = {
        "version": "stage2k-v2-q-audit-v1",
        "created_utc": utc_now(),
        "run_dir": str(run_dir),
        "state_mode": args.state_mode,
        "state_dim": int(state_dim),
        "episodes": sorted(set(episodes)),
        "state_count": int(len(states)),
        "behavior_policy": "deterministic Min-Latency (checkpoint-independent)",
        "best_last_greedy_agreement": agreement,
        "checkpoints": stats,
        "model_usage": model_usage,
        "final_test_read": False,
    }
    atomic_write_json(output, payload)
    print(json.dumps({
        "output": str(output),
        "state_count": len(states),
        "agreement": agreement,
        "best": {
            key: stats["best"][key] for key in (
                "episode", "q_mean", "q_max", "q_std_mean",
                "q_top2_gap_mean", "finite",
            )
        },
        "last": {
            key: stats["last"][key] for key in (
                "episode", "q_mean", "q_max", "q_std_mean",
                "q_top2_gap_mean", "finite",
            )
        },
    }, indent=2, ensure_ascii=False), flush=True)
    print("STAGE2K_V2_Q_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
