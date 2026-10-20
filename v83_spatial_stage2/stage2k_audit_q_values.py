"""Read-only Q-value audit for the Stage-2k DQN checkpoints.

Builds a fixed set of Dev states with a deterministic Min-Latency rollout and
runs the online networks of best.pt (Ep800) and last.pt (Ep2000) on exactly the
same states.  It reports Q ranges, NaN/Inf counts and greedy-action agreement.
No training and no checkpoint modification.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    DQN_DIR,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    VISUAL_PCA_PATH,
    load_split_specs,
    load_topology,
    read_json,
)
from v83_spatial_stage2.stage2k_dqn import DQNAgent, flat_quality
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import VisualProjector, masked_argmax

OUTPUT_DIR = DQN_DIR / "q_value_audit"
STEPS = 20


def q_stats(agent, state, mask):
    tensor = torch.from_numpy(
        np.asarray(state, dtype=np.float32)
    ).unsqueeze(0)
    with torch.inference_mode():
        q = agent.online(tensor)[0].detach().cpu().numpy()
    available = q[mask]
    order = np.sort(available)[::-1]
    gap = float(order[0] - order[1]) if len(order) > 1 else None
    return {
        "q_mean": float(available.mean()),
        "q_std": float(available.std()),
        "q_min": float(available.min()),
        "q_max": float(available.max()),
        "q_gap_top2": gap,
        "nan": int(np.isnan(q).sum()),
        "inf": int(np.isinf(q).sum()),
        "greedy_action": int(masked_argmax(q, mask)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    quality_payload = read_json(QUALITY_PRIOR_PATH)
    if quality_payload is None:
        raise SystemExit("Missing quality prior")
    quality = flat_quality(quality_payload)
    specs_payload = load_split_specs("dev")
    topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    projector = VisualProjector.load(VISUAL_PCA_PATH)
    device, encoder, backends = load_runtime("cpu")
    gab_config = load_gab_config()
    scenario = "balanced"
    spec = specs_payload["scenarios"][scenario]["episodes"][0]
    scene = build_scene(spec, encoder, device)
    runner = EpisodeRunner(
        spec=spec,
        topology=topologies[scenario],
        encoder=encoder,
        backends=backends,
        projector=projector,
        quality_prior=quality,
        gab_config=gab_config,
        scene=scene,
        device=device,
    )
    states = []
    masks = []
    for _ in range(STEPS):
        states.append(runner.step_data["state"].copy())
        masks.append(runner.step_data["mask"].copy())
        action = runner.select_node("min_latency")
        transition = runner.apply(action)
        if transition["done"]:
            break
    masks = np.asarray(masks, dtype=bool)

    checkpoints = {
        "best_ep800": DQN_DIR / "best.pt",
        "last_ep2000": DQN_DIR / "last.pt",
    }
    output = {
        "states_episode": spec["episode_uid"],
        "states": len(states),
        "policy_for_state_generation": "min_latency",
        "checkpoints": {},
    }
    q_rows = {}
    for name, path in checkpoints.items():
        agent = DQNAgent(device, seed=20261031)
        payload = agent.load(path)
        rows = [q_stats(agent, state, mask) for state, mask in zip(states, masks)]
        q_rows[name] = rows
        output["checkpoints"][name] = {
            "path": str(path),
            "episode": int(payload.get("episode", 0)),
            "total_steps": int(payload.get("total_steps", 0)),
            "q_mean": float(np.mean([row["q_mean"] for row in rows])),
            "q_mean_of_state_means": [
                float(row["q_mean"]) for row in rows
            ],
            "q_max": float(np.max([row["q_max"] for row in rows])),
            "q_min": float(np.min([row["q_min"] for row in rows])),
            "q_gap_top2_mean": float(np.mean([
                row["q_gap_top2"] for row in rows if row["q_gap_top2"] is not None
            ])),
            "nan_total": int(sum(row["nan"] for row in rows)),
            "inf_total": int(sum(row["inf"] for row in rows)),
            "greedy_actions": [row["greedy_action"] for row in rows],
        }
    best_q = np.asarray([
        row["q_mean"] for row in q_rows["best_ep800"]
    ])
    last_q = np.asarray([
        row["q_mean"] for row in q_rows["last_ep2000"]
    ])
    output["comparison"] = {
        "mean_abs_q_mean_difference": float(np.mean(np.abs(best_q - last_q))),
        "correlation_q_mean": float(np.corrcoef(best_q, last_q)[0, 1]),
        "greedy_action_agreement": float(np.mean([
            left["greedy_action"] == right["greedy_action"]
            for left, right in zip(
                q_rows["best_ep800"], q_rows["last_ep2000"]
            )
        ])),
    }
    path = OUTPUT_DIR / "q_value_audit.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("WROTE", path)


if __name__ == "__main__":
    main()
