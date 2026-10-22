"""Stage-2m Q-value overestimation audit on the fixed-city Dev tasks.

For every checkpoint the audit collects deterministic greedy rollouts on the
60 fixed Dev tasks, computes the Monte-Carlo discounted return of each visited
state (gamma = 1.0), and compares it with the network's Q value:

    bias_t = Q(s_t, a_t) - G_t,        G_t = sum_{k>=t} r_k

It also evaluates all checkpoints on one common Min-Latency behaviour state
set, recomputes the DQN / fixed / Double-DQN Bellman targets on the stored
transitions, and checks action-mask, terminal-state and reward invariants.

Usage:
    python v83_spatial_stage2/stage2l_q_audit.py \
        --checkpoints pre_best=runs/.../A/dqn/checkpoints/best.pt \
        --output-dir runs/.../stage2m_q_audit/pre
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    atomic_write_json,
    utc_now,
    write_csv,
)
from v83_spatial_stage2.stage2k_dqn import QNetwork
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    STATE_DIM_VISUAL_ABLATION_V2,
    VISUAL_DIM,
)
from v83_spatial_stage2.stage2k_runtime import load_gab_config, load_runtime
from v83_spatial_stage2.stage2k_state import VisualProjector, masked_argmax
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior
from v83_spatial_stage2.stage2l_fixed_city import MAP_SEED_PHASE1
from v83_spatial_stage2.stage2l_protocol import (
    CAPABILITY_PRIOR_PATH,
    load_bundle,
    resolve_device,
    set_threads,
)
from v83_spatial_stage2.stage2l_runtime import (
    FixedCityEpisodeRunner,
    FixedCitySceneCache,
)


def parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--checkpoints expects name=path"
        )
    name, path = value.split("=", 1)
    return str(name), Path(path)


class CheckpointQ:
    """Online and target Q networks loaded from a Stage-2l checkpoint."""

    def __init__(self, path, device):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" not in payload:
            raise ValueError(f"{path} has no state_dict")
        config = dict(payload.get("config", {}))
        self.path = str(path)
        self.device = torch.device(device)
        self.state_dim = int(
            config.get("state_dim", payload["state_dict"][
                "net.0.weight"
            ].shape[1])
        )
        self.state_mode = str(config.get("state_mode", "pure_mec_v2"))
        if self.state_mode not in ("pure_mec_v2", "visual_ablation_v2"):
            raise ValueError(
                f"unsupported checkpoint state_mode {self.state_mode}"
            )
        self.double_dqn = bool(config.get("double_dqn", False))
        self.legacy_target = bool(config.get("legacy_target", False))
        self.episode = payload.get("episode")
        self.validation = payload.get("validation") or payload.get(
            "last_validation"
        )
        self.total_steps = int(payload.get("total_steps", 0))
        self.online = QNetwork(state_dim=self.state_dim).to(self.device)
        self.online.load_state_dict(payload["state_dict"])
        self.online.eval()
        self.target = QNetwork(state_dim=self.state_dim).to(self.device)
        if "target_state_dict" in payload:
            self.target.load_state_dict(payload["target_state_dict"])
        else:
            self.target.load_state_dict(payload["state_dict"])
        self.target.eval()
        self.payload = payload

    @torch.inference_mode()
    def q_all(self, states):
        tensor = torch.as_tensor(
            np.asarray(states, dtype=np.float32), device=self.device
        )
        return self.online(tensor).cpu().numpy()

    @torch.inference_mode()
    def target_all(self, states):
        tensor = torch.as_tensor(
            np.asarray(states, dtype=np.float32), device=self.device
        )
        return self.target(tensor).cpu().numpy()

    def q_step(self, state):
        return self.q_all(np.asarray(state, dtype=np.float32)[None, :])[0]

    def expected_dim(self):
        return (
            STATE_DIM_VISUAL_ABLATION_V2
            if self.state_mode == "visual_ablation_v2"
            else 384
        )


def project_state(checkpoint, state):
    """Return the checkpoint's view of a 512-D collected state."""
    state = np.asarray(state, dtype=np.float32)
    if checkpoint.state_mode == "visual_ablation_v2":
        if state.size != STATE_DIM_VISUAL_ABLATION_V2:
            raise ValueError(
                f"visual checkpoint got {state.size}-D state"
            )
        return state
    if state.size == 512:
        return state[VISUAL_DIM:]
    if state.size == 384:
        return state
    raise ValueError(f"unexpected state dim {state.size}")


def make_runner(spec, bundle, encoder, backends, quality, gab_config,
                scene_cache, device, projector, visual):
    topology, role = None, None
    from v83_spatial_stage2.stage2l_protocol import topology_for

    topology, role = topology_for(bundle, "dev", spec)
    runner = FixedCityEpisodeRunner(
        spec=spec,
        topology=topology,
        encoder=encoder,
        backends=backends,
        projector=projector,
        quality_prior=quality,
        gab_config=gab_config,
        scene=scene_cache.get(spec),
        device=device,
        jointr_aligned=True,
        pure_mec_v2_state=not visual,
        visual_ablation_v2_state=visual,
    )
    return runner, role


def collect_on_policy(checkpoint, bundle, runtime, projector, visual,
                      device, limit):
    encoder, backends, quality, gab_config, scene_cache = runtime
    steps = []
    episodes = []
    for spec in bundle["dev_specs"][:limit]:
        runner, role = make_runner(
            spec, bundle, encoder, backends, quality, gab_config,
            scene_cache, device, projector, visual,
        )
        trajectory = []
        while True:
            state = runner.step_data["state"]
            view = project_state(checkpoint, state)
            mask = runner.step_data["mask"]
            q_values = checkpoint.q_step(view)
            action = int(masked_argmax(q_values, mask))
            available = np.where(mask)[0]
            step_index = int(runner.step_data["step"])
            transition = runner.apply(action)
            trajectory.append({
                "state": view.copy(),
                "full_state": np.asarray(state, dtype=np.float32).copy(),
                "mask": mask.copy(),
                "action": action,
                "reward": float(transition["reward"]),
                "done": bool(transition["done"]),
                "q_selected": float(q_values[action]),
                "q_max": float(q_values[available].max()),
                "q_mean": float(q_values[available].mean()),
                "available": int(len(available)),
                "step": step_index,
            })
            if transition["done"]:
                break
        result = runner.finalize("q_audit_greedy")
        for item in trajectory:
            item["episode_uid"] = spec["episode_uid"]
            item["band"] = spec["band"]
            item["episode_role"] = role
            steps.append(item)
        episodes.append({
            "episode_uid": spec["episode_uid"],
            "band": spec["band"],
            "steps": int(result["steps"]),
            "episode_return": float(result["episode_return"]),
            "episode_jointR": float(result["episode_jointR"]),
            "jointr_abs_error": float(result["jointr_abs_error"]),
            "success": bool(result["success"]),
            "nDTW": float(result["final_ndtw"]),
        })
        # Monte-Carlo return with gamma = 1.0.
        running = 0.0
        for item in reversed(trajectory):
            running += item["reward"]
            item["mc_return"] = running
            item["bias"] = item["q_selected"] - running
            item["bias_max"] = item["q_max"] - running
    return steps, episodes


def collect_fixed_behavior(bundle, runtime, visual, device, limit,
                           projector=None):
    encoder, backends, quality, gab_config, scene_cache = runtime
    steps = []
    episodes = []
    for spec in bundle["dev_specs"][:limit]:
        runner, role = make_runner(
            spec, bundle, encoder, backends, quality, gab_config,
            scene_cache, device, projector=projector, visual=visual,
        )
        trajectory = []
        while True:
            state = np.asarray(runner.step_data["state"], dtype=np.float32)
            mask = runner.step_data["mask"].copy()
            action = runner.select_node("min_latency")
            transition = runner.apply(action)
            trajectory.append({
                "state": state.copy(),
                "mask": mask,
                "action": int(action),
                "reward": float(transition["reward"]),
                "done": bool(transition["done"]),
                "available": int(mask.sum()),
                "next_state_is_none": transition["next_state"] is None,
                "next_mask_is_none": transition["next_mask"] is None,
            })
            if transition["done"]:
                break
        result = runner.finalize("min_latency")
        running = 0.0
        for item in reversed(trajectory):
            running += item["reward"]
            item["mc_return"] = running
            item["episode_uid"] = spec["episode_uid"]
            item["band"] = spec["band"]
            steps.append(item)
        episodes.append({
            "episode_uid": spec["episode_uid"],
            "band": spec["band"],
            "steps": int(result["steps"]),
            "episode_return": float(result["episode_return"]),
            "episode_jointR": float(result["episode_jointR"]),
            "jointr_abs_error": float(result["jointr_abs_error"]),
            "success": bool(result["success"]),
            "nDTW": float(result["final_ndtw"]),
        })
    return steps, episodes


def summarize_bias(steps):
    bias = np.asarray([item["bias"] for item in steps], dtype=np.float64)
    bias_max = np.asarray(
        [item["bias_max"] for item in steps], dtype=np.float64
    )
    q_selected = np.asarray(
        [item["q_selected"] for item in steps], dtype=np.float64
    )
    q_max = np.asarray(
        [item["q_max"] for item in steps], dtype=np.float64
    )
    q_mean = np.asarray([item["q_mean"] for item in steps])
    returns = np.asarray([item["mc_return"] for item in steps])
    correlation = float(np.corrcoef(q_selected, returns)[0, 1]) if (
        len(steps) > 1 and np.std(q_selected) > 0 and np.std(returns) > 0
    ) else float("nan")
    return {
        "n_steps": int(len(steps)),
        "q_selected_mean": float(q_selected.mean()),
        "q_max_mean": float(q_max.mean()),
        "q_available_mean": float(q_mean.mean()),
        "q_global_max": float(q_max.max()),
        "mc_return_mean": float(returns.mean()),
        "bias_mean": float(bias.mean()),
        "bias_median": float(np.median(bias)),
        "bias_std": float(bias.std()),
        "mae": float(np.abs(bias).mean()),
        "rmse": float(np.sqrt((bias ** 2).mean())),
        "overestimate_fraction": float((bias > 0).mean()),
        "bias_max_mean": float(bias_max.mean()),
        "mae_max": float(np.abs(bias_max).mean()),
        "pearson_q_vs_mc": correlation,
        "first_step_q_mean": float(np.mean([
            item["q_selected"] for item in steps if item["step"] == 0
        ])) if any(item["step"] == 0 for item in steps) else None,
        "first_step_return_mean": float(np.mean([
            item["mc_return"] for item in steps if item["step"] == 0
        ])) if any(item["step"] == 0 for item in steps) else None,
    }


def summarize_fixed(checkpoint, behavior_steps):
    rows = [project_state(checkpoint, item["state"])
            for item in behavior_steps]
    q_all = []
    for start in range(0, len(rows), 2048):
        q_all.append(checkpoint.q_all(np.asarray(rows[start:start + 2048])))
    q_all = np.concatenate(q_all, axis=0)
    masks = np.asarray([item["mask"] for item in behavior_steps], dtype=bool)
    actions = np.asarray([item["action"] for item in behavior_steps])
    returns = np.asarray(
        [item["mc_return"] for item in behavior_steps], dtype=np.float64
    )
    q_selected = q_all[np.arange(len(actions)), actions]
    q_max = np.where(masks, q_all, -np.inf).max(axis=1)
    q_mean = np.where(masks, q_all, 0.0).sum(axis=1) / masks.sum(axis=1)
    bias = q_selected - returns
    return {
        "n_steps": int(len(actions)),
        "q_selected_mean": float(q_selected.mean()),
        "q_max_mean": float(q_max.mean()),
        "q_available_mean": float(q_mean.mean()),
        "q_global_max": float(q_all.max()),
        "mc_return_mean": float(returns.mean()),
        "bias_mean": float(bias.mean()),
        "mae": float(np.abs(bias).mean()),
        "overestimate_fraction": float((bias > 0).mean()),
        "pearson_q_vs_mc": float(np.corrcoef(q_selected, returns)[0, 1])
        if np.std(q_selected) > 0 and np.std(returns) > 0 else None,
    }


def audit_targets(checkpoint, behavior_steps):
    """Recompute legacy / fixed / Double-DQN targets on stored transitions."""
    states = np.asarray([item["state"] for item in behavior_steps])
    masks = np.asarray([item["mask"] for item in behavior_steps], dtype=bool)
    rewards = np.asarray([item["reward"] for item in behavior_steps])
    dones = np.asarray([item["done"] for item in behavior_steps], dtype=bool)
    next_states = np.zeros_like(states)
    next_masks = np.zeros_like(masks)
    next_states[:-1] = states[1:]
    next_masks[:-1] = masks[1:]
    valid = ~dones
    next_states[~valid] = 0.0
    next_masks[~valid] = False

    target_q = checkpoint.target_all(next_states)
    with torch.no_grad():
        online_q = checkpoint.online(
            torch.as_tensor(next_states, dtype=torch.float32,
                            device=checkpoint.device)
        ).cpu().numpy()
    legacy = np.where(next_masks, target_q, 0.0).max(axis=1)
    any_available = next_masks.any(axis=1)
    fixed = np.where(next_masks, target_q, -1e9).max(axis=1)
    fixed = np.where(any_available, fixed, 0.0)
    dbl_action = np.where(next_masks, online_q, -1e9).argmax(axis=1)
    dbl = target_q[np.arange(len(dbl_action)), dbl_action]
    dbl = np.where(any_available, dbl, 0.0)
    gamma = 1.0
    target_legacy = rewards + gamma * legacy * (~dones)
    target_fixed = rewards + gamma * fixed * (~dones)
    target_double = rewards + gamma * dbl * (~dones)

    available = masks.sum(axis=1)
    phantom = (legacy > fixed) & (~dones)
    negative_available = (np.where(next_masks, target_q, np.inf).max(
        axis=1
    ) < 0.0) & any_available
    terminal = dones
    done_with_next_state = sum(
        bool(item["done"]) and not item["next_state_is_none"]
        for item in behavior_steps
    )
    done_with_next_mask = sum(
        bool(item["done"]) and not item["next_mask_is_none"]
        for item in behavior_steps
    )
    return {
        "checkpoint": checkpoint.path,
        "episode": checkpoint.episode,
        "n_transitions": int(len(states)),
        "done_transitions": int(terminal.sum()),
        "done_with_next_state": int(done_with_next_state),
        "done_with_next_mask": int(done_with_next_mask),
        "terminal_next_state_all_zero": bool(
            np.all(next_states[terminal] == 0.0)
        ) if terminal.any() else True,
        "terminal_next_mask_all_false": bool(
            np.all(~next_masks[terminal])
        ) if terminal.any() else True,
        "terminal_bootstrap_zero": bool(
            np.all(target_fixed[terminal] == rewards[terminal])
        ) if terminal.any() else True,
        "mask_min": int(available.min()),
        "mask_mean": float(available.mean()),
        "mask_max": int(available.max()),
        "all_false_mask": int((available == 0).sum()),
        "phantom_zero_transitions": int(phantom.sum()),
        "phantom_fraction": float(phantom.mean()),
        "negative_available_next_states": int(negative_available.sum()),
        "target_legacy_mean": float(target_legacy.mean()),
        "target_fixed_mean": float(target_fixed.mean()),
        "target_double_mean": float(target_double.mean()),
        "legacy_minus_fixed_mae": float(
            np.abs(target_legacy - target_fixed).mean()
        ),
        "legacy_minus_fixed_max": float(
            np.abs(target_legacy - target_fixed).max()
        ),
        "double_minus_fixed_mae": float(
            np.abs(target_double - target_fixed).mean()
        ),
        "double_minus_fixed_max": float(
            np.abs(target_double - target_fixed).max()
        ),
        "double_argmax_available_fraction": float(
            next_masks[np.arange(len(dbl_action)), dbl_action].mean()
        ),
    }


def write_report(path, summary):
    lines = [
        "# Stage-2m Q 值高估审计",
        "",
        f"- 生成时间：{summary['created_utc']}",
        f"- 固定城市：`map_seed={summary['map_seed']}`，配置 A",
        f"- Dev 任务：{summary['dev_episodes']} 条（固定任务池）",
        f"- 行为策略（共同状态集）：`{summary['behavior']}`",
        f"- γ = {summary['gamma']}，奖励与 JointR 严格一致",
        "",
        "## 1. 同策略 Q vs Monte-Carlo 回报（主判据）",
        "",
        "| Checkpoint | Ep | Q(s,a) mean | Q max mean | G_t mean | Bias mean | MAE | 高估比例 | Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["on_policy"]:
        m = item["metrics"]
        lines.append(
            f"| {item['name']} | {item['episode']} | "
            f"{m['q_selected_mean']:.4f} | {m['q_max_mean']:.4f} | "
            f"{m['mc_return_mean']:.4f} | {m['bias_mean']:.4f} | "
            f"{m['mae']:.4f} | {m['overestimate_fraction']:.1%} | "
            f"{m['pearson_q_vs_mc']:.3f} |"
        )
    lines += [
        "",
        "## 2. 共同 Min-Latency 状态集上的 Q（跨算法可比）",
        "",
        "| Checkpoint | Q(s,a) mean | Q max mean | G_t mean | Bias mean | MAE | 高估比例 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["fixed_behavior"]:
        m = item["metrics"]
        lines.append(
            f"| {item['name']} | {m['q_selected_mean']:.4f} | "
            f"{m['q_max_mean']:.4f} | {m['mc_return_mean']:.4f} | "
            f"{m['bias_mean']:.4f} | {m['mae']:.4f} | "
            f"{m['overestimate_fraction']:.1%} |"
        )
    lines += [
        "",
        "## 3. 实现审计（Bellman Target / Mask / 终止 / 奖励）",
        "",
        "| Checkpoint | 终止转换 | 终止带 next_state | 终止 bootstrap=0 | 掩码均值 | 全 False 掩码 | 幻影 0 目标 | legacy−fixed MAE | double−fixed MAE |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["target_audit"]:
        lines.append(
            f"| {item['checkpoint']} | {item['done_transitions']} | "
            f"{item['done_with_next_state']} | "
            f"{item['terminal_bootstrap_zero']} | {item['mask_mean']:.2f} | "
            f"{item['all_false_mask']} | "
            f"{item['phantom_zero_transitions']} | "
            f"{item['legacy_minus_fixed_mae']:.5f} | "
            f"{item['double_minus_fixed_mae']:.5f} |"
        )
    lines += [
        "",
        "奖励一致性：所有审计 Episode 的 `episode_return` 与 `episode_JointR` "
        f"最大绝对误差 = {summary['max_reward_error']:.2e}。",
        "",
        "## 4. 结论口径",
        "",
        "- `Bias mean = Q(s,a) − G_t`，`G_t` 是同一贪心策略轨迹上的实际折扣回报；",
        "  正值代表 Q 高估，负值代表 Q 低估。",
        "- 共同状态集使用确定性 Min-Latency 行为策略，G_t 是该行为策略的回报，",
        "  因此 Q* 与 G_t 的差包含策略改进项；该表只用于 DQN/DDQN 的相对比较。",
        "- 幻影 0 目标：旧实现用 0 填充不可用动作后取 max，当所有可用动作的",
        "  Q 都为负时会从虚构的 0 动作自举；本表统计该情形在审计状态上的出现次数。",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="A", choices=("A", "B", "C"))
    parser.add_argument(
        "--checkpoints", action="append", type=parse_checkpoint,
        required=True, help="name=path (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-episodes", type=int, default=60)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--visual-pca", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2k/visual_pca.npz"),
    )
    args = parser.parse_args()
    if abs(args.gamma - 1.0) > 1e-12:
        raise SystemExit("This audit is defined for gamma=1.0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_threads(args.threads)
    device = resolve_device(args.device)
    bundle = load_bundle(args.config)
    prior_payload, quality = load_capability_prior(CAPABILITY_PRIOR_PATH)
    device, encoder, backends = load_runtime(str(device))
    gab_config = load_gab_config()
    scene_cache = FixedCitySceneCache(
        encoder, device, max_entries=max(64, len(bundle["dev_specs"]))
    )
    runtime = (encoder, backends, quality, gab_config, scene_cache)

    checkpoints = [
        (name, CheckpointQ(path, device))
        for name, path in args.checkpoints
    ]
    any_visual = any(
        checkpoint.state_mode == "visual_ablation_v2"
        for _, checkpoint in checkpoints
    )
    projector = VisualProjector.load(args.visual_pca) if any_visual else None

    on_policy = []
    all_steps = []
    all_episodes = []
    for name, checkpoint in checkpoints:
        steps, episodes = collect_on_policy(
            checkpoint, bundle, runtime, projector, any_visual,
            device, int(args.limit_episodes),
        )
        all_steps.extend(steps)
        all_episodes.extend(episodes)
        on_policy.append({
            "name": name,
            "path": checkpoint.path,
            "episode": checkpoint.episode,
            "state_mode": checkpoint.state_mode,
            "double_dqn": checkpoint.double_dqn,
            "legacy_target": checkpoint.legacy_target,
            "metrics": summarize_bias(steps),
            "policy": {
                "SR": float(np.mean([ep["success"] for ep in episodes])),
                "nDTW": float(np.mean([ep["nDTW"] for ep in episodes])),
                "JointR": float(np.mean(
                    [ep["episode_jointR"] for ep in episodes]
                )),
                "mean_steps": float(np.mean(
                    [ep["steps"] for ep in episodes]
                )),
            },
        })
        csv_path = args.output_dir / f"per_step_{name}.csv"
        write_csv(
            csv_path,
            [{
                "episode_uid": item["episode_uid"],
                "band": item["band"],
                "step": item["step"],
                "available": item["available"],
                "action": item["action"],
                "reward": item["reward"],
                "q_selected": item["q_selected"],
                "q_max": item["q_max"],
                "q_mean": item["q_mean"],
                "mc_return": item["mc_return"],
                "bias": item["bias"],
                "bias_max": item["bias_max"],
            } for item in steps],
            fieldnames=[
                "episode_uid", "band", "step", "available", "action",
                "reward", "q_selected", "q_max", "q_mean", "mc_return",
                "bias", "bias_max",
            ],
        )

    behavior_steps, behavior_episodes = collect_fixed_behavior(
        bundle, runtime, any_visual, device, int(args.limit_episodes),
        projector=projector,
    )
    fixed_behavior = []
    target_audit = []
    for name, checkpoint in checkpoints:
        fixed_behavior.append({
            "name": name,
            "path": checkpoint.path,
            "episode": checkpoint.episode,
            "metrics": summarize_fixed(checkpoint, behavior_steps),
        })
        target_audit.append(audit_targets(checkpoint, behavior_steps))

    max_reward_error = max(
        [ep["jointr_abs_error"] for ep in all_episodes]
        + [ep["jointr_abs_error"] for ep in behavior_episodes]
    )
    summary = {
        "version": "stage2m-q-audit-v1",
        "created_utc": utc_now(),
        "config": args.config,
        "map_seed": int(MAP_SEED_PHASE1),
        "dev_episodes": int(args.limit_episodes),
        "behavior": "min_latency (fixed common state set)",
        "gamma": float(args.gamma),
        "checkpoints": [
            {"name": name, "path": str(path)}
            for name, path in args.checkpoints
        ],
        "on_policy": on_policy,
        "fixed_behavior": fixed_behavior,
        "target_audit": target_audit,
        "behavior_policy": {
            "SR": float(np.mean([ep["success"] for ep in behavior_episodes])),
            "nDTW": float(np.mean([ep["nDTW"] for ep in behavior_episodes])),
            "JointR": float(np.mean(
                [ep["episode_jointR"] for ep in behavior_episodes]
            )),
        },
        "max_reward_error": float(max_reward_error),
        "final_test_read": False,
        "test_pool_read": False,
    }
    atomic_write_json(args.output_dir / "q_audit.json", summary)
    write_report(args.output_dir / "REPORT.md", summary)
    print(json.dumps({
        "output": str(args.output_dir),
        "on_policy": [
            {
                "name": item["name"],
                "q_mean": item["metrics"]["q_selected_mean"],
                "mc_mean": item["metrics"]["mc_return_mean"],
                "bias": item["metrics"]["bias_mean"],
                "mae": item["metrics"]["mae"],
            }
            for item in on_policy
        ],
        "fixed_behavior": [
            {
                "name": item["name"],
                "bias": item["metrics"]["bias_mean"],
                "mae": item["metrics"]["mae"],
            }
            for item in fixed_behavior
        ],
        "max_reward_error": float(max_reward_error),
    }, indent=2, ensure_ascii=False), flush=True)
    print("STAGE2M_Q_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
