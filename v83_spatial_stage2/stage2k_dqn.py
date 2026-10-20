"""Stage-2k compact DQN association agent (MLP only).

The DQN chooses one of 20 MEC nodes from the 478-D state.  It never generates
UAV navigation actions and never updates a MEC.  The five Stage-2j MECs stay
frozen in eval mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.tensorboard import SummaryWriter

from v83_spatial_stage2.stage2k_common import (
    DQN_DIR,
    MODEL_NAMES,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    SCENARIO_TITLES,
    TENSORBOARD_DIR,
    VISUAL_PCA_PATH,
    atomic_write_json,
    load_progress,
    load_split_specs,
    load_topology,
    read_json,
    sha256_file,
    summarize_episodes,
    update_progress,
    utc_now,
    validation_specs,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    SceneCache,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import (
    STATE_DIM,
    VisualProjector,
    masked_argmax,
)


SEED = 20261031
HIDDEN_DIM = 256
GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
REPLAY_CAPACITY = 50_000
WARMUP_STEPS = 2_000
TARGET_UPDATE_STEPS = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 200_000
MAX_EPISODES = 4000
MIN_EPISODES = 800
VALIDATION_EVERY = 200
VALIDATION_IMPROVEMENT = 0.002
PATIENCE_VALIDATIONS = 5
EXTRA_VALIDATIONS = 3
GRAD_CLIP = 10.0


class QNetwork(nn.Module):
    """Standard 478 -> 256 -> 256 -> 20 MLP Q-network."""

    def __init__(self, state_dim=STATE_DIM, action_dim=20, hidden=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY, state_dim=STATE_DIM,
                 action_dim=20, seed=SEED):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, action_dim), dtype=bool)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done, next_mask):
        self.states[self.position] = np.asarray(state, dtype=np.float32)
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.next_states[self.position] = np.asarray(
            next_state if next_state is not None else np.zeros(
                self.state_dim, dtype=np.float32
            ),
            dtype=np.float32,
        )
        self.dones[self.position] = float(done)
        self.next_masks[self.position] = np.asarray(
            next_mask if next_mask is not None else np.zeros(
                self.action_dim, dtype=bool
            ),
            dtype=bool,
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "state": torch.from_numpy(self.states[indices]).to(device),
            "action": torch.from_numpy(self.actions[indices]).to(device),
            "reward": torch.from_numpy(self.rewards[indices]).to(device),
            "next_state": torch.from_numpy(
                self.next_states[indices]
            ).to(device),
            "done": torch.from_numpy(self.dones[indices]).to(device),
            "next_mask": torch.from_numpy(
                self.next_masks[indices]
            ).to(device),
        }

    def __len__(self):
        return self.size

    def state_dict(self):
        """Serializable state for strict resume (valid entries only)."""
        count = int(self.size)
        return {
            "capacity": int(self.capacity),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "size": count,
            "position": int(self.position),
            "states": self.states[:count].copy(),
            "actions": self.actions[:count].copy(),
            "rewards": self.rewards[:count].copy(),
            "next_states": self.next_states[:count].copy(),
            "dones": self.dones[:count].copy(),
            "next_masks": self.next_masks[:count].copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, payload):
        self.capacity = int(payload["capacity"])
        self.state_dim = int(payload["state_dim"])
        self.action_dim = int(payload["action_dim"])
        self.states = np.zeros(
            (self.capacity, self.state_dim), dtype=np.float32
        )
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, self.state_dim), dtype=np.float32
        )
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.zeros(
            (self.capacity, self.action_dim), dtype=bool
        )
        count = int(payload["size"])
        if count > self.capacity:
            raise ValueError("Replay buffer payload exceeds capacity")
        self.states[:count] = payload["states"][:count]
        self.actions[:count] = payload["actions"][:count]
        self.rewards[:count] = payload["rewards"][:count]
        self.next_states[:count] = payload["next_states"][:count]
        self.dones[:count] = payload["dones"][:count]
        self.next_masks[:count] = payload["next_masks"][:count]
        self.size = count
        self.position = int(payload["position"])
        self.rng.bit_generator.state = payload["rng_state"]


class DQNAgent:
    def __init__(self, device, seed=SEED):
        self.device = torch.device(device)
        torch.manual_seed(int(seed))
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad = False
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=LEARNING_RATE
        )
        self.buffer = ReplayBuffer(seed=seed)
        self.rng = np.random.default_rng(int(seed))
        self.total_steps = 0
        self.update_steps = 0

    def epsilon(self):
        fraction = min(
            1.0, float(self.total_steps) / float(EPSILON_DECAY_STEPS)
        )
        return float(
            EPSILON_START + (EPSILON_END - EPSILON_START) * fraction
        )

    def act(self, state, mask, epsilon=None):
        available = np.where(np.asarray(mask, dtype=bool))[0]
        if len(available) == 0:
            raise ValueError("No available MEC")
        epsilon = self.epsilon() if epsilon is None else float(epsilon)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(available))
        return self.act_greedy(state, mask)

    @torch.inference_mode()
    def act_greedy(self, state, mask):
        tensor = torch.from_numpy(
            np.asarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        q_values = self.online(tensor)[0].detach().cpu().numpy()
        return masked_argmax(q_values, mask)

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None
        batch = self.buffer.sample(BATCH_SIZE, self.device)
        with torch.no_grad():
            next_q = self.target(batch["next_state"])
            next_mask = batch["next_mask"]
            next_q = next_q.masked_fill(~next_mask, 0.0)
            next_max = next_q.max(dim=1).values
            target = batch["reward"] + GAMMA * next_max * (
                1.0 - batch["done"]
            )
        q = self.online(batch["state"]).gather(
            1, batch["action"].unsqueeze(1)
        ).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), GRAD_CLIP)
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % TARGET_UPDATE_STEPS == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def save(self, path, extra=None):
        payload = {
            "state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": int(self.total_steps),
            "update_steps": int(self.update_steps),
            "architecture": "mlp_478_256_256_20",
            "state_dim": STATE_DIM,
            "action_dim": 20,
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "replay_capacity": REPLAY_CAPACITY,
            "seed": SEED,
            "final_test_read": False,
        }
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load(self, path):
        payload = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self.online.load_state_dict(payload["state_dict"], strict=True)
        self.target.load_state_dict(
            payload.get("target_state_dict", payload["state_dict"]),
            strict=True,
        )
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(
                payload["optimizer_state_dict"]
            )
        self.total_steps = int(payload.get("total_steps", 0))
        self.update_steps = int(payload.get("update_steps", 0))
        return payload


def flat_quality(quality):
    return {
        name: float(quality["models"][name]["mean_ndtw"])
        for name in MODEL_NAMES
    }


def interleaved_training_order(spec_payload):
    by_scenario = {
        scenario: spec_payload["scenarios"][scenario]["episodes"]
        for scenario in SCENARIOS
    }
    order = []
    max_len = max(len(items) for items in by_scenario.values())
    for index in range(max_len):
        for scenario in SCENARIOS:
            items = by_scenario[scenario]
            if index < len(items):
                order.append(items[index]["episode_uid"])
    if len(order) != sum(
        len(items) for items in by_scenario.values()
    ):
        raise RuntimeError("Training order does not cover all episodes")
    return order


def spec_lookup(spec_payload):
    return {
        item["episode_uid"]: item
        for scenario in SCENARIOS
        for item in spec_payload["scenarios"][scenario]["episodes"]
    }


def dev_validation_specs():
    """Fixed 60 Dev Validation episodes used for DQN model selection.

    Training specs must never be used here.  The interrupted 2k run passed
    ``load_split_specs("train")`` to ``validation_specs``, so its recorded
    "validation" was in-sample; see STAGE2K_ASSOCIATION_RL_REPORT.md §7.2.
    """
    return validation_specs(load_split_specs("dev"))


def evaluate_dqn(
    agent,
    specs,
    topologies,
    encoder,
    backends,
    projector,
    quality,
    gab_config,
    device,
    scene_cache=None,
):
    rows = []
    for spec in specs:
        runner = EpisodeRunner(
            spec=spec,
            topology=topologies[spec["scenario"]],
            encoder=encoder,
            backends=backends,
            projector=projector,
            quality_prior=quality,
            gab_config=gab_config,
            scene_cache=scene_cache,
            device=device,
        )
        while True:
            state = runner.step_data["state"]
            mask = runner.step_data["mask"]
            action = agent.act_greedy(state, mask)
            transition = runner.apply(action)
            if transition["done"]:
                break
        rows.append(runner.finalize("dqn"))
    return rows


def should_early_stop(episode, no_improve, validation_joints):
    if int(episode) < MIN_EPISODES:
        return False, None
    if int(no_improve) >= PATIENCE_VALIDATIONS + EXTRA_VALIDATIONS:
        return True, (
            "no new best JointR after "
            f"{PATIENCE_VALIDATIONS}+{EXTRA_VALIDATIONS} validations"
        )
    if int(no_improve) >= PATIENCE_VALIDATIONS and len(validation_joints) >= 3:
        trend = float(np.polyfit(
            np.arange(3, dtype=np.float64),
            np.asarray(validation_joints[-3:], dtype=np.float64),
            1,
        )[0])
        if trend <= 1e-9:
            return True, (
                "no JointR improvement over 5 validations and last-3 "
                "trend is flat/declining"
            )
    return False, None


def plot_training_curves(history, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray([
        row["episode"] for row in history["train"]
    ], dtype=np.float64)
    returns = np.asarray([
        row["return"] for row in history["train"]
    ], dtype=np.float64)
    if len(returns):
        window = 100
        ma = np.convolve(
            returns, np.ones(window) / window, mode="valid"
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, returns, color="#c9c8c2", linewidth=0.8,
                label="episode return")
        ax.plot(
            episodes[window - 1:], ma, color="#2a78d6", linewidth=2,
            label="100-episode moving average",
        )
        ax.set_xlabel("training episode")
        ax.set_ylabel("episode return")
        ax.grid(color="#e4e3df", linewidth=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_return_ma100.png", dpi=180)
        fig.savefig(output_dir / "training_return_ma100.svg")
        plt.close(fig)
    validation = history["validation"]
    if validation:
        x = np.asarray([row["episode"] for row in validation])
        y = np.asarray([row["joint_reward"] for row in validation])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color="#2a78d6", marker="o", linewidth=2)
        ax.set_xlabel("training episode")
        ax.set_ylabel("validation JointR")
        ax.grid(color="#e4e3df", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(output_dir / "validation_jointr.png", dpi=180)
        fig.savefig(output_dir / "validation_jointr.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--output-dir", type=Path, default=DQN_DIR)
    args = parser.parse_args()

    progress = load_progress()
    if progress["dqn"]["status"] == "completed" and not args.resume:
        print("DQN training already completed; skipping.")
        return
    if progress["baseline"]["status"] != "completed":
        raise SystemExit(
            "Baseline is not complete; run stage2k_baselines.py first"
        )

    quality_payload = read_json(QUALITY_PRIOR_PATH)
    if quality_payload is None:
        raise SystemExit("Missing quality prior")
    quality = flat_quality(quality_payload)
    train_payload = load_split_specs("train")
    lookup = spec_lookup(train_payload)
    order_path = args.output_dir / "training_order.json"
    if order_path.is_file():
        order = json.loads(order_path.read_text(encoding="utf-8"))["order"]
    else:
        order = interleaved_training_order(train_payload)
        atomic_write_json(order_path, {
            "created_utc": utc_now(),
            "episodes": len(order),
            "order": order,
        })
    if len(order) != len(lookup):
        raise RuntimeError("Training order/lookup size mismatch")

    topologies = {
        scenario: load_topology(scenario, "train")
        for scenario in SCENARIOS
    }
    projector = VisualProjector.load(VISUAL_PCA_PATH)
    device, encoder, backends = load_runtime(args.device)
    gab_config = load_gab_config()

    design_path = Path("v83_spatial_stage2/STAGE2K_STATE_REWARD_DESIGN.md")
    if not design_path.is_file():
        print("WARNING: state/reward design document is not present yet")

    agent = DQNAgent(device, seed=SEED)
    start_episode = int(progress["dqn"]["current_episode"] or 0)
    last_path = args.output_dir / "last.pt"
    if last_path.is_file() and start_episode > 0:
        payload = agent.load(last_path)
        start_episode = max(
            start_episode, int(payload.get("episode", 0))
        )
        print(
            f"Resumed DQN at episode {start_episode} "
            "(replay buffer is re-initialized)",
            flush=True,
        )

    writer = SummaryWriter(str(TENSORBOARD_DIR))
    history = {
        "train": [],
        "validation": [],
        "config": {
            "seed": SEED,
            "state_dim": STATE_DIM,
            "action_dim": 20,
            "hidden_dim": HIDDEN_DIM,
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "replay_capacity": REPLAY_CAPACITY,
            "warmup_steps": WARMUP_STEPS,
            "target_update_steps": TARGET_UPDATE_STEPS,
            "epsilon": [EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS],
            "max_episodes": args.max_episodes,
            "min_episodes": MIN_EPISODES,
            "validation_every": VALIDATION_EVERY,
            "patience_validations": PATIENCE_VALIDATIONS,
            "extra_validations": EXTRA_VALIDATIONS,
            "final_test_read": False,
        },
    }
    history_path = args.output_dir / "training_history.json"
    if history_path.is_file() and start_episode > 0:
        existing = json.loads(history_path.read_text(encoding="utf-8"))
        history["train"] = existing.get("train", [])
        history["validation"] = existing.get("validation", [])

    best_joint = progress["dqn"]["best_validation_joint"]
    best_joint = float(best_joint) if best_joint is not None else -1e9
    no_improve = int(progress["dqn"]["consecutive_no_improve"] or 0)
    best_path = args.output_dir / "best.pt"
    validation_specs_list = dev_validation_specs()
    scene_cache = SceneCache(encoder, device, max_entries=60)
    started = time.time()
    stop_reason = None
    stop_episode = None
    last_episode = int(start_episode)

    for episode_index in range(start_episode, min(len(order), args.max_episodes)):
        last_episode = int(episode_index + 1)
        spec = lookup[order[episode_index]]
        topology = topologies[spec["scenario"]]
        runner = EpisodeRunner(
            spec=spec,
            topology=topology,
            encoder=encoder,
            backends=backends,
            projector=projector,
            quality_prior=quality,
            gab_config=gab_config,
            device=device,
        )
        state = runner.step_data["state"]
        mask = runner.step_data["mask"]
        done = False
        episode_return = 0.0
        nav_sum = 0.0
        latency_sum = 0.0
        e2e_values = []
        losses = []
        while not done:
            epsilon = agent.epsilon()
            action = agent.act(state, mask, epsilon=epsilon)
            transition = runner.apply(action)
            agent.buffer.add(
                state,
                action,
                transition["reward"],
                transition["next_state"],
                transition["done"],
                transition["next_mask"],
            )
            if agent.total_steps >= WARMUP_STEPS:
                loss = agent.train_step()
                if loss is not None:
                    if not np.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite DQN loss at episode "
                            f"{episode_index + 1}"
                        )
                    losses.append(loss)
            agent.total_steps += 1
            episode_return += float(transition["reward"])
            nav_sum += float(transition["nav_reward"])
            latency_sum += float(transition["latency_penalty"])
            e2e_values.append(float(transition["e2e_ms"]))
            state = transition["next_state"]
            mask = transition["next_mask"]
            done = bool(transition["done"])
        episode_metrics = {
            "episode": int(episode_index + 1),
            "episode_uid": spec["episode_uid"],
            "scenario": spec["scenario"],
            "band": spec["band"],
            "return": float(episode_return),
            "navigation_reward": float(nav_sum),
            "latency_penalty": float(latency_sum),
            "mean_e2e_ms": float(np.mean(e2e_values)) if e2e_values else 0.0,
            "epsilon": float(agent.epsilon()),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "success": bool(runner.success),
            "steps": len(runner.step_records),
            "buffer": len(agent.buffer),
            "total_steps": int(agent.total_steps),
        }
        history["train"].append(episode_metrics)
        writer.add_scalar("train/episode_return", episode_metrics["return"],
                          episode_metrics["episode"])
        writer.add_scalar("train/navigation_reward",
                          episode_metrics["navigation_reward"],
                          episode_metrics["episode"])
        writer.add_scalar("train/latency_penalty",
                          episode_metrics["latency_penalty"],
                          episode_metrics["episode"])
        writer.add_scalar("train/mean_e2e_ms",
                          episode_metrics["mean_e2e_ms"],
                          episode_metrics["episode"])
        writer.add_scalar("train/epsilon", episode_metrics["epsilon"],
                          episode_metrics["episode"])
        writer.add_scalar("train/loss", episode_metrics["loss"],
                          episode_metrics["episode"])
        if episode_metrics["episode"] % 10 == 0:
            print(
                f"[DQN] Episode {episode_metrics['episode']}/{len(order)} "
                f"return={episode_metrics['return']:.4f} "
                f"nav={episode_metrics['navigation_reward']:.4f} "
                f"lat={episode_metrics['latency_penalty']:.4f} "
                f"e2e={episode_metrics['mean_e2e_ms']:.1f}ms "
                f"eps={episode_metrics['epsilon']:.3f} "
                f"loss={episode_metrics['loss']:.5f}",
                flush=True,
            )

        if episode_metrics["episode"] % VALIDATION_EVERY == 0:
            validation_rows = evaluate_dqn(
                agent,
                validation_specs_list,
                {scenario: load_topology(scenario, "dev")
                 for scenario in SCENARIOS},
                encoder,
                backends,
                projector,
                quality,
                gab_config,
                device,
                scene_cache=scene_cache,
            )
            summary = summarize_episodes(validation_rows)
            validation_record = {
                "episode": episode_metrics["episode"],
                "joint_reward": summary["JointR"],
                "SR": summary["SR"],
                "nDTW": summary["nDTW"],
                "E2E_ms": summary["E2E_ms"],
                "p95_ms": summary["P95_ms"],
                "mean_steps": summary["mean_steps"],
                "model_usage": summary["model_usage"],
                "timestamp": utc_now(),
            }
            history["validation"].append(validation_record)
            writer.add_scalar("validation/joint_reward",
                              summary["JointR"], episode_metrics["episode"])
            writer.add_scalar("validation/SR", summary["SR"],
                              episode_metrics["episode"])
            writer.add_scalar("validation/nDTW", summary["nDTW"],
                              episode_metrics["episode"])
            writer.add_scalar("validation/E2E_ms", summary["E2E_ms"],
                              episode_metrics["episode"])
            writer.add_scalar("validation/p95_ms", summary["P95_ms"],
                              episode_metrics["episode"])
            writer.add_scalar("validation/mean_steps", summary["mean_steps"],
                              episode_metrics["episode"])
            for name, count in summary["model_usage"].items():
                writer.add_scalar(
                    f"validation/model_usage/{name}", count,
                    episode_metrics["episode"],
                )
            improved = summary["JointR"] > best_joint + VALIDATION_IMPROVEMENT
            if improved:
                best_joint = float(summary["JointR"])
                no_improve = 0
                agent.save(
                    best_path,
                    extra={
                        "episode": episode_metrics["episode"],
                        "validation": validation_record,
                        "best_joint": best_joint,
                    },
                )
            else:
                no_improve += 1
            agent.save(
                last_path,
                extra={
                    "episode": episode_metrics["episode"],
                    "validation": validation_record,
                    "best_joint": best_joint,
                    "no_improve": no_improve,
                },
            )
            atomic_write_json(history_path, history)
            update_progress(
                "dqn",
                status="training",
                current_episode=episode_metrics["episode"],
                best_validation_joint=best_joint,
                consecutive_no_improve=no_improve,
                best_checkpoint=str(best_path) if best_path.is_file() else None,
                last_checkpoint=str(last_path),
            )
            print(
                f"[Validation] Episode {episode_metrics['episode']} "
                f"JointR={summary['JointR']:.4f} SR={summary['SR']:.4f} "
                f"nDTW={summary['nDTW']:.4f} E2E={summary['E2E_ms']:.1f}ms "
                f"best={best_joint:.4f} no_improve={no_improve}",
                flush=True,
            )
            stop, reason = should_early_stop(
                episode_metrics["episode"], no_improve,
                [row["joint_reward"] for row in history["validation"]],
            )
            if stop:
                stop_reason = reason
                stop_episode = episode_metrics["episode"]
                break

    final_episode = int(last_episode)
    if stop_reason is None:
        stop_reason = (
            "reached episode cap" if final_episode >= args.max_episodes
            else "training loop ended"
        )
        stop_episode = final_episode
    # Convergence is declared only by the pre-declared early-stop rule.
    # Reaching the 4,000-episode cap without triggering it is reported as
    # not converged, even if the last validation happened to be near best.
    converged = bool(
        stop_reason.startswith("no ")
        or stop_reason.startswith("no new")
    )
    agent.save(
        last_path,
        extra={
            "episode": final_episode,
            "best_joint": best_joint,
            "no_improve": no_improve,
        },
    )
    if not best_path.is_file():
        agent.save(
            best_path,
            extra={
                "episode": final_episode,
                "best_joint": best_joint,
                "validation": history["validation"][-1]
                if history["validation"] else None,
            },
        )
    atomic_write_json(history_path, history)
    plot_training_curves(history, args.output_dir / "plots")
    training_result = {
        "version": "stage2k-dqn-v1",
        "created_utc": utc_now(),
        "episodes_trained": int(final_episode),
        "max_episodes": int(args.max_episodes),
        "min_episodes": int(MIN_EPISODES),
        "best_validation_joint": float(best_joint),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "stop_reason": stop_reason,
        "stop_episode": int(stop_episode) if stop_episode else None,
        "converged": bool(converged),
        "consecutive_no_improve": int(no_improve),
        "total_steps": int(agent.total_steps),
        "elapsed_s": time.time() - started,
        "architecture": "mlp_478_256_256_20",
        "seed": SEED,
        "quality_prior_sha256": sha256_file(QUALITY_PRIOR_PATH),
        "visual_pca_sha256": sha256_file(VISUAL_PCA_PATH),
        "state_design_sha256": (
            sha256_file(design_path) if design_path.is_file() else None
        ),
        "final_test_read": False,
        "history": history,
    }
    atomic_write_json(args.output_dir / "training_result.json", training_result)
    update_progress(
        "dqn",
        status="completed",
        current_episode=int(final_episode),
        best_validation_joint=float(best_joint),
        consecutive_no_improve=int(no_improve),
        best_checkpoint=str(best_path),
        last_checkpoint=str(last_path),
        training_finished_utc=utc_now(),
        converged=bool(converged),
    )
    writer.close()
    print(json.dumps({
        "episodes_trained": final_episode,
        "best_validation_joint": best_joint,
        "stop_reason": stop_reason,
        "converged": converged,
        "best_checkpoint": str(best_path),
        "elapsed_s": time.time() - started,
    }, indent=2), flush=True)
    print("STAGE2K_DQN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
