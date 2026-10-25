"""Standard MLP DQN agent for the JointR-aligned Stage-2k v2 run.

The network, replay buffer, target network and optimizer are the same
components as Stage-2k v1.  The only algorithmic difference is ``gamma=1.0``
and the full resumable checkpoint (online/target/optimizer/replay/RNG).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_dqn import (
    BATCH_SIZE,
    EPSILON_DECAY_STEPS,
    EPSILON_END,
    EPSILON_START,
    GRAD_CLIP,
    LEARNING_RATE,
    QNetwork,
    REPLAY_CAPACITY,
    ReplayBuffer,
    SEED,
    TARGET_UPDATE_STEPS,
)
from v83_spatial_stage2.stage2k_state import STATE_DIM, masked_argmax

ACTION_DIM = 20
GAMMA_V2 = 1.0


def epsilon_at(total_steps, decay_steps=EPSILON_DECAY_STEPS):
    """Linear epsilon schedule shared by training and tests."""
    decay_steps = int(decay_steps)
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive")
    fraction = min(1.0, float(total_steps) / float(decay_steps))
    return float(EPSILON_START + (EPSILON_END - EPSILON_START) * fraction)


class DQNAgentV2:
    def __init__(self, device, seed=SEED, gamma=GAMMA_V2,
                 state_dim=STATE_DIM, state_mode="full",
                 action_dim=ACTION_DIM,
                 epsilon_decay_steps=None, double_dqn=False,
                 legacy_target=False):
        self.device = torch.device(device)
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.state_mode = str(state_mode)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.double_dqn = bool(double_dqn)
        # ``legacy_target`` reproduces the pre-audit target that filled
        # masked next actions with 0.0 instead of -inf.  It is kept only to
        # run the before/after control; new runs must use the fixed target.
        self.legacy_target = bool(legacy_target)
        if self.double_dqn and self.legacy_target:
            raise ValueError(
                "double_dqn and legacy_target are mutually exclusive"
            )
        self.epsilon_decay_steps = int(
            EPSILON_DECAY_STEPS if epsilon_decay_steps is None
            else epsilon_decay_steps
        )
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        torch.manual_seed(int(seed))
        self.online = QNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)
        self.target = QNetwork(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad = False
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=LEARNING_RATE
        )
        self.buffer = ReplayBuffer(
            capacity=REPLAY_CAPACITY, state_dim=self.state_dim,
            action_dim=self.action_dim, seed=seed,
        )
        self.rng = np.random.default_rng(int(seed))
        self.total_steps = 0
        self.update_steps = 0

    def epsilon(self):
        return epsilon_at(self.total_steps, self.epsilon_decay_steps)

    @torch.inference_mode()
    def q_values(self, state):
        tensor = torch.from_numpy(
            np.asarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        return self.online(tensor)[0].detach().cpu().numpy()

    def act_from_q(self, q_values, mask, epsilon=None):
        available = np.where(np.asarray(mask, dtype=bool))[0]
        if len(available) == 0:
            raise ValueError("No available MEC")
        epsilon = self.epsilon() if epsilon is None else float(epsilon)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(available))
        return int(masked_argmax(q_values, mask))

    def act_greedy(self, state, mask):
        return int(masked_argmax(self.q_values(state), mask))

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return None
        batch = self.buffer.sample(BATCH_SIZE, self.device)
        with torch.no_grad():
            next_mask = batch["next_mask"]
            available_next = next_mask.any(dim=1)
            target_q = self.target(batch["next_state"])
            if self.legacy_target:
                # Pre-audit behaviour: unavailable actions contribute 0.0 to
                # the max, so an all-negative next state bootstraps from a
                # fictitious zero-valued action.
                next_max = target_q.masked_fill(
                    ~next_mask, 0.0
                ).max(dim=1).values
            elif self.double_dqn:
                # Double DQN: the online network selects among available
                # actions, the target network evaluates the selected action.
                online_next = self.online(batch["next_state"])
                next_action = online_next.masked_fill(
                    ~next_mask, -1e9
                ).argmax(dim=1, keepdim=True)
                next_max = target_q.gather(1, next_action).squeeze(1)
                next_max = torch.where(
                    available_next, next_max, torch.zeros_like(next_max)
                )
            else:
                # Fixed DQN target: -inf fill so an unavailable action can
                # never win the max; only an empty mask bootstraps from 0.
                next_max = target_q.masked_fill(
                    ~next_mask, -1e9
                ).max(dim=1).values
                next_max = torch.where(
                    available_next, next_max, torch.zeros_like(next_max)
                )
            target = batch["reward"] + self.gamma * next_max * (
                1.0 - batch["done"]
            )
        q_all = self.online(batch["state"])
        q = q_all.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite DQN loss: {loss.item()}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(
            self.online.parameters(), GRAD_CLIP
        ))
        if not np.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm: {grad_norm}")
        self.optimizer.step()
        self.update_steps += 1
        if self.update_steps % TARGET_UPDATE_STEPS == 0:
            self.target.load_state_dict(self.online.state_dict())
        q_np = q_all.detach()
        return {
            "loss": float(loss.detach().cpu().item()),
            "grad_norm": grad_norm,
            "q_mean": float(q_np.mean().cpu().item()),
            "q_max": float(q_np.max().cpu().item()),
            "q_selected_mean": float(q.detach().mean().cpu().item()),
            "target_mean": float(target.detach().mean().cpu().item()),
            "td_error_abs_mean": float(
                (q.detach() - target).abs().mean().cpu().item()
            ),
        }

    def config(self):
        return {
            "architecture": (
                f"mlp_{self.state_dim}_256_256_{self.action_dim}"
            ),
            "state_dim": self.state_dim,
            "state_mode": self.state_mode,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "replay_capacity": REPLAY_CAPACITY,
            "target_update_steps": TARGET_UPDATE_STEPS,
            "gradient_clip": GRAD_CLIP,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay_steps": int(self.epsilon_decay_steps),
            "seed": int(self.seed),
            "double_dqn": bool(self.double_dqn),
            "legacy_target": bool(self.legacy_target),
            "target_masking": (
                "legacy_zero_fill" if self.legacy_target else "negative_inf"
            ),
            "reward_mode": "jointr_aligned_v2",
            "final_test_read": False,
        }

    def _rng_state(self):
        state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "agent": self.rng.bit_generator.state,
            "buffer": self.buffer.rng.bit_generator.state,
        }
        if torch.cuda.is_available() and self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        self.rng.bit_generator.state = state["agent"]
        self.buffer.rng.bit_generator.state = state["buffer"]
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _model_payload(self):
        return {
            "state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": int(self.total_steps),
            "update_steps": int(self.update_steps),
            "epsilon": float(self.epsilon()),
            "config": self.config(),
            "final_test_read": False,
        }

    def save_model(self, path, extra=None):
        payload = self._model_payload()
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def save_full(self, path, extra=None):
        payload = self._model_payload()
        payload["buffer_state"] = self.buffer.state_dict()
        payload["rng_state"] = self._rng_state()
        payload["reward_mode"] = "jointr_aligned_v2"
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_model(self, path):
        payload = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self._load_networks(payload)
        return payload

    def load_full(self, path):
        payload = torch.load(
            path, map_location=self.device, weights_only=False
        )
        if payload.get("reward_mode") != "jointr_aligned_v2":
            raise RuntimeError("Checkpoint is not a JointR-aligned v2 run")
        config = payload.get("config", {})
        if abs(float(config.get("gamma", -1.0)) - GAMMA_V2) > 1e-12:
            raise RuntimeError("Checkpoint gamma is not 1.0")
        if str(config.get("state_mode", "full")) != self.state_mode:
            raise RuntimeError(
                f"Checkpoint state_mode {config.get('state_mode')} != "
                f"{self.state_mode}"
            )
        if int(config.get("state_dim", -1)) != self.state_dim:
            raise RuntimeError(
                f"Checkpoint state_dim {config.get('state_dim')} != "
                f"{self.state_dim}"
            )
        if int(config.get("action_dim", ACTION_DIM)) != self.action_dim:
            raise RuntimeError(
                f"Checkpoint action_dim {config.get('action_dim')} != "
                f"{self.action_dim}"
            )
        if int(config.get(
            "epsilon_decay_steps", EPSILON_DECAY_STEPS
        )) != self.epsilon_decay_steps:
            raise RuntimeError(
                "Checkpoint epsilon_decay_steps "
                f"{config.get('epsilon_decay_steps')} != "
                f"{self.epsilon_decay_steps}"
            )
        self._load_networks(payload)
        self.buffer.load_state_dict(payload["buffer_state"])
        self._restore_rng_state(payload["rng_state"])
        expected_epsilon = self.epsilon()
        saved_epsilon = float(payload.get("epsilon", expected_epsilon))
        if abs(expected_epsilon - saved_epsilon) > 1e-6:
            raise RuntimeError(
                "Checkpoint epsilon does not match the schedule"
            )
        return payload

    def _load_networks(self, payload):
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
