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


class DQNAgentV2:
    def __init__(self, device, seed=SEED, gamma=GAMMA_V2):
        self.device = torch.device(device)
        self.gamma = float(gamma)
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
        self.buffer = ReplayBuffer(
            capacity=REPLAY_CAPACITY, state_dim=STATE_DIM,
            action_dim=ACTION_DIM, seed=seed,
        )
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
            next_q = self.target(batch["next_state"])
            next_q = next_q.masked_fill(~batch["next_mask"], 0.0)
            next_max = next_q.max(dim=1).values
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
        }

    def config(self):
        return {
            "architecture": "mlp_478_256_256_20",
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "gamma": self.gamma,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "replay_capacity": REPLAY_CAPACITY,
            "target_update_steps": TARGET_UPDATE_STEPS,
            "gradient_clip": GRAD_CLIP,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay_steps": EPSILON_DECAY_STEPS,
            "seed": SEED,
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
