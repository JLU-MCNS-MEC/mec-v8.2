"""Standard PPO actor-critic for the Stage-2k 20-node MEC association task.

The PPO policy is deliberately minimal so that it can be compared with the
existing DQN under the *same* state, action space, reward and data:

  * 384-D pure-MEC v2 input -> 256 -> 256 -> {20 action logits, 1 value}
  * the 20-D availability mask is applied to the logits, never fed as input
  * gamma = 1.0, GAE(lambda=0.95), PPO clipped objective (eps=0.2)
  * the only reward is the environment's JointR-aligned reward; this module
    adds no success bonus, failure penalty or any other shaping.

The MLP trunk (two 256-wide ReLU layers) intentionally matches the DQN
network width.  Actor and critic are separate linear heads on the shared
trunk; no attention/GNN/Transformer block is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTION_DIM = 20
HIDDEN_DIM = 256
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LEARNING_RATE = 3e-4
ROLLOUT_STEPS = 2048
MINIBATCH_SIZE = 256
UPDATE_EPOCHS = 10
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.05
SEED = 20261031
REWARD_MODE = "jointr_aligned_v2"
MASKED_LOGIT = -1e8

# The exact reward string recorded by the DQN v2 run.  The PPO trainer
# re-uses the environment reward; this constant is checked in the protocol
# test so the two algorithms cannot silently diverge.
JOINTR_REWARD_DESCRIPTION = (
    "r_nav_t + (-0.25/100)*(running_mean_E2E_t - "
    "running_mean_E2E_{t-1})"
)


class PPOAnomalyError(RuntimeError):
    """Raised on NaN/Inf or an unstable PPO update.

    ``diagnostics`` is written by the trainer before the process exits so a
    numerically failed run keeps enough evidence for a post-mortem.
    """

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class ActorCriticMLP(nn.Module):
    """Two-layer shared MLP with separate policy and value heads."""

    def __init__(self, state_dim, action_dim=ACTION_DIM,
                 hidden=HIDDEN_DIM):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, self.action_dim)
        self.critic = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.trunk:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, state):
        features = self.trunk(state)
        return self.actor(features), self.critic(features).squeeze(-1)


def masked_logits(logits, mask):
    """Return logits whose unavailable actions have a very small value."""
    mask = mask.to(dtype=torch.bool)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("Every decision state must have an available MEC")
    return logits.masked_fill(~mask, MASKED_LOGIT)


class RolloutBuffer:
    """On-policy rollout storage for complete episodes.

    ``compute_gae`` resets the advantage recursion at every terminal step, so
    one buffer may contain several episodes.  With gamma=1 the return of a
    complete episode telescopes to JointR exactly (the trainer checks it).
    """

    def __init__(self):
        self.states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.rewards)

    def add(self, state, mask, action, log_prob, value, reward, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.masks.append(np.asarray(mask, dtype=bool))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def clear(self):
        self.__init__()

    def arrays(self):
        if not self.rewards:
            raise RuntimeError("empty rollout")
        return {
            "states": np.asarray(self.states, dtype=np.float32),
            "masks": np.asarray(self.masks, dtype=bool),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
        }

    def compute_gae(self, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        """Return (advantages, returns) with terminal resets at done steps."""
        data = self.arrays()
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]
        advantages = np.zeros_like(rewards, dtype=np.float64)
        last_advantage = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            if dones[index]:
                delta = float(rewards[index]) - float(values[index])
                last_advantage = delta
            else:
                delta = (
                    float(rewards[index])
                    + float(gamma) * float(values[index + 1])
                    - float(values[index])
                )
                last_advantage = (
                    delta + float(gamma) * float(gae_lambda) * last_advantage
                )
            advantages[index] = last_advantage
        returns = advantages + values.astype(np.float64)
        return advantages, returns


class PPOAgent:
    """Actor-critic with a clipped PPO update."""

    def __init__(
        self,
        device,
        seed=SEED,
        state_dim=384,
        state_mode="pure_mec_v2",
        action_dim=ACTION_DIM,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_eps=CLIP_EPS,
        learning_rate=LEARNING_RATE,
        rollout_steps=ROLLOUT_STEPS,
        minibatch_size=MINIBATCH_SIZE,
        update_epochs=UPDATE_EPOCHS,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
    ):
        self.device = torch.device(device)
        self.seed = int(seed)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.state_mode = str(state_mode)
        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.learning_rate = float(learning_rate)
        self.rollout_steps = int(rollout_steps)
        self.minibatch_size = int(minibatch_size)
        self.update_epochs = int(update_epochs)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = None if target_kl is None else float(target_kl)
        if self.gamma != 1.0:
            raise ValueError("The Stage-2k PPO protocol requires gamma=1.0")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in (0, 1]")
        if not 0.0 < self.clip_eps < 1.0:
            raise ValueError("clip_eps must be in (0, 1)")
        if self.rollout_steps < self.minibatch_size:
            raise ValueError("rollout_steps must be >= minibatch_size")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.network = ActorCriticMLP(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.learning_rate, eps=1e-5
        )
        self.rng = np.random.default_rng(self.seed)
        self.total_steps = 0
        self.update_steps = 0
        self.rollout_updates = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, state, mask):
        state_np = np.asarray(state, dtype=np.float32)
        if not np.isfinite(state_np).all():
            raise PPOAnomalyError(
                "Non-finite state passed to the policy",
                {
                    "non_finite": int((~np.isfinite(state_np)).sum()),
                    "state_dim": int(state_np.size),
                },
            )
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.shape != (self.action_dim,) or not mask_np.any():
            raise PPOAnomalyError(
                "Invalid availability mask passed to the policy",
                {
                    "available": int(mask_np.sum()),
                    "mask_shape": list(mask_np.shape),
                    "action_dim": int(self.action_dim),
                },
            )
        state_tensor = torch.from_numpy(
            state_np
        ).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(
            mask_np
        ).unsqueeze(0).to(self.device)
        logits, value = self.network(state_tensor)
        if not (
            torch.isfinite(logits).all() and torch.isfinite(value).all()
        ):
            raise PPOAnomalyError(
                "Non-finite policy logits/value",
                {"state_dim": self.state_dim},
            )
        distribution = Categorical(
            logits=masked_logits(logits, mask_tensor)
        )
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        if not torch.isfinite(log_prob).all():
            raise PPOAnomalyError(
                "Non-finite sampled log-probability",
                {"log_prob": float(log_prob.item())},
            )
        return {
            "action": int(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
            "entropy": float(entropy.item()),
        }

    @torch.no_grad()
    def greedy_action(self, state, mask):
        state_np = np.asarray(state, dtype=np.float32)
        if not np.isfinite(state_np).all():
            raise RuntimeError("Non-finite state during greedy evaluation")
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.shape != (self.action_dim,) or not mask_np.any():
            raise RuntimeError(
                "Invalid availability mask during evaluation: "
                f"{mask_np.shape} vs action_dim={self.action_dim}"
            )
        state_tensor = torch.from_numpy(
            state_np
        ).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(
            mask_np
        ).unsqueeze(0).to(self.device)
        logits, _ = self.network(state_tensor)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite policy logits during evaluation")
        available = masked_logits(logits, mask_tensor)
        return int(torch.argmax(available, dim=-1).item())

    @torch.no_grad()
    def value(self, state):
        state_tensor = torch.from_numpy(
            np.asarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        _, value = self.network(state_tensor)
        return float(value.item())

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self, rollout):
        if len(rollout) == 0:
            raise ValueError("Cannot update from an empty rollout")
        advantages, returns = rollout.compute_gae(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        if not (
            np.isfinite(advantages).all() and np.isfinite(returns).all()
        ):
            raise PPOAnomalyError(
                "Non-finite GAE advantage/return",
                {"rollout_steps": len(rollout)},
            )
        data = rollout.arrays()
        for key in ("states", "log_probs", "values", "rewards"):
            if not np.isfinite(data[key]).all():
                raise PPOAnomalyError(
                    f"Non-finite {key} in rollout",
                    {
                        "rollout_steps": len(rollout),
                        "non_finite": int(
                            (~np.isfinite(data[key])).sum()
                        ),
                    },
                )
        if not data["masks"].any(axis=1).all():
            raise PPOAnomalyError(
                "Rollout contains a state with no available MEC",
                {"rollout_steps": len(rollout)},
            )
        states = torch.from_numpy(data["states"]).to(self.device)
        masks = torch.from_numpy(data["masks"]).to(self.device)
        actions = torch.from_numpy(data["actions"]).to(self.device)
        old_log_probs = torch.from_numpy(
            data["log_probs"]
        ).to(self.device)
        advantages_tensor = torch.from_numpy(
            advantages.astype(np.float32)
        ).to(self.device)
        returns_tensor = torch.from_numpy(
            returns.astype(np.float32)
        ).to(self.device)
        advantage_std = float(advantages_tensor.std(unbiased=False).item())
        if advantage_std > 0.0:
            advantages_tensor = (
                advantages_tensor - advantages_tensor.mean()
            ) / (advantage_std + 1e-8)
        else:
            advantages_tensor = advantages_tensor - advantages_tensor.mean()

        count = len(rollout)
        aggregate = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "ratio_max": 0.0,
            "ratio_min": float("inf"),
            "minibatches": 0,
            "epochs_run": 0,
            "early_stop_epoch": None,
            "advantage_std_raw": advantage_std,
            "value_target_mean": float(returns.mean()),
            "value_target_std": float(returns.std()),
        }
        total_seen = 0
        for epoch in range(self.update_epochs):
            permutation = torch.from_numpy(
                self.rng.permutation(count)
            ).to(self.device)
            epoch_kl = 0.0
            epoch_batches = 0
            for start in range(0, count, self.minibatch_size):
                indices = permutation[start:start + self.minibatch_size]
                batch_size = int(indices.numel())
                logits, values = self.network(states[indices])
                if not (
                    torch.isfinite(logits).all()
                    and torch.isfinite(values).all()
                ):
                    raise PPOAnomalyError(
                        "Non-finite policy logits/value in update",
                        {
                            "epoch": int(epoch),
                            "rollout_steps": count,
                        },
                    )
                distribution = Categorical(
                    logits=masked_logits(logits, masks[indices])
                )
                new_log_probs = distribution.log_prob(actions[indices])
                entropy = distribution.entropy()
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = torch.exp(log_ratio)
                advantage = advantages_tensor[indices]
                surrogate_1 = ratio * advantage
                surrogate_2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * advantage
                policy_loss = -torch.min(
                    surrogate_1, surrogate_2
                ).mean()
                value_loss = F.mse_loss(
                    values, returns_tensor[indices]
                )
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )
                if not torch.isfinite(loss):
                    raise PPOAnomalyError(
                        "Non-finite PPO loss",
                        {
                            "episode": int(self.total_steps),
                            "epoch": int(epoch),
                            "policy_loss": float(policy_loss.detach().item()),
                            "value_loss": float(value_loss.detach().item()),
                            "entropy": float(entropy_mean.detach().item()),
                            "rollout_steps": count,
                        },
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                ))
                if not np.isfinite(grad_norm):
                    raise PPOAnomalyError(
                        "Non-finite PPO gradient norm",
                        {
                            "epoch": int(epoch),
                            "grad_norm": grad_norm,
                            "rollout_steps": count,
                        },
                    )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = float(
                        (old_log_probs[indices] - new_log_probs)
                        .mean().item()
                    )
                    clip_fraction = float(
                        ((ratio - 1.0).abs() > self.clip_eps)
                        .float().mean().item()
                    )
                    ratio_max = float(ratio.max().item())
                    ratio_min = float(ratio.min().item())
                aggregate["policy_loss"] += (
                    float(policy_loss.detach().item()) * batch_size
                )
                aggregate["value_loss"] += (
                    float(value_loss.detach().item()) * batch_size
                )
                aggregate["entropy"] += (
                    float(entropy_mean.detach().item()) * batch_size
                )
                aggregate["loss"] += (
                    float(loss.detach().item()) * batch_size
                )
                aggregate["approx_kl"] += approx_kl * batch_size
                aggregate["clip_fraction"] += clip_fraction * batch_size
                aggregate["grad_norm"] += grad_norm * batch_size
                aggregate["ratio_max"] = max(
                    aggregate["ratio_max"], ratio_max
                )
                aggregate["ratio_min"] = min(
                    aggregate["ratio_min"], ratio_min
                )
                aggregate["minibatches"] += 1
                total_seen += batch_size
                epoch_kl += approx_kl
                epoch_batches += 1
                self.update_steps += 1
            aggregate["epochs_run"] = epoch + 1
            mean_epoch_kl = epoch_kl / max(epoch_batches, 1)
            if (
                self.target_kl is not None
                and mean_epoch_kl > 1.5 * self.target_kl
            ):
                aggregate["early_stop_epoch"] = epoch
                break
        for key in (
            "policy_loss", "value_loss", "entropy", "loss", "approx_kl",
            "clip_fraction", "grad_norm",
        ):
            aggregate[key] /= max(total_seen, 1)
        if not np.isfinite(aggregate["ratio_min"]):
            aggregate["ratio_min"] = 0.0
        with torch.no_grad():
            logits, values = self.network(states)
            values_np = values.detach().cpu().numpy().astype(np.float64)
        residual = returns - values_np
        variance = float(np.var(returns))
        aggregate["value_mean"] = float(values_np.mean())
        aggregate["value_abs_max"] = float(np.abs(values_np).max())
        aggregate["explained_variance"] = (
            float(1.0 - np.var(residual) / variance)
            if variance > 0.0 else 0.0
        )
        aggregate["rollout_steps"] = count
        aggregate["total_steps"] = int(self.total_steps)
        aggregate["update_steps"] = int(self.update_steps)
        self.rollout_updates += 1
        if not np.isfinite(aggregate["approx_kl"]):
            raise PPOAnomalyError(
                "Non-finite approximate KL", dict(aggregate)
            )
        return aggregate

    # ------------------------------------------------------------------
    # Configuration and checkpoints
    # ------------------------------------------------------------------
    def config(self):
        return {
            "algorithm": "ppo",
            "architecture": (
                f"actor_critic_mlp_{self.state_dim}_256_256_"
                f"{self.action_dim}"
            ),
            "state_dim": self.state_dim,
            "state_mode": self.state_mode,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_eps": self.clip_eps,
            "learning_rate": self.learning_rate,
            "rollout_steps": self.rollout_steps,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "seed": self.seed,
            "reward_mode": REWARD_MODE,
            "reward": JOINTR_REWARD_DESCRIPTION,
            "action_masking": (
                f"{self.action_dim}-D availability mask on policy logits"
            ),
            "final_test_read": False,
        }

    def _rng_state(self):
        state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "agent": self.rng.bit_generator.state,
        }
        if torch.cuda.is_available() and self.device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state):
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        self.rng.bit_generator.state = state["agent"]
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    def _model_payload(self):
        return {
            "state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": int(self.total_steps),
            "update_steps": int(self.update_steps),
            "rollout_updates": int(self.rollout_updates),
            "config": self.config(),
            "reward_mode": REWARD_MODE,
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
        payload["rng_state"] = self._rng_state()
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _validate_payload(self, payload, require_optimizer):
        if payload.get("reward_mode") != REWARD_MODE:
            raise RuntimeError(
                f"Checkpoint reward_mode {payload.get('reward_mode')!r} "
                f"!= {REWARD_MODE!r}"
            )
        config = payload.get("config", {})
        if str(config.get("state_mode")) != self.state_mode:
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
        if abs(float(config.get("gamma", -1.0)) - self.gamma) > 1e-12:
            raise RuntimeError("Checkpoint gamma does not match PPO gamma")
        if abs(
            float(config.get("gae_lambda", -1.0)) - self.gae_lambda
        ) > 1e-12:
            raise RuntimeError("Checkpoint GAE lambda does not match")
        if abs(
            float(config.get("clip_eps", -1.0)) - self.clip_eps
        ) > 1e-12:
            raise RuntimeError("Checkpoint clip epsilon does not match")
        if require_optimizer and "optimizer_state_dict" not in payload:
            raise RuntimeError("Full checkpoint has no optimizer state")

    def load_model(self, path):
        payload = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self._validate_payload(payload, require_optimizer=False)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.network.eval()
        return payload

    def load_full(self, path):
        payload = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self._validate_payload(payload, require_optimizer=True)
        self.network.load_state_dict(payload["state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.total_steps = int(payload.get("total_steps", 0))
        self.update_steps = int(payload.get("update_steps", 0))
        self.rollout_updates = int(payload.get("rollout_updates", 0))
        self._restore_rng_state(payload["rng_state"])
        return payload
