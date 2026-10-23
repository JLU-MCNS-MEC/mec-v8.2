"""Stage-2p reward definitions: original reward vs scheme D.

Both schemes are executed by the *same* Stage-2p PPO trainer.  The original
reward is imported from the Stage-2n implementation so it is numerically the
same function that produced the Stage-2n PPO runs:

    baseline: r_t = -0.25 * (mean_t - mean_{t-1}) / 100
              R = -0.25 * mean_E2E / 100

Scheme D is imported from the verified Stage-2o implementation:

    u_t = 1 - e2e_t / tau
    r_t = u_t / H
    r_T <- r_T + (1 - T/H) * J,   J = 1 - mean_E2E / tau
    R = J = 1 - mean_E2E / tau

The cumulative returns satisfy the exact affine relation

    4 * R_baseline + 1 = R_D,

which is used to put the two reward curves on a comparable scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from v83_spatial_stage2.stage2n_common import (
    latency_return_from_mean as baseline_return_from_mean,
    latency_reward_increment as baseline_increment_stage2n,
)
from v83_spatial_stage2.stage2o.rewards import (
    H_MAX,
    TAU_MS,
    scheme_d_step_reward,
    scheme_d_terminal_correction,
)

SCHEMES = ("baseline", "D")
SCHEME_LABELS = {"baseline": "原始奖励", "D": "方案D"}
SCHEME_DESCRIPTIONS = {
    "baseline": "r_t = -0.25*(mean_t - mean_{t-1})/100; R = -0.25*mean_E2E/100",
    "D": (
        "u_t = 1 - e2e_t/tau; r_t = u_t/H; "
        "r_T += (1 - T/H)*(1 - mean_E2E/tau); R = 1 - mean_E2E/tau"
    ),
}
TOLERANCE = 1e-9
BASELINE_WEIGHT = 0.25


def validate_scheme(scheme: str) -> str:
    scheme = str(scheme)
    if scheme not in SCHEMES:
        raise ValueError(
            f"unknown Stage-2p reward scheme {scheme!r}; expected {SCHEMES}"
        )
    return scheme


def baseline_increment(e2e_ms: float, mean_prev: float, step_index: int):
    """Exact Stage-2n original reward increment."""
    return baseline_increment_stage2n(e2e_ms, mean_prev, step_index)


def expected_return(scheme: str, mean_e2e_ms: float,
                    tau: float = TAU_MS) -> float:
    scheme = validate_scheme(scheme)
    if scheme == "baseline":
        return float(baseline_return_from_mean(mean_e2e_ms))
    return float(1.0 - float(mean_e2e_ms) / float(tau))


def aligned_baseline_return(value: float) -> float:
    """Map a baseline cumulative return to the scheme-D scale."""
    return float(4.0 * float(value) + 1.0)


def episode_rewards(scheme: str, e2e_ms: Sequence[float],
                    tau: float = TAU_MS, horizon: int = H_MAX):
    """Reference per-step rewards for one completed episode."""
    scheme = validate_scheme(scheme)
    delays = [float(value) for value in e2e_ms]
    steps = len(delays)
    if steps < 1:
        raise ValueError("an episode must have at least one step")
    if steps > int(horizon):
        raise ValueError(
            f"episode length {steps} exceeds horizon {horizon}"
        )
    if not np.all(np.isfinite(np.asarray(delays, dtype=np.float64))):
        raise ValueError("non-finite E2E delay")
    if any(value < 0.0 for value in delays):
        raise ValueError("negative E2E delay")
    mean_e2e = float(np.mean(np.asarray(delays, dtype=np.float64)))
    rewards = [0.0] * steps
    correction = 0.0
    base_sum = 0.0
    if scheme == "baseline":
        mean = 0.0
        for index, delay in enumerate(delays, start=1):
            mean, rewards[index - 1] = baseline_increment(
                delay, mean, index
            )
        base_sum = float(sum(rewards))
    else:
        base = [
            scheme_d_step_reward(delay, tau=tau, horizon=horizon)
            for delay in delays
        ]
        base_sum = float(sum(base))
        correction = scheme_d_terminal_correction(
            steps, mean_e2e, tau=tau, horizon=horizon
        )
        rewards = base
        rewards[-1] += correction
    total = float(sum(rewards))
    expected = expected_return(scheme, mean_e2e, tau=tau)
    abs_error = abs(total - expected)
    if abs_error > TOLERANCE:
        raise AssertionError(
            f"scheme {scheme} return identity failed: {total} vs {expected}"
        )
    nonterminal = (
        np.asarray(rewards[:-1], dtype=np.float64)
        if steps > 1 else np.zeros(0, dtype=np.float64)
    )
    diagnostics = {
        "scheme": scheme,
        "steps": int(steps),
        "mean_e2e_ms": mean_e2e,
        "return": total,
        "expected_return": expected,
        "abs_error": float(abs_error),
        "base_sum": float(base_sum),
        "terminal_correction": float(correction),
        "terminal_reward": float(rewards[-1]),
        "terminal_share": (
            float(rewards[-1] / total) if abs(total) > 1e-12 else None
        ),
        "nonterminal_mean": (
            float(nonterminal.mean()) if nonterminal.size else 0.0
        ),
        "nonterminal_std": (
            float(nonterminal.std(ddof=0)) if nonterminal.size else 0.0
        ),
        "nonterminal_abs_max": (
            float(np.abs(nonterminal).max()) if nonterminal.size else 0.0
        ),
        "step_abs_max": float(np.abs(np.asarray(rewards)).max()),
        "tau_ms": float(tau),
        "horizon": int(horizon),
    }
    return rewards, diagnostics


@dataclass
class EpisodeRewardController:
    """Online baseline/D controller used by the unified Stage-2p PPO loop."""

    scheme: str
    tau: float = TAU_MS
    horizon: int = H_MAX
    mean_e2e_ms: float = 0.0
    step_index: int = 0
    e2e_ms: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    finalized: bool = False

    def __post_init__(self):
        self.scheme = validate_scheme(self.scheme)
        self.tau = float(self.tau)
        self.horizon = int(self.horizon)

    def step_reward(self, e2e_ms: float) -> float:
        if self.finalized:
            raise RuntimeError("episode already finalized")
        e2e_ms = float(e2e_ms)
        if not np.isfinite(e2e_ms) or e2e_ms < 0.0:
            raise ValueError(f"invalid E2E delay {e2e_ms}")
        self.step_index += 1
        self.e2e_ms.append(e2e_ms)
        if self.scheme == "baseline":
            self.mean_e2e_ms, reward = baseline_increment(
                e2e_ms, self.mean_e2e_ms, self.step_index
            )
        else:
            reward = scheme_d_step_reward(
                e2e_ms, tau=self.tau, horizon=self.horizon
            )
        self.rewards.append(float(reward))
        return float(reward)

    def finalize(self, mean_e2e_ms: float, steps: int):
        if self.finalized:
            raise RuntimeError("episode already finalized")
        steps = int(steps)
        if steps != self.step_index or steps != len(self.rewards):
            raise RuntimeError(
                f"controller has {self.step_index} steps but finalize got "
                f"{steps}"
            )
        mean_e2e_ms = float(mean_e2e_ms)
        correction = 0.0
        if self.scheme == "baseline":
            if abs(mean_e2e_ms - self.mean_e2e_ms) > 1e-7:
                raise RuntimeError(
                    "baseline running mean mismatch: "
                    f"{self.mean_e2e_ms} vs {mean_e2e_ms}"
                )
        else:
            correction = scheme_d_terminal_correction(
                steps, mean_e2e_ms, tau=self.tau, horizon=self.horizon
            )
            self.rewards[-1] += correction
        self.finalized = True
        total = float(sum(self.rewards))
        expected = expected_return(self.scheme, mean_e2e_ms, tau=self.tau)
        abs_error = abs(total - expected)
        if abs_error > TOLERANCE:
            raise AssertionError(
                f"scheme {self.scheme} return identity failed: "
                f"{total} vs {expected}"
            )
        nonterminal = (
            np.asarray(self.rewards[:-1], dtype=np.float64)
            if steps > 1 else np.zeros(0, dtype=np.float64)
        )
        base_sum = total if self.scheme == "baseline" else total - correction
        diagnostics = {
            "scheme": self.scheme,
            "steps": int(steps),
            "mean_e2e_ms": mean_e2e_ms,
            "return": total,
            "expected_return": expected,
            "abs_error": float(abs_error),
            "base_sum": float(base_sum),
            "terminal_correction": float(correction),
            "terminal_reward": float(self.rewards[-1]),
            "terminal_share": (
                float(self.rewards[-1] / total)
                if abs(total) > 1e-12 else None
            ),
            "nonterminal_mean": (
                float(nonterminal.mean()) if nonterminal.size else 0.0
            ),
            "nonterminal_std": (
                float(nonterminal.std(ddof=0))
                if nonterminal.size else 0.0
            ),
            "nonterminal_abs_max": (
                float(np.abs(nonterminal).max())
                if nonterminal.size else 0.0
            ),
            "step_abs_max": float(np.abs(np.asarray(self.rewards)).max()),
            "tau_ms": float(self.tau),
            "horizon": int(self.horizon),
        }
        return float(self.rewards[-1]), diagnostics


__all__ = [
    "BASELINE_WEIGHT",
    "H_MAX",
    "TAU_MS",
    "SCHEMES",
    "SCHEME_LABELS",
    "SCHEME_DESCRIPTIONS",
    "TOLERANCE",
    "validate_scheme",
    "baseline_increment",
    "expected_return",
    "aligned_baseline_return",
    "episode_rewards",
    "EpisodeRewardController",
]
