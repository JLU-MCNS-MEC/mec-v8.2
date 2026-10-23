"""Stage-2o reward schemes for the latency-only MEC association task.

All three schemes share the same episode-level objective ordering.  Let

    mean_E2E = (1/T) * sum_t e2e_t,      tau = 100 ms,      H = 400

and define the episode mean utility

    J = 1 - mean_E2E / tau.

Scheme B (remove only the 0.25 coefficient of Stage-2n)::

    r_t = -(mean_t - mean_{t-1}) / tau          =>  R_B = -mean_E2E / tau

Scheme D (fixed-scale per-step utility plus a terminal correction)::

    u_t = 1 - e2e_t / tau
    r_t = u_t / H                               for every step
    r_T <- r_T + (1 - T/H) * J                  only on the final transition

    =>  R_D = (1/H) * sum_t u_t + (1 - T/H) * J = J

Scheme "one" (direct episode-mean return, trajectory-level implementation)::

    r_t = 0                                     for t < T
    r_T = J                                     only on the final transition

    =>  R_one = J

Thus D and one have exactly the same episode return, while B differs from them
by the fixed constant +1:

    R_D = R_one = 1 + R_B.

The per-step rewards are *not* interchangeable: B and D provide dense
feedback, one provides a single terminal reward.  Comparisons in Stage-2o are
therefore made on realised E2E latency, convergence and stability, never on the
absolute height of the reward curve alone.

No function here reads success, navigation quality, target position or any
future latency for a non-terminal reward.  For D the non-terminal reward is a
function of the current e2e only; for one every non-terminal reward is exactly
zero.  The terminal correction/reward is assigned only to the transition whose
``done`` flag is true, so there is no future-information leakage into an
earlier decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

TAU_MS = 100.0
H_MAX = 400
SCHEMES = ("B", "D", "one")
SCHEME_LABELS = {"B": "方案B", "D": "方案D", "one": "方案一"}
SCHEME_DESCRIPTIONS = {
    "B": "r_t = -(mean_E2E_t - mean_E2E_{t-1}) / tau; R = -mean_E2E/tau",
    "D": (
        "u_t = 1 - e2e_t/tau; r_t = u_t/H; "
        "r_T += (1 - T/H)*(1 - mean_E2E/tau); R = 1 - mean_E2E/tau"
    ),
    "one": (
        "r_t = 0 (t<T); r_T = J = 1 - mean_E2E/tau; "
        "R = 1 - mean_E2E/tau"
    ),
}
RETURN_DESCRIPTIONS = {
    "B": "R = -mean_E2E/tau",
    "D": "R = 1 - mean_E2E/tau",
    "one": "R = 1 - mean_E2E/tau",
}
TOLERANCE = 1e-9


def validate_scheme(scheme: str) -> str:
    scheme = str(scheme)
    if scheme not in SCHEMES:
        raise ValueError(f"unknown reward scheme {scheme!r}; expected {SCHEMES}")
    return scheme


def scheme_b_increment(e2e_ms: float, mean_prev: float, step_index: int):
    """Return ``(new_running_mean, r_t)`` for scheme B.

    ``step_index`` is 1-based.  Starting from ``mean_prev = 0`` the rewards
    telescope exactly to ``-mean_E2E / tau``.
    """
    step_index = int(step_index)
    if step_index < 1:
        raise ValueError("step_index must be >= 1")
    e2e_ms = float(e2e_ms)
    mean_prev = float(mean_prev)
    if not np.isfinite(e2e_ms) or e2e_ms < 0.0:
        raise ValueError(f"invalid E2E delay {e2e_ms}")
    mean_new = mean_prev + (e2e_ms - mean_prev) / float(step_index)
    reward = -(mean_new - mean_prev) / TAU_MS
    return float(mean_new), float(reward)


def scheme_d_step_reward(e2e_ms: float, tau: float = TAU_MS,
                         horizon: int = H_MAX) -> float:
    """Scheme D per-step utility, before the terminal correction."""
    e2e_ms = float(e2e_ms)
    if not np.isfinite(e2e_ms) or e2e_ms < 0.0:
        raise ValueError(f"invalid E2E delay {e2e_ms}")
    return float((1.0 - e2e_ms / float(tau)) / float(horizon))


def scheme_d_terminal_correction(steps: int, mean_e2e_ms: float,
                                 tau: float = TAU_MS,
                                 horizon: int = H_MAX) -> float:
    """Scheme D terminal correction ``(1 - T/H) * (1 - mean/tau)``."""
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if steps > int(horizon):
        raise ValueError(f"steps {steps} exceeds horizon {horizon}")
    return float(
        (1.0 - float(steps) / float(horizon))
        * (1.0 - float(mean_e2e_ms) / float(tau))
    )


def scheme_one_terminal_reward(mean_e2e_ms: float,
                               tau: float = TAU_MS) -> float:
    """Scheme one terminal reward ``J = 1 - mean_E2E/tau``."""
    return float(1.0 - float(mean_e2e_ms) / float(tau))


def expected_return(scheme: str, mean_e2e_ms: float,
                    tau: float = TAU_MS) -> float:
    """Closed-form episode return for each scheme."""
    scheme = validate_scheme(scheme)
    if scheme == "B":
        return float(-float(mean_e2e_ms) / float(tau))
    return float(1.0 - float(mean_e2e_ms) / float(tau))


def episode_rewards(scheme: str, e2e_ms: Sequence[float],
                    tau: float = TAU_MS, horizon: int = H_MAX):
    """Return per-step rewards and diagnostics for one completed episode.

    The returned list has exactly ``len(e2e_ms)`` entries.  The function is
    deliberately pure: it is the reference implementation used by the unit
    tests.  The trainer uses :class:`EpisodeRewardController`, which produces
    the same numbers online (the terminal reward is only known at ``done``).
    """
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
    if scheme == "B":
        mean = 0.0
        for index, delay in enumerate(delays, start=1):
            mean, rewards[index - 1] = scheme_b_increment(
                delay, mean, index
            )
        base_sum = float(sum(rewards))
    elif scheme == "D":
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
    else:  # scheme one
        rewards[-1] = scheme_one_terminal_reward(mean_e2e, tau=tau)
        base_sum = 0.0

    total = float(sum(rewards))
    expected = expected_return(scheme, mean_e2e, tau=tau)
    abs_error = abs(total - expected)
    if abs_error > TOLERANCE:
        raise AssertionError(
            f"scheme {scheme} return identity failed: {total} vs {expected}"
        )
    terminal_reward = float(rewards[-1])
    nonterminal = np.asarray(rewards[:-1], dtype=np.float64) if steps > 1 else (
        np.zeros(0, dtype=np.float64)
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
        "terminal_reward": terminal_reward,
        "terminal_share": (
            float(terminal_reward / total) if abs(total) > 1e-12 else None
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
    """Online scheme implementation used by the training loops.

    ``step_reward`` is called once per executed decision step, including the
    terminal one.  At ``done`` the trainer calls :meth:`finalize` with the
    episode mean E2E, which patches the *last* reward only (scheme D adds the
    terminal correction, scheme one replaces the zero with ``J``).  Earlier
    rewards are never touched.
    """

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
        if self.scheme == "B":
            self.mean_e2e_ms, reward = scheme_b_increment(
                e2e_ms, self.mean_e2e_ms, self.step_index
            )
        elif self.scheme == "D":
            reward = scheme_d_step_reward(
                e2e_ms, tau=self.tau, horizon=self.horizon
            )
        else:
            reward = 0.0
        self.rewards.append(float(reward))
        return float(reward)

    def finalize(self, mean_e2e_ms: float, steps: int):
        """Patch the terminal reward and return ``(last_reward, diag)``."""
        if self.finalized:
            raise RuntimeError("episode already finalized")
        steps = int(steps)
        if steps != self.step_index or steps != len(self.rewards):
            raise RuntimeError(
                f"controller has {self.step_index} steps but finalize got "
                f"{steps}"
            )
        mean_e2e_ms = float(mean_e2e_ms)
        if self.scheme == "B":
            if abs(mean_e2e_ms - self.mean_e2e_ms) > 1e-7:
                raise RuntimeError(
                    "scheme B running mean mismatch: "
                    f"{self.mean_e2e_ms} vs {mean_e2e_ms}"
                )
            correction = 0.0
        elif self.scheme == "D":
            correction = scheme_d_terminal_correction(
                steps, mean_e2e_ms, tau=self.tau, horizon=self.horizon
            )
            self.rewards[-1] += correction
        else:
            correction = 0.0
            self.rewards[-1] = scheme_one_terminal_reward(
                mean_e2e_ms, tau=self.tau
            )
        self.finalized = True
        total = float(sum(self.rewards))
        expected = expected_return(self.scheme, mean_e2e_ms, tau=self.tau)
        abs_error = abs(total - expected)
        if abs_error > TOLERANCE:
            raise AssertionError(
                f"scheme {self.scheme} return identity failed: "
                f"{total} vs {expected}"
            )
        nonterminal = np.asarray(
            self.rewards[:-1], dtype=np.float64
        ) if steps > 1 else np.zeros(0, dtype=np.float64)
        if self.scheme == "B":
            base_sum = total
        elif self.scheme == "D":
            base_sum = total - correction
        else:
            base_sum = 0.0
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


def reward_scale_note() -> str:
    """Human-readable note attached to figures and the report."""
    return (
        "tau=100 ms, H=400, gamma=1.  "
        "B: R=-mean_E2E/tau; D and one: R=1-mean_E2E/tau.  "
        "D and one share the same episode return; B is shifted by +1.  "
        "Compare realised E2E latency, not absolute reward height."
    )


__all__ = [
    "TAU_MS",
    "H_MAX",
    "SCHEMES",
    "SCHEME_LABELS",
    "SCHEME_DESCRIPTIONS",
    "RETURN_DESCRIPTIONS",
    "TOLERANCE",
    "validate_scheme",
    "scheme_b_increment",
    "scheme_d_step_reward",
    "scheme_d_terminal_correction",
    "scheme_one_terminal_reward",
    "expected_return",
    "episode_rewards",
    "EpisodeRewardController",
    "reward_scale_note",
]
