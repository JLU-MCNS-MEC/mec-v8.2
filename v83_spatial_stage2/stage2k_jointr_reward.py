"""JointR-aligned per-step delay reward for Stage-2k v2.

The navigation reward is the existing potential difference and telescopes to
the final nDTW.  The delay reward uses the running mean E2E delay:

    mean_t = mean_{t-1} + (d_t - mean_{t-1}) / t
    r_delay_t = -0.25 / 100 * (mean_t - mean_{t-1})

Therefore ``sum_t r_delay_t = -0.25 * mean_E2E / 100`` and the undiscounted
episode return equals ``final_nDTW - 0.25 * mean_E2E / 100 = JointR``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from v83_spatial_stage2.stage2k_common import JOINTR_DELAY_WEIGHT


def delay_reward_increment(delay_ms, mean_prev, step_index):
    """Return ``(new_running_mean, r_delay)`` for one step.

    ``step_index`` is 1-based.  The running mean starts at 0.
    """
    step_index = int(step_index)
    if step_index < 1:
        raise ValueError("step_index must be >= 1")
    mean_prev = float(mean_prev)
    mean_new = mean_prev + (float(delay_ms) - mean_prev) / float(step_index)
    reward = -JOINTR_DELAY_WEIGHT / 100.0 * (mean_new - mean_prev)
    return float(mean_new), float(reward)


def aligned_delay_rewards(delays: Iterable[float]):
    """Return ``(rewards, final_mean_delay)`` for a delay sequence."""
    mean = 0.0
    rewards = []
    for step_index, delay in enumerate(delays, start=1):
        mean, reward = delay_reward_increment(delay, mean, step_index)
        rewards.append(reward)
    return np.asarray(rewards, dtype=np.float64), float(mean)


def aligned_episode_jointr(final_ndtw, mean_delay_ms):
    """The evaluation JointR, using the same unit as the evaluation code."""
    return float(
        float(final_ndtw)
        - JOINTR_DELAY_WEIGHT / 100.0 * float(mean_delay_ms)
    )


def aligned_episode_audit(nav_rewards: Sequence[float],
                          delays: Sequence[float],
                          final_ndtw: float):
    """Recompute an episode and return the consistency diagnostics."""
    nav_rewards = np.asarray(nav_rewards, dtype=np.float64)
    delays = np.asarray(delays, dtype=np.float64)
    if len(nav_rewards) != len(delays):
        raise ValueError("nav_rewards and delays must have the same length")
    delay_rewards, mean_delay = aligned_delay_rewards(delays)
    navigation_sum = float(nav_rewards.sum())
    delay_sum = float(delay_rewards.sum())
    episode_return = float(navigation_sum + delay_sum)
    jointr = aligned_episode_jointr(final_ndtw, mean_delay)
    return {
        "steps": int(len(delays)),
        "navigation_sum": navigation_sum,
        "delay_sum": delay_sum,
        "mean_delay_ms": mean_delay,
        "episode_return": episode_return,
        "episode_jointr": jointr,
        "abs_error": abs(episode_return - jointr),
        "nav_minus_ndtw": navigation_sum - float(final_ndtw),
    }
