"""Stage-2g.2 bounded near-target recovery policy.

The policy receives only runtime-observable inputs:
  * the P1-2 near-target-invisible probability (computed from current + two
    previous observable frames),
  * the frozen M0 base action and arrival/Stop decision,
  * the observable high-confidence target side and its recency.

It never reads true distance, true visibility, heading error or the oracle
action. Recovery is deliberately short: one correction turn followed by
forward steps for the remaining budget (default 2 movement steps total), then
the base policy resumes. Forward is allowed; the M0 Stop threshold and model
weights are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env import FORWARD, LEFT, RIGHT, STOP
from v83_spatial_stage2.stage2g1_common import (
    ACTION_DIM,
    COMPACT_DIM,
    one_hot_actions,
)
from v83_spatial_stage2.stage2g_probe import prior_correct


@dataclass
class ProbeRecoveryConfig:
    threshold: float = 0.13694876571430528
    recovery_steps: int = 2
    cooldown: int = 5
    side_window: int = 20
    side_margin: float = 0.05
    high_top1: float = 0.20
    high_entropy: float = 0.85


class ProbeRecoveryPolicy:
    def __init__(self, config: ProbeRecoveryConfig):
        self.config = config
        self.reset()

    def reset(self):
        self.recovering = False
        self.remaining = 0
        self.recovery_step_index = 0
        self.cooldown = 0
        self.trigger_count = 0
        self.recovery_step_count = 0
        self.events = []
        self.actions = []
        self.current_event = None
        self.last_high_side = "unknown"
        self.last_high_step = -1

    def _side(self, centered_x):
        if centered_x < -self.config.side_margin:
            return "left"
        if centered_x > self.config.side_margin:
            return "right"
        return "center"

    def _correction_action(self, step, centered_x):
        if (
            self.last_high_side in ("left", "right")
            and self.last_high_step >= 0
            and step - self.last_high_step <= self.config.side_window
        ):
            return LEFT if self.last_high_side == "left" else RIGHT
        if centered_x < -self.config.side_margin:
            return LEFT
        if centered_x > self.config.side_margin:
            return RIGHT
        return LEFT

    def _close_event(self, reason, step, reacquired_step=-1):
        if self.current_event is None:
            return
        self.current_event["exit_reason"] = reason
        self.current_event["reacquired_step"] = int(reacquired_step)
        self.current_event["actions"] = list(self.actions)
        self.current_event["recovery_steps"] = len(self.actions)
        self.events.append(self.current_event)
        self.current_event = None

    def finalize(self, step):
        if self.recovering:
            self._close_event("episode_end", step)
            self.recovering = False

    def act(self, p_near, base_action, step, high_conf, centered_x):
        """Return (action, recovering_flag, trigger_now, info)."""
        if high_conf:
            self.last_high_side = self._side(centered_x)
            self.last_high_step = int(step)

        if base_action == STOP:
            if self.recovering:
                self._close_event("stop", step)
            self.recovering = False
            return STOP, 0, 0, self._info(p_near, high_conf, step)

        if self.recovering:
            if high_conf:
                self._close_event("high_confidence", step, step)
                self.recovering = False
                self.cooldown = self.config.cooldown
                return int(base_action), 0, 0, self._info(
                    p_near, high_conf, step
                )
            if self.remaining > 0:
                if self.recovery_step_index == 0:
                    action = self._correction_action(step, centered_x)
                else:
                    action = FORWARD
                self.actions.append(int(action))
                self.recovery_step_count += 1
                self.recovery_step_index += 1
                self.remaining -= 1
                if self.remaining == 0:
                    self._close_event("budget", step)
                    self.recovering = False
                    self.cooldown = self.config.cooldown
                return int(action), 1, 0, self._info(p_near, high_conf, step)
            self._close_event("budget", step)
            self.recovering = False
            self.cooldown = self.config.cooldown
            return int(base_action), 0, 0, self._info(p_near, high_conf, step)

        if self.cooldown > 0:
            # A cooldown of N blocks the next N decisions; the trigger is
            # evaluated only after the counter reaches zero.
            self.cooldown -= 1
            return int(base_action), 0, 0, self._info(
                p_near, high_conf, step
            )
        if p_near >= self.config.threshold:
            self.trigger_count += 1
            self.recovering = True
            self.remaining = int(self.config.recovery_steps)
            self.recovery_step_index = 0
            self.actions = []
            steps_since_high = (
                int(step) - self.last_high_step
                if self.last_high_step >= 0 else 999
            )
            self.current_event = {
                "trigger_step": int(step),
                "trigger_probability": float(p_near),
                "base_action": int(base_action),
                "last_high_side": self.last_high_side,
                "steps_since_high_conf": int(steps_since_high),
                "actions": [],
                "exit_reason": "",
                "reacquired_step": -1,
            }
            action = self._correction_action(step, centered_x)
            self.actions.append(int(action))
            self.recovery_step_count += 1
            self.recovery_step_index += 1
            self.remaining -= 1
            recovering_flag = 1
            if self.remaining == 0:
                self._close_event("budget", step)
                self.recovering = False
                self.cooldown = self.config.cooldown
            return int(action), recovering_flag, 1, self._info(
                p_near, high_conf, step
            )

        return int(base_action), 0, 0, self._info(p_near, high_conf, step)

    def _info(self, p_near, high_conf, step):
        return {
            "p_near_invisible": float(p_near),
            "probe_flag": int(p_near >= self.config.threshold),
            "high_conf": int(high_conf),
            "last_high_side": self.last_high_side,
            "steps_since_high_conf": (
                int(step) - int(self.last_high_step)
                if self.last_high_step >= 0 else -1
            ),
            "cooldown": int(self.cooldown),
            "recovery_remaining": int(self.remaining),
            "recovering": int(self.recovering),
        }


def build_probe_input(current_features, history):
    """Build the P1-2 input: current + t-1 + t-2 (history most recent first)."""
    parts = [np.asarray(current_features, dtype=np.float32)]
    for frame in range(2):
        if frame < len(history):
            compact, action = history[frame]
            parts.append(np.asarray(compact, dtype=np.float32))
            parts.append(one_hot_actions(np.asarray([action]))[0])
            parts.append(np.asarray([1.0], dtype=np.float32))
        else:
            parts.append(np.zeros(COMPACT_DIM, dtype=np.float32))
            parts.append(np.zeros(ACTION_DIM, dtype=np.float32))
            parts.append(np.zeros(1, dtype=np.float32))
    return np.concatenate(parts)


def probe_predict(model, mean, std, pos_weight, features, device):
    import torch

    x = (np.asarray(features, dtype=np.float32) - mean) / std
    x = np.clip(x, -20.0, 20.0)
    tensor = torch.from_numpy(x).float().to(device).unsqueeze(0)
    with torch.no_grad():
        logit = model(tensor).squeeze(1)
        probability = float(torch.sigmoid(logit).item())
    return float(prior_correct(np.asarray([probability]), pos_weight)[0])
