"""Stage-2f lightweight Goal-Reacquisition / Movement Recovery policy.

Runtime inputs are ONLY the observable features from `stage2f_features.py`.
The policy never reads the simulator goal position, true distance, true
visibility, heading error, oracle action or success label. `FORBIDDEN_KEYS`
enforces this at the accessor level.

Behaviour:
  * track the last high-confidence Goal-Current observation and its target side
    (baseline-map centroid, no true labels);
  * trigger recovery after K consecutive low-confidence movement states, or
    when a yaw-aligned previous map is inconsistent with the current map;
  * while recovering, do not move FORWARD: turn toward the last known target
    side, or run a deterministic scan (12 left, 24 right, 24 left, ...);
  * leave recovery after a high-confidence observation.

The policy is an eval-time overlay on the frozen M0@0.7 controller. It is not a
trained model and does not modify the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from env import FORWARD, LEFT, RIGHT

FORBIDDEN_KEYS = {
    "goal_visible", "goal_px", "goal_py", "goal_half_size_px",
    "distance_m", "heading_error_deg", "oracle_move_action", "stop_ready",
    "success", "episode_index", "band", "owner_move_action",
    "coherent_peak_error_px", "baseline_peak_error_px",
}


def _get(features, key):
    if key in FORBIDDEN_KEYS:
        raise ValueError(f"Recovery policy cannot read true-label key {key}")
    return features[key]


@dataclass
class RecoveryConfig:
    high_top1: float = 0.30
    high_entropy: float = 0.75
    low_top1: float = 0.15
    low_entropy: float = 0.85
    trigger_k: int = 2
    consistency: float = 0.55
    reacquire_k: int = 1
    side_margin: float = 0.05
    scan_left_first: int = 12
    scan_right_next: int = 24
    use_side: bool = True
    max_recovery_steps: int = 0  # 0 = no limit


@dataclass
class RecoveryEvent:
    trigger_step: int
    trigger_reason: str
    low_run: int
    temporal_consistency: float
    last_high_side: str
    last_high_centroid_x: float
    steps_since_high_conf: int
    actions: list = field(default_factory=list)
    reacquired_step: int | None = None
    exit_reason: str = ""


class RecoveryPolicy:
    def __init__(self, config: RecoveryConfig | None = None):
        self.config = config or RecoveryConfig()
        self.reset()

    def reset(self):
        self.last_high_side = "unknown"
        self.last_high_centroid_x = float("nan")
        self.last_high_step = -1
        self.low_run = 0
        self.high_run = 0
        self.recovering = False
        self.recovery_steps = 0
        self.events: list[RecoveryEvent] = []
        self.current_event: RecoveryEvent | None = None
        self.trigger_count = 0
        self.recovery_step_count = 0
        self.reacquired_count = 0

    def _side_from_features(self, features):
        centroid = _get(features, "baseline_centroid_x")
        if centroid < -self.config.side_margin:
            return "left"
        if centroid > self.config.side_margin:
            return "right"
        return "center"

    def _is_high(self, features):
        return (
            _get(features, "heatmap_max_top1") >= self.config.high_top1
            and _get(features, "heatmap_max_entropy") <= self.config.high_entropy
        )

    def _is_low(self, features):
        return (
            _get(features, "heatmap_max_top1") < self.config.low_top1
            or _get(features, "heatmap_max_entropy") > self.config.low_entropy
        )

    def _is_temporal_inconsistent(self, features):
        valid = _get(features, "temporal_valid") > 0.5
        consistency = _get(features, "temporal_consistency")
        return bool(valid and consistency == consistency
                    and consistency < self.config.consistency)

    def observe(self, features, step):
        """Update confidence history; may enter/leave recovery."""
        high = self._is_high(features)
        low = self._is_low(features)
        inconsistent = self._is_temporal_inconsistent(features)
        if high:
            self.high_run += 1
            self.low_run = 0
            self.last_high_step = int(step)
            self.last_high_centroid_x = float(
                _get(features, "baseline_centroid_x")
            )
            self.last_high_side = self._side_from_features(features)
        else:
            self.high_run = 0
            self.low_run = self.low_run + 1 if low else 0

        if self.recovering:
            reacquired = self.high_run >= self.config.reacquire_k
            budget_exhausted = (
                self.config.max_recovery_steps > 0
                and self.recovery_step_count >= self.config.max_recovery_steps
            )
            if reacquired or budget_exhausted:
                self.recovering = False
                self.recovery_steps = 0
                if self.current_event is not None:
                    if reacquired:
                        self.current_event.reacquired_step = int(step)
                        self.current_event.exit_reason = "reacquired"
                        self.reacquired_count += 1
                    else:
                        self.current_event.exit_reason = "budget"
                self.current_event = None
        else:
            reason = None
            if self.low_run >= self.config.trigger_k:
                reason = "low_confidence_run"
            elif inconsistent and self.low_run >= 1:
                reason = "temporal_inconsistent"
            if reason is not None:
                self.recovering = True
                self.recovery_steps = 0
                self.trigger_count += 1
                self.current_event = RecoveryEvent(
                    trigger_step=int(step),
                    trigger_reason=reason,
                    low_run=int(self.low_run),
                    temporal_consistency=float(
                        _get(features, "temporal_consistency")
                    ),
                    last_high_side=self.last_high_side,
                    last_high_centroid_x=self.last_high_centroid_x,
                    steps_since_high_conf=(
                        int(step) - self.last_high_step
                        if self.last_high_step >= 0 else 999
                    ),
                )
                self.events.append(self.current_event)
        return high, low, inconsistent

    def _scan_action(self):
        step = self.recovery_steps
        left_first = self.config.scan_left_first
        right_next = self.config.scan_right_next
        if step < left_first:
            return LEFT
        if step < left_first + right_next:
            return RIGHT
        if step < left_first + 2 * right_next:
            return LEFT
        return RIGHT

    def act(self, features, base_action, step):
        """Return (action, recovering)."""
        self.observe(features, step)
        if not self.recovering:
            return int(base_action), False
        self.recovery_steps += 1
        self.recovery_step_count += 1
        if self.config.use_side and self.last_high_side == "left":
            action = LEFT
        elif self.config.use_side and self.last_high_side == "right":
            action = RIGHT
        else:
            action = self._scan_action()
        if self.current_event is not None:
            self.current_event.actions.append(int(action))
        return int(action), True
