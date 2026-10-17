import numpy as np


EPS = 1e-12


def _as_xy_array(points):
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(
            "Trajectory points must have shape [N, 2] or [N, >=2]."
        )

    return points[:, :2]


def remove_consecutive_duplicates(points, tolerance=1e-9):
    """
    Remove consecutive points whose Euclidean distance is nearly zero.

    This is useful for the current toy UAV environment because Left/Right
    actions change yaw but do not change the 2-D position. nDTW here is used
    to evaluate geometric trajectory fidelity, so repeated position samples
    caused purely by in-place turns are removed from the reference polyline.
    """
    points = _as_xy_array(points)

    if len(points) <= 1:
        return points.copy()

    kept = [points[0].copy()]

    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > tolerance:
            kept.append(point.copy())

    return np.asarray(kept, dtype=np.float64)


def reference_path_length(reference_points):
    reference_points = _as_xy_array(reference_points)

    if len(reference_points) <= 1:
        return 0.0

    segment_lengths = np.linalg.norm(
        np.diff(reference_points, axis=0),
        axis=1,
    )

    return float(segment_lengths.sum())


def _segment_geometry(reference_points):
    reference_points = _as_xy_array(reference_points)

    if len(reference_points) < 2:
        raise ValueError(
            "At least two reference points are required for path projection."
        )

    starts = reference_points[:-1]
    ends = reference_points[1:]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)

    cumulative_starts = np.zeros(len(lengths), dtype=np.float64)

    if len(lengths) > 1:
        cumulative_starts[1:] = np.cumsum(lengths[:-1])

    total_length = float(lengths.sum())

    return starts, ends, vectors, lengths, cumulative_starts, total_length


def _point_at_arc_length(reference_points, s):
    """
    Return the point/segment corresponding to cumulative reference-path length s.
    """
    reference_points = _as_xy_array(reference_points)
    (
        starts,
        ends,
        vectors,
        lengths,
        cumulative_starts,
        total_length,
    ) = _segment_geometry(reference_points)

    if total_length <= EPS:
        return {
            "point": reference_points[0].copy(),
            "segment_index": 0,
            "alpha": 0.0,
            "s": 0.0,
            "total_length": 0.0,
            "distance": np.nan,
        }

    s = float(np.clip(s, 0.0, total_length))

    if np.isclose(s, total_length):
        last_nonzero = np.where(lengths > EPS)[0][-1]
        return {
            "point": ends[last_nonzero].copy(),
            "segment_index": int(last_nonzero),
            "alpha": 1.0,
            "s": total_length,
            "total_length": total_length,
            "distance": np.nan,
        }

    for k, length in enumerate(lengths):
        if length <= EPS:
            continue

        start_s = cumulative_starts[k]
        end_s = start_s + length

        if start_s - EPS <= s <= end_s + EPS:
            alpha = (s - start_s) / length
            alpha = float(np.clip(alpha, 0.0, 1.0))
            point = starts[k] + alpha * vectors[k]

            return {
                "point": point,
                "segment_index": int(k),
                "alpha": alpha,
                "s": s,
                "total_length": total_length,
                "distance": np.nan,
            }

    # Numerical fallback.
    return {
        "point": reference_points[-1].copy(),
        "segment_index": len(reference_points) - 2,
        "alpha": 1.0,
        "s": total_length,
        "total_length": total_length,
        "distance": np.nan,
    }


def project_to_reference_polyline(
    position,
    reference_points,
    previous_s=0.0,
    max_forward_progress=None,
):
    """
    Continuously project the current UAV position onto the reference polyline.

    For every valid reference segment, the orthogonal projection coefficient is

        alpha_raw = ((p - p_k)^T (p_{k+1} - p_k)) / ||p_{k+1} - p_k||^2,

    and alpha is clipped to [0, 1] so the projection lies on the finite segment.
    Among candidates satisfying the forward-progress constraint, the closest
    continuous projection is selected.

    Parameters
    ----------
    position : array-like, shape [2] or [>=2]
    reference_points : array-like, shape [M, 2] or [M, >=2]
    previous_s : float
        Previous cumulative progress along the reference path.
    max_forward_progress : float or None
        Optional maximum amount by which the reference-path progress is allowed
        to increase in one decision step. This avoids spurious jumps at path
        self-intersections. Set to None to impose only monotonic progress.
    """
    p = np.asarray(position, dtype=np.float64)[:2]
    reference_points = _as_xy_array(reference_points)

    (
        starts,
        ends,
        vectors,
        lengths,
        cumulative_starts,
        total_length,
    ) = _segment_geometry(reference_points)

    if total_length <= EPS:
        return {
            "point": reference_points[0].copy(),
            "segment_index": 0,
            "alpha": 0.0,
            "s": 0.0,
            "total_length": 0.0,
            "distance": float(np.linalg.norm(p - reference_points[0])),
        }

    lower_s = float(np.clip(previous_s, 0.0, total_length))

    if max_forward_progress is None:
        upper_s = total_length
    else:
        upper_s = min(
            total_length,
            lower_s + max(float(max_forward_progress), 0.0),
        )

    best = None

    for k, length in enumerate(lengths):
        if length <= EPS:
            continue

        vector = vectors[k]
        alpha_raw = np.dot(p - starts[k], vector) / (length * length)
        alpha = float(np.clip(alpha_raw, 0.0, 1.0))
        projection = starts[k] + alpha * vector
        s_candidate = cumulative_starts[k] + alpha * length

        if s_candidate < lower_s - 1e-9:
            continue

        if s_candidate > upper_s + 1e-9:
            continue

        distance = float(np.linalg.norm(p - projection))

        candidate = {
            "point": projection,
            "segment_index": int(k),
            "alpha": alpha,
            "s": float(s_candidate),
            "total_length": total_length,
            "distance": distance,
        }

        if best is None:
            best = candidate
        elif distance < best["distance"] - 1e-12:
            best = candidate
        elif np.isclose(distance, best["distance"]):
            # Tie-break toward the smaller forward progress.
            # This is more conservative near path intersections.
            if s_candidate < best["s"]:
                best = candidate

    if best is not None:
        return best

    # If no candidate survives the forward window, keep the previous progress.
    fallback = _point_at_arc_length(reference_points, lower_s)
    fallback["distance"] = float(np.linalg.norm(p - fallback["point"]))
    return fallback


def build_reference_prefix(reference_points, projection):
    """
    Construct the reference prefix ending exactly at the continuous projection.
    """
    reference_points = _as_xy_array(reference_points)
    k = int(projection["segment_index"])
    alpha = float(projection["alpha"])
    projected_point = np.asarray(projection["point"], dtype=np.float64)[:2]

    prefix = [point.copy() for point in reference_points[: k + 1]]

    if alpha > 1e-9:
        if len(prefix) == 0 or np.linalg.norm(projected_point - prefix[-1]) > 1e-9:
            prefix.append(projected_point.copy())

    if len(prefix) == 0:
        prefix = [reference_points[0].copy()]

    return np.asarray(prefix, dtype=np.float64)


def dtw_distance(trajectory, reference):
    """
    Standard Dynamic Time Warping distance with Euclidean local cost.

    D(i,j) = ||q_i-r_j||_2 + min(D(i-1,j), D(i,j-1), D(i-1,j-1)).
    """
    trajectory = _as_xy_array(trajectory)
    reference = _as_xy_array(reference)

    if len(trajectory) == 0 or len(reference) == 0:
        raise ValueError("DTW requires two non-empty trajectories.")

    m = len(reference)

    previous = np.full(m + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0

    for point in trajectory:
        current = np.full(m + 1, np.inf, dtype=np.float64)

        for j in range(1, m + 1):
            local_cost = np.linalg.norm(point - reference[j - 1])

            current[j] = local_cost + min(
                previous[j],
                current[j - 1],
                previous[j - 1],
            )

        previous = current

    return float(previous[m])


def ndtw_score(trajectory, reference, distance_threshold):
    """
    normalized Dynamic Time Warping (nDTW):

        exp(-DTW / (|reference| * distance_threshold)).
    """
    trajectory = _as_xy_array(trajectory)
    reference = _as_xy_array(reference)

    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive.")

    dtw = dtw_distance(trajectory, reference)
    denominator = max(len(reference) * float(distance_threshold), EPS)

    return float(np.exp(-dtw / denominator))


def navigation_state_metrics(
    trajectory,
    reference_points,
    previous_s,
    distance_threshold,
    max_forward_progress=None,
):
    """
    Compute progress rho_t, prefix-nDTW q_t, and navigation potential Phi_t.
    """
    trajectory = _as_xy_array(trajectory)
    reference_points = _as_xy_array(reference_points)

    projection = project_to_reference_polyline(
        position=trajectory[-1],
        reference_points=reference_points,
        previous_s=previous_s,
        max_forward_progress=max_forward_progress,
    )

    total_length = max(projection["total_length"], EPS)
    progress = float(np.clip(projection["s"] / total_length, 0.0, 1.0))

    reference_prefix = build_reference_prefix(reference_points, projection)

    prefix_ndtw = ndtw_score(
        trajectory=trajectory,
        reference=reference_prefix,
        distance_threshold=distance_threshold,
    )

    potential = progress * prefix_ndtw

    return {
        "projection": projection,
        "reference_prefix": reference_prefix,
        "progress": progress,
        "prefix_ndtw": prefix_ndtw,
        "potential": float(potential),
    }


class NavigationRewardTracker:
    """
    Track the progress-aware navigation reward during one closed-loop episode.

    Non-terminal transition:
        r_t^nav = Phi_{t+1} - Phi_t

    Terminal transition:
        r_{T-1}^nav = nDTW(T_T, T*) - Phi_{T-1}

    With Phi_0 = 0, the undiscounted cumulative navigation reward telescopes to
    the final full-trajectory nDTW.
    """

    def __init__(
        self,
        reference_trajectory,
        start_position,
        distance_threshold,
        max_forward_progress=None,
    ):
        reference_trajectory = remove_consecutive_duplicates(
            reference_trajectory
        )

        if len(reference_trajectory) < 2:
            raise ValueError(
                "The reference trajectory must contain at least two distinct points."
            )

        self.reference_trajectory = reference_trajectory
        self.distance_threshold = float(distance_threshold)
        self.max_forward_progress = max_forward_progress

        self.trajectory = [
            np.asarray(start_position, dtype=np.float64)[:2].copy()
        ]

        self.previous_s = 0.0
        self.previous_potential = 0.0

        self.step_rewards = []
        self.progress_history = [0.0]
        self.prefix_ndtw_history = [1.0]
        self.potential_history = [0.0]
        self.projected_points = [self.reference_trajectory[0].copy()]

        self.final_ndtw = None

    def advance(self, new_position, done):
        self.trajectory.append(
            np.asarray(new_position, dtype=np.float64)[:2].copy()
        )

        metrics = navigation_state_metrics(
            trajectory=np.asarray(self.trajectory),
            reference_points=self.reference_trajectory,
            previous_s=self.previous_s,
            distance_threshold=self.distance_threshold,
            max_forward_progress=self.max_forward_progress,
        )

        current_potential = metrics["potential"]

        if done:
            self.final_ndtw = ndtw_score(
                trajectory=np.asarray(self.trajectory),
                reference=self.reference_trajectory,
                distance_threshold=self.distance_threshold,
            )

            reward = self.final_ndtw - self.previous_potential
        else:
            reward = current_potential - self.previous_potential

        self.step_rewards.append(float(reward))
        self.progress_history.append(metrics["progress"])
        self.prefix_ndtw_history.append(metrics["prefix_ndtw"])
        self.potential_history.append(current_potential)
        self.projected_points.append(metrics["projection"]["point"].copy())

        self.previous_s = metrics["projection"]["s"]
        self.previous_potential = current_potential

        return float(reward), metrics

    def summary(self):
        if self.final_ndtw is None:
            self.final_ndtw = ndtw_score(
                trajectory=np.asarray(self.trajectory),
                reference=self.reference_trajectory,
                distance_threshold=self.distance_threshold,
            )

        total_reward = float(np.sum(self.step_rewards))

        return {
            "final_ndtw": float(self.final_ndtw),
            "navigation_reward": total_reward,
            "reward_consistency_error": abs(
                total_reward - float(self.final_ndtw)
            ),
            "step_rewards": np.asarray(self.step_rewards, dtype=np.float64),
            "progress_history": np.asarray(
                self.progress_history,
                dtype=np.float64,
            ),
            "prefix_ndtw_history": np.asarray(
                self.prefix_ndtw_history,
                dtype=np.float64,
            ),
            "potential_history": np.asarray(
                self.potential_history,
                dtype=np.float64,
            ),
            "projected_points": np.asarray(
                self.projected_points,
                dtype=np.float64,
            ),
            "reference_trajectory": self.reference_trajectory.copy(),
        }
