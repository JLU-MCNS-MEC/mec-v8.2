"""Tests for the Stage-2q trajectory audit.

The tests read the committed Stage-2q CSVs; they do not need a GPU or a
checkpoint.  Run with:

    /home/ubuntu5/miniconda3/envs/DRL/bin/python -m unittest \
        v83_spatial_stage2.stage2q_trajectory_audit.test_trajectory_audit
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ENV  # noqa: E402
from env import FORWARD, LEFT, RIGHT, STOP, wrap_angle  # noqa: E402
from v83_spatial_stage2.stage2q_trajectory_audit.collect_trajectories import (  # noqa: E402
    ALL_METHODS,
    CONTROL_METHODS,
    LEARNED_METHODS,
    SELECTION_RULE,
    _action_sha256,
    _coord_sha256,
    _node_sha256,
    advance_pose,
    reconstruct_path,
)

REQUIRED_FILES = (
    "trajectory_steps.csv",
    "trajectory_summary.csv",
    "trajectory_invariance.csv",
    "dev60_task_summary.csv",
    "selected_tasks.csv",
    "checkpoint_inventory.csv",
    "city_geometry.csv",
    "city_meta.json",
    "audit_summary.json",
    "run_manifest.json",
    "trajectory_examples_2x3.png",
    "stage2n_method_overlay.png",
    "stage2o_reward_overlay.png",
    "stage2p_ppo_reward_overlay.png",
    "trajectory_difference_audit.png",
)
EXPECTED_DEV_TASKS_SHA256 = (
    "aefde2aadaf443d32c70faebd74624431d165f6ef1b27d0fe3f1480572ade41d"
)
EXPECTED_MEC_SHA256 = (
    "57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d"
)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        ("true", "1", "1.0", "yes")
    )


class Stage2QAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name for name in REQUIRED_FILES if not (BASE / name).is_file()
        ]
        if missing:
            raise unittest.SkipTest(
                "Stage-2q outputs are not generated yet: "
                + ", ".join(missing)
            )
        cls.summary = pd.read_csv(BASE / "trajectory_summary.csv")
        cls.steps = pd.read_csv(BASE / "trajectory_steps.csv")
        cls.invariance = pd.read_csv(BASE / "trajectory_invariance.csv")
        cls.selected = pd.read_csv(BASE / "selected_tasks.csv")
        cls.dev60 = pd.read_csv(BASE / "dev60_task_summary.csv")
        cls.city = pd.read_csv(BASE / "city_geometry.csv")
        cls.checkpoints = pd.read_csv(BASE / "checkpoint_inventory.csv")
        cls.city_meta = json.loads(
            (BASE / "city_meta.json").read_text("utf-8")
        )
        cls.audit = json.loads(
            (BASE / "audit_summary.json").read_text("utf-8")
        )
        cls.manifest = json.loads(
            (BASE / "run_manifest.json").read_text("utf-8")
        )
        for frame, columns in (
            (cls.summary, ("success",)),
            (cls.steps, ("success", "step_success", "step_done")),
            (
                cls.invariance,
                (
                    "actions_equal",
                    "coord_hash_equal",
                    "coords_equal_1e-9_m",
                    "steps_equal",
                    "success_equal",
                    "success",
                    "reference_success",
                    "action_length_equal",
                ),
            ),
            (cls.selected, ("success",)),
            (cls.dev60, ("success",)),
            (
                cls.checkpoints,
                ("last_pt_exists", "final_test_read"),
            ),
        ):
            for column in columns:
                if column not in frame.columns:
                    raise AssertionError(
                        f"missing column {column!r} in {frame.columns.tolist()}"
                    )
                frame[column] = _as_bool(frame[column])
        cls.learned = [item for item in ALL_METHODS
                       if item not in CONTROL_METHODS]

    # ------------------------------------------------------------------
    # Provenance / task stratification
    # ------------------------------------------------------------------
    def test_required_outputs_are_nonempty(self):
        for name in REQUIRED_FILES:
            path = BASE / name
            self.assertGreater(path.stat().st_size, 0, name)
        for name in (
            "trajectory_examples_2x3.png",
            "stage2n_method_overlay.png",
            "stage2o_reward_overlay.png",
            "stage2p_ppo_reward_overlay.png",
            "trajectory_difference_audit.png",
        ):
            self.assertGreater(
                (BASE / name).stat().st_size, 20_000, name
            )

    def test_only_dev_tasks_and_no_test_read(self):
        for frame, name in (
            (self.summary, "summary"),
            (self.steps, "steps"),
            (self.invariance, "invariance"),
            (self.dev60, "dev60"),
        ):
            uids = frame["episode_uid"].astype(str)
            self.assertTrue(
                uids.str.startswith("stage2n-dev-").all(), name
            )
            self.assertFalse(
                uids.str.contains("stage2n-test-").any(), name
            )
        self.assertFalse(bool(self.manifest["final_test_read"]))
        self.assertFalse(bool(self.manifest["test_pool_read"]))
        self.assertFalse(bool(self.audit["final_test_read"]))
        self.assertFalse(bool(self.audit["test_pool_read"]))
        self.assertFalse(bool(self.manifest["sealed_test_read"]))

    def test_task_pool_and_checkpoint_provenance(self):
        self.assertEqual(int(self.manifest["dev_tasks"]), 60)
        self.assertEqual(
            self.manifest["dev_tasks_sha256"], EXPECTED_DEV_TASKS_SHA256
        )
        self.assertEqual(
            self.manifest["mec_checkpoint_sha256"], EXPECTED_MEC_SHA256
        )
        self.assertEqual(
            int(self.city_meta["num_landmarks"]), 160
        )
        self.assertEqual(
            int(self.city_meta["map_seed"]), 10_000_001
        )
        self.assertEqual(
            float(self.city_meta["world_size_m"]), 1500.0
        )

    def test_dev60_stratification(self):
        self.assertEqual(len(self.dev60), 60)
        counts = self.dev60.groupby("band").size().to_dict()
        self.assertEqual(counts, {"short": 20, "medium": 20, "long": 20})
        success = self.dev60.groupby("band")["success"].sum().to_dict()
        self.assertEqual(success, {"short": 12, "medium": 13, "long": 11})
        self.assertEqual(int(self.dev60["success"].sum()), 36)

    def test_selected_tasks_cover_six_strata(self):
        self.assertEqual(len(self.selected), 6)
        for band in ("short", "medium", "long"):
            sub = self.selected[self.selected["band"] == band]
            self.assertEqual(len(sub), 2, band)
            self.assertEqual(int(sub["success"].sum()), 1, band)
        self.assertEqual(set(self.selected["episode_uid"]),
                         set(self.summary["episode_uid"]))

    def test_selected_tasks_follow_median_rule(self):
        dev = self.dev60.copy()
        dev["episode_index"] = (
            dev["episode_uid"].str.extract(r"(\d+)$")[0].astype(int)
        )
        for band in ("short", "medium", "long"):
            for success in (True, False):
                sub = dev[
                    (dev["band"] == band)
                    & (dev["success"] == success)
                ].sort_values(["steps", "episode_index"])
                rank = (len(sub) - 1) // 2
                expected = str(sub.iloc[rank]["episode_uid"])
                selected = self.selected[
                    (self.selected["band"] == band)
                    & (self.selected["success"] == success)
                ]
                self.assertEqual(len(selected), 1, (band, success))
                self.assertEqual(
                    str(selected.iloc[0]["episode_uid"]), expected,
                    (band, success),
                )
        self.assertIn("lower median", SELECTION_RULE)

    # ------------------------------------------------------------------
    # Cross-method invariance
    # ------------------------------------------------------------------
    def test_all_methods_present(self):
        self.assertEqual(len(ALL_METHODS), 20)
        self.assertEqual(len(LEARNED_METHODS), 18)
        self.assertEqual(len(CONTROL_METHODS), 2)
        self.assertEqual(
            set(self.summary["method_id"].unique()), set(ALL_METHODS)
        )
        self.assertEqual(
            self.summary.groupby("method_id").size().nunique(), 1
        )

    def test_actions_coordinates_steps_success_identical_selected(self):
        for uid in self.selected["episode_uid"].astype(str):
            sub = self.summary[
                self.summary["episode_uid"].astype(str) == uid
            ]
            self.assertEqual(len(sub), 20, uid)
            for column in (
                "action_sha256",
                "decision_coord_sha256",
                "path_coord_sha256",
                "steps",
                "success",
            ):
                self.assertEqual(
                    sub[column].nunique(dropna=False), 1,
                    f"{uid}: {column}",
                )

    def test_coordinate_deviation_is_zero_selected(self):
        sub = self.invariance[
            self.invariance["scope"] == "selected6_closed_loop"
        ]
        self.assertEqual(len(sub), 6 * 20)
        self.assertTrue(sub["actions_equal"].all())
        self.assertTrue(sub["coord_hash_equal"].all())
        self.assertTrue(sub["coords_equal_1e-9_m"].all())
        self.assertTrue(sub["steps_equal"].all())
        self.assertTrue(sub["success_equal"].all())
        self.assertEqual(float(sub["max_coord_deviation_m"].max()), 0.0)
        self.assertEqual(float(sub["max_heading_deviation_rad"].max()), 0.0)

    def test_lane_replay_invariance_all_dev60(self):
        sub = self.invariance[self.invariance["scope"] == "dev60_lane"]
        self.assertEqual(len(sub), 60 * 20)
        self.assertTrue(sub["actions_equal"].all())
        self.assertTrue(sub["coord_hash_equal"].all())
        self.assertTrue(sub["steps_equal"].all())
        self.assertTrue(sub["success_equal"].all())
        self.assertEqual(float(sub["max_coord_deviation_m"].max()), 0.0)

    def test_lane_vs_closed_loop_equivalence(self):
        sub = self.invariance[
            self.invariance["scope"] == "selected6_lane_vs_closed_loop"
        ]
        self.assertEqual(len(sub), 6 * 20)
        self.assertTrue(sub["actions_equal"].all())
        self.assertTrue(sub["coord_hash_equal"].all())
        self.assertTrue(sub["steps_equal"].all())
        self.assertTrue(sub["success_equal"].all())
        self.assertEqual(float(sub["max_coord_deviation_m"].max()), 0.0)

    def test_closed_loop_node_choices_match_lane_replay(self):
        lane = self.invariance[
            self.invariance["scope"] == "dev60_lane"
        ]
        closed = self.invariance[
            self.invariance["scope"] == "selected6_closed_loop"
        ]
        for uid in self.selected["episode_uid"].astype(str):
            for method_id in ALL_METHODS:
                lane_row = lane[
                    (lane["episode_uid"].astype(str) == uid)
                    & (lane["method_id"] == method_id)
                ]
                closed_row = closed[
                    (closed["episode_uid"].astype(str) == uid)
                    & (closed["method_id"] == method_id)
                ]
                self.assertEqual(len(lane_row), 1, (uid, method_id))
                self.assertEqual(len(closed_row), 1, (uid, method_id))
                lane_row = lane_row.iloc[0]
                closed_row = closed_row.iloc[0]
                self.assertEqual(
                    lane_row["node_choice_sha256"],
                    closed_row["node_choice_sha256"],
                    (uid, method_id),
                )
                self.assertAlmostEqual(
                    float(lane_row["mean_e2e_ms"]),
                    float(closed_row["mean_e2e_ms"]),
                    places=9,
                    msg=(uid, method_id),
                )

    def test_node_choices_and_e2e_really_differ(self):
        sub = self.invariance[
            self.invariance["scope"] == "selected6_closed_loop"
        ]
        self.assertGreater(
            int(sub["node_choice_mismatch_steps"].sum()), 0,
            "association policies never changed a selected node",
        )
        self.assertGreater(
            float(sub["e2e_delta_ms"].abs().max()), 1e-6,
            "association policies never changed E2E latency",
        )
        diversity = self.audit["node_diversity"]
        self.assertTrue(diversity)
        self.assertGreaterEqual(
            max(item["distinct_node_choice_sequences"]
                for item in diversity.values()),
            2,
        )
        for uid, item in diversity.items():
            self.assertGreaterEqual(item["methods"], 20)
            self.assertGreaterEqual(item["e2e_spread_ms"], 0.0)

    def test_invariance_summary_passed(self):
        self.assertTrue(bool(self.audit["invariance_passed"]))
        for scope in (
            "dev60_lane",
            "selected6_closed_loop",
            "selected6_lane_vs_closed_loop",
        ):
            item = self.audit["scopes"][scope]
            self.assertEqual(item["action_mismatches"], 0, scope)
            self.assertEqual(item["coordinate_hash_mismatches"], 0, scope)
            self.assertEqual(item["step_mismatches"], 0, scope)
            self.assertEqual(item["success_mismatches"], 0, scope)
            self.assertEqual(item["max_coord_deviation_m"], 0.0, scope)

    # ------------------------------------------------------------------
    # Step-level data integrity
    # ------------------------------------------------------------------
    def test_step_rows_match_summary(self):
        for row in self.summary.itertuples(index=False):
            sub = self.steps[
                (self.steps["episode_uid"] == row.episode_uid)
                & (self.steps["method_id"] == row.method_id)
            ].sort_values("step_index")
            self.assertEqual(len(sub), int(row.steps),
                             (row.episode_uid, row.method_id))
            self.assertEqual(
                sub["step_index"].tolist(), list(range(len(sub)))
            )
            self.assertTrue(sub["action"].between(0, 3).all())
            self.assertTrue(sub["node_index"].between(0, 19).all())
            self.assertTrue(np.isfinite(sub["e2e_ms"]).all())
            self.assertTrue((sub["e2e_ms"] > 0).all())
            self.assertTrue(
                np.allclose(sub["z_m"].to_numpy(), ENV.uav_altitude_m)
            )
            self.assertTrue(
                (sub["heading_rad"] >= -math.pi - 1e-6).all()
            )
            self.assertTrue(
                (sub["heading_rad"] <= math.pi + 1e-6).all()
            )
            first = sub.iloc[0]
            self.assertAlmostEqual(float(first["x_m"]),
                                   float(row.start_x_m), places=4)
            self.assertAlmostEqual(float(first["y_m"]),
                                   float(row.start_y_m), places=4)
            last = sub.iloc[-1]
            dx = float(row.final_x_m) - float(last["x_m"])
            dy = float(row.final_y_m) - float(last["y_m"])
            distance = math.hypot(dx, dy)
            if int(last["action"]) == FORWARD:
                self.assertLessEqual(distance, ENV.forward_step_m + 1e-3)
            else:
                self.assertLessEqual(distance, 1e-3)
            self.assertAlmostEqual(
                float(row.start_goal_distance_m),
                float(row.start_goal_distance_m), places=6,
            )

    def test_path_length_and_spl_are_physical(self):
        for row in self.summary.itertuples(index=False):
            sub = self.steps[
                (self.steps["episode_uid"] == row.episode_uid)
                & (self.steps["method_id"] == row.method_id)
            ].sort_values("step_index")
            xy = sub[["x_m", "y_m"]].to_numpy(dtype=float)
            xy = np.vstack([xy, [row.final_x_m, row.final_y_m]])
            length = float(
                np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
            )
            self.assertAlmostEqual(
                length, float(row.path_length_m), places=2
            )
            if bool(row.success):
                expected_spl = float(row.start_goal_distance_m) / max(
                    float(row.start_goal_distance_m), length, 1e-8
                )
                self.assertAlmostEqual(
                    float(row.spl), expected_spl, places=4
                )
            else:
                self.assertEqual(float(row.spl), 0.0)

    # ------------------------------------------------------------------
    # Checkpoints / historical reproduction
    # ------------------------------------------------------------------
    def test_checkpoint_inventory(self):
        self.assertEqual(len(self.checkpoints), 20)
        learned = self.checkpoints[
            self.checkpoints["method_family"] == "learned"
        ]
        self.assertEqual(len(learned), 18)
        for row in learned.itertuples(index=False):
            self.assertEqual(len(str(row.checkpoint_sha256)), 64)
            self.assertIn("best.pt", str(row.checkpoint_type))
            self.assertTrue(bool(row.last_pt_exists))
            self.assertEqual(len(str(row.model_state_sha256)), 64)
            self.assertFalse(bool(row.final_test_read))
        controls = self.checkpoints[
            self.checkpoints["method_family"] == "control"
        ]
        self.assertEqual(len(controls), 2)
        groups = self.audit["duplicate_weight_groups"]
        self.assertTrue(groups)
        group_map = {}
        for group in groups:
            for method in group["methods"]:
                group_map[method] = group["model_state_sha256"]
        self.assertIn("2n-ppo-s0", group_map)
        self.assertEqual(group_map.get("2n-ppo-s0"),
                         group_map.get("2p-base-ppo-s0"))
        self.assertIn("2o-D-ppo-s0", group_map)
        self.assertEqual(group_map.get("2o-D-ppo-s0"),
                         group_map.get("2p-D-ppo-s0"))

    def test_historical_reproduction_passed(self):
        historical = self.audit["historical_crosscheck"]
        self.assertTrue(
            bool(historical["stage2n_per_episode"]["passed"]),
            historical["stage2n_per_episode"],
        )
        self.assertEqual(
            int(historical["stage2n_per_episode"]["mismatch_count"]), 0
        )
        self.assertTrue(
            bool(historical["stage2n_aggregate"]["passed"]),
            historical["stage2n_aggregate"],
        )
        self.assertTrue(
            bool(historical["stage2o_stage2p_validation"]["passed"]),
            historical["stage2o_stage2p_validation"],
        )

    # ------------------------------------------------------------------
    # Coordinate / hash helpers
    # ------------------------------------------------------------------
    def test_coordinate_reconstruction_matches_environment_dynamics(self):
        pose = np.asarray([100.0, 100.0, 0.0], dtype=np.float32)
        forward = advance_pose(pose, FORWARD)
        self.assertAlmostEqual(float(forward[0]), 105.0, places=5)
        self.assertAlmostEqual(float(forward[1]), 100.0, places=5)
        left = advance_pose(pose, LEFT)
        self.assertAlmostEqual(
            float(left[2]), math.radians(ENV.turn_step_deg), places=6
        )
        right = advance_pose(pose, RIGHT)
        self.assertAlmostEqual(
            float(right[2]), -math.radians(ENV.turn_step_deg), places=6
        )
        stop = advance_pose(pose, STOP)
        self.assertTrue(np.allclose(stop, pose))
        boundary = advance_pose(
            np.asarray([1499.0, 10.0, 0.0], dtype=np.float32), FORWARD
        )
        self.assertEqual(float(boundary[0]), 1500.0)
        wrapped = advance_pose(
            np.asarray([10.0, 10.0, math.radians(179.0)],
                       dtype=np.float32),
            LEFT,
        )
        self.assertLessEqual(abs(float(wrapped[2])), math.pi + 1e-6)
        turn = math.radians(ENV.turn_step_deg)
        poses = np.asarray([
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, turn],
        ], dtype=np.float32)
        actions = np.asarray([FORWARD, LEFT, STOP], dtype=np.int64)
        path = reconstruct_path(poses, actions)
        self.assertEqual(path.shape, (4, 3))
        self.assertAlmostEqual(float(path[1, 0]), 5.0, places=5)
        self.assertAlmostEqual(float(path[2, 2]), turn, places=6)
        self.assertTrue(np.allclose(path[3], path[2]))

    def test_hash_helpers(self):
        actions = [0, 1, 2, 3]
        expected = hashlib.sha256(b"0,1,2,3").hexdigest()
        self.assertEqual(_action_sha256(actions), expected)
        self.assertEqual(_action_sha256(actions), _action_sha256(actions))
        self.assertNotEqual(
            _action_sha256(actions), _action_sha256([0, 1, 2, 2])
        )
        coords = np.asarray([[1.0, 2.0, 0.0], [3.0, 4.0, 0.5]],
                            dtype=np.float32)
        self.assertEqual(_coord_sha256(coords), _coord_sha256(coords))
        changed = coords.copy()
        changed[0, 0] += 1e-4
        self.assertNotEqual(_coord_sha256(coords), _coord_sha256(changed))
        self.assertEqual(_node_sha256([1, 2, 3]),
                         _node_sha256([1, 2, 3]))
        self.assertNotEqual(_node_sha256([1, 2, 3]),
                            _node_sha256([1, 2, 4]))
        self.assertTrue(SELECTION_RULE)


if __name__ == "__main__":
    unittest.main()
