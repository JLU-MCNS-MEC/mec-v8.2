"""Stage-2l fixed-city / task-generation tests.

Runs without a GPU or the frozen MECs:

    python v83_spatial_stage2/test_stage2l_fixed_city.py

The file is also pytest-compatible if pytest is available.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import STOP
from v83_spatial_stage2.stage2l_fixed_city import (
    BAND_ACTION_RANGES,
    BAND_COUNTS_60,
    BAND_COUNTS_3500,
    BANDS,
    B_TRAIN_POOL_SIZE,
    CONFIG_DESCRIPTIONS,
    FixedCityEnv,
    MAP_SEED_PHASE1,
    SCENARIOS,
    TOPOLOGY_SEEDS,
    _inspect_task,
    build_city,
    build_stage2l_topology,
    config_specs,
    load_config_manifest,
    load_task_pool,
    load_topology,
    load_training_order,
    topology_role_for,
)
from v83_spatial_stage2.stage2k_common import BAND_RANGES


def _city_signature(city):
    return [
        (
            round(item.x, 6),
            round(item.y, 6),
            round(item.physical_size_m, 6),
            item.shape,
            tuple(np.round(item.base_color, 4)),
            item.marker_type,
            tuple(np.round(item.marker_color, 4)),
            tuple(item.signature),
            bool(item.is_distractor),
        )
        for item in city.landmarks
    ]


class FixedCityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.city = build_city(MAP_SEED_PHASE1)
        cls.env = FixedCityEnv(MAP_SEED_PHASE1)

    def test_city_cache_returns_same_city(self):
        again = build_city(MAP_SEED_PHASE1)
        self.assertIs(again, self.city)
        self.assertEqual(
            self.city.fingerprint, again.fingerprint
        )

    def test_map_is_fixed_across_tasks(self):
        before = self.city.fingerprint
        signature = _city_signature(self.city)
        for task_seed in (11_000_000, 11_000_001, 12_000_000, 13_000_000):
            self.env.reset(task_seed=task_seed, band="short",
                           render_observation=False)
            self.assertEqual(self.city.fingerprint, before)
            self.assertEqual(_city_signature(self.city), signature)
            self.assertEqual(len(self.city.landmarks), 160)
            self.assertEqual(len(self.env.landmarks), 159)

    def test_goal_is_an_existing_landmark(self):
        for task_seed in range(20):
            self.env.reset(task_seed=task_seed, band="medium",
                           render_observation=False)
            info = self.env.task_info
            goal = self.city.landmarks[int(info["goal_index"])]
            self.assertAlmostEqual(goal.x, self.env.goal[0], places=5)
            self.assertAlmostEqual(goal.y, self.env.goal[1], places=5)
            self.assertEqual(
                info["city_fingerprint"], self.city.fingerprint
            )
            positions = np.asarray(
                [[item.x, item.y] for item in self.city.landmarks],
                dtype=np.float64,
            )
            minimum = float(np.min(np.linalg.norm(
                positions - self.env.goal.astype(np.float64), axis=1
            )))
            self.assertLess(minimum, 1e-3)

    def test_selecting_goal_does_not_mutate_other_landmarks(self):
        self.env.reset(task_seed=1, band="short", render_observation=False)
        first = _city_signature(self.city)
        self.env.reset(task_seed=2, band="long", render_observation=False)
        second = _city_signature(self.city)
        self.assertEqual(first, second)

    def test_task_seed_is_reproducible(self):
        self.env.reset(task_seed=424242, band="long",
                       render_observation=False)
        first = self.env.task_info
        self.env.reset(task_seed=424242, band="long",
                       render_observation=False)
        second = self.env.task_info
        self.assertEqual(first, second)
        self.assertEqual(first["start"], second["start"])
        self.assertEqual(first["goal"], second["goal"])
        self.assertAlmostEqual(first["start_yaw"], second["start_yaw"])

    def test_different_tasks_differ(self):
        tasks = []
        for seed in (1, 2, 3, 4, 5):
            self.env.reset(task_seed=seed, band="short",
                           render_observation=False)
            tasks.append(self.env.task_info)
        self.assertGreater(
            len({tuple(item["goal"]) for item in tasks}), 1
        )
        self.assertGreater(
            len({tuple(item["start"]) for item in tasks}), 1
        )

    def test_start_distance_inside_band(self):
        for band in BANDS:
            low, high = BAND_RANGES[band]
            for seed in range(10):
                self.env.reset(task_seed=1000 + seed, band=band,
                               render_observation=False)
                distance = self.env.task_info["start_goal_distance_m"]
                self.assertGreaterEqual(distance, low - 1e-3)
                self.assertLessEqual(distance, high + 1e-3)

    def test_render_changes_with_pose(self):
        self.env.reset(task_seed=77, band="short",
                       render_observation=False)
        image_a = self.env.render()
        self.env.x += 5.0 * math.cos(self.env.yaw)
        self.env.y += 5.0 * math.sin(self.env.yaw)
        self.env.step_count += 1
        image_b = self.env.render()
        self.assertEqual(image_a.shape, (336, 336, 3))
        self.assertGreater(
            float(np.abs(image_a.astype(np.int16)
                         - image_b.astype(np.int16)).mean()),
            1.0,
        )

    def test_same_frame_is_reproducible(self):
        env_a = FixedCityEnv(MAP_SEED_PHASE1)
        env_b = FixedCityEnv(MAP_SEED_PHASE1)
        env_a.reset(task_seed=99, band="medium", render_observation=False)
        env_b.reset(task_seed=99, band="medium", render_observation=False)
        image_a = env_a.render()
        image_b = env_b.render()
        np.testing.assert_array_equal(image_a, image_b)

    def test_stop_success_requires_20m(self):
        env = FixedCityEnv(MAP_SEED_PHASE1)
        env.reset(task_seed=5, band="short", render_observation=False)
        env.x = float(env.goal[0] + 10.0)
        env.y = float(env.goal[1])
        _, success, done = env.step(STOP, render_observation=False)
        self.assertTrue(success)
        self.assertTrue(done)

        env.reset(task_seed=5, band="short", render_observation=False)
        env.x = float(env.goal[0] + 21.0)
        env.y = float(env.goal[1])
        _, success, done = env.step(STOP, render_observation=False)
        self.assertFalse(success)
        self.assertTrue(done)

    def test_oracle_steps_inside_band(self):
        for band in BANDS:
            accepted = 0
            for seed in range(50):
                info = _inspect_task(self.env, 2_000_000 + seed, band)
                if info["accepted"]:
                    accepted += 1
                    low, high = BAND_ACTION_RANGES[band]
                    self.assertGreaterEqual(info["steps"], low)
                    self.assertLessEqual(info["steps"], high)
            self.assertGreater(accepted, 0)


class TaskPoolTests(unittest.TestCase):
    def test_band_counts(self):
        train = load_task_pool("train")
        dev = load_task_pool("dev")
        test = load_task_pool("test")
        self.assertEqual(len(train["tasks"]), 3500)
        self.assertEqual(len(dev["tasks"]), 60)
        self.assertEqual(len(test["tasks"]), 60)
        for payload, expected in (
            (train, BAND_COUNTS_3500),
            (dev, BAND_COUNTS_60),
            (test, BAND_COUNTS_60),
        ):
            counts = Counter(item["band"] for item in payload["tasks"])
            self.assertEqual(dict(counts), expected)
            self.assertTrue(payload.get("sealed") is False
                            if payload["split"] != "test" else
                            payload.get("sealed") is True)

    def test_seed_and_id_isolation(self):
        train = load_task_pool("train")["tasks"]
        dev = load_task_pool("dev")["tasks"]
        test = load_task_pool("test")["tasks"]

        def seeds(rows):
            return {int(item["task_seed"]) for item in rows}

        def uids(rows):
            return {item["episode_uid"] for item in rows}

        for a, b in ((train, dev), (train, test), (dev, test)):
            self.assertFalse(seeds(a) & seeds(b))
            self.assertFalse(uids(a) & uids(b))
        self.assertEqual(
            {int(item["map_seed"]) for item in train + dev + test},
            {MAP_SEED_PHASE1},
        )

    def test_global_training_order_is_shuffled(self):
        order = load_training_order()
        train = {item["episode_uid"]: item for item in
                 load_task_pool("train")["tasks"]}
        self.assertEqual(len(order["order"]), 3500)
        self.assertEqual(len(set(order["order"])), 3500)
        self.assertEqual(set(order["order"]), set(train))
        bands = [train[uid]["band"] for uid in order["order"]]
        same = sum(1 for a, b in zip(bands, bands[1:]) if a == b)
        self.assertLess(same / (len(bands) - 1), 0.45)
        # A shuffled order must not be a concatenation of three bands.
        first_hundred = Counter(bands[:100])
        self.assertGreaterEqual(len(first_hundred), 3)

    def test_config_nav_tasks_are_identical(self):
        reference = None
        for config in CONFIG_DESCRIPTIONS:
            train, manifest, _order = config_specs(config, "train")
            signature = [
                (
                    item["episode_uid"],
                    int(item["map_seed"]),
                    int(item["task_seed"]),
                    item["band"],
                    int(item["goal_index"]),
                    tuple(round(v, 6) for v in item["start"]),
                    tuple(round(v, 6) for v in item["goal"]),
                )
                for item in train
            ]
            if reference is None:
                reference = signature
            else:
                self.assertEqual(signature, reference)
            self.assertEqual(len(train), 3500)
            self.assertTrue(all(item["scenario"] in SCENARIOS
                                for item in train))
            if config == "C":
                self.assertEqual(
                    set(item["scenario"] for item in train), set(SCENARIOS)
                )
            else:
                self.assertEqual(
                    {item["scenario"] for item in train}, {"balanced"}
                )
            if config == "B":
                indices = [int(item["topology_index"]) for item in train]
                self.assertEqual(set(indices),
                                 set(range(B_TRAIN_POOL_SIZE)))

    def test_topology_separation(self):
        a = load_topology("A_fixed")
        b_dev = load_topology("B_dev")
        b_train = [load_topology(f"B_train_{i:02d}")
                   for i in range(B_TRAIN_POOL_SIZE)]
        self.assertEqual(len(a["nodes"]), 20)
        self.assertNotEqual(a["topology_seed"], b_dev["topology_seed"])
        self.assertNotIn(
            b_dev["topology_seed"],
            {item["topology_seed"] for item in b_train},
        )
        self.assertEqual(
            len({item["topology_seed"] for item in b_train}),
            B_TRAIN_POOL_SIZE,
        )
        self.assertTrue(all(item["scenario"] == "balanced"
                            for item in b_train))

    def test_config_manifest_topology_roles(self):
        for config in CONFIG_DESCRIPTIONS:
            manifest = load_config_manifest(config)
            train, _, _ = config_specs(config, "train")
            dev, _, _ = config_specs(config, "dev")
            train_roles = {
                topology_role_for(config, "train", item) for item in train
            }
            dev_roles = {
                topology_role_for(config, "dev", item) for item in dev
            }
            self.assertEqual(train_roles, set(manifest["topology_roles"]["train"]))
            self.assertEqual(dev_roles, set(manifest["topology_roles"]["dev"]))
            if config == "B":
                self.assertNotIn("B_dev", train_roles)

    def test_quality_prior_topology_uses_flat_ndtw(self):
        prior = json.loads(Path(
            "runs/v83_spatial_stage2/stage2k/capability_prior_v2.json"
        ).read_text(encoding="utf-8"))
        topology = build_stage2l_topology(
            "quality_rich", TOPOLOGY_SEEDS["C_quality_rich"], prior,
            "test_quality_rich",
        )
        for node in topology["nodes"]:
            self.assertAlmostEqual(
                node["quality_prior"],
                float(prior["models"][node["model_type"]]["ndtw"]),
                places=8,
            )


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(FixedCityTests))
    suite.addTests(loader.loadTestsFromTestCase(TaskPoolTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("STAGE2L_TESTS_OK")


if __name__ == "__main__":
    main()
