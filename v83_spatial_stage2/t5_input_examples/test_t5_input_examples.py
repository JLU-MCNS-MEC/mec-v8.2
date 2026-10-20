"""Lightweight checks for the three exported Stage-2j T5 input examples.

These tests do not train or evaluate a model.  They:

  * load every NPZ and enforce the exact Stage-2j input contract;
  * run ``stage2j_model._validate_common_inputs`` on batch-wrapped inputs;
  * rebuild the 16-step history from the archived episode and compare;
  * reproduce the 20-D GAB and 22-D anonymous map feature from source data;
  * prove that the NPZ contains no true goal coordinate, true distance,
    oracle action, success label or other diagnostic field;
  * verify metadata hashes and that the three selection rules were met.

Run from the repository root:

    python -m unittest -v v83_spatial_stage2.t5_input_examples.test_t5_input_examples
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import ToyUAVEnv  # noqa: E402
from config import TRAIN  # noqa: E402
from v83_spatial_stage2.stage2h_gab import current_evidence  # noqa: E402
from v83_spatial_stage2.stage2i_data import (  # noqa: E402
    build_gab_for_episode,
    build_landmark_map,
    gab_config_from_threshold,
    history_arrays,
    map_retrieval_features,
    retrieval_similarities,
    visual_descriptor,
)
from v83_spatial_stage2.stage2j_model import (  # noqa: E402
    _validate_common_inputs,
)
from v83_spatial_stage2.t5_input_examples.export_t5_input_examples import (  # noqa: E402
    CASE_ORDER,
    EXPECTED_ARRAYS,
    SELECTION_RULES,
    load_episode_sequence,
    reject_final_test,
    sha256_file,
)
from v83_spatial_training.eval_closedloop_spatial import load_encoder  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent
REQUIRED_VISUALS = (
    "current_rgb.png",
    "goal_reference_scales.png",
    "local_similarity_24x24.png",
    "gab_bearing_belief.png",
    "landmark_top5_map.png",
    "landmark_top5.csv",
)
FORBIDDEN_INPUT_TERMS = (
    "goal_xy", "goal_coord", "goal_position", "true_distance",
    "distance_to_goal", "oracle", "success", "is_goal", "goal_index",
)


def _shard_paths_from_metadata(metadata):
    """Unique shard paths in the provenance list, with stable indices."""
    paths = []
    for item in metadata["provenance"]["episode_natural_steps"]:
        path = PROJECT_ROOT / item["shard"]
        if path not in paths:
            paths.append(path)
    return paths


def _sequence_from_metadata(metadata, shard_paths):
    """Build the episode sequence consumed by the export helpers."""
    path_index = {path: index for index, path in enumerate(shard_paths)}
    episode_rows = {
        int(metadata["provenance"]["episode_id"]): [
            (
                path_index[PROJECT_ROOT / item["shard"]],
                int(item["row"]),
                int(item["step"]),
            )
            for item in metadata["provenance"]["episode_natural_steps"]
        ]
    }
    cache = {}
    return load_episode_sequence(
        int(metadata["provenance"]["episode_id"]),
        episode_rows,
        shard_paths,
        cache,
    ), cache


def _replay_with_poses(metadata, actions):
    """Replay the archived expert actions and record every visited pose."""
    band_id = int(metadata["provenance"]["band_id"])
    band = TRAIN.trajectory_mix[band_id]
    env = ToyUAVEnv()
    observation = env.reset(
        seed=int(metadata["provenance"]["episode_seed"]),
        goal_distance_range_m=(float(band[2]), float(band[3])),
        render_observation=True,
    )
    step = int(metadata["provenance"]["step_id"])
    poses = []
    for index in range(step):
        poses.append(env.pose.copy())
        observation, _, _ = env.step(
            int(actions[index]), render_observation=True
        )
    return env, observation, np.asarray(poses, dtype=np.float32).reshape(-1, 3)


class T5InputExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        cls.encoder = load_encoder(cls.device)
        cls.selection = json.loads(
            (OUTPUT_DIR / "selection_rules.json").read_text(encoding="utf-8")
        )
        cls.cases = {}
        for case in CASE_ORDER:
            case_dir = OUTPUT_DIR / case
            reject_final_test(case_dir)
            metadata = json.loads(
                (case_dir / "case_metadata.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (case_dir / "input_summary.json").read_text(encoding="utf-8")
            )
            diagnostics = json.loads(
                (case_dir / "diagnostics.json").read_text(encoding="utf-8")
            )
            with np.load(case_dir / "t5_inputs.npz", allow_pickle=False) as npz:
                arrays = {name: npz[name] for name in npz.files}
            shard_paths = _shard_paths_from_metadata(metadata)
            sequence, _ = _sequence_from_metadata(metadata, shard_paths)

            # Archived tokens for every state, used to reproduce history_vis.
            visuals = []
            tokens_by_step = {}
            shard_cache = {}
            for shard_index, row, step in sequence["items"]:
                if shard_index not in shard_cache:
                    shard_cache[shard_index] = torch.load(
                        shard_paths[shard_index],
                        map_location="cpu",
                        weights_only=True,
                    )
                token = shard_cache[shard_index]["tokens"][row].cpu()
                tokens_by_step[step] = token
                visuals.append(visual_descriptor(token.unsqueeze(0))[0].numpy())

            env, observation, poses = _replay_with_poses(
                metadata, sequence["actions"]
            )

            cls.cases[case] = {
                "dir": case_dir,
                "metadata": metadata,
                "summary": summary,
                "diagnostics": diagnostics,
                "arrays": arrays,
                "sequence": sequence,
                "visuals": visuals,
                "tokens_by_step": tokens_by_step,
                "env": env,
                "observation": observation,
                "poses": poses,
            }

    # ------------------------------------------------------------------
    # Loading, shapes and dtypes
    # ------------------------------------------------------------------
    def test_01_all_cases_load_with_exact_schema(self):
        self.assertEqual(set(self.cases), set(CASE_ORDER))
        for case, payload in self.cases.items():
            for filename in REQUIRED_VISUALS:
                path = payload["dir"] / filename
                self.assertTrue(path.is_file(), f"{case}: missing {filename}")
                minimum = 100 if filename.endswith(".csv") else 1000
                self.assertGreater(path.stat().st_size, minimum)
            arrays = payload["arrays"]
            self.assertEqual(set(arrays), set(EXPECTED_ARRAYS))
            for name, spec in EXPECTED_ARRAYS.items():
                array = arrays[name]
                self.assertEqual(
                    tuple(array.shape), tuple(spec["shape"]),
                    f"{case}.{name} shape",
                )
                self.assertEqual(
                    array.dtype, spec["dtype"],
                    f"{case}.{name} dtype",
                )
                self.assertTrue(
                    np.isfinite(array).all(), f"{case}.{name} non-finite"
                )
            self.assertEqual(arrays["history_mask"].shape, (16,))
            self.assertTrue(
                set(np.unique(arrays["history_mask"]).tolist()) <= {0, 1}
            )

    def test_02_stage2j_shape_validation_accepts_every_case(self):
        for case, payload in self.cases.items():
            arrays = payload["arrays"]
            batch = {
                name: torch.from_numpy(array).unsqueeze(0)
                for name, array in arrays.items()
            }
            # Must not raise; heatmap/network_state are structurally absent.
            _validate_common_inputs(
                batch["current_tokens"],
                batch["goal_tokens"],
                batch["history_vis"],
                batch["history_pose"],
                batch["history_action"],
                batch["history_mask"],
                batch["gab"],
                batch["map_feat"],
                heatmap=None,
                network_state=None,
            )
            with self.assertRaises(ValueError):
                _validate_common_inputs(
                    batch["current_tokens"],
                    batch["goal_tokens"],
                    batch["history_vis"],
                    batch["history_pose"],
                    batch["history_action"],
                    batch["history_mask"],
                    batch["gab"],
                    batch["map_feat"],
                    heatmap=torch.zeros(1, 3, 8, 8),
                    network_state=None,
                )

    # ------------------------------------------------------------------
    # History / GAB / map reproduction
    # ------------------------------------------------------------------
    def test_03_history_arrays_match_the_archived_episode(self):
        for case, payload in self.cases.items():
            arrays = payload["arrays"]
            metadata = payload["metadata"]
            step = int(metadata["provenance"]["step_id"])
            sequence = payload["sequence"]
            poses = payload["poses"]
            self.assertEqual(len(poses), step)
            full_poses = np.concatenate(
                [poses, payload["env"].pose[None, :]], axis=0
            )
            expected_vis, expected_pose, expected_action, expected_mask = (
                history_arrays(
                    full_poses,
                    sequence["actions"],
                    step,
                    payload["visuals"],
                )
            )
            self.assertTrue(np.array_equal(arrays["history_mask"], expected_mask))
            self.assertTrue(np.array_equal(arrays["history_action"], expected_action))
            # history_vis is float16.  A CPU re-pooling of the archived tokens
            # can differ from the GPU-produced sidecar by one float16 ULP;
            # allow at most two ULP-equivalent absolute steps.
            self.assertTrue(
                np.allclose(
                    arrays["history_vis"].astype(np.float32),
                    expected_vis.astype(np.float32),
                    atol=2e-4, rtol=0,
                ),
                f"{case}: history_vis mismatch",
            )
            self.assertTrue(
                np.allclose(arrays["history_pose"], expected_pose, atol=1e-6)
            )
            # The documented redundancy: history_pose[6:10] is the same
            # action one-hot that history_action stores separately.
            self.assertTrue(
                np.allclose(
                    arrays["history_pose"][:, 6:10],
                    arrays["history_action"].astype(np.float32),
                )
            )

    def test_04_gab_is_reproducible_from_archived_heatmaps(self):
        for case, payload in self.cases.items():
            arrays = payload["arrays"]
            sequence = payload["sequence"]
            config = gab_config_from_threshold(
                float(self.selection["gab_reliability_threshold"])
            )
            rebuilt = build_gab_for_episode(
                sequence["heatmaps"],
                sequence["actions"],
                config,
                step_ids=sequence["steps"],
            )
            step = int(payload["metadata"]["provenance"]["step_id"])
            self.assertTrue(
                np.allclose(rebuilt[step], arrays["gab"], atol=1e-7, rtol=0),
                f"{case}: GAB mismatch",
            )
            evidence = current_evidence(sequence["heatmaps"][step])
            self.assertAlmostEqual(
                float(evidence["confidence"]),
                float(payload["diagnostics"]["current_evidence"]["confidence"]),
                places=7,
            )

    def test_05_map_feature_and_top5_csv_are_reproducible(self):
        for case, payload in self.cases.items():
            arrays = payload["arrays"]
            env = payload["env"]
            goal_token = arrays["goal_tokens"][2].astype(np.float32)
            landmark_map = build_landmark_map(
                env, int(payload["metadata"]["provenance"]["episode_seed"]),
                self.encoder, self.device,
            )
            rebuilt = map_retrieval_features(
                env.pose,
                landmark_map["positions"],
                landmark_map["embeddings"],
                goal_token,
            )
            self.assertTrue(
                np.allclose(rebuilt, arrays["map_feat"], atol=1e-5, rtol=1e-6),
                f"{case}: map feature mismatch",
            )
            similarities = retrieval_similarities(
                landmark_map["embeddings"], goal_token
            )
            order = np.argsort(-similarities)[:5]
            with (payload["dir"] / "landmark_top5.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            x, y, yaw = map(float, env.pose)
            for rank, (row, index) in enumerate(zip(rows, order), start=1):
                self.assertEqual(int(row["rank"]), rank)
                self.assertAlmostEqual(
                    float(row["similarity"]),
                    float(similarities[index]), places=8,
                )
                dx = float(landmark_map["positions"][index, 0]) - x
                dy = float(landmark_map["positions"][index, 1]) - y
                forward = (dx * np.cos(yaw) + dy * np.sin(yaw)) / 500.0
                right = (-dx * np.sin(yaw) + dy * np.cos(yaw)) / 500.0
                distance = float(np.hypot(dx, dy)) / 500.0
                self.assertAlmostEqual(
                    float(row["relative_forward"]), forward, places=7
                )
                self.assertAlmostEqual(
                    float(row["relative_right"]), right, places=7
                )
                self.assertAlmostEqual(
                    float(row["normalized_distance"]), distance, places=7
                )

    # ------------------------------------------------------------------
    # Input isolation
    # ------------------------------------------------------------------
    def test_06_npz_contains_only_observable_model_inputs(self):
        for case, payload in self.cases.items():
            arrays = payload["arrays"]
            self.assertEqual(set(arrays), set(EXPECTED_ARRAYS))
            for key in arrays:
                lowered = key.lower()
                self.assertFalse(
                    any(term in lowered for term in FORBIDDEN_INPUT_TERMS),
                    f"{case}: forbidden input key {key}",
                )
            diagnostics = payload["diagnostics"]
            self.assertTrue(diagnostics["offline_only"])
            self.assertTrue(diagnostics["not_model_input"])
            self.assertIn("true_distance_to_goal_m", diagnostics[
                "replay_and_correctness"
            ])
            self.assertNotIn("true_distance_to_goal_m", arrays)
            # No object/string arrays that could smuggle labels.
            for name, array in arrays.items():
                self.assertNotEqual(array.dtype, object, name)

    def test_07_input_summary_matches_npz(self):
        for case, payload in self.cases.items():
            arrays = payload["arrays"]
            summary = payload["summary"]
            for name, stats in summary["arrays"].items():
                array = arrays[name].astype(np.float64)
                self.assertEqual(stats["shape"], list(array.shape))
                self.assertEqual(stats["dtype"], str(arrays[name].dtype))
                self.assertAlmostEqual(stats["min"], float(array.min()), places=6)
                self.assertAlmostEqual(stats["max"], float(array.max()), places=6)
                self.assertAlmostEqual(stats["mean"], float(array.mean()), places=6)
                self.assertAlmostEqual(stats["std"], float(array.std()), places=6)
            self.assertTrue(summary["validity"]["all_finite"])
            self.assertTrue(summary["validity"]["history_mask_pattern_ok"])
            self.assertTrue(
                summary["validity"]["history_action_one_hot_under_mask"]
            )
            self.assertTrue(
                summary["validity"][
                    "history_pose_action_block_equals_history_action"
                ]
            )

    # ------------------------------------------------------------------
    # Selection discipline and provenance
    # ------------------------------------------------------------------
    def test_08_selection_rules_are_met(self):
        rules = self.selection
        selected = rules["selected"]
        self.assertEqual(len(selected), 3)
        episodes = [item["episode_id"] for item in selected]
        self.assertEqual(len(set(episodes)), 3, "episodes must be distinct")
        by_case = {item["case"]: item for item in selected}

        case1 = by_case["case_01_clear_match"]
        self.assertGreaterEqual(case1["history_valid_slots"], 8)
        self.assertGreaterEqual(case1["current_confidence_percentile"], 99.99)

        case2 = by_case["case_02_weak_match_history"]
        self.assertGreaterEqual(case2["history_valid_slots"], 8)
        self.assertEqual(case2["gab_valid"], 1)
        self.assertGreaterEqual(
            case2["gab_confidence"], rules["gab_reliability_threshold"]
        )
        self.assertLess(case2["current_confidence_percentile"], 1.0)

        case3 = by_case["case_03_ambiguous_map"]
        self.assertGreaterEqual(case3["history_valid_slots"], 8)
        self.assertGreaterEqual(case3["map_top1"], rules["top1_similarity_median"])
        self.assertLess(case3["map_top5_spread_percentile"], 1.0)
        self.assertLess(case3["map_top5_spread"], 0.01)

        self.assertFalse(rules["constraints"]["navigation_success_used"])
        self.assertFalse(rules["constraints"]["true_distance_used"])
        self.assertFalse(rules["constraints"]["goal_coordinates_used"])
        self.assertFalse(rules["constraints"]["oracle_action_used"])
        self.assertFalse(rules["final_test_read"])

        for case in CASE_ORDER:
            metadata = self.cases[case]["metadata"]
            self.assertEqual(metadata["selection_rule"], SELECTION_RULES[case])
            self.assertEqual(
                metadata["provenance"]["episode_id"],
                by_case[case]["episode_id"],
            )
            self.assertEqual(
                metadata["provenance"]["step_id"], by_case[case]["step_id"]
            )

    def test_09_metadata_hashes_match_files(self):
        for case, payload in self.cases.items():
            metadata = payload["metadata"]
            npz_path = payload["dir"] / metadata["npz"]["file"]
            self.assertEqual(
                sha256_file(npz_path), metadata["npz"]["sha256"]
            )
            shard_path = PROJECT_ROOT / metadata["provenance"]["shard"]
            aux_path = PROJECT_ROOT / metadata["provenance"]["aux_file"]
            self.assertEqual(
                sha256_file(shard_path),
                metadata["provenance"]["shard_sha256"],
            )
            self.assertEqual(
                sha256_file(aux_path),
                metadata["provenance"]["aux_sha256"],
            )
            self.assertFalse(metadata["final_test_read"])
            self.assertEqual(metadata["provenance"]["split"], "train_600")
            self.assertIn(
                metadata["provenance"]["trajectory_band"],
                ("short", "medium", "long"),
            )

    def test_10_reconstructed_observation_agrees_with_archived_tokens(self):
        for case, payload in self.cases.items():
            tokens = payload["tokens_by_step"][
                int(payload["metadata"]["provenance"]["step_id"])
            ].float()
            archived = torch.from_numpy(
                payload["arrays"]["current_tokens"].astype(np.float32)
            )
            # The rendered observation is the same scene; the frozen encoder
            # re-encode agrees up to GPU float noise (float16 storage).
            self.assertGreater(
                float(torch.nn.functional.cosine_similarity(
                    tokens.flatten(), archived.flatten(), dim=0
                )),
                0.9999,
                case,
            )
            self.assertLess(
                float((tokens - archived).abs().max()),
                0.01,
                case,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
