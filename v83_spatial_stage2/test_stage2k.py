"""Stage-2k unit/smoke checks (no MEC training, no closed-loop test).

Run from the repository root:

    python -m unittest -v v83_spatial_stage2.test_stage2k
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2j_eval import load_checkpoint
from v83_spatial_stage2.stage2j_model import (
    build_stage2j_model,
    parameter_count,
)
from v83_spatial_stage2.stage2k_common import (
    BANDS,
    DQN_VALIDATION_BAND_COUNTS,
    MODEL_NAMES,
    SCENARIOS,
    TRAIN_EPISODES,
    build_topology,
    joint_reward,
    latency_penalty,
    load_split_specs,
    seed_isolation_report,
    validation_specs,
)
from v83_spatial_stage2.stage2k_dqn import (
    QNetwork,
    dev_validation_specs,
    plot_training_curves,
    should_early_stop,
)
from v83_spatial_stage2.stage2k_latency import (
    STAGE2J_PAYLOAD_BYTES,
    Stage2KLatency,
    model_complexity_table,
    validate_payload_against_arrays,
)
from v83_spatial_stage2.stage2k_state import (
    STATE_DIM,
    VisualProjector,
    build_state,
    candidate_features,
    global_features,
    masked_argmax,
    state_layout,
)
from v83_spatial_stage2.stage2k_common import VISUAL_PCA_PATH


EXPECTED_SHA = {
    "Small": "7349ec9d0c809a09dc2672a05053f1f89e69480b8802d6ca3b05944cea44e1f6",
    "Transformer-S": "ceac7506b3b3e18b8db1f8465dc42f9cdf5ca3c181cb39daae9905ae9ae42380",
    "Medium": "1fcce0f33dfff368b7fd73eb9749ee883b9c237f63596f2a679f625cf2d86def",
    "Transformer-M": "57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d",
    "Large": "310c9d2648dec7fe84ac1f0685aa76188090b7e4445d9034236af2c0d495f21f",
}
EXPECTED_PARAMS = {
    "Small": 611716,
    "Transformer-S": 1332868,
    "Medium": 1991044,
    "Transformer-M": 3033732,
    "Large": 5248932,
}
EXAMPLE_NPZ = (
    PROJECT_ROOT / "v83_spatial_stage2/t5_input_examples/"
    "case_01_clear_match/t5_inputs.npz"
)


class Stage2KChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(EXAMPLE_NPZ, allow_pickle=False) as npz:
            cls.example = {key: npz[key] for key in npz.files}
        with np.load(VISUAL_PCA_PATH, allow_pickle=False) as data:
            cls.pca_meta = json.loads(str(data["meta"]))
            cls.pca_components = data["components"]

    def test_01_checkpoint_files_parameters_and_output_reproducibility(self):
        import hashlib

        for name in MODEL_NAMES:
            path = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2j" / name / "best.pt"
            self.assertTrue(path.is_file(), name)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                EXPECTED_SHA[name],
            )
            model = build_stage2j_model(name)
            load_checkpoint(model, path, "cpu")
            self.assertEqual(parameter_count(model), EXPECTED_PARAMS[name])
            tensors = {
                key: torch.from_numpy(value).unsqueeze(0)
                for key, value in self.example.items()
                if key in (
                    "current_tokens", "goal_tokens", "history_vis",
                    "history_pose", "history_action", "history_mask",
                    "gab", "map_feat",
                )
            }
            with torch.inference_mode():
                first = model.forward_components(
                    tensors["current_tokens"],
                    tensors["goal_tokens"],
                    history_vis=tensors["history_vis"],
                    history_pose=tensors["history_pose"],
                    history_action=tensors["history_action"],
                    history_mask=tensors["history_mask"],
                    gab=tensors["gab"],
                    map_feat=tensors["map_feat"],
                )
                second = model.forward_components(
                    tensors["current_tokens"],
                    tensors["goal_tokens"],
                    history_vis=tensors["history_vis"],
                    history_pose=tensors["history_pose"],
                    history_action=tensors["history_action"],
                    history_mask=tensors["history_mask"],
                    gab=tensors["gab"],
                    map_feat=tensors["map_feat"],
                )
            self.assertTrue(torch.equal(
                first["move_logits"], second["move_logits"]
            ))
            self.assertTrue(torch.equal(
                first["arrival_logit"], second["arrival_logit"]
            ))
            self.assertEqual(tuple(first["move_logits"].shape), (1, 3))
            self.assertEqual(tuple(first["arrival_logit"].shape), (1,))

    def test_02_payload_is_actual_stage2j_input_bytes(self):
        report = validate_payload_against_arrays(self.example)
        self.assertEqual(STAGE2J_PAYLOAD_BYTES, 470_200)
        self.assertTrue(report["match"])
        self.assertEqual(report["actual_bytes"], 470_200)

    def test_03_analytic_flops_are_positive_and_ordered(self):
        table = model_complexity_table()
        values = [table[name]["flops"] for name in MODEL_NAMES]
        self.assertTrue(all(value > 0 for value in values))
        order = [
            "Small", "Medium", "Transformer-S",
            "Transformer-M", "Large",
        ]
        self.assertEqual(
            sorted(order, key=lambda name: table[name]["flops"]),
            order,
        )

    def test_04_latency_formula_and_state_features(self):
        latency = Stage2KLatency()
        node = {
            "model_type": "Transformer-M",
            "x": 100.0,
            "y": 0.0,
            "bandwidth_hz": 20.0e6,
            "compute_capacity_flops_per_s": 1.0e12,
            "coverage_radius_m": 500.0,
            "model_index": 3,
            "quality_prior": 0.55,
        }
        metric = latency.evaluate(node, np.asarray([0.0, 0.0]), "Transformer-M")
        self.assertAlmostEqual(
            metric["total_ms"],
            metric["extraction_ms"]
            + metric["communication_ms"]
            + metric["inference_ms"],
            places=9,
        )
        self.assertGreater(metric["extraction_ms"], 20.0)
        self.assertGreater(metric["communication_ms"], 0.0)

        nodes = [dict(node, model_index=3) for _ in range(20)]
        for index, item in enumerate(nodes):
            item["x"] = 1000.0 + index * 10.0
            item["y"] = 1000.0
        features, mask, metrics, fallback = candidate_features(
            nodes, np.asarray([0.0, 0.0]), latency
        )
        self.assertTrue(fallback)
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(features.shape, (20, 17))
        self.assertEqual(len(metrics), 20)
        state = build_state(
            np.zeros(128, dtype=np.float32),
            global_features(0, 400, np.zeros(16), 0.5,
                            np.eye(20, dtype=np.float32)[0], 0.0, -1),
            features,
        )
        self.assertEqual(state.shape, (STATE_DIM,))
        self.assertEqual(STATE_DIM, 478)

    def test_05_state_has_no_privileged_fields(self):
        fields = state_layout()["candidate_fields"]
        joined = " ".join(fields).lower()
        for term in ("goal_xy", "true_distance", "oracle", "success"):
            self.assertNotIn(term, joined)
        self.assertEqual(len(fields), 17)
        self.assertEqual(state_layout()["total"], 478)

    def test_06_pca_was_fitted_on_train_only(self):
        self.assertEqual(self.pca_meta["fitted_split"], "train_600")
        self.assertEqual(self.pca_meta["fitted_sample_type"], "natural")
        self.assertFalse(self.pca_meta["final_test_read"])
        self.assertEqual(self.pca_components.shape, (128, 3114))
        projector = VisualProjector.load(VISUAL_PCA_PATH)
        projected = projector.project(np.zeros(3114, dtype=np.float32))
        self.assertEqual(projected.shape, (128,))
        self.assertTrue(np.isfinite(projected).all())

    def test_07_specs_counts_bands_and_isolation(self):
        train = load_split_specs("train")
        dev = load_split_specs("dev")
        test = load_split_specs("test")
        self.assertEqual(
            sum(len(train["scenarios"][s]["episodes"]) for s in SCENARIOS),
            TRAIN_EPISODES,
        )
        for split in ("dev", "test"):
            payload = dev if split == "dev" else test
            self.assertEqual(
                sum(len(payload["scenarios"][s]["episodes"])
                    for s in SCENARIOS),
                180,
            )
            for scenario in SCENARIOS:
                episodes = payload["scenarios"][scenario]["episodes"]
                counts = {
                    band: sum(1 for item in episodes if item["band"] == band)
                    for band in BANDS
                }
                self.assertEqual(
                    counts, {"short": 18, "medium": 30, "long": 12}
                )
        report = seed_isolation_report(
            {"train": train, "dev": dev, "test": test}, {}
        )
        self.assertTrue(report["ok"], report["overlaps"])

    def test_08_fixed_dqn_validation_mix(self):
        dev = load_split_specs("dev")
        rows = validation_specs(dev)
        self.assertEqual(len(rows), 60)
        for scenario in SCENARIOS:
            subset = [row for row in rows if row["scenario"] == scenario]
            self.assertEqual(len(subset), 20)
            counts = {
                band: sum(1 for item in subset if item["band"] == band)
                for band in BANDS
            }
            self.assertEqual(counts, DQN_VALIDATION_BAND_COUNTS)

    def test_09_jointr_and_latency_penalty(self):
        self.assertAlmostEqual(joint_reward(0.6, 80.0), 0.6 - 0.2)
        self.assertAlmostEqual(
            latency_penalty(200.0), -0.25 * 2.0 / 200.0
        )
        self.assertAlmostEqual(
            latency_penalty(1000.0), -0.25 * 3.0 / 200.0
        )

    def test_10_dqn_mask_and_early_stop(self):
        network = QNetwork()
        q_values = network(torch.zeros(4, STATE_DIM)).detach().numpy()
        self.assertEqual(q_values.shape, (4, 20))
        mask = np.zeros(20, dtype=bool)
        mask[[2, 5, 7]] = True
        action = masked_argmax(q_values[0], mask)
        self.assertIn(action, (2, 5, 7))
        stop, reason = should_early_stop(600, 8, [0.1, 0.1, 0.1])
        self.assertFalse(stop)
        stop, reason = should_early_stop(1200, 5, [0.5, 0.5, 0.5])
        self.assertTrue(stop)
        stop, reason = should_early_stop(1600, 8, [0.5, 0.49, 0.51])
        self.assertTrue(stop)

    def test_13_dqn_validation_specs_are_dev_and_disjoint(self):
        dev_val = dev_validation_specs()
        self.assertEqual(len(dev_val), 60)
        self.assertTrue(all(item["split"] == "dev" for item in dev_val))
        train = load_split_specs("train")
        train_uids = {
            item["episode_uid"]
            for scenario in SCENARIOS
            for item in train["scenarios"][scenario]["episodes"]
        }
        self.assertFalse(
            {item["episode_uid"] for item in dev_val} & train_uids
        )
        counts = Counter(
            (item["scenario"], item["band"]) for item in dev_val
        )
        for scenario in SCENARIOS:
            for band, expected in (("short", 6), ("medium", 10), ("long", 4)):
                self.assertEqual(counts[(scenario, band)], expected)

    def test_12_plot_training_curves_uses_history_schema(self):
        history = {
            "train": [
                {"episode": index + 1, "return": float(index) / 120.0}
                for index in range(120)
            ],
            "validation": [
                {"episode": 100, "joint_reward": 0.1},
                {"episode": 200, "joint_reward": 0.2},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            plot_training_curves(history, directory)
            for name in (
                "training_return_ma100.png",
                "validation_jointr.png",
            ):
                self.assertTrue((Path(directory) / name).is_file(), name)

    def test_11_topologies_when_built(self):
        quality_path = PROJECT_ROOT / "runs/v83_spatial_stage2/stage2k/quality_prior.json"
        if not quality_path.is_file():
            self.skipTest("quality prior not built yet")
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        for scenario in SCENARIOS:
            payload = build_topology(scenario, "dev", quality)
            self.assertEqual(len(payload["nodes"]), 20)
            model_counts = {
                name: sum(
                    1 for node in payload["nodes"]
                    if node["model_type"] == name
                )
                for name in MODEL_NAMES
            }
            if scenario == "balanced":
                self.assertTrue(all(count == 4 for count in model_counts.values()))
            else:
                top3 = set(quality["rank"][:3])
                self.assertTrue(
                    all(
                        model_counts[name] == 5 for name in top3
                    )
                )
                self.assertEqual(
                    sum(model_counts.values()), 20
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
