"""CPU tests for the Stage-2j five-MEC protocol.

The tests cover architecture identity, the single T5 input contract,
deterministic initialisation, finite forward/backward, training-protocol
parity, sealed Final-Test extraction and result formatting.  They never read
the real Final Test data.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import ENCODER, GBS_MODEL_CONFIGS
from v83_spatial_stage2 import stage2i_train, stage2j_train
from v83_spatial_stage2.stage2i_model import (
    GAB_FEATURE_DIM,
    HISTORY_STEPS,
    MAP_FEATURE_DIM,
    POSE_FEATURE_DIM,
    VISUAL_DESCRIPTOR_DIM,
    Stage2ITransformer,
    parameter_count,
)
from v83_spatial_stage2.stage2j_common import (
    BANDS,
    MODEL_DIRS,
    STOP_THRESHOLD,
    TRAIN_PROTOCOL,
    cumulative_table,
    format_result_block,
    latency_summary,
    load_progress,
)
from v83_spatial_stage2.stage2j_eval import (
    extract_final_test_specs,
    verify_protocol_freeze,
    verify_sealed,
)
from v83_spatial_stage2.stage2j_model import (
    STAGE2J_CONFIGS,
    TABLE_ORDER,
    Stage2JMediumMEC,
    Stage2JSmallMEC,
    build_stage2j_model,
)
from v83_spatial_stage2.stage2j_analyze import (
    METRICS,
    bootstrap_results,
    plot_metric,
)


EXPECTED_PARAMETERS = {
    "Small": 611716,
    "Transformer-S": 1332868,
    "Medium": 1991044,
    "Transformer-M": 3033732,
    "Large": 5248932,
}


def random_inputs(batch=2):
    return {
        "current_tokens": torch.randn(
            batch, ENCODER.num_tokens, ENCODER.token_dim
        ),
        "goal_tokens": torch.randn(batch, 3, ENCODER.token_dim),
        "history_vis": torch.randn(
            batch, HISTORY_STEPS, VISUAL_DESCRIPTOR_DIM
        ),
        "history_pose": torch.randn(
            batch, HISTORY_STEPS, POSE_FEATURE_DIM
        ),
        "history_action": torch.zeros(batch, HISTORY_STEPS, 4),
        "history_mask": torch.ones(batch, HISTORY_STEPS),
        "gab": torch.randn(batch, GAB_FEATURE_DIM),
        "map_feat": torch.randn(batch, MAP_FEATURE_DIM),
    }


def build(name, seed=20264017):
    torch.manual_seed(int(seed))
    return build_stage2j_model(name)


def state_equal(left, right):
    a, b = left.state_dict(), right.state_dict()
    if set(a) != set(b):
        return False
    return all(
        a[key].shape == b[key].shape and torch.equal(a[key], b[key])
        for key in a
    )


class ModelArchitectureTests(unittest.TestCase):
    def test_native_configs_are_preserved(self):
        for name in TABLE_ORDER:
            ours = STAGE2J_CONFIGS[name]
            native = GBS_MODEL_CONFIGS[name]
            for key in ("architecture", "hidden_dim", "num_layers",
                        "num_heads"):
                self.assertEqual(ours[key], native[key], f"{name}.{key}")
            self.assertEqual(
                ours.get("ff_mult", 4), native.get("ff_mult", 4),
                f"{name}.ff_mult",
            )

    def test_parameter_counts_and_scale_order(self):
        counts = {}
        for name in TABLE_ORDER:
            counts[name] = parameter_count(build(name))
            self.assertEqual(counts[name], EXPECTED_PARAMETERS[name], name)
        ordered = [counts[name] for name in TABLE_ORDER]
        self.assertEqual(ordered, sorted(ordered))
        self.assertEqual(len(set(ordered)), len(ordered))

    def test_small_keeps_late_fusion_mlp_identity(self):
        model = build("Small")
        self.assertIsInstance(model, Stage2JSmallMEC)
        self.assertEqual(model.architecture, "late_fusion_mlp")
        self.assertTrue(hasattr(model, "move_classifier"))
        self.assertFalse(hasattr(model, "goal_current_attn"))
        self.assertFalse(hasattr(model, "fusion_encoder"))
        self.assertFalse(hasattr(model, "heat_proj"))
        self.assertFalse(hasattr(model, "net_proj"))
        linear_layers = [
            module for module in model.move_classifier
            if isinstance(module, nn.Linear)
        ]
        self.assertEqual(len(linear_layers), 2)
        self.assertEqual(
            model.fusion_dim,
            2 * ENCODER.token_dim + 128 + GAB_FEATURE_DIM + MAP_FEATURE_DIM,
        )
        self.assertTrue(hasattr(model, "history_encoder"))
        self.assertEqual(model.arrival_head.context_dim, 0)

    def test_medium_keeps_goal_cross_attention_identity(self):
        model = build("Medium")
        self.assertIsInstance(model, Stage2JMediumMEC)
        self.assertEqual(model.architecture, "goal_cross_attention")
        self.assertTrue(hasattr(model, "goal_current_attn"))
        self.assertEqual(model.goal_current_attn.embed_dim, 384)
        self.assertEqual(model.goal_current_attn.num_heads, 6)
        self.assertFalse(hasattr(model, "fusion_encoder"))
        self.assertFalse(hasattr(model, "heat_proj"))
        self.assertFalse(hasattr(model, "net_proj"))
        linear_layers = [
            module for module in model.move_classifier
            if isinstance(module, nn.Linear)
        ]
        self.assertEqual(len(linear_layers), 2)
        self.assertEqual(
            model.fusion_dim,
            3 * ENCODER.token_dim + 128 + GAB_FEATURE_DIM + MAP_FEATURE_DIM,
        )
        self.assertTrue(hasattr(model, "history_encoder"))
        self.assertEqual(model.arrival_head.context_dim, 0)

    def test_transformer_variants_keep_explicit_identity(self):
        expected = {
            "Transformer-S": (192, 1, 4, 3),
            "Transformer-M": (256, 2, 8, 4),
            "Large": (352, 2, 8, 4),
        }
        for name, (hidden, layers, heads, ff_mult) in expected.items():
            model = build(name)
            self.assertIsInstance(model, Stage2ITransformer, name)
            self.assertEqual(model.variant, "T5", name)
            self.assertEqual(model.hidden_dim, hidden, name)
            self.assertEqual(len(model.fusion_encoder.layers), layers, name)
            layer = model.fusion_encoder.layers[0]
            self.assertEqual(layer.self_attn.embed_dim, hidden, name)
            self.assertEqual(layer.self_attn.num_heads, heads, name)
            self.assertEqual(layer.linear1.in_features, hidden, name)
            self.assertEqual(layer.linear1.out_features, hidden * ff_mult,
                             name)
            self.assertFalse(hasattr(model, "heat_proj"))
            self.assertFalse(hasattr(model, "net_proj"))
            self.assertFalse(hasattr(model.context_encoder, "heatmap_net"))
            self.assertFalse(hasattr(model.context_encoder, "network_net"))
            self.assertTrue(hasattr(model, "history_encoder"))
            self.assertTrue(hasattr(model, "gab_proj"))
            self.assertTrue(hasattr(model, "map_proj"))
            self.assertEqual(model.arrival_head.context_dim, 0)

    def test_transformer_m_is_exactly_stage2i_t5(self):
        stage2j = build("Transformer-M")
        stage2i = Stage2ITransformer("T5")
        self.assertTrue(state_equal(stage2j, stage2i))
        inputs = random_inputs()
        stage2j.eval()
        stage2i.eval()
        with torch.no_grad():
            left = stage2j.forward_components(**inputs)
            right = stage2i.forward_components(**inputs)
        for key in ("move_logits", "arrival_logit", "arrival_features"):
            self.assertTrue(
                torch.equal(left[key], right[key]), key
            )

    def test_transformer_m_loads_stage2i_checkpoint_strictly(self):
        path = Path(
            "runs/v83_spatial_stage2/stage2i/T5_map_assisted/best.pt"
        )
        if not path.is_file():
            self.skipTest("Stage-2i T5 checkpoint is not present")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build("Transformer-M")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.assertEqual(parameter_count(model), 3033732)

    def test_initialisation_is_deterministic(self):
        for name in TABLE_ORDER:
            left = build(name)
            right = build(name)
            self.assertTrue(state_equal(left, right), name)


class InputContractTests(unittest.TestCase):
    def test_all_models_accept_the_same_inputs(self):
        inputs = random_inputs()
        for name in TABLE_ORDER:
            model = build(name).eval()
            with torch.no_grad():
                components = model.forward_components(**inputs)
                logits = model(**inputs)
            self.assertEqual(components["move_logits"].shape, (2, 3), name)
            self.assertEqual(components["arrival_logit"].shape, (2,), name)
            self.assertEqual(components["arrival_features"].shape, (2, 8),
                             name)
            self.assertEqual(logits.shape, (2, 4), name)
            self.assertTrue(torch.isfinite(logits).all(), name)
            # combine_move_and_arrival returns normalized log-probabilities.
            self.assertTrue(torch.allclose(
                torch.logsumexp(logits, dim=-1),
                torch.zeros(2), atol=1e-5,
            ), name)

    def test_missing_inputs_are_rejected(self):
        for name in TABLE_ORDER:
            model = build(name).eval()
            for field in (
                "history_vis", "history_pose", "history_action",
                "history_mask", "gab", "map_feat",
            ):
                inputs = random_inputs()
                inputs[field] = None
                with self.assertRaises(ValueError, msg=f"{name}/{field}"):
                    model.forward_components(**inputs)

    def test_removed_modalities_are_rejected(self):
        inputs = random_inputs()
        for name in TABLE_ORDER:
            model = build(name).eval()
            with self.assertRaises(ValueError, msg=name):
                model.forward_components(
                    heatmap=torch.randn(2, 3, 8, 8), **inputs
                )
            with self.assertRaises(ValueError, msg=name):
                model.forward_components(
                    network_state=torch.randn(2, 5), **inputs
                )

    def test_shape_validation(self):
        for name in TABLE_ORDER:
            model = build(name).eval()
            inputs = random_inputs()
            inputs["history_vis"] = torch.randn(
                2, HISTORY_STEPS - 1, VISUAL_DESCRIPTOR_DIM
            )
            with self.assertRaises(ValueError, msg=name):
                model.forward_components(**inputs)
            inputs = random_inputs()
            inputs["map_feat"] = torch.randn(2, MAP_FEATURE_DIM + 1)
            with self.assertRaises(ValueError, msg=name):
                model.forward_components(**inputs)
            inputs = random_inputs()
            inputs["gab"] = torch.randn(3, GAB_FEATURE_DIM)
            with self.assertRaises(ValueError, msg=name):
                model.forward_components(**inputs)


class GradientTests(unittest.TestCase):
    def test_each_model_has_finite_loss_and_nonzero_key_gradients(self):
        inputs = random_inputs()
        for name in TABLE_ORDER:
            model = build(name)
            model.train()
            components = model.forward_components(**inputs)
            loss = (
                components["move_logits"].sum()
                + components["arrival_logit"].sum()
            )
            self.assertTrue(torch.isfinite(loss), name)
            loss.backward()
            self.assertIsNotNone(
                model.history_encoder.visual_proj.weight.grad, name
            )
            self.assertGreater(
                float(model.history_encoder.visual_proj.weight.grad.abs().sum()),
                0.0, name,
            )
            self.assertIsNotNone(
                model.arrival_head.net[0].weight.grad, name
            )
            self.assertGreater(
                float(model.arrival_head.net[0].weight.grad.abs().sum()),
                0.0, name,
            )
            if name in ("Small", "Medium"):
                grad = model.move_classifier[0].weight.grad
            else:
                grad = model.fusion_encoder.layers[0].linear1.weight.grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)


class TrainingProtocolParityTests(unittest.TestCase):
    def test_constants_match_stage2i(self):
        pairs = (
            ("SEED", 20264017),
            ("MAX_EPOCHS", 50),
            ("PATIENCE", 8),
            ("BATCH_SIZE", 48),
            ("VAL_BATCH_SIZE", 96),
            ("INITIAL_LR", 6e-5),
            ("WEIGHT_DECAY", 1e-4),
            ("LR_PATIENCE", 3),
            ("LR_FACTOR", 0.5),
            ("MIN_LR", 1e-6),
        )
        for name, value in pairs:
            self.assertEqual(getattr(stage2j_train, name), value, name)
            self.assertEqual(
                getattr(stage2i_train, name), value, f"stage2i.{name}"
            )
        self.assertEqual(
            stage2j_train.AUX_FIELDS, stage2i_train.AUX_FIELDS["T5"]
        )
        self.assertEqual(TRAIN_PROTOCOL["seed"], 20264017)
        self.assertEqual(TRAIN_PROTOCOL["max_epochs"], 50)

    def test_forward_batch_passes_all_six_aux_fields(self):
        extras = {
            field: torch.zeros(2, 1)
            for field in stage2j_train.AUX_FIELDS
        }
        captured = {}

        class Dummy(nn.Module):
            def forward_components(self, current, goal, **kwargs):
                captured.update(kwargs)
                return {}

        batch = {
            "tokens": torch.zeros(2, 1, 1),
            "goal": torch.zeros(2, 1, 1),
            "extras": extras,
        }
        stage2j_train.forward_batch(Dummy(), batch)
        for field in stage2j_train.AUX_FIELDS:
            self.assertIs(captured[field], extras[field], field)
        self.assertIsNone(captured["heatmap"])
        self.assertIsNone(captured["network_state"])

    def test_stop_threshold_is_fixed(self):
        self.assertEqual(STOP_THRESHOLD, 0.7)
        source = Path(
            "v83_spatial_stage2/stage2j_eval.py"
        ).read_text(encoding="utf-8")
        self.assertIn("STOP_THRESHOLD", source)
        self.assertIn("arrival_probability >= STOP_THRESHOLD", source)


def _sha(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _write_synthetic_test(root, episode_count=6, shard_sizes=(2, 1, 3)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    band_names = {0: "short", 1: "medium", 2: "long"}
    max_steps = 3
    rows = []
    for episode_id in range(episode_count):
        band_id = episode_id % 3
        for step in range(max_steps):
            rows.append({
                "episode_id": episode_id,
                "episode_seed": 3_000_000 + episode_id,
                "network_layout_seed": 51_820_000 + episode_id,
                "band_id": band_id,
                "step": step,
            })
    chunks = []
    cursor = 0
    for size in shard_sizes:
        chunks.append(rows[cursor:cursor + size])
        cursor += size
    if cursor < len(rows):
        chunks.append(rows[cursor:])
    for index, chunk in enumerate(chunks):
        torch.save({
            "episode_ids": torch.tensor(
                [row["episode_id"] for row in chunk], dtype=torch.long
            ),
            "episode_seeds": torch.tensor(
                [row["episode_seed"] for row in chunk], dtype=torch.long
            ),
            "network_layout_seeds": torch.tensor(
                [row["network_layout_seed"] for row in chunk],
                dtype=torch.long,
            ),
            "trajectory_band_ids": torch.tensor(
                [row["band_id"] for row in chunk], dtype=torch.long
            ),
            "step_ids": torch.tensor(
                [row["step"] for row in chunk], dtype=torch.long
            ),
        }, root / f"shard_{index:04d}.pt")
    stats = {
        "trajectories": [
            {
                "seed": 3_000_000 + episode_id,
                "band": band_names[episode_id % 3],
                "network_layout_seed": 51_820_000 + episode_id,
                "steps": max_steps,
                "start_goal_distance_m": (
                    200.0 if episode_id % 3 == 0
                    else 300.0 if episode_id % 3 == 1 else 450.0
                ),
            }
            for episode_id in range(episode_count)
        ]
    }
    (root / "dataset_stats.json").write_text(
        json.dumps(stats), encoding="utf-8"
    )
    files = {
        path.name: _sha(path)
        for path in sorted(root.glob("*"))
        if path.name != "SEALED.json"
    }
    (root / "SEALED.json").write_text(
        json.dumps({"split": "final_test", "files": files}),
        encoding="utf-8",
    )
    return root


class FinalTestProtocolTests(unittest.TestCase):
    def test_sealed_verification_and_task_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_synthetic_test(tmp)
            seal = verify_sealed(root)
            self.assertEqual(seal["shards"], 4)
            result = extract_final_test_specs(
                root,
                seal=seal,
                expected_episodes=6,
                expected_band_counts={"short": 2, "medium": 2, "long": 2},
            )
            specs = result["specs"]
            self.assertEqual(len(specs), 6)
            self.assertEqual(
                [spec["episode_index"] for spec in specs], list(range(6))
            )
            for spec in specs:
                self.assertEqual(
                    spec["seed"], 3_000_000 + spec["episode_index"]
                )
                self.assertEqual(
                    spec["network_layout_seed"],
                    51_820_000 + spec["episode_index"],
                )
                self.assertEqual(
                    spec["band"], BANDS[spec["episode_index"] % 3]
                )

    def test_sealed_hash_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_synthetic_test(tmp)
            shard = root / "shard_0000.pt"
            shard.write_bytes(shard.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                verify_sealed(root)

    def test_inconsistent_episode_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_synthetic_test(tmp)
            data = torch.load(root / "shard_0000.pt", weights_only=True)
            data["episode_seeds"] = data["episode_seeds"] + 1
            torch.save(data, root / "shard_0000.pt")
            with self.assertRaises(ValueError):
                extract_final_test_specs(
                    root,
                    seal={"manifest_sha256": "not-checked"},
                    expected_episodes=6,
                    expected_band_counts={"short": 2, "medium": 2,
                                          "long": 2},
                )

    def test_missing_episode_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_synthetic_test(tmp, episode_count=5)
            with self.assertRaises(ValueError):
                extract_final_test_specs(
                    root,
                    seal={"manifest_sha256": "not-checked"},
                    expected_episodes=6,
                    expected_band_counts={"short": 2, "medium": 2,
                                          "long": 2},
                )

    def test_freeze_rejects_code_or_threshold_drift(self):
        model = build("Small")
        with tempfile.TemporaryDirectory() as tmp:
            config_sha = _sha(Path("config.py"))
            freeze = {
                "status": "frozen",
                "critical_files": {"config.py": config_sha},
                "models": {
                    "Small": {"parameters": parameter_count(model)}
                },
                "stop_threshold": 0.7,
            }
            path = Path(tmp) / "freeze.json"
            path.write_text(json.dumps(freeze), encoding="utf-8")
            verify_protocol_freeze(path, "Small", model)
            freeze["critical_files"]["config.py"] = "0" * 64
            path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_protocol_freeze(path, "Small", model)
            freeze["critical_files"]["config.py"] = config_sha
            freeze["stop_threshold"] = 0.5
            path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_protocol_freeze(path, "Small", model)

    def test_stage2i_dev_guard_still_rejects_final_test(self):
        from v83_spatial_stage2.stage2i_common import reject_final_test
        with self.assertRaises(ValueError):
            reject_final_test(
                "runs/v83_spatial_stage2/datasets/final_test_300"
            )


class ResultFormattingTests(unittest.TestCase):
    def test_progress_starts_with_five_pending_models(self):
        progress = load_progress(Path("/nonexistent/progress.json"))
        self.assertEqual(set(progress["models"]), set(TABLE_ORDER))
        table = cumulative_table(progress)
        self.assertIn("| MEC | Params | SR | OSR | SPL | nDTW |", table)
        self.assertEqual(table.count("Pending"), len(TABLE_ORDER))

    def test_result_block_contains_required_fields(self):
        entry = {
            "metrics": {
                "episodes": 300, "sr": 0.5, "osr": 0.8, "spl": 0.4,
                "ndtw": 0.45, "premature_stop": 0.2, "timeout": 0.1,
                "steps": 120.0,
            },
            "bands": {
                band: {"sr": 0.5, "osr": 0.8, "spl": 0.4, "ndtw": 0.45}
                for band in BANDS
            },
            "retrieval": {"clean_top1": 1.0, "clean_top5": 1.0},
            "latency": {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 2.0},
            "architecture": "late_fusion_mlp",
            "parameters": 611716,
            "best_epoch": 10,
            "training_time_min": 20.0,
        }
        block = format_result_block(
            "Small", entry, {name: False for name in TABLE_ORDER}
        )
        for text in (
            "STAGE-2J MODEL RESULT: Small", "SR:", "OSR:", "SPL:", "nDTW:",
            "Premature-Stop:", "Timeout:", "Band Results:", "Map retrieval:",
            "Top-1", "Top-5", "Inference latency", "Completed models:",
        ):
            self.assertIn(text, block)

    def test_latency_summary(self):
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["mean_ms"], 2.5)
        self.assertAlmostEqual(summary["p50_ms"], 2.5)
        self.assertGreater(summary["p95_ms"], 3.0)


class AnalysisTests(unittest.TestCase):
    def _synthetic_results(self, n=20):
        rng = np.random.default_rng(7)
        results = {}
        for index, name in enumerate(TABLE_ORDER):
            base = 0.20 + 0.05 * index
            rows = []
            for episode in range(n):
                success = int(rng.random() < base)
                rows.append({
                    "success": float(success),
                    "osr": float(int(rng.random() < base + 0.1)),
                    "spl": float(success * rng.random()),
                    "ndtw": float(rng.random() * 0.8),
                })
            results[name] = {
                "rows": rows,
                "metrics": {
                    "sr": float(np.mean([row["success"] for row in rows])),
                    "osr": float(np.mean([row["osr"] for row in rows])),
                    "spl": float(np.mean([row["spl"] for row in rows])),
                    "ndtw": float(np.mean([row["ndtw"] for row in rows])),
                },
            }
        return results

    def test_bootstrap_cis_bracket_the_mean(self):
        results = self._synthetic_results()
        bootstrap = bootstrap_results(results, reps=300, seed=11)
        for metric in METRICS:
            for name in TABLE_ORDER:
                block = bootstrap["metrics"][metric][name]
                self.assertLessEqual(block["ci95_low"], block["mean"])
                self.assertGreaterEqual(block["ci95_high"], block["mean"])
        self.assertEqual(bootstrap["episodes"], 20)
        self.assertTrue(bootstrap["shared_resamples"])
        self.assertIn("vs_Transformer-M", bootstrap["paired_deltas"])

    def test_figures_render_to_png_and_svg(self):
        results = self._synthetic_results()
        bootstrap = bootstrap_results(results, reps=100, seed=12)
        with tempfile.TemporaryDirectory() as tmp:
            for metric in METRICS:
                png, svg = plot_metric(
                    results, bootstrap, metric, output_dir=tmp
                )
                self.assertTrue(Path(png).is_file())
                self.assertTrue(Path(svg).is_file())
                self.assertGreater(Path(png).stat().st_size, 10000)
                self.assertGreater(Path(svg).stat().st_size, 5000)


class DisciplineSourceTests(unittest.TestCase):
    def test_training_rejects_final_test_paths(self):
        source = Path(
            "v83_spatial_stage2/stage2j_train.py"
        ).read_text(encoding="utf-8")
        self.assertIn("reject_final_test(args.train_dir)", source)
        self.assertIn("reject_final_test(args.val_dir)", source)

    def test_formal_test_requires_explicit_authorisation(self):
        source = Path(
            "v83_spatial_stage2/stage2j_eval.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--allow-final-test", source)
        self.assertIn("verify_protocol_freeze", source)
        self.assertIn("verify_sealed", source)
        self.assertIn('if final_test and not args.allow_final_test', source)
        self.assertIn("if final_test and args.smoke", source)

    def test_model_module_reads_no_simulator_state(self):
        source = Path(
            "v83_spatial_stage2/stage2j_model.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertNotIn("env", imported)
        forbidden_attributes = {
            "distance_to_goal", "goal_position", "oracle_action",
            "success", "is_goal", "goal",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(
                    node.attr, forbidden_attributes,
                    f"forbidden attribute access .{node.attr} at line "
                    f"{getattr(node, 'lineno', '?')}",
                )


if __name__ == "__main__":
    unittest.main()
