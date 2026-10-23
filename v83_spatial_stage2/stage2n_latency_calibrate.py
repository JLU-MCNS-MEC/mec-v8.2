"""Stage-2n inference-latency calibration for Small and Transformer-M.

This script measures the two frozen Stage-2j MEC checkpoints on a real
Stage-2j training input sample.  It does not train, does not read the sealed
Final Test set, and does not modify the simulation latency model.

Measurement protocol
--------------------
* batch size 1, FP32 model weights, ``model.eval()``,
  ``torch.inference_mode()``;
* 200 warm-up forwards before every round;
* 3 independent rounds; each round performs at least 1000 forwards for:
    - pure GPU forward (per-call CUDA events, 1000 samples),
    - pure GPU forward (grouped CUDA events, 1000 calls in groups of 10),
    - request-to-result host-visible latency (1000 calls, including result
      copy/argmax/sigmoid and CUDA synchronisation; no network),
    - module diagnostics (history encoder, move head, arrival head, and the
      Transformer-M fusion encoder);
* analytical inference from ``stage2k_latency.stage2j_model_flops`` at the
  current 2 TFLOPS assumption (read-only reuse).

Outputs go to
``v83_spatial_stage2/stage2n_evidence/latency_calibration/``.
Raw checkpoints stay under the ignored ``runs/`` directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2j_eval import load_checkpoint
from v83_spatial_stage2.stage2j_model import (
    build_stage2j_model,
    model_metadata,
    parameter_count,
)
from v83_spatial_stage2.stage2k_latency import (
    STAGE2J_INPUT_BYTES,
    STAGE2J_PAYLOAD_BYTES,
    Stage2KLatency,
    stage2j_model_flops,
)

MODELS = ("Small", "Transformer-M")
CHECKPOINTS = {
    "Small": Path("runs/v83_spatial_stage2/stage2j/Small/best.pt"),
    "Transformer-M": Path(
        "runs/v83_spatial_stage2/stage2j/Transformer-M/best.pt"
    ),
}
CONFIGS = {
    "Small": Path("runs/v83_spatial_stage2/stage2j/Small/config.json"),
    "Transformer-M": Path(
        "runs/v83_spatial_stage2/stage2j/Transformer-M/config.json"
    ),
}
EXPECTED_SHA256 = {
    "Small": "7349ec9d0c809a09dc2672a05053f1f89e69480b8802d6ca3b05944cea44e1f6",
    "Transformer-M": "57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d",
}
EXPECTED_PARAMS = {"Small": 611_716, "Transformer-M": 3_033_732}
TOKEN_SAMPLE = Path(
    "runs/v83_spatial_stage2/datasets/train_600/shard_0000.pt"
)
AUX_SAMPLE = Path(
    "runs/v83_spatial_stage2/stage2i/aux/train/aux_shard_0000.npz"
)
OUTPUT_DIR = Path(
    "v83_spatial_stage2/stage2n_evidence/latency_calibration"
)
HISTORICAL_GREEDY_E2E_MS = 49.11
ANALYTICAL_TFLOPS = 2.0
REFERENCE_DISTANCE_M = 150.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nvidia_smi(query: str):
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception as exc:  # pragma: no cover - hardware dependent
        return [f"unavailable: {exc}"]


def hardware_snapshot() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception:
        apps = []
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_query": _nvidia_smi(
            "name,driver_version,memory.total,clocks.sm,clocks.max.sm,"
            "power.draw,power.limit,temperature.gpu,utilization.gpu"
        ),
        "compute_apps": apps,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_gb": props.total_memory / 1e9,
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "allow_tf32_matmul": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def load_real_inputs(sample_index: int, device: torch.device):
    for path in (TOKEN_SAMPLE, AUX_SAMPLE):
        if "final_test" in str(path).lower():
            raise RuntimeError(f"refusing sealed Test input: {path}")
    tokens = torch.load(
        TOKEN_SAMPLE, map_location="cpu", weights_only=True
    )
    aux = np.load(AUX_SAMPLE, allow_pickle=False)
    index = int(sample_index)
    if not 0 <= index < int(tokens["tokens"].shape[0]):
        raise IndexError(sample_index)
    if not 0 <= index < int(aux["history_vis"].shape[0]):
        raise IndexError(sample_index)
    inputs = {
        "current_tokens": tokens["tokens"][index:index + 1],
        "goal_tokens": tokens["goal_tokens"][index:index + 1],
        "history_vis": torch.from_numpy(
            aux["history_vis"][index:index + 1]
        ),
        "history_pose": torch.from_numpy(
            aux["history_pose"][index:index + 1]
        ),
        "history_action": torch.from_numpy(
            aux["history_action"][index:index + 1]
        ),
        "history_mask": torch.from_numpy(
            aux["history_mask"][index:index + 1]
        ),
        "gab": torch.from_numpy(aux["gab"][index:index + 1]),
        "map_feat": torch.from_numpy(aux["map_feat"][index:index + 1]),
    }
    source_dtypes = {
        key: str(value.dtype) for key, value in inputs.items()
    }
    inputs = {
        key: value.float().contiguous().to(device)
        for key, value in inputs.items()
    }
    expected = {
        "current_tokens": (1, 576, 384),
        "goal_tokens": (1, 3, 384),
        "history_vis": (1, 16, 768),
        "history_pose": (1, 16, 10),
        "history_action": (1, 16, 4),
        "history_mask": (1, 16),
        "gab": (1, 20),
        "map_feat": (1, 22),
    }
    shapes = {key: tuple(value.shape) for key, value in inputs.items()}
    if shapes != expected:
        raise ValueError(f"input shapes {shapes} != {expected}")
    provenance = {
        "token_sample": str(TOKEN_SAMPLE),
        "token_sample_sha256": sha256_file(TOKEN_SAMPLE),
        "aux_sample": str(AUX_SAMPLE),
        "aux_sample_sha256": sha256_file(AUX_SAMPLE),
        "sample_index": index,
        "source_dtypes": source_dtypes,
        "deployed_dtype": "float32",
        "shapes": {key: list(value) for key, value in shapes.items()},
        "final_test_read": False,
    }
    return inputs, provenance


def verify_checkpoint(name: str) -> dict:
    path = CHECKPOINTS[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != EXPECTED_SHA256[name]:
        raise ValueError(
            f"{name} checkpoint SHA256 {actual} != "
            f"{EXPECTED_SHA256[name]}"
        )
    config = read_json(CONFIGS[name])
    if config.get("final_test_read") is not False:
        raise ValueError(f"{name} config claims a Final Test read")
    meta = model_metadata(name)
    if int(meta["parameters"]) != EXPECTED_PARAMS[name]:
        raise ValueError(
            f"{name} parameter count {meta['parameters']} != "
            f"{EXPECTED_PARAMS[name]}"
        )
    if int(config.get("parameters", -1)) != EXPECTED_PARAMS[name]:
        raise ValueError(f"{name} config parameter count mismatch")
    return {
        "model": name,
        "checkpoint": str(path),
        "checkpoint_resolved": str(path.resolve()),
        "checkpoint_sha256": actual,
        "config": str(CONFIGS[name]),
        "config_sha256": sha256_file(CONFIGS[name]),
        "metadata": meta,
        "parameters": int(meta["parameters"]),
        "frozen": True,
        "retrained": False,
        "final_test_read": False,
    }


def load_model(name: str, device: torch.device):
    record = verify_checkpoint(name)
    model = build_stage2j_model(name).to(device)
    load_checkpoint(model, CHECKPOINTS[name], device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    actual_params = parameter_count(model)
    if actual_params != EXPECTED_PARAMS[name]:
        raise RuntimeError(
            f"{name} loaded parameters {actual_params} != "
            f"{EXPECTED_PARAMS[name]}"
        )
    dtypes = {str(p.dtype) for p in model.parameters()}
    record["loaded_parameters"] = actual_params
    record["parameter_dtypes"] = sorted(dtypes)
    return model, record


def _stats(values: Sequence[float]) -> dict:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        raise ValueError("empty measurement")
    if not np.all(np.isfinite(array)):
        raise ValueError("non-finite measurement")
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _cv_round_means(round_means: Sequence[float]) -> float:
    values = np.asarray(round_means, dtype=np.float64)
    if values.size < 2 or values.mean() == 0.0:
        return 0.0
    return float(values.std(ddof=1) / values.mean() * 100.0)


def measure_forward_per_call(
    model,
    inputs: Mapping[str, torch.Tensor],
    iterations: int,
) -> list[float]:
    starts = [
        torch.cuda.Event(enable_timing=True) for _ in range(iterations)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True) for _ in range(iterations)
    ]
    with torch.inference_mode():
        for index in range(iterations):
            starts[index].record()
            model.forward_components(**inputs)
            ends[index].record()
    torch.cuda.synchronize()
    return [
        float(starts[index].elapsed_time(ends[index]))
        for index in range(iterations)
    ]


def _measure_callable_groups(
    callable_fn: Callable[[], object],
    groups: int,
    calls_per_group: int,
) -> list[float]:
    values = []
    with torch.inference_mode():
        for _ in range(groups):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(calls_per_group):
                callable_fn()
            end.record()
            torch.cuda.synchronize()
            values.append(
                float(start.elapsed_time(end)) / float(calls_per_group)
            )
    return values


def measure_forward_grouped(
    model,
    inputs: Mapping[str, torch.Tensor],
    groups: int = 100,
    calls_per_group: int = 10,
) -> list[float]:
    return _measure_callable_groups(
        lambda: model.forward_components(**inputs),
        groups,
        calls_per_group,
    )


def measure_request_to_result(
    model,
    inputs: Mapping[str, torch.Tensor],
    iterations: int,
) -> list[float]:
    values = []
    with torch.inference_mode():
        for _ in range(iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            components = model.forward_components(**inputs)
            arrival_probability = float(
                torch.sigmoid(components["arrival_logit"]).item()
            )
            move_action = int(
                torch.argmax(components["move_logits"], -1).item()
            )
            torch.cuda.synchronize()
            values.append((time.perf_counter() - started) * 1000.0)
            if not math.isfinite(arrival_probability):
                raise RuntimeError("non-finite arrival probability")
            if move_action not in (0, 1, 2):
                raise RuntimeError(f"invalid move action {move_action}")
    return values


def measure_queued_throughput(
    model,
    inputs: Mapping[str, torch.Tensor],
    iterations: int = 1000,
) -> dict:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            model.forward_components(**inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "iterations": int(iterations),
        "elapsed_s": float(elapsed),
        "throughput_per_s": float(iterations / elapsed),
    }


def prepare_module_probes(name: str, model, inputs):
    """Build callables for the module diagnostics.

    These call the actual model submodules.  They are not summed into the
    full forward: projections, token construction, residual glue and the
    goal-current attention block are outside the measured pieces.
    """
    with torch.inference_mode():
        history = model.history_encoder(
            inputs["history_vis"],
            inputs["history_pose"],
            inputs["history_action"],
            inputs["history_mask"],
        )
        if name == "Small":
            fusion = torch.cat(
                [
                    inputs["current_tokens"].mean(dim=1),
                    inputs["goal_tokens"].mean(dim=1),
                    history,
                    inputs["gab"],
                    inputs["map_feat"],
                ],
                dim=-1,
            )
            probes = {
                "module_history_encoder": (
                    lambda: model.history_encoder(
                        inputs["history_vis"],
                        inputs["history_pose"],
                        inputs["history_action"],
                        inputs["history_mask"],
                    ),
                    10,
                ),
                "module_move_head": (
                    lambda: model.move_classifier(fusion), 100
                ),
                "module_arrival_head": (
                    lambda: model.arrival_head(
                        inputs["current_tokens"],
                        inputs["goal_tokens"],
                        None,
                    ),
                    10,
                ),
            }
        else:
            current = (
                model.visual_proj(inputs["current_tokens"])
                + model.current_pos
            )
            goal = (
                model.visual_proj(inputs["goal_tokens"])
                + model.goal_pos
            )
            kv_current = torch.cat(
                [
                    current,
                    torch.zeros(
                        current.size(0), 1, current.size(-1),
                        device=current.device,
                    ),
                ],
                dim=1,
            )
            matched, _ = model.goal_current_attn(
                query=goal, key=kv_current, value=kv_current,
                need_weights=False,
            )
            matched = model.goal_current_norm(goal + matched)
            matched = model.goal_current_ffn_norm(
                matched + model.goal_current_ffn(matched)
            )
            tokens = [
                model.cls_token.expand(1, -1, -1),
                current + model.current_type,
                matched + model.goal_type,
                model.hist_proj(history).unsqueeze(1)
                + model.hist_type,
                model.gab_proj(inputs["gab"]).unsqueeze(1)
                + model.gab_type,
                model.map_proj(inputs["map_feat"]).unsqueeze(1)
                + model.map_type,
            ]
            sequence = torch.cat(tokens, dim=1)
            encoded = model.fusion_encoder(sequence)
            final = model.final_norm(encoded[:, 0])
            probes = {
                "module_history_encoder": (
                    lambda: model.history_encoder(
                        inputs["history_vis"],
                        inputs["history_pose"],
                        inputs["history_action"],
                        inputs["history_mask"],
                    ),
                    10,
                ),
                "module_move_head": (
                    lambda: model.move_head(final), 100
                ),
                "module_arrival_head": (
                    lambda: model.arrival_head(
                        inputs["current_tokens"],
                        inputs["goal_tokens"],
                        None,
                    ),
                    10,
                ),
                "module_fusion_encoder": (
                    lambda: model.fusion_encoder(sequence), 10
                ),
            }
    return probes


def analytical_record(name: str) -> dict:
    flops = float(stage2j_model_flops(name))
    analytical_inference_ms = 1000.0 * flops / (ANALYTICAL_TFLOPS * 1e12)
    latency = Stage2KLatency()
    per_distance = {}
    for distance in (50.0, 100.0, 150.0, 200.0, 500.0):
        node = {
            "x": 0.0,
            "y": 0.0,
            "model_type": name,
            "bandwidth_hz": 30.0e6,
            "compute_capacity_flops_per_s": ANALYTICAL_TFLOPS * 1e12,
            "coverage_radius_m": 500.0,
        }
        metric = latency.evaluate(node, [distance, 0.0], name)
        per_distance[f"{distance:.0f}"] = {
            "communication_ms": float(metric["communication_ms"]),
            "inference_ms": float(metric["inference_ms"]),
            "total_ms": float(metric["total_ms"]),
            "uplink_rate_mbps": float(metric["uplink_rate_mbps"]),
            "snr_db": float(metric["snr_db"]),
        }
    reference = per_distance[f"{REFERENCE_DISTANCE_M:.0f}"]
    return {
        "model": name,
        "flops": flops,
        "gflops": flops / 1e9,
        "analytical_tflops": ANALYTICAL_TFLOPS,
        "analytical_inference_ms": float(analytical_inference_ms),
        "extraction_ms": float(latency.extraction_ms),
        "payload_bytes": int(STAGE2J_PAYLOAD_BYTES),
        "payload_breakdown": dict(STAGE2J_INPUT_BYTES),
        "reference_distance_m": REFERENCE_DISTANCE_M,
        "communication_ms_at_reference": float(
            reference["communication_ms"]
        ),
        "analytical_total_ms_at_reference": float(reference["total_ms"]),
        "per_distance": per_distance,
        "latency_model": {
            "module": "v83_spatial_stage2.stage2k_latency.Stage2KLatency",
            "underlying_channel": "latency_model.LatencyModel",
            "modified": False,
        },
    }


def make_figures(results: Mapping, output_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface = "#fcfcfb"
    ink = "#0b0b0b"
    secondary = "#52514e"
    grid = "#e1e0d9"
    blue = "#2a78d6"
    orange = "#eb6834"
    violet = "#4a3aa7"
    neutral = "#b9b7ae"
    plt.rcParams.update({
        "figure.facecolor": surface,
        "axes.facecolor": surface,
        "savefig.facecolor": surface,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": ink,
        "axes.edgecolor": grid,
        "xtick.color": secondary,
        "ytick.color": secondary,
        "axes.grid": True,
        "grid.color": grid,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    })
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = list(results["models"])
    comparison_rows = []
    for model in models:
        summary = results["models"][model]["summary"]
        comparison_rows.append({
            "model": model,
            "analytical_inference_ms": results["analytical"][model][
                "analytical_inference_ms"
            ],
            "measured_forward_ms": summary["forward_cuda_grouped"][
                "mean_ms"
            ],
            "measured_request_ms": summary["request_to_result"]["mean_ms"],
            "measured_forward_p95_ms": summary["forward_cuda_percall"][
                "p95_ms"
            ],
            "measured_request_p95_ms": summary["request_to_result"][
                "p95_ms"
            ],
        })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    positions = np.arange(len(models))
    width = 0.36
    forward = [
        results["models"][model]["summary"]["forward_cuda_grouped"][
            "mean_ms"
        ] for model in models
    ]
    request = [
        results["models"][model]["summary"]["request_to_result"]["mean_ms"]
        for model in models
    ]
    forward_std = [
        float(np.std(
            results["models"][model]["summary"]["forward_cuda_grouped"][
                "round_means_ms"
            ], ddof=1,
        )) for model in models
    ]
    request_std = [
        float(np.std(
            results["models"][model]["summary"]["request_to_result"][
                "round_means_ms"
            ], ddof=1,
        )) for model in models
    ]
    axes[0].bar(
        positions - width / 2, forward, width, color=blue,
        yerr=forward_std, capsize=3, label="GPU forward (grouped)",
    )
    axes[0].bar(
        positions + width / 2, request, width, color=orange,
        yerr=request_std, capsize=3, label="Request→result",
    )
    for index, (fwd, req) in enumerate(zip(forward, request)):
        axes[0].text(
            index - width / 2,
            fwd + forward_std[index] + 0.02, f"{fwd:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
        axes[0].text(
            index + width / 2,
            req + request_std[index] + 0.02, f"{req:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Measured latency (mean ± std across rounds)")
    axes[0].legend(loc="upper left", fontsize=8)
    analytical = [
        results["analytical"][model]["analytical_inference_ms"]
        for model in models
    ]
    measured = forward
    axes[1].bar(
        positions - width / 2, analytical, width, color=violet,
        label="Analytical @2 TFLOPS",
    )
    axes[1].bar(
        positions + width / 2, measured, width, color=blue,
        label="Measured GPU forward",
    )
    axes[1].set_yscale("log")
    for index, (ana, mea) in enumerate(zip(analytical, measured)):
        axes[1].text(
            index - width / 2, ana * 1.15, f"{ana:.4f}",
            ha="center", va="bottom", fontsize=8,
        )
        axes[1].text(
            index + width / 2, mea * 1.15, f"{mea:.3f}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel("Latency (ms, log scale)")
    axes[1].set_title("Analytical vs measured inference")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    with (output_dir / "figure_latency_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(comparison_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    module_rows = []
    module_names = []
    for model in models:
        for name in results["models"][model]["module_summary"]:
            if name not in module_names:
                module_names.append(name)
    for model in models:
        for name in module_names:
            item = results["models"][model]["module_summary"].get(name)
            if item is not None:
                module_rows.append({
                    "model": model,
                    "module": name,
                    "mean_ms": item["mean_ms"],
                    "p95_ms": item["p95_ms"],
                })
    fig, axis = plt.subplots(figsize=(9.2, 4.3))
    positions = np.arange(len(module_names))
    width = 0.36
    for offset, (model, color) in enumerate(
        zip(models, (blue, orange))
    ):
        values = []
        errors = []
        for name in module_names:
            item = results["models"][model]["module_summary"].get(name)
            values.append(item["mean_ms"] if item else 0.0)
            errors.append(item["std_ms"] if item else 0.0)
        axis.bar(
            positions + (offset - 0.5) * width, values, width,
            color=color, yerr=errors, capsize=2, label=model,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [name.replace("module_", "") for name in module_names],
        rotation=20, ha="right",
    )
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Module diagnostics (not additive; see report)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_module_breakdown.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    with (output_dir / "figure_module_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model", "module", "mean_ms", "p95_ms"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(module_rows)
    return comparison_rows, module_rows


def _write_csv(path: Path, rows: Sequence[Mapping]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_measurement(args):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this calibration")
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, input_provenance = load_real_inputs(args.sample_index, device)
    hardware_before = hardware_snapshot()
    results = {
        "stage": "stage2n_inference_latency_calibration",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": (
            "Measure real Stage-2j Small and Transformer-M inference latency; "
            "compare with the unchanged 2 TFLOPS analytical model."
        ),
        "hardware_before": hardware_before,
        "input": input_provenance,
        "measurement": {
            "batch_size": 1,
            "precision": "float32",
            "torch_inference_mode": True,
            "model_eval": True,
            "warmup_per_round": int(args.warmup),
            "rounds": int(args.rounds),
            "forward_iterations_per_round": int(args.iterations),
            "request_iterations_per_round": int(args.iterations),
            "grouped_forward_groups": int(args.groups),
            "grouped_calls_per_group": int(args.calls_per_group),
            "module_groups": int(args.groups),
            "module_calls_per_group": 10,
            "module_large_group_calls": 100,
            "historical_greedy_e2e_ms": HISTORICAL_GREEDY_E2E_MS,
            "network_measured": False,
            "test_read": False,
            "training": False,
        },
        "models": {},
        "analytical": {},
    }
    raw_rows = []
    round_rows = []
    for model_name in args.models:
        if model_name not in MODELS:
            raise ValueError(model_name)
        print(f"[calibrate] loading {model_name}", flush=True)
        model, checkpoint_record = load_model(model_name, device)
        results["models"][model_name] = {
            "checkpoint": checkpoint_record,
            "rounds": {},
            "summary": {},
            "module_summary": {},
        }
        results["analytical"][model_name] = analytical_record(model_name)
        probes = prepare_module_probes(model_name, model, inputs)
        with torch.inference_mode():
            reference_components = model.forward_components(**inputs)
        reference = {
            "move_logits": [
                float(value) for value in
                reference_components["move_logits"][0].detach().cpu()
            ],
            "arrival_logit": float(
                reference_components["arrival_logit"][0].detach().cpu()
            ),
            "finite": bool(
                torch.isfinite(reference_components["move_logits"]).all()
                and torch.isfinite(
                    reference_components["arrival_logit"]
                ).all()
            ),
        }
        results["models"][model_name]["reference_output"] = reference
        print(
            f"[calibrate] {model_name}: params="
            f"{checkpoint_record['parameters']} flops="
            f"{results['analytical'][model_name]['gflops']:.6f} GFLOP "
            f"sha256={checkpoint_record['checkpoint_sha256'][:12]}...",
            flush=True,
        )
        for round_index in range(1, int(args.rounds) + 1):
            with torch.inference_mode():
                for _ in range(int(args.warmup)):
                    model.forward_components(**inputs)
            torch.cuda.synchronize()
            before = hardware_snapshot()
            print(
                f"[calibrate] {model_name} round {round_index}/"
                f"{args.rounds}: warmup={args.warmup}, "
                f"forward={args.iterations}, request={args.iterations}",
                flush=True,
            )
            per_call = measure_forward_per_call(
                model, inputs, int(args.iterations)
            )
            grouped = measure_forward_grouped(
                model, inputs, int(args.groups),
                int(args.calls_per_group),
            )
            request = measure_request_to_result(
                model, inputs, int(args.iterations)
            )
            throughput = measure_queued_throughput(
                model, inputs, int(args.iterations)
            )
            module_values = {}
            for module_name, (callable_fn, group_size) in probes.items():
                module_values[module_name] = _measure_callable_groups(
                    callable_fn,
                    int(args.groups),
                    int(group_size),
                )
            after = hardware_snapshot()
            metrics = {
                "forward_cuda_percall": per_call,
                "forward_cuda_grouped": grouped,
                "request_to_result": request,
            }
            metrics.update(module_values)
            round_record = {
                "round": round_index,
                "metrics": {},
                "throughput": throughput,
                "hardware_before": before,
                "hardware_after": after,
            }
            for metric_name, values in metrics.items():
                stats = _stats(values)
                round_record["metrics"][metric_name] = stats
                round_rows.append({
                    "model": model_name,
                    "round": round_index,
                    "metric": metric_name,
                    **stats,
                })
                for sample_index, value in enumerate(values):
                    raw_rows.append({
                        "model": model_name,
                        "round": round_index,
                        "metric": metric_name,
                        "sample_index": sample_index,
                        "ms": float(value),
                    })
            results["models"][model_name]["rounds"][
                str(round_index)
            ] = round_record
            print(
                f"[calibrate] {model_name} round {round_index}: "
                f"forward={round_record['metrics']['forward_cuda_grouped']['mean_ms']:.4f} ms, "
                f"percall={round_record['metrics']['forward_cuda_percall']['mean_ms']:.4f} ms, "
                f"request={round_record['metrics']['request_to_result']['mean_ms']:.4f} ms",
                flush=True,
            )
        # Overall summary across raw samples, with CV of round means.
        summary = {}
        for metric_name in results["models"][model_name]["rounds"][
            "1"
        ]["metrics"]:
            all_values = []
            round_means = []
            for round_index in range(1, int(args.rounds) + 1):
                values = [
                    row["ms"] for row in raw_rows
                    if row["model"] == model_name
                    and row["round"] == round_index
                    and row["metric"] == metric_name
                ]
                all_values.extend(values)
                round_means.append(float(np.mean(values)))
            stats = _stats(all_values)
            stats["round_means_ms"] = round_means
            stats["cv_across_round_means_pct"] = _cv_round_means(
                round_means
            )
            summary[metric_name] = stats
        results["models"][model_name]["summary"] = summary
        results["models"][model_name]["module_summary"] = {
            key: value for key, value in summary.items()
            if key.startswith("module_")
        }
        # Analytical comparison and historical-reference percentages.
        analytical = results["analytical"][model_name]
        forward_mean = summary["forward_cuda_grouped"]["mean_ms"]
        forward_pcall_mean = summary["forward_cuda_percall"]["mean_ms"]
        request_mean = summary["request_to_result"]["mean_ms"]
        analytical_inference = analytical["analytical_inference_ms"]
        extraction = analytical["extraction_ms"]
        comm = analytical["communication_ms_at_reference"]
        results["analytical"][model_name].update({
            "measured_forward_mean_ms": forward_mean,
            "measured_forward_percall_mean_ms": forward_pcall_mean,
            "measured_forward_p95_ms": summary[
                "forward_cuda_percall"
            ]["p95_ms"],
            "measured_request_mean_ms": request_mean,
            "measured_request_p95_ms": summary[
                "request_to_result"
            ]["p95_ms"],
            "analytical_minus_forward_ms": (
                analytical_inference - forward_mean
            ),
            "analytical_vs_forward_pct": (
                100.0 * (analytical_inference - forward_mean)
                / forward_mean
            ),
            "analytical_vs_request_pct": (
                100.0 * (analytical_inference - request_mean)
                / request_mean
            ),
            "effective_tflops_forward": (
                analytical["flops"] / (forward_mean / 1000.0) / 1e12
            ),
            "effective_tflops_percall": (
                analytical["flops"] / (forward_pcall_mean / 1000.0)
                / 1e12
            ),
            "historical_reference_ms": HISTORICAL_GREEDY_E2E_MS,
            "pct_of_historical_analytical_inference": (
                100.0 * analytical_inference / HISTORICAL_GREEDY_E2E_MS
            ),
            "pct_of_historical_measured_forward": (
                100.0 * forward_mean / HISTORICAL_GREEDY_E2E_MS
            ),
            "pct_of_historical_measured_request": (
                100.0 * request_mean / HISTORICAL_GREEDY_E2E_MS
            ),
            "pct_of_historical_extraction": (
                100.0 * extraction / HISTORICAL_GREEDY_E2E_MS
            ),
            "pct_of_historical_communication": (
                100.0 * comm / HISTORICAL_GREEDY_E2E_MS
            ),
            "simulated_full_mec_measured_ms": (
                extraction + comm + forward_mean
            ),
            "simulated_full_mec_analytical_ms": (
                extraction + comm + analytical_inference
            ),
            "pct_of_historical_simulated_full_measured": (
                100.0 * (extraction + comm + forward_mean)
                / HISTORICAL_GREEDY_E2E_MS
            ),
        })
        del model
        torch.cuda.empty_cache()
    results["hardware_after"] = hardware_snapshot()
    # Files.
    _write_csv(output_dir / "latency_measurements.csv", round_rows)
    _write_csv(output_dir / "latency_raw_samples.csv", raw_rows)
    summary_rows = []
    for model_name in args.models:
        for metric_name, stats in results["models"][model_name][
            "summary"
        ].items():
            row = {
                "model": model_name,
                "metric": metric_name,
                **stats,
            }
            summary_rows.append(row)
    _write_csv(output_dir / "latency_summary.csv", summary_rows)
    comparison_rows, module_rows = make_figures(results, output_dir)
    results["figure_data"] = {
        "latency_comparison": comparison_rows,
        "module_breakdown": module_rows,
    }
    (output_dir / "latency_calibration.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_dir, results)
    print(
        f"[calibrate] wrote {output_dir}/latency_calibration_report.md",
        flush=True,
    )
    print("STAGE2N_LATENCY_CALIBRATION_COMPLETE", flush=True)
    return results


def _fmt(value, digits=4):
    return f"{float(value):.{digits}f}"


def write_report(output_dir: Path, results: Mapping):
    models = list(results["models"])
    lines = [
        "# Stage-2n 推理时延校准报告",
        "",
        f"- 时间：{results['created_utc']}",
        "- 对象：Stage-2j `Small` 与 `Transformer-M`，原始最佳检查点，未重训",
        "- 输入：Stage-2j 真实训练样本（`train_600/shard_0000.pt` + "
        "`aux/train/aux_shard_0000.npz`，index 0）",
        "- 精度：FP32 权重与 FP32 输入；batch size = 1；"
        "`model.eval()` + `torch.inference_mode()`",
        f"- 预热：每轮 {results['measurement']['warmup_per_round']} 次；"
        f"3 轮独立测量，每轮 "
        f"{results['measurement']['forward_iterations_per_round']} 次前向",
        "- 本轮只做时延校准：未训练、未读取封存 Test、未修改既有模型/奖励/"
        "任务池/训练结果",
        "",
        "## 0. 结论摘要",
        "",
    ]
    for model in models:
        s = results["models"][model]["summary"]
        a = results["analytical"][model]
        lines += [
            f"- **{model}**：GPU 前向 "
            f"**{_fmt(s['forward_cuda_grouped']['mean_ms'])} ms**"
            f"（P95 {_fmt(s['forward_cuda_percall']['p95_ms'])} ms），"
            f"请求到结果 **{_fmt(s['request_to_result']['mean_ms'])} ms**"
            f"（P95 {_fmt(s['request_to_result']['p95_ms'])} ms）；"
            f"2 TFLOPS 解析推理 {_fmt(a['analytical_inference_ms'], 6)} ms，"
            f"解析/实测偏差 **{a['analytical_vs_forward_pct']:+.1f}%**。",
        ]
    lines += [
        "",
        "**五个问题的直接回答**",
        "",
        "1. **实际推理时延**：见上；Small 约 "
        f"{_fmt(results['models']['Small']['summary']['forward_cuda_grouped']['mean_ms'], 3)} ms，"
        "Transformer-M 约 "
        f"{_fmt(results['models']['Transformer-M']['summary']['forward_cuda_grouped']['mean_ms'], 3)} ms"
        "（GPU 前向，均值）。",
        "2. **差异是否值得做异质关联**：两个模型 GPU 前向差约 "
        f"{_fmt(abs(results['models']['Transformer-M']['summary']['forward_cuda_grouped']['mean_ms'] - results['models']['Small']['summary']['forward_cuda_grouped']['mean_ms']), 3)} ms，"
        "在 49.11 ms 历史 E2E 参考中约占 "
        f"{abs(results['models']['Transformer-M']['summary']['forward_cuda_grouped']['mean_ms'] - results['models']['Small']['summary']['forward_cuda_grouped']['mean_ms']) / HISTORICAL_GREEDY_E2E_MS * 100:.1f}%"
        "；若模型差异只体现在这一量级，异质关联的时延收益有限，"
        "需要与通信/算力差异一起评估。",
        "3. **解析模型偏差**：Small 的 2 TFLOPS 解析值严重低估实际调用"
        "成本（启动/调度开销主导）；Transformer-M 的解析值偏高，"
        "因为实测有效吞吐高于 2 TFLOPS。具体百分比见第 5 节。",
        "4. **推理占 E2E 的比例**：推理只占历史 49.11 ms 参考的很小部分；"
        "提取（共享视觉编码器约 22.7 ms 解析）和通信（150 m 约 21.1 ms 解析）"
        "才是主要项。**优先优化通信与特征提取，而不是 MEC 前向。**",
        "5. **是否用于下一轮仿真校准**：建议新增独立校准 profile，"
        "不要覆盖 `stage2k_latency.py` 的原始 2 TFLOPS 模型；"
        "在本服务器 GPU 上使用实测值时，必须标注这不是 UAV/MEC 目标硬件，"
        "并保留启动开销下限。详见第 8 节。",
        "",
        "## 1. 硬件与软件",
        "",
        f"- GPU：`{results['hardware_before']['device_name']}`，"
        f"compute capability {results['hardware_before']['compute_capability']}，"
        f"{results['hardware_before']['sm_count']} SM，"
        f"{results['hardware_before']['total_memory_gb']:.1f} GB",
        f"- PyTorch：`{results['hardware_before']['torch']}`；"
        f"CUDA：`{results['hardware_before']['torch_cuda']}`；"
        f"cuDNN：`{results['hardware_before']['cudnn']}`",
        f"- TF32 matmul：{results['hardware_before']['allow_tf32_matmul']}；"
        f"float32_matmul_precision："
        f"`{results['hardware_before']['float32_matmul_precision']}`",
        "- **重要限制**：这是服务器 RTX 5090，不是实际 UAV/MEC 目标硬件；"
        "结果不能直接等同于目标硬件性能。",
        "",
        "## 2. 检查点与模型",
        "",
        "| 模型 | 参数量 | FLOPs (GFLOP) | 检查点 SHA256 | 配置 |",
        "|---|---:|---:|---|---|",
    ]
    for model in models:
        c = results["models"][model]["checkpoint"]
        a = results["analytical"][model]
        lines.append(
            f"| {model} | {c['parameters']:,} | {a['gflops']:.6f} | "
            f"`{c['checkpoint_sha256'][:16]}…` | "
            f"`{c['config']}` |"
        )
    lines += [
        "",
        "## 3. 测量方法",
        "",
        "- **纯模型前向（分组 CUDA Event）**：输入已在 GPU；100 组 ×10 次"
        "连续前向，事件包住每组，除以 10；用于主均值/吞吐。",
        "- **纯模型前向（逐次 CUDA Event）**：1000 个独立事件对，用于 "
        "P50/P90/P95/P99 分布；包含单次调用间的 GPU 空闲/启动间隙。",
        "- **请求到结果**：`perf_counter` 包住前向 + sigmoid/argmax "
        "结果取回 + `cuda.synchronize()`；1000 次/轮；不包含网络上行。",
        "- **模块分项**：直接调用 `history_encoder`、`move_head`/"
        "`move_classifier`、`arrival_head`，以及 Transformer-M 的"
        "`fusion_encoder`；分组 CUDA Event；**分项不可相加**，因为"
        "投影、token 拼接、goal-current attention 和残差 glue 未包含。",
        "- 每个模型每轮先预热 200 次；3 轮独立测量。",
        "",
        "## 4. 测量结果（ms）",
        "",
        "### 4.1 总体（3 轮合并；CV 为三轮均值的变异系数）",
        "",
        "| 模型 | 指标 | Mean | P50 | P90 | P95 | P99 | Std | CV(轮均值) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for metric in (
            "forward_cuda_grouped", "forward_cuda_percall",
            "request_to_result",
        ):
            s = results["models"][model]["summary"][metric]
            lines.append(
                f"| {model} | `{metric}` | {_fmt(s['mean_ms'])} | "
                f"{_fmt(s['p50_ms'])} | {_fmt(s['p90_ms'])} | "
                f"{_fmt(s['p95_ms'])} | {_fmt(s['p99_ms'])} | "
                f"{_fmt(s['std_ms'])} | "
                f"{s['cv_across_round_means_pct']:.2f}% |"
            )
    lines += [
        "",
        "### 4.2 逐轮均值",
        "",
        "| 模型 | 轮次 | GPU 前向（分组） | 请求到结果 |",
        "|---|---:|---:|---:|",
    ]
    for model in models:
        for round_index in range(1, 4):
            r = results["models"][model]["rounds"][str(round_index)]
            lines.append(
                f"| {model} | {round_index} | "
                f"{_fmt(r['metrics']['forward_cuda_grouped']['mean_ms'])} | "
                f"{_fmt(r['metrics']['request_to_result']['mean_ms'])} |"
            )
    lines += [
        "",
        "### 4.3 吞吐率",
        "",
        "| 模型 | 排队前向吞吐（次/秒） | 前向耗时（s） |",
        "|---|---:|---:|",
    ]
    for model in models:
        # Queued throughput is in the last round record.
        r = results["models"][model]["rounds"]["3"]["throughput"]
        lines.append(
            f"| {model} | {r['throughput_per_s']:.1f} | "
            f"{r['elapsed_s']:.3f} |"
        )
    lines += [
        "",
        "## 5. 解析模型对比",
        "",
        "解析值来自现有 `Stage2KLatency`，假设 2 TFLOPS；通信/提取为解析值，"
        f"参考距离 {REFERENCE_DISTANCE_M:.0f} m。",
        "",
        "| 模型 | 解析推理 (ms) | 实测 GPU 前向 (ms) | 请求→结果 (ms) | 前向 P95 (ms) | 解析/前向偏差 | 实测有效 TFLOP/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        a = results["analytical"][model]
        s = results["models"][model]["summary"]
        lines.append(
            f"| {model} | {a['analytical_inference_ms']:.6f} | "
            f"{s['forward_cuda_grouped']['mean_ms']:.4f} | "
            f"{s['request_to_result']['mean_ms']:.4f} | "
            f"{s['forward_cuda_percall']['p95_ms']:.4f} | "
            f"{a['analytical_vs_forward_pct']:+.1f}% | "
            f"{a['effective_tflops_forward']:.2f} |"
        )
    lines += [
        "",
        "### 5.1 与历史 49.11 ms 的比例（仅量级参考）",
        "",
        "| 模型 | 解析推理占比 | 实测前向占比 | 请求占比 | 解析提取占比 | 解析通信占比（150 m） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        a = results["analytical"][model]
        lines.append(
            f"| {model} | {a['pct_of_historical_analytical_inference']:.3f}% | "
            f"{a['pct_of_historical_measured_forward']:.3f}% | "
            f"{a['pct_of_historical_measured_request']:.3f}% | "
            f"{a['pct_of_historical_extraction']:.2f}% | "
            f"{a['pct_of_historical_communication']:.2f}% |"
        )
    lines += [
        "",
        "**注意**：49.11 ms 是上一轮 Greedy-Min-E2E 的历史仿真值，"
        "不是本轮异质部署的实测 E2E；这些比例只用于量级判断。",
        "",
        "## 6. 从解析到完整 MEC 的分层",
        "",
        "| 项目 | Small | Transformer-M | 说明 |",
        "|---|---:|---:|---|",
    ]
    for label, key in (
        ("纯推理（实测 GPU 前向）", "measured_forward_mean_ms"),
        ("请求到结果（本地，无网络）", "measured_request_mean_ms"),
    ):
        lines.append(
            f"| {label} | {results['analytical']['Small'][key]:.4f} | "
            f"{results['analytical']['Transformer-M'][key]:.4f} | 实测 |"
        )
    for label, key in (
        ("共享视觉编码器提取（解析）", "extraction_ms"),
        ("150 m 上行通信（解析）", "communication_ms_at_reference"),
    ):
        lines.append(
            f"| {label} | {results['analytical']['Small'][key]:.3f} | "
            f"{results['analytical']['Transformer-M'][key]:.3f} | 解析，未实测 |"
        )
    lines += [
        "",
        "**完整 MEC 处理**：解析提取 + 解析通信 + 实测前向 = "
        f"Small {results['analytical']['Small']['simulated_full_mec_measured_ms']:.3f} ms，"
        f"Transformer-M {results['analytical']['Transformer-M']['simulated_full_mec_measured_ms']:.3f} ms "
        "（150 m 假设；这是解析/实测混合值，不是端到端实测）。",
        "",
        "## 7. 模块分项（诊断，不可相加）",
        "",
        "| 模型 | 模块 | Mean (ms) | P95 (ms) |",
        "|---|---|---:|---:|",
    ]
    for model in models:
        for name, s in results["models"][model][
            "module_summary"
        ].items():
            lines.append(
                f"| {model} | `{name}` | {s['mean_ms']:.4f} | "
                f"{s['p95_ms']:.4f} |"
            )
    lines += [
        "",
        "分项之和通常小于完整 forward，因为视觉投影、goal-current attention、"
        "token 拼接、LayerNorm/残差等未包含在单独模块内；**不要相加后宣称"
        "等于完整 forward**。",
        "",
        "## 8. 校准建议与影响分析",
        "",
        "1. **不要覆盖原始解析模型**。`stage2k_latency.py` 的 2 TFLOPS "
        "FLOPs 模型是已有 Stage-2n 结果的一部分；直接修改会让旧结果不可复现。",
        "2. **新增独立 profile**：例如 "
        "`measured_rtx5090_fp32_v1`，字段为 "
        "`forward_ms_small=…`、`forward_ms_transformer_m=…`、"
        "`request_overhead_ms=…`。保留 `analytical_2tflops` 作为默认/对照。",
        "3. **加入启动开销下限**：Small 的解析值只有 "
        f"{results['analytical']['Small']['analytical_inference_ms']:.6f} ms，"
        f"而实测前向 {results['analytical']['Small']['measured_forward_mean_ms']:.4f} ms；"
        "任何校准都必须包含约 0.4–0.5 ms 的启动/调度地板，否则会严重低估。",
        "4. **影响量级**：Transformer-M 的解析推理比实测高约 "
        f"{results['analytical']['Transformer-M']['analytical_vs_forward_pct']:.1f}%；"
        "若把其解析推理替换为实测值，150 m 模拟完整 MEC 约变化 "
        f"{results['analytical']['Transformer-M']['simulated_full_mec_measured_ms'] - results['analytical']['Transformer-M']['simulated_full_mec_analytical_ms']:+.3f} ms，"
        "相对 49.11 ms 约 "
        f"{100*abs(results['analytical']['Transformer-M']['simulated_full_mec_measured_ms'] - results['analytical']['Transformer-M']['simulated_full_mec_analytical_ms'])/HISTORICAL_GREEDY_E2E_MS:.2f}%。"
        "因此推理校准不会改变“通信/提取主导”的结论。",
        "5. **不等同目标硬件**：RTX 5090 是服务器 GPU；UAV/MEC 上的 CPU、"
        "功耗、散热、驱动和并发都会改变结果。下一轮异质实验若使用实测值，"
        "应做硬件缩放/敏感性分析，而不是把 RTX 5090 数字当作 UAV 实测。",
        "",
        "## 9. 复现命令与产物",
        "",
        "```bash",
        "python v83_spatial_stage2/stage2n_latency_calibrate.py",
        "python v83_spatial_stage2/test_stage2n_latency_calibrate.py",
        "```",
        "",
        "- `latency_calibration_report.md`（本文件）",
        "- `latency_measurements.csv`（逐轮、逐模型统计）",
        "- `latency_raw_samples.csv`（原始逐次/逐组样本）",
        "- `latency_summary.csv`（总体统计与 CV）",
        "- `latency_calibration.json`（硬件、检查点、输入、参数与全部结果）",
        "- `figures/latency_comparison.png`（300 dpi）",
        "- `figures/latency_module_breakdown.png`（300 dpi）",
        "- 原始 checkpoint 与训练日志保留在 `runs/`，未改动",
        "",
        "## 10. 限制",
        "",
        "- 单个真实训练样本；时延对输入数值不敏感，但样本多样性未覆盖。",
        "- 只测 batch 1、FP32；未测 FP16/TensorRT/多并发。",
        "- 服务器 RTX 5090，不是 UAV/MEC 目标硬件。",
        "- 网络上行/提取为解析值，未做真实网络或 UAV 端测量。",
        "- 本轮不读取封存 Test-400，不训练 DQN/PPO。",
    ]
    (output_dir / "latency_calibration_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=list(MODELS)
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--calls-per-group", type=int, default=10)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.warmup < 200:
        raise SystemExit("warmup must be >= 200")
    if args.rounds < 3:
        raise SystemExit("rounds must be >= 3")
    if args.iterations < 1000:
        raise SystemExit("iterations must be >= 1000")
    run_measurement(args)


if __name__ == "__main__":
    main()
