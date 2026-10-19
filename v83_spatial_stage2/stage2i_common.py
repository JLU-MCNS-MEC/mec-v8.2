"""Shared Stage-2i constants, progress bookkeeping and reporting helpers."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

STAGE2I_ROOT = Path("runs/v83_spatial_stage2/stage2i")
PROGRESS_PATH = STAGE2I_ROOT / "progress.json"
RESULTS_MD = STAGE2I_ROOT / "RESULTS.md"
TRAIN_AUX_DIR = STAGE2I_ROOT / "aux"
ORDER = ("T5", "T4", "T3", "T2", "T1", "T0")

MODEL_DIRS = {
    "T5": "T5_map_assisted",
    "T4": "T4_history_pose_gab",
    "T3": "T3_no_heatmap_no_network",
    "T2": "T2_no_network",
    "T1": "T1_no_heatmap",
    "T0": "T0_baseline",
}

MODEL_TITLES = {
    "T5": "Transformer-M + History + Pose + GAB + Landmark Graph",
    "T4": "Transformer-M + History + Pose + GAB",
    "T3": "Transformer-M (Current + Goal only)",
    "T2": "Transformer-M (Current + Goal + Heatmap)",
    "T1": "Transformer-M (Current + Goal + Network State)",
    "T0": "Transformer-M original M0 inputs",
}


def model_dir(variant, root=STAGE2I_ROOT):
    return Path(root) / MODEL_DIRS[variant]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    )
    try:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def load_progress(path=PROGRESS_PATH):
    payload = read_json(path, default=None)
    if payload is None:
        payload = {
            "created_utc": utc_now(),
            "order": list(ORDER),
            "models": {
                variant: {
                    "variant": variant,
                    "train_status": "pending",
                    "eval_status": "pending",
                    "best_epoch": None,
                    "checkpoint": None,
                    "training_time_s": None,
                    "metrics": None,
                }
                for variant in ORDER
            },
        }
    for variant in ORDER:
        payload.setdefault("models", {}).setdefault(
            variant,
            {"variant": variant, "train_status": "pending",
             "eval_status": "pending"},
        )
    return payload


def update_progress(variant, **fields):
    payload = load_progress()
    payload["updated_utc"] = utc_now()
    payload["models"][variant].update(fields)
    atomic_write_json(PROGRESS_PATH, payload)
    return payload


def reject_final_test(path):
    if "final_test" in str(path):
        raise ValueError(f"Final Test is sealed; refusing to read {path}")


def format_result_block(variant, metrics, bands, completed):
    """Exact Stage-2i terminal result format requested by the protocol."""
    lines = [
        f"STAGE-2i MODEL RESULT: {variant}",
        "",
        f"Model: {MODEL_TITLES[variant]}",
        "Train: 600 Episodes",
        "Evaluation: Dev Val 180 Episodes",
        "",
        f"SR: {100*metrics['sr']:.2f}% "
        f"({int(round(metrics['sr']*metrics['episodes']))}/{metrics['episodes']})",
        f"OSR: {100*metrics['osr']:.2f}%",
        f"SPL: {metrics['spl']:.4f}",
        f"nDTW: {metrics['ndtw']:.4f}",
        f"Premature-Stop: {100*metrics['premature_stop']:.2f}%",
        f"Timeout: {100*metrics['timeout']:.2f}%",
        f"Average Steps: {metrics['steps']:.1f}",
        f"Best Epoch: {metrics['best_epoch']}",
        f"Training Time: {metrics['training_time_min']:.0f} min",
        "",
        "Band Results:",
    ]
    for label, band in (("Short:  ", "short"), ("Medium: ", "medium"),
                        ("Long:   ", "long")):
        entry = bands.get(band, {})
        lines.append(
            f"{label} SR={100*entry.get('sr', 0.0):.2f}% | "
            f"OSR={100*entry.get('osr', 0.0):.2f}% | "
            f"SPL={entry.get('spl', 0.0):.4f} | "
            f"nDTW={entry.get('ndtw', 0.0):.4f}"
        )
    lines.extend([
        "",
        "Conclusion:",
        metrics.get("conclusion", "Pending final paired analysis against T0."),
        "",
        "Completed models:",
    ])
    for name in ORDER:
        status = "DONE" if completed.get(name) else "PENDING"
        lines.append(f"{name}: {status}")
    return "\n".join(lines)


def cumulative_table(progress):
    """Markdown table of every evaluated model so far."""
    header = (
        "| Model | SR | OSR | SPL | nDTW | Premature-Stop | Timeout | "
        "Steps | Best Epoch | Train min |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for variant in ORDER:
        entry = progress["models"].get(variant, {})
        metrics = entry.get("metrics")
        if not metrics:
            rows.append(f"| {variant} | pending | — | — | — | — | — | — | — | — |")
            continue
        rows.append(
            f"| {variant} | {100*metrics['sr']:.2f}% | "
            f"{100*metrics['osr']:.2f}% | {metrics['spl']:.4f} | "
            f"{metrics['ndtw']:.4f} | "
            f"{100*metrics['premature_stop']:.2f}% | "
            f"{100*metrics['timeout']:.2f}% | "
            f"{metrics['steps']:.1f} | {metrics['best_epoch']} | "
            f"{metrics['training_time_min']:.0f} |"
        )
    return "\n".join(rows)
