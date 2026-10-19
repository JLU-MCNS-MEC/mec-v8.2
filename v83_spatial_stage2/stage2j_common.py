"""Shared Stage-2j constants, progress bookkeeping and result formatting."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from v83_spatial_stage2.stage2j_model import EXECUTION_ORDER, TABLE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2J_ROOT = Path("runs/v83_spatial_stage2/stage2j")
STAGE2I_ROOT = Path("runs/v83_spatial_stage2/stage2i")
TRAIN_DIR = Path("runs/v83_spatial_stage2/datasets/train_600")
VAL_DIR = Path("runs/v83_spatial_stage2/datasets/dev_val_180")
FINAL_TEST_DIR = Path("runs/v83_spatial_stage2/datasets/final_test_300")
AUX_ROOT = STAGE2I_ROOT / "aux"
T5_DIR = STAGE2I_ROOT / "T5_map_assisted"
T5_CHECKPOINT = T5_DIR / "best.pt"
T5_TRAINING_RESULT = T5_DIR / "training_result.json"
T5_EXPECTED_SHA256 = (
    "57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d"
)
PROGRESS_PATH = STAGE2J_ROOT / "progress.json"
RESULTS_MD = STAGE2J_ROOT / "RESULTS.md"
COMMITTED_RESULTS_MD = Path("v83_spatial_stage2/STAGE2J_RESULTS.md")
FREEZE_PATH = STAGE2J_ROOT / "protocol_freeze.json"
COMMITTED_FREEZE_PATH = Path("v83_spatial_stage2/stage2j_protocol_freeze.json")
PROTOCOL_PATH = Path("v83_spatial_stage2/STAGE2J_PROTOCOL.md")
LOGS_DIR = STAGE2J_ROOT / "logs"
SPECS_DIR = STAGE2J_ROOT / "specs"

BANDS = ("short", "medium", "long")
BAND_EPISODES = 100
TEST_EPISODES = 300
STOP_THRESHOLD = 0.7

TRAIN_PROTOCOL = {
    "train_dataset": str(TRAIN_DIR),
    "dev_val_dataset": str(VAL_DIR),
    "aux_root": str(AUX_ROOT),
    "max_epochs": 50,
    "patience": 8,
    "batch_size": 48,
    "val_batch_size": 96,
    "learning_rate": 6e-5,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "lr_scheduler": "ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)",
    "seed": 20264017,
    "amp": True,
    "frozen_shared_rgb_encoder": True,
    "loss": "movement CE (soft class weights) + Arrival BCE; no pos_weight",
    "checkpoint_score": "0.7*macro_f1 + 0.3*mean(natural,hard)_arrival_f1",
    "stop_threshold_for_offline_metrics": 0.5,
    "stop_threshold_for_closed_loop": STOP_THRESHOLD,
}

EVAL_PROTOCOL = {
    "final_test_dataset": str(FINAL_TEST_DIR),
    "episodes": TEST_EPISODES,
    "bands": {band: BAND_EPISODES for band in BANDS},
    "stop_threshold": STOP_THRESHOLD,
    "weights_frozen": True,
    "no_test_tuning": True,
    "task_list_source": "sealed token shards (episode_id, episode_seed, "
                        "network_layout_seed, trajectory_band_id)",
    "seal_verification": "SHA256 of every file listed in SEALED.json",
    "oracle_usage": "evaluation-only SPL/nDTW reference; never model input",
    "metrics": ["SR", "OSR", "SPL", "nDTW", "Premature-Stop", "Timeout",
                "Average Steps"],
}

MODEL_DIRS = {name: name for name in TABLE_ORDER}
MODEL_TITLES = {
    "Small": "Small - late-fusion MLP",
    "Transformer-S": "Transformer-S - explicit Transformer "
                     "(192D, 1 layer, 4 heads, ff x3)",
    "Medium": "Medium - Goal-Current cross-attention (384D, 6 heads)",
    "Transformer-M": "Transformer-M - explicit Transformer "
                     "(256D, 2 layers, 8 heads, ff x4)",
    "Large": "Large - explicit Transformer "
             "(352D, 2 layers, 8 heads, ff x4)",
}


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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_final_test(path):
    if "final_test" in str(path):
        raise ValueError(f"Final Test is sealed; refusing to read {path}")


def model_dir(name, root=STAGE2J_ROOT):
    return Path(root) / MODEL_DIRS[name]


def _new_model_entry(name):
    return {
        "model": name,
        "architecture": None,
        "parameters": None,
        "trainable_parameters": None,
        "train_status": "pending",
        "test_status": "pending",
        "best_epoch": None,
        "best_score": None,
        "checkpoint": None,
        "checkpoint_sha256": None,
        "training_time_s": None,
        "training_time_min": None,
        "metrics": None,
        "bands": None,
        "retrieval": None,
        "latency": None,
        "eval_output_dir": None,
        "eval_started_utc": None,
        "eval_finished_utc": None,
        "final_test_read": False,
    }


def load_progress(path=PROGRESS_PATH):
    payload = read_json(path, default=None)
    if payload is None:
        payload = {
            "created_utc": utc_now(),
            "order": list(TABLE_ORDER),
            "execution_order": list(EXECUTION_ORDER),
            "models": {name: _new_model_entry(name) for name in TABLE_ORDER},
        }
    payload.setdefault("order", list(TABLE_ORDER))
    payload.setdefault("execution_order", list(EXECUTION_ORDER))
    payload.setdefault("models", {})
    for name in TABLE_ORDER:
        entry = payload["models"].setdefault(name, _new_model_entry(name))
        for key, value in _new_model_entry(name).items():
            entry.setdefault(key, value)
    return payload


def update_progress(model, **fields):
    payload = load_progress()
    payload["updated_utc"] = utc_now()
    payload["models"][model].update(fields)
    atomic_write_json(PROGRESS_PATH, payload)
    return payload


def is_completed(progress, model):
    entry = progress["models"].get(model, {})
    return bool(entry.get("metrics")) and entry.get("test_status") in (
        "tested", "reused_tested"
    )


def cumulative_table(progress):
    header = (
        "| MEC | Params | SR | OSR | SPL | nDTW | Premature | Timeout | "
        "Train min |\n"
        "|-----|--------|----|-----|-----|------|-----------|---------|"
        "-----------|"
    )
    rows = [header]
    for name in TABLE_ORDER:
        entry = progress["models"].get(name, {})
        metrics = entry.get("metrics")
        params = entry.get("parameters")
        params_text = f"{int(params):,}" if params else "..."
        if not metrics:
            rows.append(
                f"| {name} | {params_text} | Pending | - | - | - | - | - | - |"
            )
            continue
        train_min = entry.get("training_time_min")
        if train_min is None and entry.get("training_time_s") is not None:
            train_min = float(entry["training_time_s"]) / 60.0
        train_text = f"{float(train_min):.0f}" if train_min is not None else "-"
        rows.append(
            f"| {name} | {params_text} | {100*metrics['sr']:.2f}% | "
            f"{100*metrics['osr']:.2f}% | {metrics['spl']:.4f} | "
            f"{metrics['ndtw']:.4f} | "
            f"{100*metrics['premature_stop']:.2f}% | "
            f"{100*metrics['timeout']:.2f}% | {train_text} |"
        )
    return "\n".join(rows)


def band_table(bands):
    header = (
        "| Band | SR | OSR | SPL | nDTW |\n"
        "|---|---:|---:|---:|---:|"
    )
    rows = [header]
    for band in BANDS:
        entry = (bands or {}).get(band, {})
        rows.append(
            f"| {band.capitalize()} | {100*entry.get('sr', 0.0):.2f}% | "
            f"{100*entry.get('osr', 0.0):.2f}% | "
            f"{entry.get('spl', 0.0):.4f} | {entry.get('ndtw', 0.0):.4f} |"
        )
    return "\n".join(rows)


def latency_summary(times_ms):
    values = np.asarray(list(times_ms), dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def format_result_block(model, entry, completed):
    metrics = entry["metrics"]
    bands = entry.get("bands") or {}
    retrieval = entry.get("retrieval") or {}
    latency = entry.get("latency") or {}
    train_min = entry.get("training_time_min")
    if train_min is None and entry.get("training_time_s") is not None:
        train_min = float(entry["training_time_s"]) / 60.0
    lines = [
        f"STAGE-2J MODEL RESULT: {model}",
        "",
        f"Model: {MODEL_TITLES[model]}",
        f"Architecture: {entry.get('architecture')}",
        f"Parameters: {int(entry['parameters']):,}"
        if entry.get("parameters") else "Parameters: pending",
        "Train: 600 Episodes",
        "Evaluation: Final Test 300 Episodes (100 Short / 100 Medium / 100 Long)",
        "",
        f"SR: {100*metrics['sr']:.2f}% "
        f"({int(round(metrics['sr']*metrics['episodes']))}/{metrics['episodes']})",
        f"OSR: {100*metrics['osr']:.2f}%",
        f"SPL: {metrics['spl']:.4f}",
        f"nDTW: {metrics['ndtw']:.4f}",
        f"Premature-Stop: {100*metrics['premature_stop']:.2f}%",
        f"Timeout: {100*metrics['timeout']:.2f}%",
        f"Average Steps: {metrics['steps']:.1f}",
        f"Best Epoch: {entry.get('best_epoch')}",
        (
            f"Training Time: {float(train_min):.0f} min"
            if train_min is not None else "Training Time: reused"
        ),
        "",
        "Band Results:",
    ]
    for label, band in (("Short:  ", "short"), ("Medium: ", "medium"),
                        ("Long:   ", "long")):
        item = bands.get(band, {})
        lines.append(
            f"{label} SR={100*item.get('sr', 0.0):.2f}% | "
            f"OSR={100*item.get('osr', 0.0):.2f}% | "
            f"SPL={item.get('spl', 0.0):.4f} | "
            f"nDTW={item.get('ndtw', 0.0):.4f}"
        )
    lines.extend([
        "",
        "Map retrieval:",
        f"Top-1: {100*retrieval.get('clean_top1', 0.0):.2f}% | "
        f"Top-5: {100*retrieval.get('clean_top5', 0.0):.2f}%",
        "",
        "Inference latency (MEC forward, batch=1):",
        f"mean={latency.get('mean_ms', 0.0):.3f} ms | "
        f"p50={latency.get('p50_ms', 0.0):.3f} ms | "
        f"p95={latency.get('p95_ms', 0.0):.3f} ms",
        "",
        "Completed models:",
    ])
    for name in TABLE_ORDER:
        status = "DONE" if completed.get(name) else "PENDING"
        lines.append(f"{name}: {status}")
    return "\n".join(lines)


def append_results_snapshot(progress, note):
    """Write the complete table, and append the snapshot to RESULTS.md."""
    timestamp = utc_now()
    table = cumulative_table(progress)
    snapshot = (
        f"\n### {timestamp} - {note}\n\n"
        "Evaluation: Final Test 300 episodes (100/100/100), stop threshold 0.7.\n\n"
        f"{table}\n"
    )
    header = (
        "# Stage-2j cumulative results - Final Test 300\n\n"
        f"Last updated: {timestamp}\n\n"
        "Status: all numbers are actual Final Test 300 closed-loop results. "
        "`Pending` means that model has not completed its formal test.\n\n"
        "## Update history\n"
    )
    existing = ""
    if RESULTS_MD.is_file():
        existing = RESULTS_MD.read_text(encoding="utf-8")
    if not existing.startswith("# Stage-2j cumulative results"):
        existing = header
    # Keep the header's "Last updated" current without dropping history.
    lines = existing.splitlines()
    if lines and lines[0].startswith("# Stage-2j cumulative results"):
        rebuilt = [lines[0], "", f"Last updated: {timestamp}", ""]
        rebuilt.extend(lines[3:])
        existing = "\n".join(rebuilt).rstrip() + "\n"
    else:
        existing = header
    content = existing + snapshot
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(content, encoding="utf-8")
    COMMITTED_RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    COMMITTED_RESULTS_MD.write_text(content, encoding="utf-8")
    return content


def ensure_dirs():
    for path in (STAGE2J_ROOT, LOGS_DIR, SPECS_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
