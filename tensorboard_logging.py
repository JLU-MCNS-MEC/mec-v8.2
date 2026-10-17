"""TensorBoard logging utilities shared by the V8.2 training scripts.

Only records metrics already computed by training. No model, loss, optimizer,
random seed or data sampling changes are made here.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Any

from torch.utils.tensorboard import SummaryWriter


DEFAULT_LOG_ROOT = Path("logs") / "tensorboard"


def new_run_id() -> str:
    """Unique across independent launches, including concurrent runs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-pid{os.getpid()}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_").lower()


def make_writer(
    experiment: str,
    model_name: str,
    *,
    run_id: str,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> SummaryWriter:
    """One event directory per experiment, process run, and MEC model."""
    log_dir = (
        Path(log_root) / _slug(experiment) / str(run_id) / _slug(model_name)
    )
    return SummaryWriter(log_dir=str(log_dir), flush_secs=30)


def _metric_tag(name: str) -> str | None:
    if name in ("epoch", "model_seed"):
        return None
    if name.startswith("train_"):
        return "train/" + name[len("train_"):]
    if name.startswith("val_"):
        return "val/" + name[len("val_"):]
    if name == "lr":
        return "optimizer/lr"
    if name in ("checkpoint_score", "arrival_selection_f1"):
        return "selection/" + name
    return "metrics/" + name


def write_epoch_metrics(
    writer: SummaryWriter,
    row: Mapping[str, Any],
    epoch: int,
) -> None:
    """Record scalar values from the existing per-epoch CSV row unchanged.

    Metric values are kept in their native units (F1/accuracy are 0..1).
    Non-numeric and non-finite values are omitted.
    """
    for name, value in row.items():
        tag = _metric_tag(name)
        if tag is None or value is None or isinstance(value, bool):
            continue
        try:
            scalar = float(value)
        except (ValueError, TypeError, OverflowError):
            continue
        if math.isfinite(scalar):
            writer.add_scalar(tag, scalar, global_step=int(epoch))
    writer.flush()
