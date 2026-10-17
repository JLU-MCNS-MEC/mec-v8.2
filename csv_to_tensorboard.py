"""Convert already-saved V8.2 per-model CSV histories to TensorBoard events.

This tool never edits the input CSVs, model checkpoints or source code.
Repeated imports of an unchanged CSV are skipped automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from tensorboard_logging import DEFAULT_LOG_ROOT, write_epoch_metrics


DEFAULT_SOURCE_DIRS = (
    Path("gbs_v82_input_redesign_results"),
    Path("gbs_v82_epoch_compare_results"),
)
CSV_SUFFIX = "_training_history.csv"


def convert_csv_file(
    csv_file: Path,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> tuple[Path, bool]:
    """Return (event_directory, created_new_events)."""
    csv_file = Path(csv_file)
    if not csv_file.name.endswith(CSV_SUFFIX):
        raise ValueError(f"Not a model training-history CSV: {csv_file}")
    model_name = csv_file.name[: -len(CSV_SUFFIX)]
    raw_csv = csv_file.read_bytes()
    fingerprint = hashlib.sha256(raw_csv).hexdigest()[:16]

    # Hash-based folders preserve separate snapshots if the CSV later changes.
    event_dir = (
        Path(log_root)
        / "historical"
        / csv_file.parent.name
        / model_name
        / fingerprint
    )
    if any(event_dir.glob("events.out.tfevents.*")):
        return event_dir, False

    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "epoch" not in reader.fieldnames:
            raise ValueError(f"CSV is missing the epoch column: {csv_file}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no history rows: {csv_file}")

    with SummaryWriter(log_dir=str(event_dir)) as writer:
        for row in rows:
            epoch = int(row["epoch"])
            write_epoch_metrics(writer, row, epoch)
    return event_dir, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dirs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_SOURCE_DIRS),
        help="Directories containing *_training_history.csv files",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="TensorBoard root folder (default: logs/tensorboard)",
    )
    args = parser.parse_args()

    found = 0
    created = 0
    for source_dir in args.source_dirs:
        if not source_dir.is_dir():
            print(f"[MISSING] {source_dir} (skipping)")
            continue
        for csv_file in sorted(source_dir.glob(f"*{CSV_SUFFIX}")):
            found += 1
            output, is_new = convert_csv_file(csv_file, args.log_root)
            created += int(is_new)
            print(
                f"[{'IMPORTED' if is_new else 'SKIPPED'}] "
                f"{csv_file} -> {output}"
            )

    print(f"CSV files found: {found}; new imports: {created}")
    if found == 0:
        print("No training-history CSVs found. Check the source directories.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
