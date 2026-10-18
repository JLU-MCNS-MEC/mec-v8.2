"""Non-destructive migration of completed PyCharm V8.2 artifacts to Workspace.

Usage:
  python migrate_v82_workspace.py             # dry run (default)
  python migrate_v82_workspace.py --apply     # copy missing files, never overwrite
  python migrate_v82_workspace.py --apply --results-only
  python migrate_v82_workspace.py --results-only --archive-subdir runs/archive_pycharm_v82_20261018
  python migrate_v82_workspace.py --apply --results-only --archive-subdir runs/archive_pycharm_v82_20261018

This NEVER changes old /tmp artifacts, Git-tracked source code, or checkpoints
at the destination. It does not remove source files. Run only after all
training/evaluation jobs in the source directory have finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE = Path("/tmp/pycharm_project_479/test_v8.2")
RESULT_DIRS = (
    "gbs_v82_input_redesign_results",
    "gbs_v82_epoch_compare_results",
    "fixed_mec_capability_v82_results",
    "fixed_mec_epoch_compare_results",
)
DATA_DIRS = (
    "token_dataset_train",
    "token_dataset_val",
    "token_dataset_val_hard",
    "token_dataset_test",
)
OTHER_FILES = ("shared_encoder.pt",)


def same_contents(source: Path, destination: Path) -> bool:
    """Verify existing destination files instead of blindly overwriting."""
    if source.stat().st_size != destination.stat().st_size:
        return False
    def checksum(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return checksum(source) == checksum(destination)


def file_plan(source_root: Path, target_root: Path, names: tuple[str, ...]):
    planned = []
    for name in names:
        src = source_root / name
        if not src.exists():
            print(f"[NOT FOUND] {src}")
            continue
        if src.is_symlink():
            raise RuntimeError(f"Refusing symbolic-link source: {src}")
        if src.is_file():
            entries = [(src, target_root / name)]
        elif src.is_dir():
            entries = []
            for src_file in sorted(src.rglob("*")):
                if src_file.is_symlink():
                    raise RuntimeError(f"Refusing symbolic-link source: {src_file}")
                if src_file.is_file():
                    entries.append(
                        (src_file, target_root / name / src_file.relative_to(src))
                    )
        else:
            raise RuntimeError(f"Unsupported source type: {src}")

        for src_file, dst_file in entries:
            if dst_file.is_symlink():
                raise RuntimeError(f"Refusing symbolic-link destination: {dst_file}")
            if dst_file.exists():
                if not dst_file.is_file() or not same_contents(src_file, dst_file):
                    raise RuntimeError(
                        f"Destination conflict (different content): {dst_file}\n"
                        "Nothing was overwritten. Resolve the conflict manually."
                    )
                status = "identical"
            else:
                status = "copy"
            planned.append((src_file, dst_file, status))
        print(f"[FOUND] {name}: {len(entries)} files")
    return planned


def safe_copy(src: Path, dst: Path) -> None:
    """Copy to a private temp path and atomically place the new file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if same_contents(src, dst):
            return
        raise RuntimeError(f"Destination conflict: {dst}")
    with tempfile.NamedTemporaryFile(
        dir=str(dst.parent), prefix=f".{dst.name}.migration-", delete=False
    ) as temp:
        tmp_path = Path(temp.name)
    try:
        shutil.copy2(src, tmp_path)
        if tmp_path.stat().st_size != src.stat().st_size:
            raise RuntimeError(f"Copy size mismatch for {src}")
        # A single operator owns this Workspace; recheck just before rename.
        if dst.exists():
            raise RuntimeError(f"Destination created during migration: {dst}")
        os.replace(str(tmp_path), str(dst))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--dest", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Copy missing files. Without this flag, only inspect and plan."
    )
    parser.add_argument(
        "--results-only", action="store_true",
        help="Migrate results and shared_encoder.pt but skip large token datasets."
    )
    parser.add_argument(
        "--archive-subdir", type=Path, default=None,
        help=("Place the copied artifacts under an ignored runs/ subdirectory "
              "inside the Git workspace, instead of mixing results in its root. "
              "Example: runs/archive_pycharm_v82_20261018")
    )
    args = parser.parse_args()

    source_root = args.source.expanduser().resolve()
    target_root = args.dest.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Source directory missing: {source_root}")
    if not target_root.is_dir() or not (target_root / ".git").exists():
        raise SystemExit(f"Destination is not a Git workspace: {target_root}")
    if source_root == target_root or source_root in target_root.parents:
        raise SystemExit("Source and destination overlap: refusing migration")

    artifact_root = target_root
    if args.archive_subdir is not None:
        archive_subdir = args.archive_subdir
        if (
            archive_subdir.is_absolute()
            or ".." in archive_subdir.parts
            or len(archive_subdir.parts) < 2
            or archive_subdir.parts[0] != "runs"
        ):
            raise SystemExit(
                "--archive-subdir must be a relative path below runs/, "
                "for example runs/archive_pycharm_v82_20261018"
            )
        artifact_root = (target_root / archive_subdir).resolve()
        if target_root not in artifact_root.parents:
            raise SystemExit("Archive path escapes the workspace: refusing migration")
    print(f"Artifact destination: {artifact_root}")

    names = RESULT_DIRS + OTHER_FILES
    if not args.results_only:
        names += DATA_DIRS

    plan = file_plan(source_root, artifact_root, names)
    missing = [(src, dst) for src, dst, action in plan if action == "copy"]
    total = sum(src.stat().st_size for src, _ in missing)
    print(f"Files: {len(plan)} total; {len(missing)} missing; "
          f"{total/1024**3:.2f} GiB to copy")
    if not plan:
        raise SystemExit("No matching source artifacts found; nothing migrated.")
    if not args.apply:
        print("[DRY RUN] No file changed. Run with --apply after review.")
        return
    free = shutil.disk_usage(target_root).free
    if free < total + min(1024**3, total // 20):
        raise SystemExit(
            f"Insufficient free space: need ~{total/1024**3:.2f} GiB, "
            f"available {free/1024**3:.2f} GiB"
        )
    for index, (src, dst) in enumerate(missing, 1):
        safe_copy(src, dst)
        if index == 1 or index % 100 == 0 or index == len(missing):
            print(f"[COPIED] {index}/{len(missing)}: {dst.name}")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_root),
        "workspace_root": str(target_root),
        "destination": str(artifact_root),
        "copied_files": len(missing),
        "already_identical_files": len(plan) - len(missing),
        "copied_bytes": total,
        "files": [
            {
                "relative_path": str(dst.relative_to(target_root)),
                "source_bytes": src.stat().st_size,
                "action": status,
            }
            for src, dst, status in plan
        ],
    }
    report_dir = target_root / "logs" / "migrations"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_file = report_dir / f"workspace_migration_{stamp}.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SUCCESS] No source files were deleted or overwritten.")
    print(f"Manifest: {report_file}")


if __name__ == "__main__":
    main()
