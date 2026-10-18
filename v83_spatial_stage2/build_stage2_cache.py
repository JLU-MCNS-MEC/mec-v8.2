"""Build Stage-2 spatial caches (Train 600, Dev Val 180 natural+hard).

Final Test is deliberately not cached or used. The existing V8.2/V8.3 caches
are not modified; this writes a new cache directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_training.spatial_cache import build_split_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("runs/v83_spatial_stage2/datasets"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    train_dir = args.data_root / "train_600"
    dev_dir = args.data_root / "dev_val_180"
    if not (train_dir / "dataset_stats.json").is_file():
        raise SystemExit(f"Missing {train_dir}")
    if not (dev_dir / "dataset_stats.json").is_file():
        raise SystemExit(f"Missing {dev_dir}")
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite cache {args.output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "data_root": str(args.data_root), "caches": {}}
    manifest["caches"]["train"] = build_split_cache(
        "train", train_dir, args.output_dir, device
    )
    manifest["caches"]["val"] = build_split_cache(
        "val", dev_dir, args.output_dir, device
    )
    manifest["caches"]["val_hard"] = build_split_cache(
        "val_hard", dev_dir, args.output_dir, device
    )
    manifest["sizes_bytes"] = {
        f.name: f.stat().st_size for f in args.output_dir.glob("*.npz")
    }
    (args.output_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("STAGE2_CACHE_DONE", json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
