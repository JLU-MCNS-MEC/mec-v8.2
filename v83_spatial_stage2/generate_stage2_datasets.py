"""Generate V8.3 Stage-2 balanced token datasets.

Train 600 = 200 Short / 200 Medium / 200 Long
Dev Val 180 = 60 / 60 / 60
Final Test 300 = 100 / 100 / 100 (sealed; never used for tuning)

New trajectory seed ranges and derived network-layout seeds are disjoint from
all archived/V8.2/V8.3 splits. The final test is generated once and sealed with
SHA256 hashes and read-only files.

The existing V8.2 training protocol/data/token_data implementation is NOT
modified on disk. A process-local dataclasses.replace sets equal band weights
for this generation process only. Output directories are guarded against
overwrite; token_data's own delete-on-existing behavior is never invoked for an
existing directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import token_data
from config import TRAIN as ORIGINAL_TRAIN


EQUAL_MIX = tuple(
    (name, 1.0 / len(ORIGINAL_TRAIN.trajectory_mix), dmin, dmax, amin, amax)
    for name, _weight, dmin, dmax, amin, amax in ORIGINAL_TRAIN.trajectory_mix
)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seal(directory: Path, split: str, seed_start: int):
    files = sorted(directory.glob("*.pt")) + [directory / "dataset_stats.json"]
    hashes = {f.name: _sha256(f) for f in files if f.is_file()}
    seal = {
        "split": split,
        "seed_start": int(seed_start),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": hashes,
        "note": "SEALED FINAL TEST. Do not use for model selection, threshold tuning, or diagnostics.",
    }
    (directory / "SEALED.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
    for f in files:
        if f.is_file():
            os.chmod(f, 0o444)
    for f in sorted(directory.glob("*.json")):
        os.chmod(f, 0o444)
    print(f"Sealed {directory} ({len(hashes)} files)", flush=True)


def generate_split(name, target_episodes, seed_start, include_hard, output_root, expected_per_band):
    out = Path(output_root) / f"{name}_{target_episodes}"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {out}")
    t0 = time.time()
    print(f"\n=== Generating {name}: {target_episodes} episodes, seed_start={seed_start}, hard={include_hard} ===", flush=True)
    token_data.generate_token_dataset(
        output_dir=str(out),
        target_episodes=int(target_episodes),
        seed_start=int(seed_start),
        include_hard_states=bool(include_hard),
    )
    elapsed = time.time() - t0
    stats = json.loads((out / "dataset_stats.json").read_text(encoding="utf-8"))
    counts = stats.get("trajectory_band_counts", {})
    if counts != expected_per_band:
        raise RuntimeError(f"{name}: band counts {counts} != expected {expected_per_band}")
    if int(stats.get("num_episodes", -1)) != int(target_episodes):
        raise RuntimeError(f"{name}: episode count mismatch")
    size_bytes = sum(f.stat().st_size for f in out.glob("*") if f.is_file())
    result = {
        "split": name,
        "episodes": int(target_episodes),
        "seed_start": int(seed_start),
        "include_hard_states": bool(include_hard),
        "band_counts": counts,
        "samples": int(stats.get("total_samples", 0)),
        "arrival_label_counts": stats.get("arrival_label_counts"),
        "arrival_train_mask_counts": stats.get("arrival_train_mask_counts"),
        "sample_type_counts": stats.get("sample_type_counts"),
        "elapsed_s": elapsed,
        "size_bytes": size_bytes,
        "path": str(out),
    }
    print(f"Generated {name}: {elapsed:.1f}s, {size_bytes/1e9:.2f} GB, stats={result['sample_type_counts']}", flush=True)
    if name == "final_test":
        _seal(out, name, seed_start)
        result["sealed"] = True
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("runs/v83_spatial_stage2/datasets"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        # 6 episodes = 2 per band; separate seed ranges from all planned splits.
        specs = [
            ("train_smoke", 6, 4_100_000, True, {"short": 2, "medium": 2, "long": 2}),
            ("dev_val_smoke", 6, 4_200_000, True, {"short": 2, "medium": 2, "long": 2}),
            ("final_test_smoke", 6, 4_300_000, False, {"short": 2, "medium": 2, "long": 2}),
        ]
        root = Path("runs/v83_spatial_stage2") / f"smoke_datasets_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    else:
        specs = [
            ("train", 600, 1_000_000, True, {"short": 200, "medium": 200, "long": 200}),
            ("dev_val", 180, 2_000_000, True, {"short": 60, "medium": 60, "long": 60}),
            ("final_test", 300, 3_000_000, False, {"short": 100, "medium": 100, "long": 100}),
        ]
        root = args.output_root
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Process-local only: equal band weights, identical band ranges/actions.
    token_data.TRAIN = dataclasses.replace(
        ORIGINAL_TRAIN, trajectory_mix=EQUAL_MIX
    )
    print("Equal band mix:", EQUAL_MIX, flush=True)
    results = []
    for name, episodes, seed_start, include_hard, expected in specs:
        results.append(generate_split(
            name, episodes, seed_start, include_hard, root, expected
        ))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "equal_band_mix": EQUAL_MIX,
        "requested": results,
        "total_size_bytes": sum(r["size_bytes"] for r in results),
    }
    (root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nGENERATION_COMPLETE")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Output root: {root}", flush=True)


if __name__ == "__main__":
    main()
