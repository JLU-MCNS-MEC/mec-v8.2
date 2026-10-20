"""Generate the Stage-2k train / dev / sealed test episode lists.

Only the toy environment's oracle expert is used to accept natural episodes;
no token shard or Stage-2j Final Test data is read.  The test list is generated
and sealed now, but is not evaluated until Stage F.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    SCENARIOS,
    BANDS,
    build_split_specs,
    save_split_specs,
    seed_isolation_report,
    specs_path,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "dev", "test"]
    )
    args = parser.parse_args()

    payloads = {}
    for split in args.splits:
        print(f"[specs] generating {split}", flush=True)
        specs = build_split_specs(split)
        payloads[split] = save_split_specs(
            split, specs, sealed=(split == "test")
        )
        path = specs_path(split)
        counts = {
            scenario: {
                band: sum(
                    1 for episode in specs[scenario]
                    if episode["band"] == band
                )
                for band in BANDS
            }
            for scenario in SCENARIOS
        }
        print(
            f"[specs] {split}: {sum(len(v) for v in specs.values())} episodes, "
            f"{path} ({path.stat().st_size} bytes)",
            flush=True,
        )
        print(json.dumps(counts, indent=2), flush=True)

    report = seed_isolation_report(payloads, {})
    print(json.dumps(report, indent=2), flush=True)
    if not report["ok"]:
        raise SystemExit("Seed overlap detected; refusing to continue")
    print("STAGE2K_SPECS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
