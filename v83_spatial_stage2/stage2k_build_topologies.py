"""Build the three fixed Stage-2k deployment scenarios for each split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    SPLITS,
    build_topology,
    load_split_specs,
    read_json,
    save_topology,
    seed_isolation_report,
)


def main():
    quality = read_json(QUALITY_PRIOR_PATH)
    if quality is None:
        raise SystemExit("Run stage2k_build_quality_prior.py first")
    spec_payloads = {}
    for split in SPLITS:
        try:
            spec_payloads[split] = load_split_specs(split)
        except FileNotFoundError:
            print(f"[topologies] no {split} specs yet")
    topology_payloads = {}
    for split in SPLITS:
        if split not in spec_payloads:
            continue
        for scenario in SCENARIOS:
            payload = build_topology(scenario, split, quality)
            path = save_topology(payload)
            topology_payloads[(split, scenario)] = payload
            print(
                f"[topologies] {split}/{scenario}: {path} "
                f"seed={payload['topology_seed']}",
                flush=True,
            )
    report = seed_isolation_report(spec_payloads, topology_payloads)
    print(json.dumps(report, indent=2), flush=True)
    if not report["ok"]:
        raise SystemExit("Seed overlap detected")
    print("STAGE2K_TOPOLOGIES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
