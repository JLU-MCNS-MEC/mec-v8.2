"""Build the Stage-2l fixed city, task pools, order and deployment topologies.

Run once before training:

    python v83_spatial_stage2/stage2l_build.py

The script writes everything under
``runs/v83_spatial_stage2/stage2l_fixed_city/`` and refuses to overwrite a
frozen protocol unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2l_fixed_city import (
    STAGE2L_ROOT,
    build_all,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if protocol_freeze.json already exists.",
    )
    args = parser.parse_args()
    freeze = STAGE2L_ROOT / "protocol_freeze.json"
    if freeze.is_file() and not args.force:
        raise SystemExit(
            f"{freeze} already exists; use --force to regenerate"
        )
    result = build_all(verbose=True)
    print(json.dumps(result["isolation"]["checks"], indent=2), flush=True)
    print("STAGE2L_BUILD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
