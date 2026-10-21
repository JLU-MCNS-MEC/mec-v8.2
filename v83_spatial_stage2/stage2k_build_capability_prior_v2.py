"""Build the frozen v2 MEC capability prior from Dev Val 180.

Input : runs/v83_spatial_stage2/stage2k/quality_dev_val/<model>/metrics.json
Output: runs/v83_spatial_stage2/stage2k/capability_prior_v2.json

The Stage-2j Dev Val 180 uses episode seeds 2,000,000+, separate from the
Stage-2k Association Train/Dev/Test seeds.  The Stage-2j Final Test 300 token
directory is never opened; a prior that claims ``final_test_read != false`` is
rejected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import QUALITY_DIR
from v83_spatial_stage2.stage2k_v2_common import (
    CAPABILITY_PRIOR_PATH,
    build_capability_prior,
    write_capability_prior,
)


def main():
    payload = build_capability_prior(QUALITY_DIR)
    write_capability_prior(CAPABILITY_PRIOR_PATH, payload)
    print(json.dumps({
        "path": str(CAPABILITY_PRIOR_PATH),
        "split": payload["split"],
        "episodes": payload["episodes"],
        "metrics": payload["metrics"],
        "final_test_read": payload["final_test_read"],
        "models": {
            name: {
                metric: payload["models"][name][metric]
                for metric in payload["metrics"]
            }
            for name in payload["models"]
        },
        "rank_by_ndtw": payload["rank_by_ndtw"],
    }, indent=2, ensure_ascii=False), flush=True)
    print("STAGE2K_CAPABILITY_PRIOR_V2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
