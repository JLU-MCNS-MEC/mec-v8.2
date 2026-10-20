"""Fit the frozen 128-D visual PCA on Train-600 natural states only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import VISUAL_PCA_PATH
from v83_spatial_stage2.stage2k_state import fit_visual_pca_from_train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir", type=Path,
        default=Path("runs/v83_spatial_stage2/datasets/train_600"),
    )
    parser.add_argument(
        "--aux-dir", type=Path,
        default=Path("runs/v83_spatial_stage2/stage2i/aux/train"),
    )
    parser.add_argument("--output", type=Path, default=VISUAL_PCA_PATH)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    payload = fit_visual_pca_from_train(
        train_dir=args.train_dir,
        aux_dir=args.aux_dir,
        output_path=args.output,
        n_samples=args.samples,
        seed=args.seed,
    )
    summary = {
        "path": str(args.output),
        "n_samples": payload["n_samples"],
        "summary_dim": payload["summary_dim"],
        "components_dim": payload["components_dim"],
        "explained_variance_total": payload[
            "explained_variance_total"
        ],
        "explained_variance_first_10": payload[
            "explained_variance_ratio"
        ][:10].tolist(),
        "fitted_split": payload["fitted_split"],
        "final_test_read": payload["final_test_read"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    print("STAGE2K_PCA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
