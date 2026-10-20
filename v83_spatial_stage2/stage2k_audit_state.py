"""Audit the actual Stage-2k 478-D state on one Dev episode per scenario.

This is a read-only inspection.  It builds a single EpisodeRunner, prints the
20 candidate feature rows before/after normalization, checks the action-index
mapping and the previous-selection flag, and saves the full tables under
``runs/v83_spatial_stage2/stage2k/state_audit/``.  No training is started and
no checkpoint is modified.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    QUALITY_PRIOR_PATH,
    SCENARIOS,
    VISUAL_PCA_PATH,
    load_split_specs,
    load_topology,
    read_json,
)
from v83_spatial_stage2.stage2k_runtime import (
    EpisodeRunner,
    build_scene,
    load_gab_config,
    load_runtime,
)
from v83_spatial_stage2.stage2k_state import (
    CANDIDATE_DIM,
    GLOBAL_DIM,
    STATE_DIM,
    VISUAL_DIM,
    VisualProjector,
    state_layout,
)

AUDIT_DIR = Path("runs/v83_spatial_stage2/stage2k/state_audit")
FIELD_NAMES = state_layout()["candidate_fields"]
RAW_NAMES = (
    "dx_m", "dy_m", "distance_m", "bandwidth_mhz", "compute_tflops",
    "uplink_mbps", "communication_ms", "inference_ms", "e2e_ms",
    "quality_prior",
)


def flat_quality(quality):
    return {
        name: float(quality["models"][name]["mean_ndtw"])
        for name in quality["models"]
    }


def build_debug_rows(runner, quality):
    nodes = runner.nodes
    metrics = runner.step_data["latency_metrics"]
    features = runner.step_data["candidate_features"]
    mask = runner.step_data["mask"]
    uav = np.asarray(runner.step_data["pose"], dtype=np.float64)[:2]
    rows = []
    for index, node in enumerate(nodes):
        dx = float(node["x"]) - float(uav[0])
        dy = float(node["y"]) - float(uav[1])
        metric = metrics[index]
        rows.append({
            "index": index,
            "node_id": node["node_id"],
            "model_type": node["model_type"],
            "model_index": int(node["model_index"]),
            "available": int(mask[index]),
            "dx_m": dx,
            "dy_m": dy,
            "distance_m": float(np.hypot(dx, dy)),
            "bandwidth_mhz": float(node["bandwidth_hz"]) / 1e6,
            "compute_tflops": float(
                node["compute_capacity_flops_per_s"]
            ) / 1e12,
            "uplink_mbps": (
                float(metric["uplink_rate_mbps"]) if metric else None
            ),
            "communication_ms": (
                float(metric["communication_ms"]) if metric else None
            ),
            "inference_ms": (
                float(metric["inference_ms"]) if metric else None
            ),
            "e2e_ms": float(metric["total_ms"]) if metric else None,
            "quality_prior": float(
                quality.get(node["model_type"], node.get("quality_prior", 0.0))
            ),
            **{
                f"f_{name}": float(features[index, position])
                for position, name in enumerate(FIELD_NAMES)
            },
        })
    return rows


def field_stats(rows):
    available = [row for row in rows if row["available"]]
    stats = {}
    for position, name in enumerate(FIELD_NAMES):
        values = np.asarray(
            [row[f"f_{name}"] for row in available], dtype=np.float64
        )
        stats[name] = {
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std()) if len(values) else None,
            "n_available": int(len(values)),
        }
    return stats


def checks_for(runner, rows, quality):
    nodes = runner.nodes
    features = runner.step_data["candidate_features"]
    mask = runner.step_data["mask"]
    metrics = runner.step_data["latency_metrics"]
    checks = {
        "state_dim": int(runner.step_data["state"].shape[0]),
        "state_dim_expected": int(STATE_DIM),
        "candidate_shape": list(features.shape),
        "visual_dim": int(VISUAL_DIM),
        "global_dim": int(GLOBAL_DIM),
        "candidate_dim": int(CANDIDATE_DIM),
        "num_available": int(mask.sum()),
        "over_distance_radius": int(np.sum([
            row["distance_m"] > 500.0 for row in rows
        ])),
        "e2e_clipped_nodes": int(np.sum([
            row["e2e_ms"] is not None and row["e2e_ms"] > 300.0
            for row in rows
        ])),
        "all_finite": bool(np.all(np.isfinite(runner.step_data["state"]))),
        "one_hot_match": True,
        "quality_match": True,
        "e2e_match": True,
        "unavailable_network_zero": True,
        "mask_matches_distance": True,
    }
    for index, node in enumerate(nodes):
        expected_one_hot = np.zeros(len(MODEL_NAMES), dtype=np.float32)
        expected_one_hot[int(node["model_index"])] = 1.0
        if not np.allclose(features[index, 12:17], expected_one_hot):
            checks["one_hot_match"] = False
        if not np.isclose(
            features[index, 10],
            quality.get(node["model_type"], node.get("quality_prior", 0.0)),
            atol=1e-6,
        ):
            checks["quality_match"] = False
        available = metrics[index] is not None
        if bool(mask[index]) != bool(available):
            checks["mask_matches_distance"] = False
        if available:
            expected_e2e = min(metrics[index]["total_ms"] / 100.0, 3.0)
            if not np.isclose(features[index, 9], expected_e2e, atol=1e-6):
                checks["e2e_match"] = False
        else:
            if not np.allclose(features[index, 6:10], 0.0):
                checks["unavailable_network_zero"] = False
    checks["ok"] = bool(
        checks["state_dim"] == STATE_DIM
        and checks["candidate_shape"] == [20, CANDIDATE_DIM]
        and checks["all_finite"]
        and checks["one_hot_match"]
        and checks["quality_match"]
        and checks["e2e_match"]
        and checks["unavailable_network_zero"]
        and checks["mask_matches_distance"]
    )
    return checks


def print_table(rows):
    print(
        " idx node model             av  dx_m   dy_m  dist  bwMHz  TFLOPS "
        "upMbps comm  inf   e2e  quality  e2e_n prev onehot"
    )
    for row in rows:
        onehot = "".join(
            str(int(row[f"f_model_{name}"])) for name in MODEL_NAMES
        )
        uplink = (
            f"{row['uplink_mbps']:6.1f}" if row["uplink_mbps"] is not None
            else "  mask"
        )
        print(
            f"{row['index']:3d} {row['node_id']:4s} {row['model_type']:<16s} "
            f"{row['available']:2d} {row['dx_m']:7.1f} {row['dy_m']:7.1f} "
            f"{row['distance_m']:5.1f} {row['bandwidth_mhz']:5.1f} "
            f"{row['compute_tflops']:6.2f} {uplink:>7s} "
            f"{row['communication_ms'] if row['communication_ms'] is not None else 0:5.1f} "
            f"{row['inference_ms'] if row['inference_ms'] is not None else 0:5.2f} "
            f"{row['e2e_ms'] if row['e2e_ms'] is not None else 0:6.1f} "
            f"{row['quality_prior']:.4f} {row['f_e2e_ms_norm']:5.3f} "
            f"{int(row['f_previous_selected'])} {onehot}"
        )


def main():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    quality_payload = read_json(QUALITY_PRIOR_PATH)
    if quality_payload is None:
        raise SystemExit("Missing quality prior")
    quality = flat_quality(quality_payload)
    specs_payload = load_split_specs("dev")
    topologies = {
        scenario: load_topology(scenario, "dev") for scenario in SCENARIOS
    }
    projector = VisualProjector.load(VISUAL_PCA_PATH)
    device, encoder, backends = load_runtime("cuda")
    gab_config = load_gab_config()

    summary = {
        "state_dim_expected": int(STATE_DIM),
        "candidate_fields": FIELD_NAMES,
        "quality_prior": quality,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        spec = specs_payload["scenarios"][scenario]["episodes"][0]
        scene = build_scene(spec, encoder, device)
        runner = EpisodeRunner(
            spec=spec,
            topology=topologies[scenario],
            encoder=encoder,
            backends=backends,
            projector=projector,
            quality_prior=quality,
            gab_config=gab_config,
            scene=scene,
            device=device,
        )
        rows = build_debug_rows(runner, quality)
        checks = checks_for(runner, rows, quality)
        fieldnames = list(rows[0].keys())
        path = AUDIT_DIR / f"{scenario}_step000.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n=== {scenario} | {spec['episode_uid']} | step 0 ===")
        print("layout:", json.dumps(state_layout(), ensure_ascii=False))
        print_table(rows)
        print("field stats over available nodes:")
        for name, item in field_stats(rows).items():
            if item["n_available"] == 0:
                continue
            print(
                f"  {name:24s} min={item['min']:.6f} "
                f"max={item['max']:.6f} mean={item['mean']:.6f} "
                f"std={item['std']:.6f} n={item['n_available']}"
            )
        print("checks:", json.dumps(checks, ensure_ascii=False))

        # One step with the fastest available node.  This verifies the action
        # index -> node mapping and the previous_selected flag in the next
        # state.  It does not run a full episode.
        fastest = runner.select_node("min_latency")
        before_global = runner.step_data["state"][128:138].copy()
        transition = runner.apply(fastest)
        next_features = (
            runner.step_data["candidate_features"]
            if not transition["done"] else None
        )
        previous_flags = (
            next_features[:, 11].tolist() if next_features is not None else None
        )
        after_global = (
            runner.step_data["state"][128:138].copy()
            if not transition["done"] else None
        )
        step_check = {
            "selected_index": int(fastest),
            "selected_node_id": topologies[scenario]["nodes"][fastest]["node_id"],
            "selected_model": topologies[scenario]["nodes"][fastest]["model_type"],
            "transition_node_index": int(transition["node_index"]),
            "index_matches": int(fastest) == int(transition["node_index"]),
            "next_previous_selected_flags": previous_flags,
            "next_previous_selected_is_one": (
                previous_flags is not None
                and int(np.sum(previous_flags)) == 1
                and previous_flags[fastest] == 1.0
            ),
            "step_e2e_ms": float(transition["e2e_ms"]),
            "global_before": before_global.tolist(),
            "global_after": after_global.tolist() if after_global is not None else None,
            "global_previous_e2e_expected": min(
                float(transition["e2e_ms"]) / 100.0, 3.0
            ),
        }
        print("one-step action check:", json.dumps(step_check, ensure_ascii=False))
        summary["scenarios"][scenario] = {
            "episode_uid": spec["episode_uid"],
            "candidate_csv": str(path),
            "checks": checks,
            "field_stats": field_stats(rows),
            "step_check": step_check,
        }

    summary_path = AUDIT_DIR / "state_audit.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWROTE {summary_path}")


if __name__ == "__main__":
    main()
