"""Stage-2j Final Test 300 aggregation, bootstrap CIs and paper figures.

Reads only the five completed ``eval_final_test_300`` directories.  Produces:

  * summary.json / summary.md
  * band_sr_table.md (Short / Medium / Long)
  * bootstrap_ci.csv and summary.csv
  * four 300-dpi PNG + SVG bar charts in
    ``v83_spatial_stage2/figures/stage2j/``

Error bars are 95% percentile CIs from 10,000 episode-level bootstrap
resamples, using one shared resampling index set for all five models.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2j_common import (
    BANDS,
    MODEL_TITLES,
    STAGE2J_ROOT,
    TABLE_ORDER,
    model_dir,
)


FIGURE_DIR = Path("v83_spatial_stage2/figures/stage2j")
EVAL_DIR_NAME = "eval_final_test_300"
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20264017

# Validated two-slot emphasis palette (dataviz skill reference palette).
COLOR_NEW = "#2a78d6"
COLOR_REFERENCE = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

METRICS = ("sr", "osr", "spl", "ndtw")
METRIC_TITLES = {
    "sr": "Success Rate (SR)",
    "osr": "Oracle Success Rate (OSR)",
    "spl": "Success weighted by Path Length (SPL)",
    "ndtw": "Normalized Dynamic Time Warping (nDTW)",
}
METRIC_YLIM = {"sr": (0.0, 100.0), "osr": (0.0, 100.0),
               "spl": (0.0, 1.0), "ndtw": (0.0, 1.0)}
METRIC_FILENAME = {
    "sr": "MEC_SR_Comparison",
    "osr": "MEC_OSR_Comparison",
    "spl": "MEC_SPL_Comparison",
    "ndtw": "MEC_nDTW_Comparison",
}

EPISODE_FIELDS = (
    "success", "osr", "spl", "ndtw", "premature_stop",
    "max_steps_termination", "steps", "min_distance_m",
)


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_results(root=STAGE2J_ROOT):
    progress = read_json(Path(root) / "progress.json", {})
    results = {}
    missing = []
    for name in TABLE_ORDER:
        directory = Path(root) / name / EVAL_DIR_NAME
        episode_path = directory / f"{name}_per_episode.csv"
        metrics_path = directory / "metrics.json"
        if not episode_path.is_file() or not metrics_path.is_file():
            missing.append(name)
            continue
        rows = read_csv_rows(episode_path)
        if len(rows) != 300:
            raise ValueError(
                f"{name}: expected 300 test episodes, found {len(rows)}"
            )
        for row in rows:
            row["episode_index"] = int(row["episode_index"])
            for field in EPISODE_FIELDS:
                row[field] = float(row[field])
        rows.sort(key=lambda row: row["episode_index"])
        metrics = read_json(metrics_path)
        entry = progress.get("models", {}).get(name, {})
        training_result = read_json(
            Path(root) / name / "training_result.json", {}
        )
        best_epoch = entry.get("best_epoch")
        if best_epoch is None:
            best_epoch = training_result.get("best_epoch")
        training_time_min = entry.get("training_time_min")
        if training_time_min is None:
            training_time_s = training_result.get("training_time_s")
            training_time_min = (
                float(training_time_s) / 60.0
                if training_time_s is not None else 0.0
            )
        results[name] = {
            "model": name,
            "title": MODEL_TITLES[name],
            "architecture": metrics["meta"]["architecture"],
            "parameters": int(metrics["meta"]["parameters"]),
            "best_epoch": best_epoch,
            "training_time_min": float(training_time_min),
            "metrics": metrics["overall"],
            "bands": metrics["bands"],
            "retrieval": metrics["retrieval"],
            "latency": metrics["latency"],
            "rows": rows,
            "episode_indices": [row["episode_index"] for row in rows],
            "eval_dir": str(directory),
            "checkpoint_sha256": metrics["meta"]["checkpoint_sha256"],
            "sealed_manifest_sha256": metrics["meta"].get(
                "sealed_manifest_sha256"),
            "test_spec_sha256": metrics["meta"].get("test_spec_sha256"),
        }
    if missing:
        return results, missing
    reference = results[TABLE_ORDER[0]]["episode_indices"]
    for name in TABLE_ORDER:
        if results[name]["episode_indices"] != reference:
            raise ValueError(f"{name}: episode index set/order differs")
    return results, []


def _metric_vector(rows, metric):
    if metric in ("sr", "osr"):
        return np.asarray([row["success" if metric == "sr" else "osr"]
                           for row in rows], dtype=np.float64)
    return np.asarray([row[metric] for row in rows], dtype=np.float64)


def bootstrap_results(results, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    n = len(results[TABLE_ORDER[0]]["rows"])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, n, size=(int(reps), n))
    output = {
        "reps": int(reps),
        "seed": int(seed),
        "episodes": int(n),
        "shared_resamples": True,
        "metrics": {},
    }
    for metric in METRICS:
        block = {}
        for name in TABLE_ORDER:
            values = _metric_vector(results[name]["rows"], metric)
            boots = values[indices].mean(axis=1)
            block[name] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.percentile(boots, 2.5)),
                "ci95_high": float(np.percentile(boots, 97.5)),
                "n": int(n),
            }
        output["metrics"][metric] = block
    # Paired deltas against Transformer-M and Small, same resamples.
    paired = {"vs_Transformer-M": {}, "vs_Small": {}}
    for base, key in (("Transformer-M", "vs_Transformer-M"),
                      ("Small", "vs_Small")):
        for other in TABLE_ORDER:
            if other == base:
                continue
            entry = {}
            for metric in METRICS:
                a = _metric_vector(results[other]["rows"], metric)
                b = _metric_vector(results[base]["rows"], metric)
                diff = a - b
                boots = diff[indices].mean(axis=1)
                entry[metric] = {
                    "delta": float(diff.mean()),
                    "ci95_low": float(np.percentile(boots, 2.5)),
                    "ci95_high": float(np.percentile(boots, 97.5)),
                }
            paired[key][other] = entry
    output["paired_deltas"] = paired
    return output


def band_table_markdown(results):
    lines = [
        "## Short / Medium / Long SR",
        "",
        "| MEC | Short SR | Medium SR | Long SR |",
        "|---|---:|---:|---:|",
    ]
    for name in TABLE_ORDER:
        bands = results[name]["bands"]
        lines.append(
            f"| {name} | {100*bands['short']['sr']:.2f}% | "
            f"{100*bands['medium']['sr']:.2f}% | "
            f"{100*bands['long']['sr']:.2f}% |"
        )
    lines.extend(["", "## Full band metrics", ""])
    lines.append(
        "| MEC | Band | SR | OSR | SPL | nDTW | Premature | Timeout |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name in TABLE_ORDER:
        for band in BANDS:
            item = results[name]["bands"][band]
            lines.append(
                f"| {name} | {band.capitalize()} | "
                f"{100*item['sr']:.2f}% | {100*item['osr']:.2f}% | "
                f"{item['spl']:.4f} | {item['ndtw']:.4f} | "
                f"{100*item['premature_stop']:.2f}% | "
                f"{100*item['timeout']:.2f}% |"
            )
    return "\n".join(lines)


def summary_markdown(results, bootstrap):
    lines = [
        "# Stage-2j summary - Final Test 300",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "All numbers are closed-loop Final Test 300 results. "
        "Stop threshold fixed at 0.7; no test-time tuning.",
        "",
        "## Overall",
        "",
        "| MEC | Params | SR | OSR | SPL | nDTW | Premature | Timeout | "
        "Steps | Best Epoch | Train min | MEC forward ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in TABLE_ORDER:
        item = results[name]
        metrics = item["metrics"]
        latency = item["latency"]["mec_forward"]["mean_ms"]
        lines.append(
            f"| {name} | {item['parameters']:,} | "
            f"{100*metrics['sr']:.2f}% | {100*metrics['osr']:.2f}% | "
            f"{metrics['spl']:.4f} | {metrics['ndtw']:.4f} | "
            f"{100*metrics['premature_stop']:.2f}% | "
            f"{100*metrics['timeout']:.2f}% | {metrics['steps']:.1f} | "
            f"{item['best_epoch']} | {item['training_time_min']:.0f} | "
            f"{latency:.3f} |"
        )
    lines.extend(["", "## Bootstrap 95% CI", ""])
    lines.append("| MEC | SR CI | OSR CI | SPL CI | nDTW CI |")
    lines.append("|---|---:|---:|---:|---:|")

    def _fmt(metric, block):
        scale = 100.0 if metric in ("sr", "osr") else 1.0
        return (
            f"{scale*block['mean']:.2f}"
            f" [{scale*block['ci95_low']:.2f}, "
            f"{scale*block['ci95_high']:.2f}]"
        )

    for name in TABLE_ORDER:
        blocks = bootstrap["metrics"]
        lines.append(
            f"| {name} | {_fmt('sr', blocks['sr'][name])} | "
            f"{_fmt('osr', blocks['osr'][name])} | "
            f"{_fmt('spl', blocks['spl'][name])} | "
            f"{_fmt('ndtw', blocks['ndtw'][name])} |"
        )
    lines.extend(["", band_table_markdown(results), ""])
    lines.extend(["## Retrieval and latency", ""])
    lines.append(
        "| MEC | Top-1 | Top-5 | Mean goal cosine | MEC forward ms | "
        "p50 ms | p95 ms | Decision pipeline ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in TABLE_ORDER:
        item = results[name]
        retrieval = item["retrieval"]
        latency = item["latency"]
        lines.append(
            f"| {name} | {100*retrieval['clean_top1']:.2f}% | "
            f"{100*retrieval['clean_top5']:.2f}% | "
            f"{retrieval['clean_mean_goal_similarity']:.7f} | "
            f"{latency['mec_forward']['mean_ms']:.3f} | "
            f"{latency['mec_forward']['p50_ms']:.3f} | "
            f"{latency['mec_forward']['p95_ms']:.3f} | "
            f"{latency['decision_pipeline']['mean_ms']:.3f} |"
        )
    lines.extend(["", "## Paired deltas vs Transformer-M", ""])
    lines.append("| MEC | Delta SR | Delta SPL | Delta nDTW |")
    lines.append("|---|---:|---:|---:|")
    for name in TABLE_ORDER:
        if name == "Transformer-M":
            continue
        entry = bootstrap["paired_deltas"]["vs_Transformer-M"][name]
        lines.append(
            f"| {name} | {100*entry['sr']['delta']:+.2f} pp | "
            f"{entry['spl']['delta']:+.4f} | "
            f"{entry['ndtw']['delta']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _setup_matplotlib():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT,
        "ytick.color": TEXT_MUTED,
        "svg.fonttype": "none",
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })


def _x_labels():
    return ["Small", "Transformer-\nS", "Medium", "Transformer-\nM", "Large"]


def plot_metric(results, bootstrap, metric, output_dir=FIGURE_DIR):
    _setup_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = [
        results[name]["metrics"][metric] * (100.0 if metric in ("sr", "osr")
                                            else 1.0)
        for name in TABLE_ORDER
    ]
    ci = bootstrap["metrics"][metric]
    scale = 100.0 if metric in ("sr", "osr") else 1.0
    lows = [scale * ci[name]["ci95_low"] for name in TABLE_ORDER]
    highs = [scale * ci[name]["ci95_high"] for name in TABLE_ORDER]
    colors = [
        COLOR_REFERENCE if name == "Transformer-M" else COLOR_NEW
        for name in TABLE_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.45), dpi=300)
    x = np.arange(len(TABLE_ORDER))
    bars = ax.bar(
        x, values, width=0.56, color=colors, edgecolor="none", zorder=3,
    )
    ax.errorbar(
        x, values,
        yerr=[np.array(values) - np.array(lows),
              np.array(highs) - np.array(values)],
        fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1.0,
        capsize=4, capthick=1.0, zorder=4,
    )

    ymin, ymax = METRIC_YLIM[metric]
    ax.set_ylim(ymin, ymax)
    headroom = (ymax - ymin) * 0.035
    for xi, value, high in zip(x, values, highs):
        label = f"{value:.2f}%" if metric in ("sr", "osr") else f"{value:.4f}"
        ax.text(
            xi, min(high + headroom, ymax - headroom), label,
            ha="center", va="bottom", fontsize=9.5, color=TEXT,
            fontweight="semibold", zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(_x_labels(), fontsize=9.2, color=TEXT)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.tick_params(axis="y", length=0, pad=6, labelsize=8.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_ylabel(
        "Percent" if metric in ("sr", "osr") else "Score (0-1)",
        fontsize=9, color=TEXT_SECONDARY,
    )

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_NEW, edgecolor="none",
              label="Newly trained MEC"),
        Patch(facecolor=COLOR_REFERENCE, edgecolor="none",
              label="Transformer-M (Stage-2i T5 reference)"),
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        ncol=2, frameon=False, fontsize=8.2, handlelength=1.2,
        handleheight=0.9, borderaxespad=0.2, columnspacing=1.4,
    )

    fig.text(
        0.075, 0.955, METRIC_TITLES[metric], fontsize=13.5,
        fontweight="bold", color=TEXT, ha="left", va="top",
    )
    fig.text(
        0.075, 0.905,
        "Final Test 300 episodes (100 Short / 100 Medium / 100 Long) - "
        "five MEC architectures, identical T5 inputs",
        fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="top",
    )
    fig.text(
        0.075, 0.03,
        "Error bars: 95% episode bootstrap CI (10,000 resamples). "
        "Stop threshold fixed at 0.7; no test-time tuning.",
        fontsize=7.5, color=TEXT_MUTED, ha="left", va="bottom",
    )
    fig.subplots_adjust(top=0.80, bottom=0.21, left=0.095, right=0.975)

    png = output_dir / f"{METRIC_FILENAME[metric]}.png"
    svg = output_dir / f"{METRIC_FILENAME[metric]}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    results, missing = load_results(STAGE2J_ROOT)
    if missing:
        print(
            "Pending models (no final-test result): " + ", ".join(missing),
            flush=True,
        )
        print("Figures and summary will be generated after all five finish.",
              flush=True)
        return None
    bootstrap = bootstrap_results(
        results, reps=int(args.bootstrap_reps), seed=int(args.bootstrap_seed)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "Stage-2j Final Test 300",
        "episodes": 300,
        "bands": {band: 100 for band in BANDS},
        "stop_threshold": 0.7,
        "order": list(TABLE_ORDER),
        "models": {
            name: {
                key: value for key, value in results[name].items()
                if key not in ("rows", "episode_indices")
            }
            for name in TABLE_ORDER
        },
        "bootstrap": bootstrap,
        "retrieval_all_models_identical": all(
            results[name]["retrieval"] == results["Small"]["retrieval"]
            for name in TABLE_ORDER
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    markdown = summary_markdown(results, bootstrap)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    band_md = band_table_markdown(results)
    (output_dir / "band_sr_table.md").write_text(band_md, encoding="utf-8")
    (Path("v83_spatial_stage2") / "STAGE2J_BAND_SR_TABLE.md").write_text(
        band_md + "\n", encoding="utf-8"
    )

    ci_rows = []
    for name in TABLE_ORDER:
        row = {"model": name}
        for metric in METRICS:
            block = bootstrap["metrics"][metric][name]
            row[f"{metric}_mean"] = block["mean"]
            row[f"{metric}_ci95_low"] = block["ci95_low"]
            row[f"{metric}_ci95_high"] = block["ci95_high"]
        ci_rows.append(row)
    write_csv(output_dir / "bootstrap_ci.csv", ci_rows)
    write_csv(
        output_dir / "summary.csv",
        [
            {
                "model": name,
                **{
                    key: results[name]["metrics"][key]
                    for key in (
                        "sr", "osr", "spl", "ndtw", "premature_stop",
                        "timeout", "steps",
                    )
                },
                "parameters": results[name]["parameters"],
                "best_epoch": results[name]["best_epoch"],
                "training_time_min": results[name]["training_time_min"],
                "mec_forward_ms": results[name]["latency"][
                    "mec_forward"]["mean_ms"],
                "retrieval_top1": results[name]["retrieval"]["clean_top1"],
                "retrieval_top5": results[name]["retrieval"]["clean_top5"],
            }
            for name in TABLE_ORDER
        ],
    )

    # Committed copies next to the figures.
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "summary.md", "bootstrap_ci.csv",
                 "summary.csv"):
        (FIGURE_DIR / name).write_text(
            (output_dir / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    for metric in METRICS:
        png, svg = plot_metric(results, bootstrap, metric)
        print(f"figure: {png} | {svg}", flush=True)

    print(markdown, flush=True)
    print(f"\nSummary written to {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=STAGE2J_ROOT / "analysis")
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int,
                        default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
