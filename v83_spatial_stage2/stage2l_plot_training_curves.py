"""Extract, plot and analyse the four completed Stage-2l training runs.

Data sources (read-only):
  runs/v83_spatial_stage2/stage2l_fixed_city/{A,C}/{dqn,ppo}/training_history.json
  runs/v83_spatial_stage2/stage2l_fixed_city/{A,C}/{dqn,ppo}/progress.json
  runs/v83_spatial_stage2/stage2l_fixed_city/specs/training_order.json
  runs/v83_spatial_stage2/stage2l_fixed_city/specs/train_tasks.json

Outputs:
  v83_spatial_stage2/figures/stage2l_curves/  (PNG + SVG)
  v83_spatial_stage2/figures/stage2l_curves/data/  (CSV)
  v83_spatial_stage2/STAGE2L_TRAINING_CURVES_ANALYSIS.md

No training process, checkpoint or history file is modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path("runs/v83_spatial_stage2/stage2l_fixed_city")
OUT_DIR = Path("v83_spatial_stage2/figures/stage2l_curves")
DATA_DIR = OUT_DIR / "data"
REPORT_PATH = Path("v83_spatial_stage2/STAGE2L_TRAINING_CURVES_ANALYSIS.md")

AGENTS = (
    ("A", "dqn", "A/DQN"),
    ("A", "ppo", "A/PPO"),
    ("C", "dqn", "C/DQN"),
    ("C", "ppo", "C/PPO"),
)
AGENT_LABEL = {(c, a): label for c, a, label in AGENTS}
CONFIG_TITLE = {
    "A": "A：固定城市 + Balanced 单张固定 MEC 拓扑",
    "C": "C：固定城市 + Balanced / Quality-rich / Tradeoff-stress 三种部署，各一张固定拓扑",
}
CONFIG_SHORT = {
    "A": "A · Balanced 单固定拓扑",
    "C": "C · 三部署场景固定拓扑",
}

# Validated light-mode categorical palette (dataviz skill validator):
# blue/orange/aqua pass all-pairs; the 5-color model palette passes adjacent.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
RAW = "#b9b7ae"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd5"
MODEL_ORDER = ("Small", "Transformer-S", "Medium", "Transformer-M", "Large")
MODEL_COLORS = dict(zip(MODEL_ORDER, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA)))

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "svg.fonttype": "path",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data():
    order_payload = json.loads(
        (RUN_ROOT / "specs" / "training_order.json").read_text(encoding="utf-8")
    )
    order = list(order_payload["order"])
    task_payload = json.loads(
        (RUN_ROOT / "specs" / "train_tasks.json").read_text(encoding="utf-8")
    )
    specs = {item["episode_uid"]: item for item in task_payload["tasks"]}
    if set(order) != set(specs):
        raise RuntimeError("Training order and task specs differ")

    frames: Dict[tuple, pd.DataFrame] = {}
    validations: Dict[tuple, pd.DataFrame] = {}
    for config, algorithm, label in AGENTS:
        base = RUN_ROOT / config / algorithm
        history = json.loads(
            (base / "training_history.json").read_text(encoding="utf-8")
        )
        df = pd.DataFrame(history["train"])
        if len(df) != 3500:
            raise RuntimeError(f"{label} has {len(df)} train rows")
        if list(df["episode_uid"]) != order:
            raise RuntimeError(f"{label} does not share the frozen task order")
        df["label"] = label
        df["config"] = config
        df["algorithm"] = algorithm
        df["ma100"] = df["episode_return"].rolling(100, min_periods=100).mean()
        df["oracle_steps"] = [
            int(specs[uid]["oracle_steps"]) for uid in df["episode_uid"]
        ]
        df["start_goal_distance_m"] = [
            float(specs[uid]["start_goal_distance_m"])
            for uid in df["episode_uid"]
        ]
        df["block"] = ((df["episode"] - 1) // 200).astype(int)
        df["block_start"] = df["block"] * 200 + 1
        df["block_end"] = np.minimum((df["block"] + 1) * 200, 3500)
        frames[(config, algorithm)] = df

        val = pd.DataFrame(history["validation"])
        val["label"] = label
        val["config"] = config
        val["algorithm"] = algorithm
        val["best"] = val["joint_reward"] == val["joint_reward"].max()
        val["final"] = val["episode"] == 3500
        # Keep only the first row if max is tied.
        best_indices = val.index[val["best"]].tolist()
        if len(best_indices) > 1:
            val.loc[best_indices[1:], "best"] = False
        validations[(config, algorithm)] = val

    # The band sequence must be identical across all four agents.
    base_band = list(frames[("A", "dqn")]["band"])
    for key, df in frames.items():
        if list(df["band"]) != base_band:
            raise RuntimeError(f"{AGENT_LABEL[key]} band order differs")
    return order, specs, frames, validations


def block_task_table(order, specs) -> pd.DataFrame:
    rows = []
    for block in range(18):
        uids = order[block * 200:(block + 1) * 200]
        bands = [specs[uid]["band"] for uid in uids]
        rows.append({
            "block": block,
            "start": block * 200 + 1,
            "end": min((block + 1) * 200, 3500),
            "short": bands.count("short"),
            "medium": bands.count("medium"),
            "long": bands.count("long"),
            "oracle_steps_mean": float(np.mean([
                specs[uid]["oracle_steps"] for uid in uids
            ])),
            "start_goal_distance_mean": float(np.mean([
                specs[uid]["start_goal_distance_m"] for uid in uids
            ])),
        })
    return pd.DataFrame(rows)


def per_block_metrics(frames, validations, task_table) -> pd.DataFrame:
    out = task_table.copy()
    for (config, algorithm), df in frames.items():
        label = AGENT_LABEL[(config, algorithm)]
        grp = df.groupby("block")
        out[f"{label}_reward_mean"] = grp["episode_return"].mean().values
        out[f"{label}_success"] = grp["success"].mean().values
        out[f"{label}_steps_mean"] = grp["steps"].mean().values
        out[f"{label}_e2e_mean"] = grp["mean_episode_e2e_ms"].mean().values
        out[f"{label}_ndtw_mean"] = grp["final_nDTW"].mean().values
        out[f"{label}_ma100_end"] = grp["ma100"].last().values
    return out


def correlations(frames, block_table) -> pd.DataFrame:
    keys = list(frames)
    rows = []
    block_series = {
        key: frames[key].groupby("block")["episode_return"].mean()
        for key in keys
    }
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            x_block = block_series[left].values
            y_block = block_series[right].values
            x_ep = frames[left]["episode_return"].values
            y_ep = frames[right]["episode_return"].values
            rows.append({
                "pair": f"{AGENT_LABEL[left]} vs {AGENT_LABEL[right]}",
                "level": "per_200_block",
                "pearson": float(pearsonr(x_block, y_block)[0]),
                "spearman": float(spearmanr(x_block, y_block).correlation),
            })
            rows.append({
                "pair": f"{AGENT_LABEL[left]} vs {AGENT_LABEL[right]}",
                "level": "per_episode",
                "pearson": float(pearsonr(x_ep, y_ep)[0]),
                "spearman": float(spearmanr(x_ep, y_ep).correlation),
            })

    long_fraction = (
        block_table.assign(long=(block_table["long"]).astype(float))
        .assign(long_frac=block_table["long"] / (
            block_table["short"] + block_table["medium"] + block_table["long"]
        ))
    )["long_frac"].values
    for (config, algorithm), df in frames.items():
        label = AGENT_LABEL[(config, algorithm)]
        rew = df.groupby("block")["episode_return"].mean().values
        steps = df.groupby("block")["steps"].mean().values
        success = df.groupby("block")["success"].mean().values
        for name, values in (
            ("reward_vs_actual_steps", steps),
            ("reward_vs_success", success),
            ("reward_vs_oracle_steps", block_table["oracle_steps_mean"].values),
            ("reward_vs_long_fraction", long_fraction),
            ("reward_vs_start_distance", block_table["start_goal_distance_mean"].values),
        ):
            rows.append({
                "pair": f"{label} · {name}",
                "level": "per_200_block",
                "pearson": float(pearsonr(rew, values)[0]),
                "spearman": float(spearmanr(rew, values).correlation),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stats and CSV
# ---------------------------------------------------------------------------


def make_stats(frames, validations, block_table, correlations_df):
    stats = {}
    for (config, algorithm), df in frames.items():
        label = AGENT_LABEL[(config, algorithm)]
        valid = df.dropna(subset=["ma100"])
        peak_row = valid.loc[valid["ma100"].idxmax()]
        late = valid[valid["episode"] >= 2500]
        late_row = late.loc[late["ma100"].idxmax()]
        val = validations[(config, algorithm)]
        best_row = val.loc[val["joint_reward"].idxmax()]
        final_row = val.iloc[-1]
        dqn = algorithm == "dqn"
        stats[label] = {
            "config": config,
            "algorithm": algorithm,
            "peak_ma100": float(peak_row["ma100"]),
            "peak_episode": int(peak_row["episode"]),
            "late_peak_ma100": float(late_row["ma100"]),
            "late_peak_episode": int(late_row["episode"]),
            "final_ma100": float(valid["ma100"].iloc[-1]),
            "final_100_mean": float(df["episode_return"].tail(100).mean()),
            "best_val_jointr": float(best_row["joint_reward"]),
            "best_val_episode": int(best_row["episode"]),
            "final_val_jointr": float(final_row["joint_reward"]),
            "final_val_episode": int(final_row["episode"]),
            "best_checkpoint": str(
                RUN_ROOT / config / algorithm / "checkpoints" / "best.pt"
            ),
            "final_checkpoint": str(
                RUN_ROOT / config / algorithm / "checkpoints" / "last.pt"
            ),
            "early_success": float(df["success"].head(200).mean()),
            "late_success": float(df["success"].tail(200).mean()),
            "early_steps": float(df["steps"].head(200).mean()),
            "late_steps": float(df["steps"].tail(200).mean()),
            "early_reward": float(df["episode_return"].head(200).mean()),
            "late_reward": float(df["episode_return"].tail(200).mean()),
        }
        if dqn:
            stats[label].update({
                "epsilon_start": float(df["epsilon"].iloc[0]),
                "epsilon_final": float(df["epsilon"].iloc[-1]),
                "q_mean_early": float(df["q_mean"].head(200).mean()),
                "q_mean_peak": float(df["q_mean"].max()),
                "q_mean_late": float(df["q_mean"].tail(200).mean()),
                "q_max_peak": float(df["q_max"].max()),
                "loss_early": float(df["loss"].iloc[200:400].mean()),
                "loss_late": float(df["loss"].tail(200).mean()),
            })
        else:
            stats[label].update({
                "entropy_early": float(df["policy_entropy"].head(200).mean()),
                "entropy_late": float(df["policy_entropy"].tail(200).mean()),
                "value_mean_early": float(df["value_mean"].head(200).mean()),
                "value_mean_late": float(df["value_mean"].tail(200).mean()),
                "approx_kl_early": float(val["approx_kl"].head(3).mean()),
                "approx_kl_late": float(val["approx_kl"].tail(3).mean()),
                "value_loss_early": float(val["update_value_loss"].head(3).mean()),
                "value_loss_late": float(val["update_value_loss"].tail(3).mean()),
                "policy_loss_early": float(val["update_policy_loss"].head(3).mean()),
                "policy_loss_late": float(val["update_policy_loss"].tail(3).mean()),
            })
    return stats


def write_csvs(frames, validations, block_table, correlations_df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    for (config, algorithm), df in frames.items():
        for _, row in df.iterrows():
            raw_rows.append({
                "config": config,
                "algorithm": algorithm,
                "agent": AGENT_LABEL[(config, algorithm)],
                "episode": int(row["episode"]),
                "episode_uid": row["episode_uid"],
                "band": row["band"],
                "scenario": row["scenario"],
                "topology_role": row["topology_role"],
                "steps": int(row["steps"]),
                "success": int(bool(row["success"])),
                "episode_return": float(row["episode_return"]),
                "ma100": (
                    None if pd.isna(row["ma100"]) else float(row["ma100"])
                ),
                "oracle_steps": int(row["oracle_steps"]),
                "start_goal_distance_m": float(row["start_goal_distance_m"]),
                "mean_episode_e2e_ms": float(row["mean_episode_e2e_ms"]),
                "final_nDTW": float(row["final_nDTW"]),
                "epsilon": (
                    float(row["epsilon"]) if "epsilon" in row
                    and not pd.isna(row["epsilon"]) else None
                ),
                "q_mean": (
                    float(row["q_mean"]) if "q_mean" in row
                    and not pd.isna(row["q_mean"]) else None
                ),
                "q_max": (
                    float(row["q_max"]) if "q_max" in row
                    and not pd.isna(row["q_max"]) else None
                ),
                "td_loss": (
                    float(row["loss"]) if "loss" in row
                    and not pd.isna(row["loss"]) else None
                ),
                "policy_entropy": (
                    float(row["policy_entropy"]) if "policy_entropy" in row
                    and not pd.isna(row["policy_entropy"]) else None
                ),
                "value_mean": (
                    float(row["value_mean"]) if "value_mean" in row
                    and not pd.isna(row["value_mean"]) else None
                ),
            })
    pd.DataFrame(raw_rows).to_csv(
        DATA_DIR / "training_curves_raw.csv", index=False, float_format="%.6f"
    )

    val_rows = []
    for (config, algorithm), val in validations.items():
        for _, row in val.iterrows():
            base = {
                "config": config,
                "algorithm": algorithm,
                "agent": AGENT_LABEL[(config, algorithm)],
                "episode": int(row["episode"]),
                "joint_reward": float(row["joint_reward"]),
                "SR": float(row["SR"]),
                "OSR": float(row["OSR"]),
                "SPL": float(row["SPL"]),
                "nDTW": float(row["nDTW"]),
                "E2E_ms": float(row["E2E_ms"]),
                "P95_ms": float(row["P95_ms"]),
                "mean_steps": float(row["mean_steps"]),
                "is_best": int(bool(row["best"])),
                "is_final": int(bool(row["final"])),
                "best_checkpoint": (
                    str(RUN_ROOT / config / algorithm
                        / "checkpoints" / "best.pt")
                ),
                "final_checkpoint": (
                    str(RUN_ROOT / config / algorithm
                        / "checkpoints" / "last.pt")
                ),
            }
            for metric in (
                "update_policy_loss", "update_value_loss", "update_entropy",
                "approx_kl", "clip_fraction", "explained_variance",
            ):
                if metric in row and not pd.isna(row[metric]):
                    base[metric] = float(row[metric])
            for model in MODEL_ORDER:
                base[f"usage_{model}"] = float(
                    row.get("model_usage_ratio", {}).get(model, 0.0)
                )
            val_rows.append(base)
    pd.DataFrame(val_rows).to_csv(
        DATA_DIR / "validation_curves.csv", index=False, float_format="%.6f"
    )

    block_table.to_csv(
        DATA_DIR / "per200_task_mix.csv", index=False, float_format="%.6f"
    )
    correlations_df.to_csv(
        DATA_DIR / "correlations.csv", index=False, float_format="%.6f"
    )

    # DQN / PPO per-100 blocks and validation diagnostics.
    dqn_rows, ppo_rows = [], []
    for (config, algorithm), df in frames.items():
        if algorithm != "dqn":
            continue
        df = df.copy()
        df["block100"] = (df["episode"] - 1) // 100
        for block, grp in df.groupby("block100"):
            dqn_rows.append({
                "config": config,
                "agent": AGENT_LABEL[(config, algorithm)],
                "block100": int(block),
                "start": int(block * 100 + 1),
                "end": int(min((block + 1) * 100, 3500)),
                "epsilon_mean": float(grp["epsilon"].mean()),
                "q_mean": float(grp["q_mean"].mean()),
                "q_max": float(grp["q_max"].max()),
                "td_loss": float(grp["loss"].mean()),
            })
    for (config, algorithm), df in frames.items():
        if algorithm != "ppo":
            continue
        df = df.copy()
        df["block100"] = (df["episode"] - 1) // 100
        for block, grp in df.groupby("block100"):
            ppo_rows.append({
                "config": config,
                "agent": AGENT_LABEL[(config, algorithm)],
                "block100": int(block),
                "start": int(block * 100 + 1),
                "end": int(min((block + 1) * 100, 3500)),
                "policy_entropy": float(grp["policy_entropy"].mean()),
                "value_mean": float(grp["value_mean"].mean()),
            })
    pd.DataFrame(dqn_rows).to_csv(
        DATA_DIR / "diagnostics_dqn.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(ppo_rows).to_csv(
        DATA_DIR / "diagnostics_ppo.csv", index=False, float_format="%.6f"
    )

    # Model usage long table.
    usage_rows = []
    for (config, algorithm), val in validations.items():
        for _, row in val.iterrows():
            for model in MODEL_ORDER:
                usage_rows.append({
                    "config": config,
                    "algorithm": algorithm,
                    "agent": AGENT_LABEL[(config, algorithm)],
                    "episode": int(row["episode"]),
                    "model": model,
                    "ratio": float(
                        row.get("model_usage_ratio", {}).get(model, 0.0)
                    ),
                })
    pd.DataFrame(usage_rows).to_csv(
        DATA_DIR / "model_usage.csv", index=False, float_format="%.6f"
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)


def _save(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT_DIR / f"{stem}.png", dpi=180, bbox_inches="tight",
        facecolor=SURFACE,
    )
    fig.savefig(
        OUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor=SURFACE,
    )
    plt.close(fig)


def _agent_color(algorithm):
    return BLUE if algorithm == "dqn" else ORANGE


def _best_final_text(stats):
    return (
        f"best Val JointR = {stats['best_val_jointr']:.4f} @ Ep"
        f"{stats['best_val_episode']}；final = {stats['final_val_jointr']:.4f} @ Ep"
        f"{stats['final_val_episode']}"
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_agent_training(config, algorithm, df, stats):
    label = AGENT_LABEL[(config, algorithm)]
    color = _agent_color(algorithm)
    fig, ax = plt.subplots(figsize=(12.6, 5.6), facecolor=SURFACE)
    _style(ax)
    ax.plot(
        df["episode"], df["episode_return"], color=RAW,
        linewidth=0.55, alpha=0.42, label="逐 Episode Reward",
    )
    valid = df.dropna(subset=["ma100"])
    ax.plot(
        valid["episode"], valid["ma100"], color=color,
        linewidth=2.1, label="100-Episode 移动平均 (MA100)",
    )
    ax.axhline(0.0, color=GRID, linewidth=0.8)
    peak = stats["peak_ma100"]
    peak_ep = stats["peak_episode"]
    final = stats["final_ma100"]
    ax.scatter([peak_ep], [peak], marker="*", s=150, color=color,
               zorder=5, edgecolor=SURFACE, linewidth=0.8)
    ax.annotate(
        f"MA100 峰值 {peak:.4f} @ Ep{peak_ep}\n（峰值 ≠ 收敛）",
        xy=(peak_ep, peak), xytext=(10, 18), textcoords="offset points",
        fontsize=8.2, color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )
    ax.scatter([3500], [final], marker="s", s=55, color=color,
               zorder=5, edgecolor=SURFACE, linewidth=0.8)
    ax.annotate(
        f"最终 MA100 {final:.4f}",
        xy=(3500, final), xytext=(-88, -20), textcoords="offset points",
        fontsize=8.2, color=color,
    )
    text = (
        f"{CONFIG_TITLE[config]}\n"
        f"{algorithm.upper()}；{_best_final_text(stats)}\n"
        f"晚期(≥2500) MA100 峰值 {stats['late_peak_ma100']:.4f} @ Ep"
        f"{stats['late_peak_episode']}；逐 Episode 原始值仍保留"
    )
    ax.text(
        0.012, 0.035, text, transform=ax.transAxes, fontsize=8.0,
        color=MUTED, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                  edgecolor=GRID, alpha=0.92),
    )
    ax.set_title(
        f"{label} 真实训练奖励曲线（固定城市，3500 Episodes）",
        fontsize=12.5, pad=8,
    )
    ax.set_xlabel("训练 Episode", fontsize=9.5)
    ax.set_ylabel("Episode Reward（JointR 对齐）", fontsize=9.5)
    ax.set_xlim(0, 3510)
    ax.legend(loc="upper right", frameon=False, fontsize=8.8)
    fig.tight_layout()
    _save(fig, f"{config.lower()}_{algorithm}_training_curve")


def plot_ma100_comparison(frames, stats):
    fig, axes = plt.subplots(
        2, 1, figsize=(12.6, 7.8), sharex=True, sharey=True,
        facecolor=SURFACE,
    )
    for ax, config in zip(axes, ("A", "C")):
        _style(ax)
        for algorithm in ("dqn", "ppo"):
            df = frames[(config, algorithm)]
            valid = df.dropna(subset=["ma100"])
            label = AGENT_LABEL[(config, algorithm)]
            s = stats[label]
            ax.plot(
                valid["episode"], valid["ma100"],
                color=_agent_color(algorithm), linewidth=2.0,
                label=(
                    f"{algorithm.upper()}（best Val {s['best_val_jointr']:.4f}"
                    f" @ Ep{s['best_val_episode']}）"
                ),
            )
            ax.annotate(
                f"{algorithm.upper()} {s['final_ma100']:.3f}",
                xy=(3500, s["final_ma100"]), xytext=(5, 0),
                textcoords="offset points", fontsize=7.8,
                color=_agent_color(algorithm), va="center",
            )
        ax.axhline(0.0, color=GRID, linewidth=0.8)
        ax.set_title(CONFIG_TITLE[config], fontsize=11.0, pad=6)
        ax.set_ylabel("MA100 Reward", fontsize=9.5)
        ax.legend(loc="upper left", frameon=False, fontsize=8.4)
    axes[-1].set_xlabel("训练 Episode", fontsize=9.5)
    axes[-1].set_xlim(0, 3510)
    fig.suptitle(
        "A/C 四臂 MA100 训练奖励综合对比（训练曲线，不是验证曲线）",
        fontsize=13.0, color=TEXT, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "ac_ma100_comparison")


def plot_validation(validations, stats):
    fig, axes = plt.subplots(
        2, 1, figsize=(12.6, 7.8), sharex=True, sharey=True,
        facecolor=SURFACE,
    )
    for ax, config in zip(axes, ("A", "C")):
        _style(ax)
        for algorithm in ("dqn", "ppo"):
            val = validations[(config, algorithm)]
            label = AGENT_LABEL[(config, algorithm)]
            s = stats[label]
            color = _agent_color(algorithm)
            ax.plot(
                val["episode"], val["joint_reward"],
                color=color, linewidth=1.7, marker="o", markersize=3.8,
                label=(
                    f"{algorithm.upper()}（best {s['best_val_jointr']:.4f}"
                    f" @ Ep{s['best_val_episode']}）"
                ),
            )
            best = val.loc[val["joint_reward"].idxmax()]
            final = val.iloc[-1]
            ax.scatter([best["episode"]], [best["joint_reward"]],
                       marker="*", s=150, color=color, zorder=5,
                       edgecolor=SURFACE, linewidth=0.8)
            ax.scatter([final["episode"]], [final["joint_reward"]],
                       marker="s", s=55, color=color, zorder=5,
                       edgecolor=SURFACE, linewidth=0.8)
            ax.annotate(
                f"best {best['joint_reward']:.4f} @{int(best['episode'])}",
                xy=(best["episode"], best["joint_reward"]),
                xytext=(5, 9), textcoords="offset points",
                fontsize=7.5, color=color,
            )
            ax.annotate(
                f"final {final['joint_reward']:.4f}",
                xy=(final["episode"], final["joint_reward"]),
                xytext=((-72, 12) if algorithm == "dqn" else (-72, -20)),
                textcoords="offset points",
                fontsize=7.5, color=color,
            )
        ax.axhline(0.0, color=GRID, linewidth=0.8)
        ax.set_title(
            f"{config} · Dev60 Validation（{CONFIG_TITLE[config].split('：', 1)[1]}）",
            fontsize=10.6, pad=6,
        )
        ax.set_ylabel("Validation JointR", fontsize=9.5)
        ax.legend(loc="upper right", frameon=False, fontsize=8.2)
    axes[-1].set_xlabel("训练 Episode（每 200 Episodes 一个验证点）", fontsize=9.5)
    axes[-1].set_xlim(0, 3560)
    fig.suptitle(
        "Dev60 Validation JointR：每 200 Episodes 验证；★ best.pt，■ last.pt（不与训练 Reward 混合）",
        fontsize=12.2, color=TEXT, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "ac_validation_jointr")


def plot_dqn_diagnostics(frames, validations, stats):
    fig, axes = plt.subplots(
        2, 3, figsize=(14.2, 7.8), sharex=True, facecolor=SURFACE,
    )
    for row, config in enumerate(("A", "C")):
        df = frames[(config, "dqn")].copy()
        df["block100"] = (df["episode"] - 1) // 100
        grp = df.groupby("block100")
        x = grp["episode"].mean()
        eps = grp["epsilon"].mean()
        qmean = grp["q_mean"].mean()
        qmax = grp["q_max"].max()
        loss = grp["loss"].mean()
        best_ep = stats[AGENT_LABEL[(config, "dqn")]]["best_val_episode"]
        for col in range(3):
            _style(axes[row][col])
            axes[row][col].axvline(
                best_ep, color=MUTED, linestyle=":", linewidth=0.9,
                alpha=0.8,
            )
        axes[row][0].plot(x, eps, color=BLUE, linewidth=1.8)
        axes[row][0].set_ylabel(
            f"{config}/DQN\nε", fontsize=9.5,
        )
        axes[row][1].plot(x, qmean, color=BLUE, linewidth=1.8,
                          label="Q mean")
        axes[row][1].plot(x, qmax, color=ORANGE, linewidth=1.8,
                          label="Q max")
        axes[row][1].legend(frameon=False, fontsize=8.2, loc="upper left")
        axes[row][2].plot(x, loss, color=BLUE, linewidth=1.8)
        if row == 0:
            axes[row][0].set_title("ε（探索率）", fontsize=10)
            axes[row][1].set_title("Q mean / Q max", fontsize=10)
            axes[row][2].set_title("TD Loss（SmoothL1）", fontsize=10)
        if row == 1:
            axes[row][0].set_xlabel("训练 Episode", fontsize=9)
            axes[row][1].set_xlabel("训练 Episode", fontsize=9)
            axes[row][2].set_xlabel("训练 Episode", fontsize=9)
    fig.suptitle(
        "DQN 诊断（每 100 Episode 均值；虚线=best Dev checkpoint）：ε、Q 值、TD Loss",
        fontsize=12.2, color=TEXT, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "dqn_diagnostics")


def plot_ppo_diagnostics(frames, validations, stats):
    fig, axes = plt.subplots(
        2, 3, figsize=(14.2, 7.8), sharex=True, facecolor=SURFACE,
    )
    for row, config in enumerate(("A", "C")):
        df = frames[(config, "ppo")].copy()
        df["block100"] = (df["episode"] - 1) // 100
        grp = df.groupby("block100")
        x = grp["episode"].mean()
        ent = grp["policy_entropy"].mean()
        val = validations[(config, "ppo")]
        best_ep = stats[AGENT_LABEL[(config, "ppo")]]["best_val_episode"]
        for col in range(3):
            _style(axes[row][col])
            axes[row][col].axvline(
                best_ep, color=MUTED, linestyle=":", linewidth=0.9,
                alpha=0.8,
            )
        axes[row][0].plot(x, ent, color=BLUE, linewidth=1.8)
        axes[row][0].set_ylabel(
            f"{config}/PPO\nEntropy", fontsize=9.5,
        )
        axes[row][1].plot(
            val["episode"], val["approx_kl"], color=ORANGE,
            linewidth=1.7, marker="o", markersize=3.8,
        )
        axes[row][2].plot(
            val["episode"], val["update_value_loss"], color=ORANGE,
            linewidth=1.7, marker="o", markersize=3.8,
        )
        if row == 0:
            axes[row][0].set_title("策略熵（训练，100-Ep 均值）", fontsize=10)
            axes[row][1].set_title("Approx KL（验证日志，每 200）", fontsize=10)
            axes[row][2].set_title("Value Loss（验证日志，每 200）", fontsize=10)
        if row == 1:
            axes[row][0].set_xlabel("训练 Episode", fontsize=9)
            axes[row][1].set_xlabel("训练 Episode", fontsize=9)
            axes[row][2].set_xlabel("训练 Episode", fontsize=9)
    fig.suptitle(
        "PPO 诊断（虚线=best Dev checkpoint）：策略熵、Approx KL、Value Loss",
        fontsize=12.2, color=TEXT, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "ppo_diagnostics")


def plot_task_mix(block_table, frames):
    fig, axes = plt.subplots(
        2, 2, figsize=(13.8, 8.4), facecolor=SURFACE,
    )
    x = np.arange(len(block_table))
    labels = [
        f"{int(row.start)}–{int(row.end)}" for row in block_table.itertuples()
    ]
    ax = axes[0][0]
    _style(ax)
    bottom = np.zeros(len(block_table))
    for band, color in (("short", BLUE), ("medium", ORANGE), ("long", AQUA)):
        values = block_table[band].values
        ax.bar(x, values, bottom=bottom, color=color, width=0.78,
               label={"short": "Short", "medium": "Medium", "long": "Long"}[band],
               edgecolor=SURFACE, linewidth=0.4)
        bottom += values
    ax.set_title("每 200 Episode 的 Band 组成（四臂完全相同）", fontsize=10)
    ax.set_ylabel("Episode 数", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.2)
    ax.legend(
        frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=0.92,
        fontsize=7.6, ncol=3, loc="upper center",
    )

    ax = axes[0][1]
    _style(ax)
    oracle = block_table["oracle_steps_mean"].values
    distance = block_table["start_goal_distance_mean"].values
    ax.plot(x, oracle / oracle.mean(), color=BLUE, linewidth=1.8,
            marker="o", markersize=3.5, label="oracle 专家步数 / 均值")
    ax.plot(x, distance / distance.mean(), color=ORANGE, linewidth=1.8,
            marker="o", markersize=3.5, label="起终点距离 / 均值")
    ax.axhline(1.0, color=GRID, linewidth=0.8)
    ax.set_title("任务难度（归一化到各自均值）", fontsize=10)
    ax.set_ylabel("相对均值", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.2)
    ax.legend(frameon=False, fontsize=8)

    for col, config in enumerate(("A", "C")):
        ax = axes[1][col]
        _style(ax)
        for algorithm in ("dqn", "ppo"):
            df = frames[(config, algorithm)]
            steps = df.groupby("block")["steps"].mean().values
            ax.plot(x, steps, color=_agent_color(algorithm), linewidth=1.8,
                    marker="o", markersize=3.4,
                    label=f"{algorithm.upper()} 实际步数")
        ax.plot(x, block_table["oracle_steps_mean"].values,
                color=MUTED, linewidth=1.3, linestyle="--",
                label="oracle 专家步数")
        ax.set_title(f"{config} 配置的实际导航步数", fontsize=10)
        ax.set_ylabel("每 200 Episode 平均步数", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "任务顺序、Band 组成、专家难度与实际导航步数（四臂共享同一任务顺序）",
        fontsize=12.2, color=TEXT, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "task_mix_steps")


def plot_model_usage(validations):
    fig, axes = plt.subplots(
        2, 2, figsize=(13.2, 8.0), facecolor=SURFACE,
    )
    for ax, (config, algorithm) in zip(axes.flat, (
        ("A", "dqn"), ("A", "ppo"), ("C", "dqn"), ("C", "ppo")
    )):
        _style(ax)
        val = validations[(config, algorithm)]
        episodes = val["episode"].values
        bottom = np.zeros(len(val))
        for model in MODEL_ORDER:
            ratios = np.asarray([
                row.get("model_usage_ratio", {}).get(model, 0.0)
                for row in val.to_dict("records")
            ], dtype=float)
            ax.bar(
                episodes, ratios, bottom=bottom, width=105,
                color=MODEL_COLORS[model], label=model,
                edgecolor=SURFACE, linewidth=0.35,
            )
            bottom += ratios
        ax.set_ylim(0, 1.0)
        ax.set_title(
            f"{AGENT_LABEL[(config, algorithm)]} · MEC 使用比例（验证时）",
            fontsize=9.8,
        )
        ax.set_ylabel("比例", fontsize=8.8)
        ax.set_xlabel("训练 Episode（验证点）", fontsize=8.8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=5, frameon=False,
        fontsize=8.4, bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        "MEC 模型使用比例随验证 Episode 的变化（每 200 Episodes）",
        fontsize=12.2, color=TEXT, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, "model_usage")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value, digits=4):
    return f"{value:.{digits}f}"


def _markdown_table(stats):
    lines = [
        "| Agent | MA100 峰值 | 峰值 Episode | 晚期(≥2500)峰值 | 最终 MA100 | Best Val JointR | Best Ep | Final Val JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, _, label in AGENTS:
        s = stats[label]
        lines.append(
            f"| {label} | {s['peak_ma100']:.4f} | {s['peak_episode']} | "
            f"{s['late_peak_ma100']:.4f} @ {s['late_peak_episode']} | "
            f"{s['final_ma100']:.4f} | {s['best_val_jointr']:.4f} | "
            f"{s['best_val_episode']} | {s['final_val_jointr']:.4f} |"
        )
    return "\n".join(lines)


def _model_usage_table(validations):
    lines = [
        "| Agent | 阶段 | Small | Transformer-S | Medium | Transformer-M | Large |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for config, algorithm in (
        ("A", "dqn"), ("A", "ppo"), ("C", "dqn"), ("C", "ppo")
    ):
        val = validations[(config, algorithm)]
        for name, rows in (
            ("早期(前3次)", val.head(3)),
            ("中期(第7–9次)", val.iloc[6:9]),
            ("晚期(后3次)", val.tail(3)),
        ):
            means = {
                model: np.mean([
                    row.get("model_usage_ratio", {}).get(model, 0.0)
                    for row in rows.to_dict("records")
                ])
                for model in MODEL_ORDER
            }
            lines.append(
                f"| {AGENT_LABEL[(config, algorithm)]} | {name} | "
                + " | ".join(f"{means[m]:.3f}" for m in MODEL_ORDER)
                + " |"
            )
    return "\n".join(lines)


def write_report(frames, validations, stats, block_table, corr_df):
    lines = [
        "# Stage-2l 训练奖励曲线与中后期下降分析",
        "",
        "日期：2026-10-21  ",
        "分支：`experiment/v83-spatial-foundation`  ",
        "数据：`runs/v83_spatial_stage2/stage2l_fixed_city/`（只读）",
        "",
        "本报告只读取训练历史、验证记录和评测产物；**没有停止或修改任何训练进程、"
        "Checkpoint 或历史实验数据**。四个已完成 Agent 为 A/DQN、A/PPO、"
        "C/DQN、C/PPO；B 仍暂停，不在本报告的 3500-Episode 曲线中。",
        "",
        "## 1. 数据与定义",
        "",
        "- 逐 Episode Reward：`training_history.json` 的 `episode_return`，"
        "与 JointR 对齐（导航 reward + 运行平均 E2E 惩罚）。",
        "- MA100：从 Episode 100 起，最近 100 个训练 Episode 的算术平均。",
        "- Validation JointR：每 200 Episodes 在 60 条固定 Dev 任务上的 "
        "`joint_reward`；共 18 个点（200…3400、3500）。",
        "- 训练 Reward 与 Validation JointR 分开绘制，验证点不做逐 Episode 插值。",
        "- 本报告中的“晚期”通常指 Episode ≥ 2500；DQN/PPO 诊断表的"
        "“早期/晚期”是前/后 200 个训练 Episode（PPO 的 KL/Value Loss 为"
        "前/后 3 个验证点）；MEC 使用比例的早期/中期/晚期为前 3 / 第 7–9 /"
        " 后 3 个验证点。",
        "",
        "## 2. 四个 Agent 的训练奖励曲线",
        "",
        "每条图同时包含原始逐 Episode Reward（浅灰）与 MA100（彩色）；"
        "星号标出 MA100 峰值，方块标出最终 MA100。峰值只表示中途最高，"
        "**不是收敛**。",
        "",
        "### C/DQN（重点）",
        "",
        "![C/DQN 训练曲线](figures/stage2l_curves/c_dqn_training_curve.png)",
        "",
        "![C/DQN 训练曲线 SVG](figures/stage2l_curves/c_dqn_training_curve.svg)",
        "",
        "### C/PPO（重点）",
        "",
        "![C/PPO 训练曲线](figures/stage2l_curves/c_ppo_training_curve.png)",
        "",
        "![C/PPO 训练曲线 SVG](figures/stage2l_curves/c_ppo_training_curve.svg)",
        "",
        "### A/DQN 与 A/PPO",
        "",
        "![A/DQN 训练曲线](figures/stage2l_curves/a_dqn_training_curve.png)",
        "",
        "![A/PPO 训练曲线](figures/stage2l_curves/a_ppo_training_curve.png)",
        "",
        "## 3. A/C 四臂 MA100 综合对比",
        "",
        "![A/C MA100 对比](figures/stage2l_curves/ac_ma100_comparison.png)",
        "",
        "图中只画训练 MA100；验证曲线见下一节。A 为单张 Balanced 固定拓扑；"
        "C 为 Balanced / Quality-rich / Tradeoff-stress 三种部署、每种一张固定拓扑。",
        "",
        "## 4. Dev60 Validation JointR（每 200 Episodes）",
        "",
        "![Validation JointR](figures/stage2l_curves/ac_validation_jointr.png)",
        "",
        "★ 标出对应 best checkpoint 的验证点，■ 标出最终（Ep3500）验证点；"
        "best checkpoint 文件名固定为 `best.pt`，最后 checkpoint 为 `last.pt`。"
        "验证点只有 18 个，连线仅辅助阅读，不能当作逐 Episode 训练曲线。",
        "",
        "## 5. 奖励变化汇总",
        "",
        _markdown_table(stats),
        "",
        "关键观察：",
        "",
        "- A/DQN、C/DQN、C/PPO 的全局 MA100 峰值出现在 Ep3097–3161；"
        "A/PPO 的全局峰值在 Ep680，但 Ep3120 附近还有接近的晚期峰值 0.3162。",
        "- 四个 Agent 在 3001–3200 这一 200-Episode 区间同时出现较高的训练奖励，"
        "随后在 3201–3400、3401–3500 同时回落。",
        "- **训练 MA100 的峰值与 Dev Validation 的 best 并不对齐**："
        "A/PPO 的 Dev best 在 Ep1800（0.1801），训练 MA100 的全局峰值却在 Ep680；"
        "C/PPO 的 Dev best 在 Ep1800（0.1636），训练 MA100 峰值在 Ep3161。"
        "因此训练曲线峰值不能作为模型选择或收敛证据。",
        "- 四臂的 Dev best 都低于其训练 MA100 峰值，且 final Dev JointR 更低；"
        "这与“训练奖励上升但 Dev 泛化下降”一致，但不能仅凭这一点断言原因。",
        "",
        "## 6. 中后期下降诊断",
        "",
        "### 6.1 DQN：ε、Q 值与 TD Loss",
        "",
        "![DQN 诊断](figures/stage2l_curves/dqn_diagnostics.png)",
        "",
        "| Agent | ε 初始→最终 | Q mean 早期→晚期 | Q mean 峰值 | Q max 峰值 | TD Loss 早期→晚期 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, _, label in AGENTS:
        s = stats[label]
        if "epsilon_start" not in s:
            continue
        lines.append(
            f"| {label} | {s['epsilon_start']:.3f}→{s['epsilon_final']:.3f} | "
            f"{s['q_mean_early']:.2f}→{s['q_mean_late']:.2f} | "
            f"{s['q_mean_peak']:.2f} | {s['q_max_peak']:.2f} | "
            f"{s['loss_early']:.5f}→{s['loss_late']:.5f} |"
        )
    lines += [
        "",
        "可证实：",
        "",
        "- ε 从约 1.0 衰减到 0.05–0.07，后期探索显著减少。",
        "- Q mean / Q max 在训练中大幅上升（A/DQN 的 Q mean 峰值约 23.7，"
        "C/DQN 约 42.6），远高于实际 Episode Reward 的量级；TD Loss 在后期"
        "也明显高于早期。",
        "- 这些是训练日志中直接可读的数值，不是猜测。",
        "",
        "推测（需要额外实验）：",
        "",
        "- Q 值持续增大而 Dev JointR 没有同步提高，可能包含 Q 过估计、"
        "replay 分布偏移或对近期任务块的过拟合；目前的日志不足以区分这些机制。",
        "- 不能仅凭 Q 值上升判断“学到了更好的关联”；它也可能只是 target 变大。",
        "",
        "### 6.2 PPO：熵、KL 与 Value Loss",
        "",
        "![PPO 诊断](figures/stage2l_curves/ppo_diagnostics.png)",
        "",
        "| Agent | 熵 早期→晚期 | Approx KL 早期→晚期 | Value Loss 早期→晚期 | Policy Loss 早期→晚期 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, _, label in AGENTS:
        s = stats[label]
        if "entropy_early" not in s:
            continue
        lines.append(
            f"| {label} | {s['entropy_early']:.3f}→{s['entropy_late']:.3f} | "
            f"{s['approx_kl_early']:.4f}→{s['approx_kl_late']:.4f} | "
            f"{s['value_loss_early']:.4f}→{s['value_loss_late']:.4f} | "
            f"{s['policy_loss_early']:.4f}→{s['policy_loss_late']:.4f} |"
        )
    lines += [
        "",
        "可证实：",
        "",
        "- 两臂 PPO 的策略熵都下降（A 1.62→1.25，C 1.56→1.13），策略变得更确定。",
        "- Approx KL 保持在约 0.01–0.02，没有出现数值爆炸；Value Loss 没有"
        "同步恶化（A 0.015→0.008，C 0.011→0.009）。",
        "- 因此 PPO 后期的 Dev 下降不是“训练发散/NaN”造成的；更可能是策略"
        "分布变窄后与 Dev 任务不匹配。",
        "",
        "推测（需要额外实验）：",
        "",
        "- 熵下降可能使 PPO 逐渐锁定到少数 MEC 类型；从验证时的模型使用比例"
        "可以看到晚期偏向 Small（A）或 Large（C）的迹象，但这不是因果证明。",
        "- 训练任务顺序是全局随机打乱的，PPO 可能在后期块上过拟合；需要"
        "用独立 Dev/新拓扑做对照实验才能确认。",
        "",
        "### 6.3 MEC 使用比例",
        "",
        "![MEC 使用比例](figures/stage2l_curves/model_usage.png)",
        "",
        _model_usage_table(validations),
        "",
        "### 6.4 任务顺序、Band 组成与实际步数",
        "",
        "![任务混合与步数](figures/stage2l_curves/task_mix_steps.png)",
        "",
        "四个 Agent 使用**完全相同的 training_order.json、map_seed 和 task_seed**；"
        "四臂的历史配置都记录 `seed=20261031`（DQN 与 PPO 各自算法内部使用），"
        "但算法/网络不同，不能把相同 seed 当作相同随机过程。"
        "每个 200-Episode 块的 Short/Medium/Long 数量、oracle 专家步数和"
        "起终点距离完全相同。以下为最后三个块与早期块的对比：",
        "",
        "| 块 | Short | Medium | Long | oracle 步数均值 | 起终点距离均值 (m) | A/DQN 实际步数 | A/PPO 实际步数 | C/DQN 实际步数 | C/PPO 实际步数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for block in (0, 3, 11, 14, 15, 16, 17):
        row = block_table.iloc[block]
        lines.append(
            f"| {int(row.start)}–{int(row.end)} | {int(row.short)} | "
            f"{int(row.medium)} | {int(row.long)} | "
            f"{row.oracle_steps_mean:.1f} | {row.start_goal_distance_mean:.1f} | "
            + " | ".join(
                f"{frames[(c, a)].groupby('block')['steps'].mean().iloc[block]:.1f}"
                for c, a in (("A", "dqn"), ("A", "ppo"),
                             ("C", "dqn"), ("C", "ppo"))
            )
            + " |"
        )
    lines += [
        "",
        "可证实：",
        "",
        "- 任务顺序与 Band 组成在四臂间逐 Episode 完全相同；因此四臂在"
        "3001–3200 同时出现高奖励、随后同时回落，至少部分来自共享任务序列。",
        "- 3001–3200 块的实际导航步数在四臂中都较低（A/DQN 136.2、"
        "A/PPO 130.7、C/DQN 129.0、C/PPO 118.0），而 3201–3400 和"
        "3401–3500 明显升高；奖励与实际步数呈强负相关（块级 Pearson："
        "A/DQN -0.64、A/PPO -0.57、C/DQN -0.58、C/PPO -0.74），"
        "与成功率正相关（0.52–0.65）。",
        "- 因此“后期奖励下降”与后期实际步数增加/成功率下降是一致的；"
        "这是可观察的相关性。",
        "",
        "推测（需要额外实验）：",
        "",
        "- 3001–3200 块本身略“容易”（oracle 步数均值 84.4 vs 相邻块 86–91），"
        "可能放大了这次峰值；但 oracle 难度与奖励的相关性很弱且方向不一致"
        "（A/DQN +0.20、A/PPO +0.02、C/DQN -0.06、C/PPO -0.28），"
        "所以不能把峰值简单归因于任务更容易。",
        "- 共享任务顺序/相同基础随机种子是已证实的事实；它们对相似奖励变化的"
        "因果贡献只能由跨块相关性和对照实验估计，不能仅凭相关性断言。",
        "",
        "### 6.5 跨 Agent 奖励相关",
        "",
        "块级（18 个 200-Episode 块）与逐 Episode 的 Pearson/Spearman 相关：",
        "",
        "| 对比 | 块级 Pearson | 块级 Spearman | 逐 Episode Pearson |",
        "|---|---:|---:|---:|",
    ]
    for pair in sorted(set(corr_df["pair"])):
        if " vs " not in pair:
            continue
        sub = corr_df[corr_df["pair"] == pair]
        block = sub[sub["level"] == "per_200_block"]
        ep = sub[sub["level"] == "per_episode"]
        if block.empty:
            continue
        lines.append(
            f"| {pair} | {float(block['pearson'].iloc[0]):.3f} | "
            f"{float(block['spearman'].iloc[0]):.3f} | "
            f"{float(ep['pearson'].iloc[0]):.3f} |"
        )
    lines += [
        "",
        "块级相关最高的是 C/DQN vs C/PPO（0.788）和 A/PPO vs C/PPO"
        "（0.727）；A/DQN vs A/PPO 为 0.601。这与共享任务顺序、共享"
        "map/task seed 以及共同的后期块结构一致，但不能证明算法本身相同。",
        "",
        "## 7. 结论（明确不称收敛）",
        "",
        "- 四个 Agent 都跑满 3500 Episodes；训练 MA100 的峰值出现在中后期"
        "（约 Ep3100）或早期（A/PPO Ep680），但 Dev Validation 的 best"
        "分别出现在 Ep2400/1800/3200/1800，两者不对齐。",
        "- **训练奖励峰值不是收敛**：所有 best checkpoint 的 Dev JointR 都"
        "低于训练 MA100 峰值，final Dev JointR 更低；四臂的最终状态均为"
        "`plateaued_not_converged`。",
        "- 固定 Dev Baseline 对照（上一轮）显示：A/PPO best 0.1801 低于"
        "Max-Quality 0.2179 与 Heuristic 0.2131；C/PPO best 0.1636 低于"
        "Min-Latency 0.1833 与 Heuristic 0.1843。因此当前学习策略还没有"
        "优于全部非特权 baseline；这与“训练曲线看起来很高”并不矛盾。",
        "- 已证实：共享任务顺序/Band 组成、后期实际步数上升、DQN ε 衰减与"
        "Q 值/ TD Loss 上升、PPO 熵下降、训练与 Dev 不对齐。",
        "- 推测：Q 过估计/replay 分布偏移、PPO 策略分布变窄与后期任务块"
        "过拟合；这些需要独立 Dev/新拓扑/种子重复实验验证，不能作为已证实结论。",
        "- 没有使用 Final Test；没有修改奖励、算法、Checkpoint 或历史数据。",
        "",
        "## 8. 产物与复现",
        "",
        "图（PNG/SVG）：`v83_spatial_stage2/figures/stage2l_curves/`  ",
        "原始数据 CSV：`v83_spatial_stage2/figures/stage2l_curves/data/`  ",
        "复现：",
        "",
        "```bash",
        "python v83_spatial_stage2/stage2l_plot_training_curves.py",
        "```",
        "",
        "生成的 CSV：`training_curves_raw.csv`（14,000 行逐 Episode 数据）、"
        "`validation_curves.csv`、`per200_task_mix.csv`、"
        "`diagnostics_dqn.csv`、`diagnostics_ppo.csv`、`model_usage.csv`、"
        "`correlations.csv`。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    order, specs, frames, validations = load_data()
    task_table = block_task_table(order, specs)
    block_table = per_block_metrics(frames, validations, task_table)
    corr_df = correlations(frames, block_table)
    stats = make_stats(frames, validations, block_table, corr_df)
    write_csvs(frames, validations, block_table, corr_df)

    for config, algorithm, label in AGENTS:
        plot_agent_training(
            config, algorithm, frames[(config, algorithm)], stats[label]
        )
    plot_ma100_comparison(frames, stats)
    plot_validation(validations, stats)
    plot_dqn_diagnostics(frames, validations, stats)
    plot_ppo_diagnostics(frames, validations, stats)
    plot_task_mix(block_table, frames)
    plot_model_usage(validations)
    write_report(frames, validations, stats, block_table, corr_df)
    print("STAGE2L_TRAINING_CURVES_COMPLETE", flush=True)
    print(json.dumps({
        "agents": [label for _, _, label in AGENTS],
        "figures": sorted(path.name for path in OUT_DIR.glob("*.png")),
        "report": str(REPORT_PATH),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
