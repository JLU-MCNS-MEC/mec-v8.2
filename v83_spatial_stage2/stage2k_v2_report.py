"""Chinese reports and real curves for the Stage-2k v2 A/B ablations.

Reads only completed training/eval artifacts under ``runs/``.  It never reads
or evaluates the Association Final Test.  Figures are generated from the
recorded per-episode/per-validation points; no interpolation or synthetic
points are added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    SCENARIOS,
    atomic_write_json,
    utc_now,
)

PURE_RUN = Path("runs/v83_spatial_stage2/stage2k_pure_mec_v2")
VISUAL_RUN = Path("runs/v83_spatial_stage2/stage2k_visual_ablation_v2")
FIG_PURE = Path("v83_spatial_stage2/figures/stage2k_pure_mec_v2")
FIG_VISUAL = Path("v83_spatial_stage2/figures/stage2k_visual_ablation_v2")
FIG_CMP = Path("v83_spatial_stage2/figures/stage2k_v2_ablation")
REPORT_PURE = Path("v83_spatial_stage2/STAGE2K_PURE_MEC_V2_REPORT.md")
REPORT_VISUAL = Path("v83_spatial_stage2/STAGE2K_VISUAL_ABLATION_V2_REPORT.md")
REPORT_CMP = Path("v83_spatial_stage2/STAGE2K_V2_ABLATION_COMPARISON.md")
DECISION_PATH = Path("runs/v83_spatial_stage2/stage2k_v2_final_test_decision.json")
DEV60_BASELINE_THRESHOLD = 0.2828457269663921
SEED = 20261031

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
COLORS = {"A": BLUE, "B": ORANGE}
LABELS = {
    "A": "A: Pure MEC (384-D)",
    "B": "B: A + 128-D visual (512-D)",
}
SHORT_LABELS = {"A": "A Pure MEC", "B": "B Visual ablation"}


def configure_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "Noto Sans CJK SC",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
    })


def load_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < window:
        return output
    kernel = np.ones(window, dtype=np.float64) / float(window)
    output[window - 1:] = np.convolve(values, kernel, mode="valid")
    return output


def load_run(run_dir):
    history = load_json(run_dir / "training_history.json")
    result = load_json(run_dir / "training_result.json")
    if history is None or result is None:
        raise SystemExit(f"{run_dir} is missing training artifacts")
    evals = {}
    for name in ("best", "last"):
        summary = load_json(run_dir / f"dev180_{name}" / "summary.json")
        if summary is not None:
            evals[name] = summary
    q_audit = load_json(run_dir / "q_audit" / "q_audit.json")
    train = history["train"]
    validations = history.get("validation", [])
    episodes = np.asarray([row["episode"] for row in train], dtype=np.float64)
    returns = np.asarray(
        [row["episode_return"] for row in train], dtype=np.float64
    )
    q_mean = np.asarray([row["q_mean"] for row in train], dtype=np.float64)
    q_max = np.asarray([row["q_max"] for row in train], dtype=np.float64)
    epsilon = np.asarray([row["epsilon"] for row in train], dtype=np.float64)
    best_validation = (
        max(validations, key=lambda row: row["joint_reward"])
        if validations else None
    )
    return {
        "run_dir": run_dir,
        "history": history,
        "result": result,
        "evals": evals,
        "q_audit": q_audit,
        "train": train,
        "validations": validations,
        "best_validation": best_validation,
        "last_validation": validations[-1] if validations else None,
        "arrays": {
            "episodes": episodes,
            "returns": returns,
            "ma100": moving_average(returns, 100),
            "q_mean": q_mean,
            "q_max": q_max,
            "q_mean_ma25": moving_average(q_mean, 25),
            "q_max_ma25": moving_average(q_max, 25),
            "epsilon": epsilon,
            "total_steps": np.asarray(
                [row["total_steps"] for row in train], dtype=np.float64
            ),
        },
    }


def plot_training_return(runs, path):
    keys = [key for key in ("A", "B") if key in runs]
    fig, axes = plt.subplots(
        1, len(keys), figsize=(5.2 * len(keys), 3.8),
        sharey=True, squeeze=False,
    )
    axes = axes.ravel()
    for ax, key in zip(axes, keys):
        run = runs[key]
        arrays = run["arrays"]
        color = COLORS[key]
        ax.plot(
            arrays["episodes"], arrays["returns"],
            color=GRID, linewidth=0.7, zorder=0, label="Raw return",
        )
        ax.plot(
            arrays["episodes"], arrays["ma100"],
            color=color, linewidth=2.0, zorder=2,
            label="100-episode MA",
        )
        ax.set_title(LABELS[key], loc="left")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode return = JointR")
        ax.legend(loc="lower right")
    fig.suptitle(
        "训练原始 Return 与 100-Episode 移动平均（真实记录）",
        fontsize=11, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_validation_curves(runs, path):
    metrics = [
        ("joint_reward", "Validation JointR", 1.0),
        ("SR", "SR", 1.0),
        ("OSR", "OSR", 1.0),
        ("SPL", "SPL", 1.0),
        ("nDTW", "nDTW", 1.0),
        ("E2E_ms", "Mean E2E (ms)", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2))
    for ax, (key, title, ymax) in zip(axes.ravel(), metrics):
        for name in [k for k in ("A", "B") if k in runs]:
            rows = runs[name]["validations"]
            if not rows:
                continue
            ax.plot(
                [row["episode"] for row in rows],
                [row.get(key, np.nan) for row in rows],
                color=COLORS[name], marker="o", markersize=4.8,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=SHORT_LABELS[name],
            )
        ax.set_title(title)
        ax.set_xlabel("Training episode")
        if ymax is not None:
            ax.set_ylim(0.0, ymax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=2,
            frameon=False, bbox_to_anchor=(0.5, 0.005),
        )
    fig.suptitle(
        "固定 Dev60 上的 Validation 曲线（每个点对应一次真实评估）",
        fontsize=11, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_q_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharex=True)
    for name in [k for k in ("A", "B") if k in runs]:
        arrays = runs[name]["arrays"]
        color = COLORS[name]
        axes[0].plot(
            arrays["episodes"], arrays["q_mean"],
            color=color, alpha=0.22, linewidth=0.6,
        )
        axes[0].plot(
            arrays["episodes"], arrays["q_mean_ma25"],
            color=color, linewidth=2.0,
            label=f"{SHORT_LABELS[name]} (25-ep MA)",
        )
        axes[1].plot(
            arrays["episodes"], arrays["q_max"],
            color=color, alpha=0.22, linewidth=0.6,
        )
        axes[1].plot(
            arrays["episodes"], arrays["q_max_ma25"],
            color=color, linewidth=2.0,
            label=f"{SHORT_LABELS[name]} (25-ep MA)",
        )
    axes[0].set_title("Q mean（逐 Episode，细线原始 + 25-ep MA）")
    axes[1].set_title("Q max（逐 Episode，细线原始 + 25-ep MA）")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Q value")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_epsilon(runs, path):
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    for name in [k for k in ("A", "B") if k in runs]:
        arrays = runs[name]["arrays"]
        ax.plot(
            arrays["episodes"], arrays["epsilon"],
            color=COLORS[name], label=LABELS[name],
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("ε (actual)")
    ax.set_title("实际 ε：1.0 → 0.05，500,000 环境交互步预算")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def metric_value(summary, key):
    return float(summary["overall"][key])


def plot_dev180_bars(runs, path):
    metrics = [
        ("SR", "SR", True),
        ("OSR", "OSR", True),
        ("SPL", "SPL", False),
        ("nDTW", "nDTW", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    x = np.arange(4)
    width = 0.2
    series = []
    for name in [k for k in ("A", "B") if k in runs]:
        for which in ("best", "last"):
            summary = runs[name]["evals"].get(which)
            if summary is not None:
                series.append((name, which, summary))
    if not series:
        for ax in axes.ravel():
            ax.text(
                0.5, 0.5, "Dev180 evaluation pending",
                ha="center", va="center", transform=ax.transAxes,
                color=MUTED,
            )
        fig.suptitle(
            "Dev Validation 180：best vs last Checkpoint 的四项指标",
            fontsize=11, fontweight="bold", color=INK,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return
    for ax, (metric, title, as_percent) in zip(axes.ravel(), metrics):
        for index, (name, which, summary) in enumerate(series):
            value = metric_value(summary, metric)
            height = value * 100.0 if as_percent else value
            base = COLORS[name]
            hatch = None if which == "best" else "//"
            alpha = 1.0 if which == "best" else 0.45
            bars = ax.bar(
                index - (len(series) - 1) * width / 2.0,
                height, width=width, color=base, alpha=alpha,
                hatch=hatch, edgecolor=SURFACE, linewidth=0.8,
                label=f"{name} {which}",
            )
            ax.annotate(
                f"{height:.1f}" if as_percent else f"{height:.3f}",
                xy=(bars[0].get_x() + bars[0].get_width() / 2.0, height),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5,
                color=INK_SECONDARY,
            )
        ax.set_xticks([])
        ax.set_title(title)
        if as_percent:
            ax.set_ylim(0, 100)
            ax.set_ylabel("Percent")
        else:
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
    )
    fig.suptitle(
        "Dev Validation 180：best vs last Checkpoint 的四项指标",
        fontsize=11, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mec_usage(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    y = np.arange(len(MODEL_NAMES))
    height = 0.36
    for ax, which in zip(axes, ("best", "last")):
        for offset, name in (
            (offset, name) for offset, name in ((-0.5, "A"), (0.5, "B"))
            if name in runs
        ):
            summary = runs[name]["evals"].get(which)
            if summary is None:
                continue
            ratios = summary["overall"]["model_usage_ratio"]
            values = [100.0 * float(ratios.get(model, 0.0))
                      for model in MODEL_NAMES]
            ax.barh(
                y + offset * height, values, height=height,
                color=COLORS[name], label=SHORT_LABELS[name],
            )
        ax.set_yticks(y)
        ax.set_yticklabels(MODEL_NAMES)
        ax.set_xlabel("Share of association steps (%)")
        ax.set_title(f"Dev180 {which} Checkpoint")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="lower right")
    fig.suptitle(
        "MEC 类型选择频率（同一批 Dev180）",
        fontsize=11, fontweight="bold", color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(value, digits=4, percent=False, missing="—"):
    if value is None:
        return missing
    if percent:
        return f"{100.0 * float(value):.2f}%"
    return f"{float(value):.{digits}f}"


def training_summary(run):
    result = run["result"]
    last = run["train"][-1]
    best_val = run["best_validation"]
    last_val = run["last_validation"]
    return {
        "episodes": int(result["episodes_trained"]),
        "total_steps": int(result.get("total_steps", last["total_steps"])),
        "epsilon_final": float(result.get(
            "epsilon_final", last["epsilon"]
        )),
        "loss_final": float(result.get("loss_final", last["loss"])),
        "grad_final": float(result.get(
            "grad_norm_final", last["grad_norm"]
        )),
        "q_mean_final": float(result.get(
            "q_mean_final", last["q_mean"]
        )),
        "q_max_final": float(result.get(
            "q_max_final", last["q_max"]
        )),
        "best_episode": int(best_val["episode"]) if best_val else None,
        "best_joint": float(best_val["joint_reward"]) if best_val else None,
        "last_joint": float(last_val["joint_reward"]) if last_val else None,
        "status": result.get("status"),
        "converged": bool(result.get("converged")),
        "stop_reason": result.get("stop_reason"),
        "early_stop_disabled": bool(result.get("early_stop_disabled")),
        "reached_cap": bool(result.get("reached_cap")),
    }


def gate_decision(run):
    summary = training_summary(run)
    competitive = (
        summary["best_joint"] is not None
        and summary["best_joint"] > DEV60_BASELINE_THRESHOLD
    )
    if summary["converged"] and competitive:
        decision = "eligible_for_final_test"
    elif summary["converged"] and not competitive:
        decision = "retained_converged_below_baseline"
    else:
        decision = "retained_not_converged"
    return {
        "converged": summary["converged"],
        "best_dev60_joint": summary["best_joint"],
        "competitiveness_threshold": DEV60_BASELINE_THRESHOLD,
        "competitive": bool(competitive),
        "final_test_eligible": bool(
            summary["converged"] and competitive
        ),
        "decision": decision,
        "final_test_read": False,
    }


def canonical_hash(payload):
    import hashlib
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_parity(runs):
    orders = {}
    validations = {}
    for key, run in runs.items():
        order_payload = load_json(run["run_dir"] / "training_order.json")
        isolation = load_json(
            run["run_dir"] / "validation_isolation.json"
        )
        orders[key] = order_payload["order"]
        validations[key] = [
            item["episode_uid"] for item in isolation["validation_episodes"]
        ]
    ca = runs["A"]["result"]["config"]
    cb = runs["B"]["result"]["config"]
    arch_a = str(ca["architecture"]).split("_")
    arch_b = str(cb["architecture"]).split("_")
    payload = {
        "version": "stage2k-v2-protocol-parity-v1",
        "created_utc": utc_now(),
        "checks": {
            "training_order_equal": orders["A"] == orders["B"],
            "training_order_length": len(orders["A"]),
            "training_order_sha256": canonical_hash(orders["A"]),
            "validation_episodes_equal": (
                validations["A"] == validations["B"]
            ),
            "validation_episode_count": len(validations["A"]),
            "validation_uids_sha256": canonical_hash(validations["A"]),
            "seed_equal": ca["seed"] == cb["seed"],
            "seed": ca["seed"],
            "train_topology_split_equal": (
                ca["train_topology_split"] == cb["train_topology_split"]
            ),
            "train_topology_split": ca["train_topology_split"],
            "validation_topology_split_equal": (
                ca["validation_topology_split"]
                == cb["validation_topology_split"]
            ),
            "validation_topology_split": ca[
                "validation_topology_split"
            ],
            "reward_equal": ca["reward"] == cb["reward"],
            "epsilon_decay_steps_equal": (
                ca["epsilon_decay_steps"] == cb["epsilon_decay_steps"]
            ),
            "epsilon_decay_steps": ca["epsilon_decay_steps"],
            "gamma_equal": ca["gamma"] == cb["gamma"],
            "gamma": ca["gamma"],
            "hidden_layers_equal": arch_a[2:] == arch_b[2:],
            "architecture_A": ca["architecture"],
            "architecture_B": cb["architecture"],
            "capability_prior_sha_equal": (
                ca["capability_prior"]["sha256"]
                == cb["capability_prior"]["sha256"]
            ),
            "capability_prior_sha256": ca["capability_prior"]["sha256"],
            "early_stop_disabled_both": (
                bool(ca["early_stop_disabled"])
                and bool(cb["early_stop_disabled"])
            ),
        },
        "final_test_read": False,
    }
    checks = payload["checks"]
    payload["all_parity_checks_pass"] = all([
        checks["training_order_equal"],
        checks["validation_episodes_equal"],
        checks["seed_equal"],
        checks["train_topology_split_equal"],
        checks["validation_topology_split_equal"],
        checks["reward_equal"],
        checks["epsilon_decay_steps_equal"],
        checks["gamma_equal"],
        checks["hidden_layers_equal"],
        checks["capability_prior_sha_equal"],
        checks["early_stop_disabled_both"],
    ])
    return payload


def write_decision(run, key):
    decision = gate_decision(run)
    decision.update({
        "version": "stage2k-v2-final-test-decision-v1",
        "created_utc": utc_now(),
        "variant": key,
        "run_dir": str(run["run_dir"]),
        "state_mode": run["result"]["config"]["state_mode"],
    })
    atomic_write_json(run["run_dir"] / "final_test_decision.json", decision)
    return decision


def run_markdown(key, run, figures, decision):
    config = run["result"]["config"]
    summary = training_summary(run)
    lines = [
        f"# Stage-2k v2 {'A：Pure MEC' if key == 'A' else 'B：视觉消融'} 实验报告",
        "",
        f"- 日期：{run['result']['created_utc']}",
        f"- 运行目录：`{run['run_dir']}`",
        f"- 状态模式：`{config['state_mode']}`，"
        f"{config['state_dim']} 维 MLP 输入，"
        f"{config['state_dim'] + 20} 维决策状态",
        f"- 网络：`{config['architecture']}`",
        f"- TensorBoard：`runs/tensorboard/"
        f"association_stage2k_{config['state_mode']}/`",
        "- 数据隔离：未读取 Stage-2j Final Test 300，未读取 "
        "Association Final Test 180。",
        "",
        "## 0. TL;DR",
        "",
        f"- 实际训练 **{summary['episodes']} / 3500** Episodes，"
        f"**{summary['total_steps']:,}** 环境交互步；"
        f"最终 ε **{summary['epsilon_final']:.4f}**。",
        f"- 最佳 Dev60 JointR **{fmt(summary['best_joint'])}** "
        f"@ Ep{summary['best_episode']}；最后 Dev60 JointR "
        f"**{fmt(summary['last_joint'])}**。",
        f"- 训练状态：**{summary['status']}**；"
        f"converged = **{summary['converged']}**；"
        f"早停禁用 = **{summary['early_stop_disabled']}**。",
        f"- 收敛门槛与 Final Test 决策：**{decision['decision']}**"
        f"（Final Test read = **{decision['final_test_read']}**）。",
    ]
    q_audit = run["q_audit"]
    if q_audit is not None:
        q_best = q_audit["checkpoints"]["best"]
        q_last = q_audit["checkpoints"]["last"]
        ratio = (
            q_last["q_mean"] / q_best["q_mean"]
            if q_best["q_mean"] else None
        )
        lines.append(
            f"- Q 审计（固定 Dev 状态）：best→last Q mean "
            f"**{q_best['q_mean']:.3f} → {q_last['q_mean']:.3f}**"
            + (f"（{ratio:.1f}×）" if ratio is not None else "")
            + f"，Q max **{q_best['q_max']:.3f} → "
            f"{q_last['q_max']:.3f}**；贪心一致率 "
            f"**{100*q_audit['best_last_greedy_agreement']:.2f}%**。"
        )
        if q_last["q_mean"] > 100.0 or q_last["q_max"] > 500.0:
            lines.append(
                "- **Q 尺度警告**：last Checkpoint 的 Q 尺度达到 "
                f"{q_last['q_mean']:.1f}（mean）/ {q_last['q_max']:.1f}"
                "（max），远超奖励量级，属于明显的 Q 过估计/尺度漂移；"
                "虽然未触发 1e6 数值中止阈值，但**不能**据此认为策略收敛或可部署。"
            )
        elif ratio is not None and ratio > 5.0:
            lines.append(
                "- Q 尺度在 best→last 之间有明显增长，但不构成收敛证据；"
                "收敛判定只看固定 Dev60 Validation 的稳定平台。"
            )
    lines += [
        "",
        "## 1. 状态空间",
        "",
        f"`{config['state_mode']}` 的布局（由 `state_layout` 固化）：",
        "",
        "```json",
        json.dumps(config["state_layout"], ensure_ascii=False, indent=2),
        "```",
        "",
        f"- 能力先验：`{config['capability_prior']['path']}`，"
        f"SHA256 `{config['capability_prior']['sha256']}`；"
        f"split = `{config['capability_prior']['split']}`，"
        f"metrics = `{config['capability_prior']['metrics']}`，"
        f"final_test_read = `{config['capability_prior']['final_test_read']}`。",
    ]
    if config.get("visual_projector"):
        lines.append(
            f"- 视觉上下文：`{config['visual_projector']['path']}`，"
            f"SHA256 `{config['visual_projector']['sha256']}`；"
            f"{config['visual_projector']['usage']}。"
        )
    lines += [
        "",
        "## 2. 训练配置",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| Episodes | {summary['episodes']} / {config['max_episodes']} |",
        f"| ε 预算 | {config['epsilon_decay_steps']:,} 环境交互步 |",
        f"| γ | {config['gamma']} |",
        f"| 学习率 / Batch / Replay | {config['learning_rate']} / "
        f"{config['batch_size']} / {config['replay_capacity']:,} |",
        f"| 训练任务 | {config['sampling_frame']} |",
        f"| 训练顺序 | `{config['order_mode']}`，seed = "
        f"{config['order_seed'] if 'order_seed' in config else SEED + 20261031} |",
        f"| 训练拓扑 / 验证拓扑 | `{config['train_topology_split']}` / "
        f"`{config['validation_topology_split']}` |",
        f"| 验证 | 每 {config['validation_every']} Episodes，"
        f"{config['validation_source']} |",
        f"| 早停 | disabled = {config['early_stop_disabled']}；"
        "仅 NaN/Inf、Q 发散、程序错误中止 |",
        f"| 奖励 | `{config['reward']}` |",
        "",
        "## 3. 训练曲线",
        "",
        f"![training return]({figures['return']})",
        "",
        f"![validation]({figures['validation']})",
        "",
        f"![epsilon]({figures['epsilon']})",
        "",
        f"![Q]({figures['q']})",
        "",
        "## 4. Dev60 Validation 记录",
        "",
        "| Episode | JointR | SR | OSR | SPL | nDTW | E2E (ms) | "
        "Loss | Q mean | Q max | ε |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run["validations"]:
        lines.append(
            f"| {row['episode']} | {row['joint_reward']:.4f} | "
            f"{100*row['SR']:.2f}% | {100*row.get('OSR', np.nan):.2f}% | "
            f"{row.get('SPL', np.nan):.4f} | {row['nDTW']:.4f} | "
            f"{row['E2E_ms']:.2f} | {row['train_loss_mean']:.4f} | "
            f"{row['train_q_mean']:.3f} | {row['train_q_max']:.3f} | "
            f"{run['train'][row['episode']-1]['epsilon']:.4f} |"
        )
    lines += [
        "",
        "## 5. Dev180 best / last",
        "",
        "| Checkpoint | Episode | SR | OSR | SPL | nDTW | E2E (ms) | "
        "P95 (ms) | JointR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for which in ("best", "last"):
        summary_eval = run["evals"].get(which)
        if summary_eval is None:
            lines.append(f"| {which} | — | — | — | — | — | — | — | — |")
            continue
        overall = summary_eval["overall"]
        lines.append(
            f"| {which} | {summary_eval['checkpoint_episode']} | "
            f"{100*overall['SR']:.2f}% | {100*overall['OSR']:.2f}% | "
            f"{overall['SPL']:.4f} | {overall['nDTW']:.4f} | "
            f"{overall['E2E_ms']:.2f} | {overall['P95_ms']:.2f} | "
            f"**{overall['JointR']:.4f}** |"
        )
    lines += [
        "",
        f"![Dev180 bars]({figures['dev180']})",
        "",
        "## 6. MEC 选择频率（Dev180）",
        "",
        f"![MEC usage]({figures['mec']})",
        "",
        "| MEC | best 使用率 | last 使用率 |",
        "|---|---:|---:|",
    ]
    for model in MODEL_NAMES:
        best = run["evals"].get("best")
        last = run["evals"].get("last")
        best_ratio = (
            best["overall"]["model_usage_ratio"].get(model, 0.0)
            if best else None
        )
        last_ratio = (
            last["overall"]["model_usage_ratio"].get(model, 0.0)
            if last else None
        )
        lines.append(
            f"| {model} | {fmt(best_ratio, percent=True)} | "
            f"{fmt(last_ratio, percent=True)} |"
        )
    lines += [
        "",
        "## 7. Q 稳定性（只读审计）",
        "",
    ]
    if run["q_audit"] is None:
        lines.append(
            "Q 审计尚未运行；训练曲线中的 Q mean/max 已在上图给出。"
        )
    else:
        audit = run["q_audit"]
        best = audit["checkpoints"]["best"]
        last = audit["checkpoints"]["last"]
        lines += [
            f"- 固定状态数：{audit['state_count']}；"
            f"行为策略：{audit['behavior_policy']}。",
            f"- best：Ep{best['episode']}，Q mean {best['q_mean']:.4f}，"
            f"Q max {best['q_max']:.4f}，Q std {best['q_std_mean']:.4f}，"
            f"top-2 gap {best['q_top2_gap_mean']:.4f}。",
            f"- last：Ep{last['episode']}，Q mean {last['q_mean']:.4f}，"
            f"Q max {last['q_max']:.4f}，Q std {last['q_std_mean']:.4f}，"
            f"top-2 gap {last['q_top2_gap_mean']:.4f}。",
            f"- best/last 贪心动作一致率："
            f"**{100*audit['best_last_greedy_agreement']:.2f}%**。",
        ]
    lines += [
        "",
        "## 8. 收敛与 Final Test 决策",
        "",
        "预注册标准：完成 3500 Episodes；数值有限；至少 5 次 Dev 验证；"
        "最近 3 次 JointR 极差 ≤ 0.01；最后值 ≥ 最佳 − 0.005；"
        "且不再形成新最佳。",
        "",
        f"- `converged`：**{decision['converged']}**",
        f"- 最佳 Dev60 JointR：**{fmt(decision['best_dev60_joint'])}**；"
        f"竞争力阈值：**{decision['competitiveness_threshold']:.4f}**"
        "（同批 Dev60 最强非特权 Baseline，Min-Latency）",
        f"- `competitive`：**{decision['competitive']}**",
        f"- `final_test_eligible`：**{decision['final_test_eligible']}**",
        f"- 决策：**{decision['decision']}**；"
        f"Final Test read = **{decision['final_test_read']}**",
        "",
        "## 9. 结论",
        "",
        f"- 实际训练 {summary['episodes']} Episodes；收敛状态 "
        f"`{summary['status']}`；最佳 Checkpoint 为 Ep"
        f"{summary['best_episode']}（Dev60 JointR "
        f"{fmt(summary['best_joint'])}）。",
        f"- Dev180 best/last 与 Q 稳定性见上表；未读取一次性 Final Test。",
        "",
        "## 10. 产物",
        "",
        f"- 训练目录：`{run['run_dir']}`",
        f"- Checkpoints：`{run['run_dir']}/checkpoints/best.pt`、"
        f"`last.pt`、`last_full.pt`",
        f"- TensorBoard：`runs/tensorboard/"
        f"association_stage2k_{config['state_mode']}/`",
        f"- 曲线：`{FIG_PURE if key == 'A' else FIG_VISUAL}`",
        "- 大型权重、Replay Buffer、TensorBoard Event 与逐 Episode CSV "
        "保留在 `runs/` 下，不提交 Git。",
    ]
    return "\n".join(lines)


def comparison_markdown(runs, decisions, figures, parity):
    summaries = {key: training_summary(run) for key, run in runs.items()}
    lines = [
        "# Stage-2k v2 视觉消融对照：A Pure MEC vs B +128-D Visual",
        "",
        f"- 日期：{utc_now()}",
        "- 分支：`experiment/v83-spatial-foundation`",
        "- 两个变体共享 Train 3500 任务清单、随机顺序、种子、Train/Dev 拓扑、"
        "奖励、MLP 结构与 Dev60 验证。",
        "- 未读取 Stage-2j Final Test 300；Final Test 180 是否读取由预注册门槛决定。",
        "",
        "## 0. TL;DR",
        "",
        "| 变体 | 实际 Episodes | 最终 ε | 最佳 Dev60 JointR | 最佳 Episode | "
        "最后 Dev60 JointR | converged | 决策 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for key in ("A", "B"):
        item = summaries[key]
        lines.append(
            f"| {key} | {item['episodes']} | "
            f"{item['epsilon_final']:.4f} | {fmt(item['best_joint'])} | "
            f"{item['best_episode']} | {fmt(item['last_joint'])} | "
            f"{item['converged']} | {decisions[key]['decision']} |"
        )
    lines += [
        "",
        "## 1. 状态设计对照表",
        "",
        "| 项 | A Pure MEC | B + Visual |",
        "|---|---|---|",
        "| MLP 输入维度 | **384** | **512** |",
        "| 决策状态维度 | 404（+20 mask） | 532（+20 mask） |",
        "| 网络 | MLP 384→256→256→20 | MLP 512→256→256→20 |",
        "| UAV 位置 | 2 | 2 |",
        "| Episode 步数 t/400 | 1 | 1 |",
        "| running_mean_E2E_ms/100 | 1 | 1 |",
        "| 候选 MEC 特征 | 20 × 19 | 20 × 19（同 A） |",
        "| 每 MEC 能力指标 | SR/OSR/SPL/nDTW | SR/OSR/SPL/nDTW（同 A） |",
        "| 可用性 | 独立 20-D 动作 Mask | 独立 20-D 动作 Mask |",
        "| 视觉上下文 | **无** | **128-D 冻结 VisualProjector** |",
        "| GAB / 热图 / 新模块 | 无 | 无；投影前 GAB/map 块置为训练均值 |",
        "| 导航动作 / 目标位置 / Oracle / 成功标签 | 排除 | 排除 |",
        "",
        "## 2. 训练协议一致性",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| DQN seed | {SEED} |",
        f"| 顺序 seed | {SEED + 20261031} |",
        "| Train 任务数 | 3500（1167/1167/1166） |",
        "| 训练拓扑 | Train topology |",
        "| 验证 | 固定 Dev60，Dev topology |",
        "| ε 预算 | 500,000 环境交互步 |",
        "| Episodes | 3500，无早停 |",
        "| 奖励 | 与 JointR 严格一致，无成功/失败/特权奖励 |",
        "",
        f"- 全部一致性检查通过：**{parity['all_parity_checks_pass']}**",
        f"- Train order：{parity['checks']['training_order_length']} 条，"
        f"数组逐项相同 = {parity['checks']['training_order_equal']}，"
        f"SHA256 `{parity['checks']['training_order_sha256']}`。",
        f"- Dev60 验证集：{parity['checks']['validation_episode_count']} 条，"
        f"逐项相同 = {parity['checks']['validation_episodes_equal']}，"
        f"SHA256 `{parity['checks']['validation_uids_sha256']}`。",
        f"- 种子/拓扑/奖励/ε 预算/γ/隐藏层/能力先验："
        f"{parity['checks']['seed_equal']} / "
        f"{parity['checks']['train_topology_split_equal']} / "
        f"{parity['checks']['reward_equal']} / "
        f"{parity['checks']['epsilon_decay_steps_equal']} / "
        f"{parity['checks']['gamma_equal']} / "
        f"{parity['checks']['hidden_layers_equal']} / "
        f"{parity['checks']['capability_prior_sha_equal']}。",
        "",
        "由于策略动作不同，闭环轨迹和实际环境步数会不同，这是预期行为。",
        "",
        "## 3. 训练结果",
        "",
        "| 指标 | A | B |",
        "|---|---:|---:|",
    ]
    rows = [
        ("实际 Episodes", "episodes", "{:d}"),
        ("环境交互步", "total_steps", "{:,}"),
        ("最终 ε", "epsilon_final", "{:.4f}"),
        ("最终 TD Loss", "loss_final", "{:.4f}"),
        ("最终梯度范数", "grad_final", "{:.4f}"),
        ("最终 Q mean", "q_mean_final", "{:.4f}"),
        ("最终 Q max", "q_max_final", "{:.4f}"),
        ("最佳 Dev60 JointR", "best_joint", "{:.4f}"),
        ("最佳 Episode", "best_episode", "{:d}"),
        ("最后 Dev60 JointR", "last_joint", "{:.4f}"),
        ("训练状态", "status", "{}"),
    ]
    for label, key, fmt_string in rows:
        values = []
        for name in ("A", "B"):
            value = summaries[name][key]
            values.append(
                fmt_string.format(value) if value is not None else "—"
            )
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += [
        "",
        f"![training return]({figures['return']})",
        "",
        f"![validation]({figures['validation']})",
        "",
        f"![epsilon]({figures['epsilon']})",
        "",
        f"![Q]({figures['q']})",
        "",
        "## 4. Dev180 best / last 四项指标",
        "",
        "| 变体 | Checkpoint | Episode | SR | OSR | SPL | nDTW | "
        "E2E (ms) | JointR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "B"):
        for which in ("best", "last"):
            summary = runs[key]["evals"].get(which)
            if summary is None:
                lines.append(
                    f"| {key} | {which} | — | — | — | — | — | — | — |"
                )
                continue
            overall = summary["overall"]
            lines.append(
                f"| {key} | {which} | {summary['checkpoint_episode']} | "
                f"{100*overall['SR']:.2f}% | {100*overall['OSR']:.2f}% | "
                f"{overall['SPL']:.4f} | {overall['nDTW']:.4f} | "
                f"{overall['E2E_ms']:.2f} | **{overall['JointR']:.4f}** |"
            )
    lines += [
        "",
        "### 4.1 B − A 的配对差值（同一 Dev180 清单）",
        "",
        "| Checkpoint | ΔSR (pp) | ΔOSR (pp) | ΔSPL | ΔnDTW | "
        "ΔE2E (ms) | ΔJointR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for which in ("best", "last"):
        a = runs["A"]["evals"][which]["overall"]
        b = runs["B"]["evals"][which]["overall"]
        lines.append(
            f"| {which} | {100*(b['SR']-a['SR']):+.2f} | "
            f"{100*(b['OSR']-a['OSR']):+.2f} | "
            f"{b['SPL']-a['SPL']:+.4f} | "
            f"{b['nDTW']-a['nDTW']:+.4f} | "
            f"{b['E2E_ms']-a['E2E_ms']:+.2f} | "
            f"{b['JointR']-a['JointR']:+.4f} |"
        )
    lines += [
        "",
        "同一 Dev180 上的 Baseline JointR：Random 0.2010、Min-Latency 0.2477、"
        "Max-Quality 0.2256、Heuristic 0.2617、Oracle 0.4236（特权）。"
        "A best 0.1116 与 B best 0.1687 都低于所有非特权 Baseline 的点估计；"
        "B 相对 A 的提升是真实的同清单差值，但尚未达到任何非特权 Baseline 的水平。",
        "",
        "| best Checkpoint | ΔJointR vs Random [95% CI] | vs Min-Latency | "
        "vs Max-Quality | vs Heuristic |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("A", "B"):
        paired = {
            item["strategy"]: item["delta_JointR"]
            for item in runs[key]["evals"]["best"]["paired"]
        }
        cells = []
        for strategy in ("random", "min_latency", "max_quality", "heuristic"):
            item = paired[strategy]
            cells.append(
                f"{item['mean']:+.4f} [{item['low']:+.4f}, {item['high']:+.4f}]"
            )
        lines.append(
            f"| {key} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "A best 对四个非特权 Baseline 的 95% CI 都完全位于 0 以下；"
        "B best 对 Min-Latency、Max-Quality、Heuristic 的 CI 也完全位于 0 以下，"
        "但对 Random 的 CI 为 [−0.0836, +0.0171]，跨 0，"
        "因此只能报告“点估计低于 Random”，不能声称显著优于 Random。"
        "两个变体都没有击败任何非特权 Baseline。",
        "",
        f"![Dev180 bars]({figures['dev180']})",
        "",
        "## 5. Q 值稳定性",
        "",
        "| 变体 | Checkpoint | Episode | Q mean | Q max | Q std | "
        "Top-2 gap | best/last 贪心一致率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "B"):
        audit = runs[key]["q_audit"]
        if audit is None:
            lines.append(f"| {key} | — | — | — | — | — | — | — |")
            continue
        for which in ("best", "last"):
            item = audit["checkpoints"][which]
            lines.append(
                f"| {key} | {which} | {item['episode']} | "
                f"{item['q_mean']:.4f} | {item['q_max']:.4f} | "
                f"{item['q_std_mean']:.4f} | "
                f"{item['q_top2_gap_mean']:.4f} | "
                f"{100*audit['best_last_greedy_agreement']:.2f}% |"
            )
    lines += [
        "",
        "Q 审计使用同一组固定 Dev 状态（确定性 Min-Latency 行为策略），"
        "best/last 在完全相同的状态上前向；不同变体的 Q 尺度不直接可比，"
        "稳定性只在同一变体内部比较。",
        "",
        f"- A 的训练 Q mean 从 Ep200 的 "
        f"{runs['A']['train'][199]['q_mean']:.3f} 升到 Ep3500 的 "
        f"{runs['A']['train'][-1]['q_mean']:.3f}；best→last 审计 Q mean "
        f"{runs['A']['q_audit']['checkpoints']['best']['q_mean']:.2f}→"
        f"{runs['A']['q_audit']['checkpoints']['last']['q_mean']:.2f}，"
        "属于中等尺度漂移。",
        f"- B 的训练 Q mean 从 Ep200 的 "
        f"{runs['B']['train'][199]['q_mean']:.3f} 升到 Ep3500 的 "
        f"{runs['B']['train'][-1]['q_mean']:.3f}（约 "
        f"{runs['B']['train'][-1]['q_mean']/runs['B']['train'][199]['q_mean']:.0f}×），"
        f"TD Loss 从 {runs['B']['train'][199]['loss']:.4f} 升到 "
        f"{runs['B']['train'][-1]['loss']:.2f}，梯度范数从 "
        f"{runs['B']['train'][199]['grad_norm']:.3f} 升到 "
        f"{runs['B']['train'][-1]['grad_norm']:.1f}。best→last 审计 Q mean "
        f"{runs['B']['q_audit']['checkpoints']['best']['q_mean']:.2f}→"
        f"{runs['B']['q_audit']['checkpoints']['last']['q_mean']:.2f}、Q max "
        f"{runs['B']['q_audit']['checkpoints']['best']['q_max']:.2f}→"
        f"{runs['B']['q_audit']['checkpoints']['last']['q_max']:.2f}。"
        "**这是明显的 Q 过估计/尺度漂移，不是收敛。** 它未触发预注册的 "
        "1e6 数值中止阈值，但足以否定 B 作为可部署候选。",
        "",
        "## 6. MEC 选择频率",
        "",
        f"![MEC usage]({figures['mec']})",
        "",
        "| MEC | A best | A last | B best | B last |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        values = []
        for key in ("A", "B"):
            for which in ("best", "last"):
                summary = runs[key]["evals"].get(which)
                ratio = (
                    summary["overall"]["model_usage_ratio"].get(model, 0.0)
                    if summary else None
                )
                values.append(fmt(ratio, percent=True))
        lines.append(
            f"| {model} | {values[0]} | {values[1]} | "
            f"{values[2]} | {values[3]} |"
        )
    lines += [
        "",
        "## 7. 收敛与 Final Test 门槛",
        "",
        "| 变体 | converged | 最佳 Dev60 JointR | 阈值 | competitive | "
        "final_test_eligible | 决策 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for key in ("A", "B"):
        decision = decisions[key]
        lines.append(
            f"| {key} | {decision['converged']} | "
            f"{fmt(decision['best_dev60_joint'])} | "
            f"{decision['competitiveness_threshold']:.4f} | "
            f"{decision['competitive']} | "
            f"{decision['final_test_eligible']} | "
            f"{decision['decision']} |"
        )
    lines += [
        "",
        "门槛在 `STAGE2K_V2_PROTOCOL.md` 中预注册：G1 收敛；G2 最佳 Dev60 "
        "JointR 严格大于同批 Dev60 最强非特权 Baseline（Min-Latency "
        f"{DEV60_BASELINE_THRESHOLD:.4f}）。任一不满足即保留一次性 Final Test。",
        "",
        "## 8. 解释与限制",
        "",
        "- A/B 的闭环轨迹因策略动作不同而不同，因此实际环境交互步数和最终 ε "
        "不会逐点相同；训练任务清单、顺序、种子、拓扑、奖励和验证集相同。",
        "- B 的 128-D 视觉上下文来自现有冻结 `VisualProjector`；投影前把 summary "
        "的 GAB(20) 与 map(22) 块替换为训练均值，因此没有新增 GAB/热图模块，"
        "但保留了 current/goal/history 视觉 token 的池化信息。",
        "- 能力先验来自 Stage-2j Dev Val 180（seeds 2,000,000+），与 Stage-2k "
        "Train/Dev/Test（5M/6M/7M）不相交，且 `final_test_read=false`。",
        "- Q 值尺度随状态维度、奖励尺度和探索分布变化；本报告不把 Q 上升"
        "当作收敛。B 的 Q 绝对尺度（审计约 505–741，训练末 1726）和约 669× "
        "的增长已超出可解释的尺度差异，标记为明显的 Q 过估计/尺度漂移；"
        "A 也有约 60× 的增长，但绝对尺度较小。两者都未形成收敛平台。",
        "- 若未满足 G1/G2，Final Test 180 保持未读取；本报告不给出测试集成绩。",
        "",
        "## 9. 产物",
        "",
        f"- A：`{PURE_RUN}`",
        f"- B：`{VISUAL_RUN}`",
        f"- 曲线：`{FIG_CMP}`",
        "- 大型权重、Replay Buffer、TensorBoard Event 与逐 Episode CSV "
        "保留在 `runs/` 下，不提交 Git。",
    ]
    return "\n".join(lines)


def main():
    configure_style()
    for path in (FIG_PURE, FIG_VISUAL, FIG_CMP):
        path.mkdir(parents=True, exist_ok=True)
    runs = {"A": load_run(PURE_RUN), "B": load_run(VISUAL_RUN)}
    decisions = {
        "A": write_decision(runs["A"], "A"),
        "B": write_decision(runs["B"], "B"),
    }
    parity = protocol_parity(runs)
    atomic_write_json(
        Path("runs/v83_spatial_stage2/stage2k_v2_protocol_parity.json"),
        parity,
    )
    atomic_write_json(DECISION_PATH, {
        "version": "stage2k-v2-final-test-decisions-v1",
        "created_utc": utc_now(),
        "variants": decisions,
        "final_test_read": False,
    })

    figures = {}
    per_run_figures = {}
    for key in ("A", "B"):
        target = FIG_PURE if key == "A" else FIG_VISUAL
        figure_paths = {
            "return": target / "dqn_return_raw_ma100.png",
            "validation": target / "dqn_validation_metrics.png",
            "epsilon": target / "dqn_epsilon.png",
            "q": target / "dqn_qvalue_loss.png",
            "dev180": target / "dev180_best_last.png",
            "mec": target / "mec_usage.png",
        }
        plot_training_return({key: runs[key]}, figure_paths["return"])
        plot_validation_curves({key: runs[key]}, figure_paths["validation"])
        plot_epsilon({key: runs[key]}, figure_paths["epsilon"])
        plot_q_curves({key: runs[key]}, figure_paths["q"])
        plot_dev180_bars({key: runs[key]}, figure_paths["dev180"])
        plot_mec_usage({key: runs[key]}, figure_paths["mec"])
        per_run_figures[key] = {
            name: str(path.relative_to(Path("v83_spatial_stage2")))
            for name, path in figure_paths.items()
        }

    comparison_paths = {
        "return": FIG_CMP / "training_return.png",
        "validation": FIG_CMP / "validation_metrics.png",
        "epsilon": FIG_CMP / "epsilon.png",
        "q": FIG_CMP / "qvalue.png",
        "dev180": FIG_CMP / "dev180_best_last.png",
        "mec": FIG_CMP / "mec_usage.png",
    }
    plot_training_return(runs, comparison_paths["return"])
    plot_validation_curves(runs, comparison_paths["validation"])
    plot_epsilon(runs, comparison_paths["epsilon"])
    plot_q_curves(runs, comparison_paths["q"])
    plot_dev180_bars(runs, comparison_paths["dev180"])
    plot_mec_usage(runs, comparison_paths["mec"])
    figures = {
        name: str(path.relative_to(Path("v83_spatial_stage2")))
        for name, path in comparison_paths.items()
    }

    REPORT_PURE.write_text(
        run_markdown("A", runs["A"], per_run_figures["A"], decisions["A"]),
        encoding="utf-8",
    )
    REPORT_VISUAL.write_text(
        run_markdown("B", runs["B"], per_run_figures["B"], decisions["B"]),
        encoding="utf-8",
    )
    REPORT_CMP.write_text(
        comparison_markdown(runs, decisions, figures, parity),
        encoding="utf-8",
    )
    print(json.dumps({
        "reports": [
            str(REPORT_PURE), str(REPORT_VISUAL), str(REPORT_CMP),
        ],
        "decision": str(DECISION_PATH),
        "figures": {
            "A": per_run_figures["A"],
            "B": per_run_figures["B"],
            "comparison": figures,
        },
        "summaries": {
            key: training_summary(run) for key, run in runs.items()
        },
        "decisions": decisions,
        "protocol_parity": parity,
    }, indent=2, ensure_ascii=False), flush=True)
    print("STAGE2K_V2_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
