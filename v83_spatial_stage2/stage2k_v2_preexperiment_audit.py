"""Stage-2k v2 pre-experiment audit: epsilon, Q scale and reward curves.

This script is read-only.  It loads the existing training histories of the
Stage-2k DQN runs, reports the actual epsilon schedule, Q mean/max growth and
raw/100-episode-moving-average rewards, and writes real (non-interpolated)
curves plus a JSON summary and a Chinese markdown report.

No checkpoint is modified, no training is launched, and no Final Test is read.
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
from matplotlib import font_manager

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now


OUTPUT_DIR = Path("v83_spatial_stage2/figures/stage2k_v2_preexperiment")
REPORT_PATH = Path("v83_spatial_stage2/STAGE2K_V2_PREEXPERIMENT_AUDIT.md")
SUMMARY_PATH = OUTPUT_DIR / "preexperiment_audit.json"

COLORS = {
    "stage2k_v1": "#2a78d6",
    "jointr_v2_full": "#eb6834",
    "no_visual": "#1baf7a",
    "pure_mec_v1": "#eda100",
}
LABELS = {
    "stage2k_v1": "Stage-2k v1 (478-D, stopped Ep2000)",
    "jointr_v2_full": "JointR v2 full (478-D, stopped Ep1320)",
    "no_visual": "No-visual (342-D, stopped Ep2450)",
    "pure_mec_v1": "Pure MEC v1 (322-D, stopped Ep1600)",
}
RUNS = (
    {
        "key": "stage2k_v1",
        "history": "runs/v83_spatial_stage2/stage2k/dqn/training_history.json",
        "result": "runs/v83_spatial_stage2/stage2k/dqn/training_result.json",
        "epsilon_decay_steps": 200_000,
        "q_available": False,
    },
    {
        "key": "jointr_v2_full",
        "history": "runs/v83_spatial_stage2/stage2k_jointr_v2/training_history.json",
        "result": "runs/v83_spatial_stage2/stage2k_jointr_v2/training_result.json",
        "epsilon_decay_steps": 200_000,
        "q_available": True,
    },
    {
        "key": "no_visual",
        "history": "runs/v83_spatial_stage2/stage2k_no_visual/training_history.json",
        "result": "runs/v83_spatial_stage2/stage2k_no_visual/training_result.json",
        "epsilon_decay_steps": 200_000,
        "q_available": True,
    },
    {
        "key": "pure_mec_v1",
        "history": "runs/v83_spatial_stage2/stage2k_pure_mec_state/training_history.json",
        "result": "runs/v83_spatial_stage2/stage2k_pure_mec_state/training_result.json",
        "epsilon_decay_steps": 200_000,
        "q_available": True,
    },
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


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


def moving_average(values, window=100):
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < window:
        return output
    kernel = np.ones(window, dtype=np.float64) / window
    output[window - 1:] = np.convolve(values, kernel, mode="valid")
    return output


def load_run(config):
    history = json.loads(Path(config["history"]).read_text(encoding="utf-8"))
    rows = history["train"]
    validations = history.get("validation", [])
    result_payload = {}
    if Path(config["result"]).is_file():
        result_payload = json.loads(
            Path(config["result"]).read_text(encoding="utf-8")
        )
    episodes = np.asarray([row["episode"] for row in rows], dtype=np.float64)
    returns = np.asarray([
        row.get("episode_return", row.get("return", np.nan))
        for row in rows
    ], dtype=np.float64)
    epsilon = np.asarray([row["epsilon"] for row in rows], dtype=np.float64)
    total_steps = np.asarray(
        [row["total_steps"] for row in rows], dtype=np.float64
    )
    q_mean = (
        np.asarray([row["q_mean"] for row in rows], dtype=np.float64)
        if config["q_available"] else None
    )
    q_max = (
        np.asarray([row["q_max"] for row in rows], dtype=np.float64)
        if config["q_available"] else None
    )
    loss = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    grad = (
        np.asarray([row["grad_norm"] for row in rows], dtype=np.float64)
        if "grad_norm" in rows[0] else None
    )
    ma100 = moving_average(returns, 100)
    peak_index = int(np.nanargmax(ma100)) if np.isfinite(ma100).any() else None
    q_summary = None
    if q_mean is not None:
        # Episode 1 is dominated by random exploration (Q is still near the
        # random initialisation), so ratios from Ep1 are unstable.  Use the
        # first validation-adjacent Episode (200) as the anchor.
        anchor_index = int(np.argmax(episodes >= 200))
        q_summary = {
            "first_episode": int(episodes[0]),
            "anchor_episode": int(episodes[anchor_index]),
            "last_episode": int(episodes[-1]),
            "q_mean_first": float(q_mean[0]),
            "q_mean_anchor": float(q_mean[anchor_index]),
            "q_mean_last": float(q_mean[-1]),
            "q_mean_max": float(np.max(q_mean)),
            "q_max_first": float(q_max[0]),
            "q_max_anchor": float(q_max[anchor_index]),
            "q_max_last": float(q_max[-1]),
            "q_max_max": float(np.max(q_max)),
            "q_mean_ratio": float(
                q_mean[-1] / q_mean[anchor_index]
            ) if q_mean[anchor_index] > 0 else None,
            "q_max_ratio": float(
                q_max[-1] / q_max[anchor_index]
            ) if q_max[anchor_index] > 0 else None,
        }
    validation_rows = [{
        "episode": int(row["episode"]),
        "joint_reward": float(row["joint_reward"]),
        "SR": float(row.get("SR", np.nan)),
        "nDTW": float(row.get("nDTW", np.nan)),
        "E2E_ms": float(row.get("E2E_ms", np.nan)),
    } for row in validations]
    best_validation = (
        max(validation_rows, key=lambda item: item["joint_reward"])
        if validation_rows else None
    )
    return {
        "key": config["key"],
        "label": LABELS[config["key"]],
        "history_path": config["history"],
        "result_path": config["result"],
        "episodes": int(episodes[-1]),
        "total_steps": int(total_steps[-1]),
        "epsilon_decay_steps": int(config["epsilon_decay_steps"]),
        "epsilon_first": float(epsilon[0]),
        "epsilon_last": float(epsilon[-1]),
        "epsilon_min": float(np.min(epsilon)),
        "epsilon_reached_floor_at_step": int(
            total_steps[int(np.argmax(epsilon <= 0.0500001))]
        ) if np.any(epsilon <= 0.0500001) else None,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "ma100_peak": (
            float(ma100[peak_index]) if peak_index is not None else None
        ),
        "ma100_peak_episode": (
            int(episodes[peak_index]) if peak_index is not None else None
        ),
        "ma100_last": (
            float(ma100[-1]) if np.isfinite(ma100[-1]) else None
        ),
        "loss_first": float(loss[0]),
        "loss_last": float(loss[-1]),
        "loss_mean": float(np.mean(loss)),
        "grad_mean": float(np.mean(grad)) if grad is not None else None,
        "q": q_summary,
        "validations": validation_rows,
        "best_validation": best_validation,
        "last_validation": validation_rows[-1] if validation_rows else None,
        "stop_reason": result_payload.get("stop_reason"),
        "status": result_payload.get("status"),
        "converged": result_payload.get("converged"),
        "early_stop_triggered": result_payload.get("early_stop_triggered"),
        "arrays": {
            "episodes": episodes,
            "total_steps": total_steps,
            "returns": returns,
            "ma100": ma100,
            "epsilon": epsilon,
            "q_mean": q_mean,
            "q_max": q_max,
        },
    }


def plot_epsilon(runs):
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    max_steps = max(run["total_steps"] for run in runs)
    schedule_steps = np.linspace(0.0, float(max_steps), 2000)
    schedule = np.where(
        schedule_steps <= 200_000.0,
        1.0 - 0.95 * schedule_steps / 200_000.0,
        0.05,
    )
    ax.plot(
        schedule_steps / 1000.0, schedule,
        color=INK_SECONDARY, linewidth=1.6,
        label="ε schedule: 1.0 → 0.05 over 200k steps",
    )
    for run in runs:
        ax.scatter(
            [run["total_steps"] / 1000.0], [run["epsilon_last"]],
            s=46, color=COLORS[run["key"]], edgecolor=SURFACE,
            linewidth=1.5, zorder=3,
            label=f"{run['label']}, ε={run['epsilon_last']:.3f}",
        )
    ax.axvline(
        200.0, color=MUTED, linestyle=(0, (3, 3)), linewidth=1.0,
        zorder=0,
    )
    ax.text(
        203.0, 0.55, "200k interaction steps",
        color=INK_SECONDARY, fontsize=8,
    )
    ax.set_xlabel("Environment interaction steps (thousands)")
    ax.set_ylabel("ε (actual)")
    ax.set_title("实际 ε 衰减：1.0 → 0.05，名义衰减步数 200k")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    path = OUTPUT_DIR / "epsilon_vs_steps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_q(runs):
    selected = [run for run in runs if run["q"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharex=True)
    for run in selected:
        arrays = run["arrays"]
        axes[0].plot(
            arrays["episodes"], arrays["q_mean"],
            color=COLORS[run["key"]], label=run["label"],
        )
        axes[1].plot(
            arrays["episodes"], arrays["q_max"],
            color=COLORS[run["key"]], label=run["label"],
        )
    axes[0].set_title("Q mean（逐 Episode 训练网络，可用动作）")
    axes[1].set_title("Q max（逐 Episode 训练网络，可用动作）")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Q value")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    path = OUTPUT_DIR / "q_mean_qmax.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_returns(runs):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.3), sharex=False)
    for ax, run in zip(axes.ravel(), runs):
        arrays = run["arrays"]
        color = COLORS[run["key"]]
        ax.plot(
            arrays["episodes"], arrays["returns"],
            color=GRID, linewidth=0.7, zorder=0,
            label="Raw return",
        )
        ax.plot(
            arrays["episodes"], arrays["ma100"],
            color=color, linewidth=2.0, zorder=2,
            label="100-episode MA",
        )
        peak = run["ma100_peak_episode"]
        if peak is not None:
            peak_value = run["ma100_peak"]
            ax.scatter(
                [peak], [peak_value], s=42, color=color,
                edgecolor=SURFACE, linewidth=1.5, zorder=3,
            )
            ax.annotate(
                f"peak {peak_value:.3f} @ Ep{peak}",
                xy=(peak, peak_value),
                xytext=(6, 8), textcoords="offset points",
                color=INK_SECONDARY, fontsize=7.5,
            )
        ax.set_title(run["label"], loc="left")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode return = JointR")
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("训练原始 Return 与 100-Episode 移动平均（真实记录，无插值）",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUTPUT_DIR / "return_raw_ma100.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_validation(runs):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for run in runs:
        rows = run["validations"]
        if not rows:
            continue
        source = (
            "train-subset val" if run["key"] == "stage2k_v1"
            else "fixed Dev60"
        )
        ax.plot(
            [row["episode"] for row in rows],
            [row["joint_reward"] for row in rows],
            color=COLORS[run["key"]], marker="o", markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0,
            label=f"{run['label']} [{source}]",
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Validation JointR")
    ax.set_title(
        "训练期 Validation JointR（Stage-2k v1 为 train 子集；其余为固定 Dev60）"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = OUTPUT_DIR / "validation_jointr.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_markdown(summary):
    lines = [
        "# Stage-2k v2 前置审计：ε、Q 值与奖励曲线",
        "",
        f"- 生成时间（UTC）：{summary['created_utc']}",
        "- 分支：`experiment/v83-spatial-foundation`",
        "- 性质：**只读审计**；未启动训练、未修改 Checkpoint、未读取 Final Test。",
        "- 数据来源：各运行目录下的 `training_history.json` 与 "
        "`training_result.json`；曲线直接由逐 Episode 真实记录生成，没有插值或补点。",
        "",
        "## 0. 结论摘要",
        "",
        "1. 现有 DQN 的 ε 名义上从 **1.0** 线性衰减到 **0.05**，衰减预算为 "
        "**200,000 个环境交互步**；`stage2k_jointr_v2`、`no_visual`、"
        "`pure_mec_v1` 的 ε 都实际到达了 0.05。",
        "2. Q mean/max 在训练后期**持续增大**，而不是收敛到稳定平台。"
        "例如 JointR v2 full 在 Ep1200 的 Q mean/max 为 "
        f"**{summary['runs']['jointr_v2_full']['q']['q_mean_last']:.2f}/"
        f"{summary['runs']['jointr_v2_full']['q']['q_max_last']:.2f}**；"
        "pure MEC v1 在 Ep1600 为 "
        f"**{summary['runs']['pure_mec_v1']['q']['q_mean_last']:.2f}/"
        f"{summary['runs']['pure_mec_v1']['q']['q_max_last']:.2f}**。",
        "3. 训练 return 的 100-Episode 移动平均有峰值，但峰值**不是收敛**。"
        "峰值之后 return 可以明显下降；而且训练任务按 band/scenario 采样，"
        "窗口组成会改变 return，不能用峰值或早停本身证明收敛。",
        "4. 现有运行的停止都来自外部停止或基于验证集的早停。"
        "JointR v2 full、no_visual、pure MEC v1 的 `converged` 均为 **false**；"
        "Stage-2k v1 虽保存了 `converged=true`，但其验证用的是 **train 子集**，"
        "已被 `STAGE2K_DQN_CONVERGENCE_AUDIT.md` 判定为无效的 in-sample 标志。"
        "因此 v2 不应沿用 200k ε 预算或“奖励下降即停”的行为。",
        "",
        "## 1. ε 衰减审计",
        "",
        "| 运行 | 实际 Episodes | 环境交互步 | 名义衰减步数 | ε 首/末 | 首次到 0.05 的步数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("stage2k_v1", "jointr_v2_full", "no_visual", "pure_mec_v1"):
        run = summary["runs"][key]
        lines.append(
            f"| {run['label']} | {run['episodes']} | {run['total_steps']:,} | "
            f"{run['epsilon_decay_steps']:,} | "
            f"{run['epsilon_first']:.4f} / {run['epsilon_last']:.4f} | "
            f"{run['epsilon_reached_floor_at_step'] or '—'} |"
        )
    lines += [
        "",
        "说明：ε 按**环境交互步**而不是 Episode 数衰减；每个 Episode 的步数"
        "随 band 和策略变化，因此同一个 Episode 编号在不同 run 中的 ε 不同。",
        "",
        "![epsilon](figures/stage2k_v2_preexperiment/epsilon_vs_steps.png)",
        "",
        "## 2. Q 值增长审计",
        "",
        "| 运行 | Q mean Ep200 → 末 | 增长倍数 | Q max Ep200 → 末 | 增长倍数 | 全局 Q max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("jointr_v2_full", "no_visual", "pure_mec_v1"):
        run = summary["runs"][key]
        q = run["q"]
        lines.append(
            f"| {run['label']} | {q['q_mean_anchor']:.3f} → "
            f"{q['q_mean_last']:.3f} | {q['q_mean_ratio']:.1f}× | "
            f"{q['q_max_anchor']:.3f} → {q['q_max_last']:.3f} | "
            f"{q['q_max_ratio']:.1f}× | {q['q_max_max']:.2f} |"
        )
    lines += [
        "",
        "`stage2k_v1` 的逐 Episode 历史没有保存 Q 字段，因此不纳入 Q 曲线；"
        "其独立 Q 审计（Ep800 vs Ep2000）记录在 "
        "`STAGE2K_DQN_CONVERGENCE_AUDIT.md`。",
        "",
        "![q](figures/stage2k_v2_preexperiment/q_mean_qmax.png)",
        "",
        "观察：",
        "",
        "- JointR v2 full 的 Q mean 在 1320 Episodes 内从 "
        f"{summary['runs']['jointr_v2_full']['q']['q_mean_anchor']:.3f} "
        f"升到 {summary['runs']['jointr_v2_full']['q']['q_mean_last']:.3f}，"
        f"Q max 升到 {summary['runs']['jointr_v2_full']['q']['q_max_last']:.2f}；",
        "- pure MEC v1 的 Q 尺度明显更小，但同样持续上升；",
        "- Q 值上升与 ε 降到 0.05 同时发生，不能用“Q 还在涨”判断策略在变好，"
        "也不能仅凭 Q 尺度判断发散；需要同时看 Dev Validation、TD Loss 和梯度。",
        "",
        "## 3. 奖励曲线审计",
        "",
        "| 运行 | Return 均值 ± std | 100-MA 峰值 | 峰值 Episode | 100-MA 末值 | Loss 首 → 末 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("stage2k_v1", "jointr_v2_full", "no_visual", "pure_mec_v1"):
        run = summary["runs"][key]
        lines.append(
            f"| {run['label']} | {run['return_mean']:.4f} ± "
            f"{run['return_std']:.4f} | {run['ma100_peak']:.4f} | "
            f"{run['ma100_peak_episode']} | {run['ma100_last']:.4f} | "
            f"{run['loss_first']:.4f} → {run['loss_last']:.4f} |"
        )
    lines += [
        "",
        "![returns](figures/stage2k_v2_preexperiment/return_raw_ma100.png)",
        "",
        "关键点：",
        "",
        "- 训练 return 的峰值出现在早期/中期，之后并不单调；",
        "- 100-MA 峰值不能被称为“收敛”，因为峰值后仍然可以出现长期下降；",
        "- 训练任务的 band/scenario 组成会改变 return 的可比性，"
        "真正的模型选择必须使用固定 Dev Validation。",
        "",
        "## 4. Validation 与停止状态",
        "",
        "| 运行 | 验证次数 | 最佳 JointR (Episode) | 最后 JointR | 停止原因 | converged |",
        "|---|---:|---:|---:|---|---|",
    ]
    for key in ("stage2k_v1", "jointr_v2_full", "no_visual", "pure_mec_v1"):
        run = summary["runs"][key]
        best = run["best_validation"]
        last = run["last_validation"]
        lines.append(
            f"| {run['label']} | {len(run['validations'])} | "
            f"{best['joint_reward']:.4f} (Ep{best['episode']}) | "
            f"{last['joint_reward']:.4f} | {run['stop_reason'] or '—'} | "
            f"{run['converged']} |"
        )
    lines += [
        "",
        "![validation](figures/stage2k_v2_preexperiment/validation_jointr.png)",
        "",
        "现有 `converged` 判定要求最近的验证点形成稳定平台（最近 3 次极差"
        " ≤ 0.01，且最后值不低于最佳值 − 0.005），并且不再处于实质提升。"
        "早停本身不会令 `converged=true`。",
        "",
        "**关于 Stage-2k v1 的 `converged=true`**：该运行训练时的 Validation "
        "读取的是 60 个 **train Episode** + dev topology，是 in-sample 曲线；"
        "独立审计（`STAGE2K_DQN_CONVERGENCE_AUDIT.md`）已指出其 best/last "
        "Checkpoint 在真实 Dev180 上反而 last 更好。因此该标志不能作为"
        "“已收敛”的证据，v2 的收敛判定必须使用固定且独立的 60 个 Dev Episodes。",
        "",
        "## 5. 对 v2 的直接约束",
        "",
        "1. 两个变体共用 **EPSILON_DECAY_STEPS=500,000**；"
        "3500 Episodes 预计约 50–55 万环境交互步，ε 会在训练后段才降到 0.05。",
        "2. **不因普通奖励下降早停**；只有 NaN/Inf、Q 明显发散或程序错误才中止。",
        "3. 每 200 Episodes 在固定 60 个独立 Dev Episodes 上评估；"
        "最佳 Checkpoint 由 Dev JointR 选择，最后 Checkpoint 始终保留。",
        "4. 收敛判定只看固定 Dev Validation 的稳定平台，不看训练 return 峰值；"
        "未收敛则保留 Final Test，不消耗测试集。",
        "",
        "## 6. 产物",
        "",
        f"- 审计 JSON：`{SUMMARY_PATH}`",
        f"- 报告：`{REPORT_PATH}`",
        "- 曲线目录：`v83_spatial_stage2/figures/stage2k_v2_preexperiment/`",
        "",
        "所有旧权重、日志、TensorBoard 与报告均未修改或覆盖。",
    ]
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    runs = [load_run(config) for config in RUNS]
    paths = {
        "epsilon": str(plot_epsilon(runs)),
        "q": str(plot_q(runs)),
        "returns": str(plot_returns(runs)),
        "validation": str(plot_validation(runs)),
    }
    summary = {
        "version": "stage2k-v2-preexperiment-audit-v1",
        "created_utc": utc_now(),
        "final_test_read": False,
        "runs": {run["key"]: {
            key: value for key, value in run.items() if key != "arrays"
        } for run in runs},
        "figures": paths,
    }
    atomic_write_json(SUMMARY_PATH, summary)
    REPORT_PATH.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "summary": str(SUMMARY_PATH),
        "report": str(REPORT_PATH),
        "figures": paths,
        "runs": {
            key: {
                "episodes": summary["runs"][key]["episodes"],
                "epsilon_last": summary["runs"][key]["epsilon_last"],
                "q": summary["runs"][key]["q"],
            }
            for key in summary["runs"]
        },
    }, indent=2, ensure_ascii=False))
    print("STAGE2K_V2_PREEXPERIMENT_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
