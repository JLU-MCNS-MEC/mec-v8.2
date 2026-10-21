"""Build a unified TensorBoard dashboard for six completed Stage-2k/2l agents.

Reads only:
  * Stage-2k V2 Pure MEC DQN (384-D)
  * Stage-2k V2 Pure MEC + Visual DQN (512-D)
  * Stage-2l A / DQN, A / PPO, C / DQN, C / PPO

The original ``training_history.json`` files are authoritative.  PPO update
losses / KL / clip / explained variance are re-logged from the original
TensorBoard event files at their real update steps.  No original event,
checkpoint or history file is modified.  New summary event files are written
under the dashboard directory only.

Usage:
    python v83_spatial_stage2/stage2l_tensorboard_dashboard.py
    python v83_spatial_stage2/stage2l_tensorboard_dashboard.py --figures
    python v83_spatial_stage2/stage2l_tensorboard_dashboard.py --figures-only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from v83_spatial_stage2.stage2k_common import atomic_write_json, utc_now


DASHBOARD_ROOT = Path(
    "runs/tensorboard/rl_stage2k_stage2l_dashboard"
)
EVENTS_ROOT = DASHBOARD_ROOT / "events"
REPORT_PATH = DASHBOARD_ROOT / "LOAD_REPORT.json"
README_PATH = DASHBOARD_ROOT / "README.md"
LOG_PATH = DASHBOARD_ROOT / "tensorboard.log"
FIGURES_ROOT = DASHBOARD_ROOT / "figures"
EXPECTED_TRAIN_EPISODES = 3500
EXPECTED_VALIDATION_POINTS = 18

STAGE2L_ROOT = Path("runs/v83_spatial_stage2/stage2l_fixed_city")

RUNS = (
    {
        "key": "01_Stage2k_V2_PureMEC_DQN_384D",
        "label": "Stage-2k V2 · Pure MEC DQN (384D)",
        "history": Path("runs/v83_spatial_stage2/stage2k_pure_mec_v2"),
        "tb_dir": Path("runs/tensorboard/association_stage2k_pure_mec_v2"),
        "algorithm": "dqn",
        "state": "Pure MEC v2：384-D MLP 输入 + 20-D 可用性 mask",
        "environment": (
            "Stage-2k V2 ToyUAVEnv：每个 Episode 重新生成城市与目标；"
            "不是 Stage-2l 固定城市"
        ),
        "topology": (
            "Balanced / Quality-rich / Tradeoff-stress 三部署场景；"
            "每个场景一张固定 Train 拓扑，验证使用各自的 Dev 拓扑"
        ),
        "validation": "60 Dev episodes（6/10/4 per scenario）",
        "protocol": "stage2k-v2-3500-no-early-stop",
    },
    {
        "key": "02_Stage2k_V2_PureMEC_Visual_DQN_512D",
        "label": "Stage-2k V2 · Pure MEC + Visual DQN (512D)",
        "history": Path("runs/v83_spatial_stage2/stage2k_visual_ablation_v2"),
        "tb_dir": Path("runs/tensorboard/association_stage2k_visual_ablation_v2"),
        "algorithm": "dqn",
        "state": (
            "128-D frozen VisualProjector + 384-D Pure MEC v2 = 512-D "
            "MLP 输入 + 20-D 可用性 mask"
        ),
        "environment": (
            "Stage-2k V2 ToyUAVEnv：每个 Episode 重新生成城市与目标；"
            "不是 Stage-2l 固定城市"
        ),
        "topology": (
            "Balanced / Quality-rich / Tradeoff-stress 三部署场景；"
            "每个场景一张固定 Train 拓扑，验证使用各自的 Dev 拓扑"
        ),
        "validation": "60 Dev episodes（6/10/4 per scenario）",
        "protocol": "stage2k-v2-3500-no-early-stop",
    },
    {
        "key": "03_Stage2l_A_DQN",
        "label": "Stage-2l A · DQN",
        "history": STAGE2L_ROOT / "A" / "dqn",
        "tb_dir": STAGE2L_ROOT / "A" / "dqn" / "tensorboard",
        "algorithm": "dqn",
        "state": "Pure MEC v2：384-D MLP 输入 + 20-D 可用性 mask",
        "environment": (
            "Stage-2l 固定城市：map_seed=10,000,001；task_seed 决定起点、"
            "航向与目标；城市地标/道路/背景固定"
        ),
        "topology": "Balanced 单张固定拓扑 A_fixed（训练与验证共用）",
        "validation": "60 条独立 Dev 任务（20 Short / 20 Medium / 20 Long）",
        "protocol": "stage2l-fixed-city-v1",
    },
    {
        "key": "04_Stage2l_A_PPO",
        "label": "Stage-2l A · PPO",
        "history": STAGE2L_ROOT / "A" / "ppo",
        "tb_dir": STAGE2L_ROOT / "A" / "ppo" / "tensorboard",
        "algorithm": "ppo",
        "state": "Pure MEC v2：384-D MLP 输入 + 20-D 可用性 mask",
        "environment": (
            "Stage-2l 固定城市：map_seed=10,000,001；task_seed 决定起点、"
            "航向与目标；城市地标/道路/背景固定"
        ),
        "topology": "Balanced 单张固定拓扑 A_fixed（训练与验证共用）",
        "validation": "60 条独立 Dev 任务（20 Short / 20 Medium / 20 Long）",
        "protocol": "stage2l-fixed-city-v1",
    },
    {
        "key": "05_Stage2l_C_DQN",
        "label": "Stage-2l C · DQN",
        "history": STAGE2L_ROOT / "C" / "dqn",
        "tb_dir": STAGE2L_ROOT / "C" / "dqn" / "tensorboard",
        "algorithm": "dqn",
        "state": "Pure MEC v2：384-D MLP 输入 + 20-D 可用性 mask",
        "environment": (
            "Stage-2l 固定城市：map_seed=10,000,001；task_seed 决定起点、"
            "航向与目标；城市地标/道路/背景固定"
        ),
        "topology": (
            "Balanced / Quality-rich / Tradeoff-stress 三种部署场景；"
            "每种各一张固定拓扑（C_balanced / C_quality_rich / "
            "C_tradeoff_stress）"
        ),
        "validation": "60 条独立 Dev 任务（20 Short / 20 Medium / 20 Long）",
        "protocol": "stage2l-fixed-city-v1",
    },
    {
        "key": "06_Stage2l_C_PPO",
        "label": "Stage-2l C · PPO",
        "history": STAGE2L_ROOT / "C" / "ppo",
        "tb_dir": STAGE2L_ROOT / "C" / "ppo" / "tensorboard",
        "algorithm": "ppo",
        "state": "Pure MEC v2：384-D MLP 输入 + 20-D 可用性 mask",
        "environment": (
            "Stage-2l 固定城市：map_seed=10,000,001；task_seed 决定起点、"
            "航向与目标；城市地标/道路/背景固定"
        ),
        "topology": (
            "Balanced / Quality-rich / Tradeoff-stress 三种部署场景；"
            "每种各一张固定拓扑（C_balanced / C_quality_rich / "
            "C_tradeoff_stress）"
        ),
        "validation": "60 条独立 Dev 任务（20 Short / 20 Medium / 20 Long）",
        "protocol": "stage2l-fixed-city-v1",
    },
)


# Fixed categorical slots 1-6 from the validated reference palette.  DQN is
# solid and PPO dashed as a second, non-color identity channel; a per-figure
# CSV table is also written because three of the six hues are below 3:1 on the
# light surface.
RUN_COLORS = {
    "01_Stage2k_V2_PureMEC_DQN_384D": "#2a78d6",
    "02_Stage2k_V2_PureMEC_Visual_DQN_512D": "#eb6834",
    "03_Stage2l_A_DQN": "#1baf7a",
    "04_Stage2l_A_PPO": "#eda100",
    "05_Stage2l_C_DQN": "#e87ba4",
    "06_Stage2l_C_PPO": "#008300",
}
RUN_LEGEND_LABELS = {
    "01_Stage2k_V2_PureMEC_DQN_384D": "2k-V2 Pure 384D (DQN)",
    "02_Stage2k_V2_PureMEC_Visual_DQN_512D": "2k-V2 +Visual 512D (DQN)",
    "03_Stage2l_A_DQN": "2l-A Balanced (DQN)",
    "04_Stage2l_A_PPO": "2l-A Balanced (PPO)",
    "05_Stage2l_C_DQN": "2l-C 3 topologies (DQN)",
    "06_Stage2l_C_PPO": "2l-C 3 topologies (PPO)",
}
MA100_TAGS = {
    "train/jointr_ma100",
    "train/navigation_reward_ma100",
    "train/latency_penalty_ma100",
    "train/sr_ma100",
    "train/ndtw_ma100",
    "train/e2e_ms_ma100",
}
PPO_UPDATE_TAGS = {
    "train/update_policy_loss": "ppo/policy_loss",
    "train/update_value_loss": "ppo/value_loss",
    "train/update_approx_kl": "ppo/approx_kl",
    "train/update_clip_fraction": "ppo/clip_fraction",
    "train/update_explained_variance": "ppo/explained_variance",
    "train/update_entropy": "ppo/update_entropy",
}


@dataclass(frozen=True)
class FigureSpec:
    key: str
    title: str
    tag: str
    scope: str
    ylabel: str
    kind: str = "train"
    xlabel: str = "Training episode"
    percent: bool = False
    symlog: float | None = None
    note: str = ""


FIGURE_SPECS = (
    FigureSpec(
        "train_jointr",
        "Training JointR (per episode)",
        "train/jointr",
        "all",
        "JointR",
    ),
    FigureSpec(
        "train_jointr_ma100",
        "Training JointR (100-episode moving average)",
        "train/jointr_ma100",
        "all",
        "JointR (MA100)",
        kind="ma100",
    ),
    FigureSpec(
        "train_navigation_reward",
        "Training navigation reward (per episode)",
        "train/navigation_reward",
        "all",
        "Navigation reward",
    ),
    FigureSpec(
        "train_navigation_reward_ma100",
        "Training navigation reward (100-episode moving average)",
        "train/navigation_reward_ma100",
        "all",
        "Navigation reward (MA100)",
        kind="ma100",
    ),
    FigureSpec(
        "train_latency_penalty",
        "Training latency penalty (per episode)",
        "train/latency_penalty",
        "all",
        "Latency penalty",
    ),
    FigureSpec(
        "train_latency_penalty_ma100",
        "Training latency penalty (100-episode moving average)",
        "train/latency_penalty_ma100",
        "all",
        "Latency penalty (MA100)",
        kind="ma100",
    ),
    FigureSpec(
        "train_sr",
        "Training success rate (per episode)",
        "train/sr",
        "all",
        "Success rate (%)",
        percent=True,
    ),
    FigureSpec(
        "train_sr_ma100",
        "Training success rate (100-episode moving average)",
        "train/sr_ma100",
        "all",
        "Success rate (%)",
        kind="ma100",
        percent=True,
    ),
    FigureSpec(
        "train_ndtw",
        "Training nDTW (per episode)",
        "train/ndtw",
        "all",
        "nDTW",
    ),
    FigureSpec(
        "train_ndtw_ma100",
        "Training nDTW (100-episode moving average)",
        "train/ndtw_ma100",
        "all",
        "nDTW (MA100)",
        kind="ma100",
    ),
    FigureSpec(
        "train_e2e_ms",
        "Training E2E latency (per episode)",
        "train/e2e_ms",
        "all",
        "E2E (ms)",
    ),
    FigureSpec(
        "train_e2e_ms_ma100",
        "Training E2E latency (100-episode moving average)",
        "train/e2e_ms_ma100",
        "all",
        "E2E (ms, MA100)",
        kind="ma100",
    ),
    FigureSpec(
        "validation_jointr",
        "Dev validation JointR",
        "validation/jointr",
        "all",
        "JointR",
        kind="validation",
        xlabel="Validation episode",
    ),
    FigureSpec(
        "validation_sr",
        "Dev validation success rate",
        "validation/sr",
        "all",
        "Success rate (%)",
        kind="validation",
        xlabel="Validation episode",
        percent=True,
    ),
    FigureSpec(
        "validation_ndtw",
        "Dev validation nDTW",
        "validation/ndtw",
        "all",
        "nDTW",
        kind="validation",
        xlabel="Validation episode",
    ),
    FigureSpec(
        "validation_e2e_ms",
        "Dev validation E2E latency",
        "validation/e2e_ms",
        "all",
        "E2E (ms)",
        kind="validation",
        xlabel="Validation episode",
    ),
    FigureSpec(
        "dqn_td_loss",
        "DQN TD loss",
        "dqn/td_loss",
        "dqn",
        "TD loss (SmoothL1)",
    ),
    FigureSpec(
        "dqn_q_mean",
        "DQN mean Q value",
        "dqn/q_mean",
        "dqn",
        "Mean Q",
        note="Visual DQN reaches q_mean ~= 2,134 (linear); see the symlog variant.",
    ),
    FigureSpec(
        "dqn_q_mean_symlog",
        "DQN mean Q value (symlog)",
        "dqn/q_mean",
        "dqn",
        "Mean Q (symlog)",
        symlog=1.0,
        note="Symlog y-axis (linthresh=1); all 3,500 real points per run are retained.",
    ),
    FigureSpec(
        "dqn_q_max",
        "DQN maximum Q value",
        "dqn/q_max",
        "dqn",
        "Max Q",
        note="Visual DQN reaches q_max ~= 2,446 (linear); see the symlog variant.",
    ),
    FigureSpec(
        "dqn_q_max_symlog",
        "DQN maximum Q value (symlog)",
        "dqn/q_max",
        "dqn",
        "Max Q (symlog)",
        symlog=1.0,
        note="Symlog y-axis (linthresh=1); all 3,500 real points per run are retained.",
    ),
    FigureSpec(
        "dqn_grad_norm",
        "DQN gradient norm",
        "dqn/grad_norm",
        "dqn",
        "Gradient norm (pre-clip)",
    ),
    FigureSpec(
        "dqn_epsilon",
        "DQN epsilon schedule",
        "dqn/epsilon",
        "dqn",
        "Epsilon",
    ),
    FigureSpec(
        "ppo_policy_loss",
        "PPO policy loss",
        "ppo/policy_loss",
        "ppo",
        "Policy loss",
        kind="update",
        xlabel="Training episode (PPO update)",
    ),
    FigureSpec(
        "ppo_value_loss",
        "PPO value loss",
        "ppo/value_loss",
        "ppo",
        "Value loss",
        kind="update",
        xlabel="Training episode (PPO update)",
    ),
    FigureSpec(
        "ppo_policy_entropy",
        "PPO policy entropy (per episode)",
        "ppo/policy_entropy",
        "ppo",
        "Policy entropy",
    ),
    FigureSpec(
        "ppo_update_entropy",
        "PPO update entropy (update point)",
        "ppo/update_entropy",
        "ppo",
        "Update entropy",
        kind="update",
        xlabel="Training episode (PPO update)",
    ),
    FigureSpec(
        "ppo_approx_kl",
        "PPO approximate KL",
        "ppo/approx_kl",
        "ppo",
        "Approx. KL",
        kind="update",
        xlabel="Training episode (PPO update)",
    ),
    FigureSpec(
        "ppo_clip_fraction",
        "PPO clip fraction",
        "ppo/clip_fraction",
        "ppo",
        "Clip fraction",
        kind="update",
        xlabel="Training episode (PPO update)",
    ),
    FigureSpec(
        "ppo_explained_variance",
        "PPO explained variance",
        "ppo/explained_variance",
        "ppo",
        "Explained variance",
        kind="update",
        xlabel="Training episode (PPO update)",
    ),
)

VALIDATION_TAGS = {
    "jointr": ("joint_reward", "JointR"),
    "sr": ("SR", "sr"),
    "osr": ("OSR", "osr"),
    "spl": ("SPL", "spl"),
    "ndtw": ("nDTW", "ndtw"),
    "e2e_ms": ("E2E_ms", "e2e_ms"),
}


def _first_value(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _rolling(values, window=100):
    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < window:
        return out
    cumulative = np.cumsum(values)
    out[window - 1] = cumulative[window - 1] / window
    out[window:] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def _read_history(run):
    path = run["history"] / "training_history.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = list(payload.get("train", []))
    validation = list(payload.get("validation", []))
    if len(train) != 3500:
        raise RuntimeError(f"{run['label']} has {len(train)} train rows")
    return payload, train, validation


def _read_tb_events(tb_dir):
    """Return {tag: [(step, value), ...]} from the original event dir."""
    result = {}
    if not Path(tb_dir).is_dir():
        return result
    try:
        accumulator = EventAccumulator(
            str(tb_dir), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        tags = list(accumulator.Tags().get("scalars", []))
        for tag in tags:
            result[tag] = [
                (int(event.step), float(event.value))
                for event in accumulator.Scalars(tag)
            ]
    except Exception as exc:  # keep the dashboard generation non-fatal
        result["__error__"] = str(exc)
    return result


def _write_run_events(run, writer, history, train, validation):
    config = history.get("config", {})
    description = (
        f"## {run['label']}\n\n"
        f"- 算法：{run['algorithm'].upper()}\n"
        f"- 状态：{run['state']}\n"
        f"- 环境：{run['environment']}\n"
        f"- MEC 拓扑：{run['topology']}\n"
        f"- 验证：{run['validation']}\n"
        f"- 协议：{run['protocol']}\n"
        f"- 训练 Episodes：{len(train)}；验证点：{len(validation)}\n"
        f"- 历史配置：`seed={config.get('seed')}`, "
        f"`state_mode={config.get('state_mode')}`, "
        f"`state_dim={config.get('state_dim')}`\n"
    )
    writer.add_text("meta/experiment", description, 0)
    writer.add_text(
        "meta/config",
        json.dumps(config, indent=2, ensure_ascii=False),
        0,
    )

    jointr = []
    nav_rewards = []
    penalties = []
    sr = []
    ndtw = []
    e2e = []
    for row in train:
        episode = int(row["episode"])
        value = _first_value(row, ("episode_jointR", "episode_return"))
        if value is None:
            continue
        jointr.append(float(value))
        sr.append(float(bool(row.get("success", False))))
        ndtw.append(float(row.get("final_nDTW", 0.0)))
        e2e.append(float(row.get("mean_episode_e2e_ms", 0.0)))
        writer.add_scalar("train/jointr", float(value), episode)
        nav = row.get("navigation_reward")
        pen = row.get("latency_penalty")
        if nav is not None:
            nav_rewards.append(float(nav))
            writer.add_scalar("train/navigation_reward", float(nav), episode)
        if pen is not None:
            penalties.append(float(pen))
            writer.add_scalar("train/latency_penalty", float(pen), episode)
        writer.add_scalar("train/sr", float(bool(row.get("success", False))), episode)
        writer.add_scalar("train/ndtw", float(row.get("final_nDTW", 0.0)), episode)
        writer.add_scalar(
            "train/e2e_ms", float(row.get("mean_episode_e2e_ms", 0.0)), episode
        )
        if run["algorithm"] == "dqn":
            for source, tag in (
                ("loss", "dqn/td_loss"),
                ("q_mean", "dqn/q_mean"),
                ("q_max", "dqn/q_max"),
                ("grad_norm", "dqn/grad_norm"),
                ("epsilon", "dqn/epsilon"),
            ):
                if source in row and row[source] is not None:
                    writer.add_scalar(tag, float(row[source]), episode)
        else:
            if "policy_entropy" in row and row["policy_entropy"] is not None:
                writer.add_scalar(
                    "ppo/policy_entropy", float(row["policy_entropy"]), episode
                )

    for offset, (values, tag) in enumerate((
        (jointr, "train/jointr_ma100"),
        (nav_rewards, "train/navigation_reward_ma100"),
        (penalties, "train/latency_penalty_ma100"),
        (sr, "train/sr_ma100"),
        (ndtw, "train/ndtw_ma100"),
        (e2e, "train/e2e_ms_ma100"),
    )):
        ma = _rolling(values, 100)
        for index, value in enumerate(ma):
            if np.isfinite(value):
                writer.add_scalar(tag, float(value), index + 1)

    for row in validation:
        episode = int(row["episode"])
        for tag, keys in VALIDATION_TAGS.items():
            value = _first_value(row, keys)
            if value is not None:
                writer.add_scalar(
                    f"validation/{tag}", float(value), episode
                )


def _write_ppo_updates(run, writer, tb_events, validation):
    if run["algorithm"] != "ppo":
        return [], []
    logged = []
    missing = []
    for source_tag, target_tag in PPO_UPDATE_TAGS.items():
        events = tb_events.get(source_tag, [])
        if events:
            for step, value in events:
                writer.add_scalar(target_tag, value, step)
            logged.append({
                "tag": target_tag,
                "source": f"original TB {source_tag}",
                "points": len(events),
            })
        else:
            # Fallback: validation records carry these metrics every 200
            # episodes.  Keep the cadence explicit in the tag.
            metric = target_tag.split("/", 1)[1]
            source_key = {
                "policy_loss": "update_policy_loss",
                "value_loss": "update_value_loss",
                "approx_kl": "approx_kl",
                "clip_fraction": "clip_fraction",
                "explained_variance": "explained_variance",
                "update_entropy": "update_entropy",
            }.get(metric)
            if source_key is None:
                missing.append(target_tag)
                continue
            points = 0
            for row in validation:
                if source_key in row and row[source_key] is not None:
                    writer.add_scalar(
                        f"{target_tag}_validation_cadence",
                        float(row[source_key]),
                        int(row["episode"]),
                    )
                    points += 1
            if points:
                logged.append({
                    "tag": f"{target_tag}_validation_cadence",
                    "source": f"validation record {source_key}",
                    "points": points,
                    "note": "每 200 Episodes 的验证点，不是逐 Episode 更新",
                })
            else:
                missing.append(target_tag)
    return logged, missing


def _write_readme(run_reports, port, session, log_path):
    lines = [
        "# Stage-2k / Stage-2l TensorBoard 汇总",
        "",
        f"生成时间：{utc_now()}  ",
        f"启动会话：`tmux` session `{session}`  ",
        f"监听：`127.0.0.1:{port}`  ",
        f"日志：`{log_path}`",
        "",
        "## 六组实验",
        "",
        "| Run | 算法 | 状态 | 环境 | MEC 拓扑 | 验证 |",
        "|---|---|---|---|---|---|",
    ]
    for item in run_reports:
        lines.append(
            f"| `{item['key']}` | {item['label']} | {item['state']} | "
            f"{item['environment']} | {item['topology']} | "
            f"{item['validation']} |"
        )
    lines += [
        "",
        "## 曲线标签",
        "",
        "### 所有 Agent",
        "",
        "- `train/jointr`：逐 Episode JointR（真实记录）",
        "- `train/jointr_ma100`：从 Episode 100 起的 100-Episode 移动平均",
        "- `train/navigation_reward`、`train/latency_penalty`（逐 Episode）",
        "- `train/navigation_reward_ma100`、`train/latency_penalty_ma100`",
        "- `train/sr`、`train/ndtw`、`train/e2e_ms`（逐 Episode）",
        "- `train/sr_ma100`、`train/ndtw_ma100`、`train/e2e_ms_ma100`",
        "",
        "### DQN",
        "",
        "- `dqn/td_loss`、`dqn/q_mean`、`dqn/q_max`、`dqn/grad_norm`、`dqn/epsilon`",
        "",
        "### PPO",
        "",
        "- `ppo/policy_entropy`：逐 Episode（来自 training_history）",
        "- `ppo/policy_loss`、`ppo/value_loss`、`ppo/approx_kl`、"
        "`ppo/clip_fraction`、`ppo/explained_variance`、`ppo/update_entropy`："
        "来自原始 TensorBoard 的 `train/update_*` 事件，步数是真实更新 Episode；"
        "若原始事件缺失，则使用验证记录并加 `_validation_cadence` 后缀。",
        "",
        "### Validation（每 200 Episodes）",
        "",
        "- `validation/jointr`、`validation/sr`、`validation/osr`、"
        "`validation/spl`、`validation/ndtw`、`validation/e2e_ms`",
        "- 这些点只有 18 个（200…3400、3500），不是逐 Episode 曲线。",
        "",
        "## 数据来源与限制",
        "",
        "- 训练曲线优先读取 `training_history.json`；PPO 更新诊断从原 Event 重放。",
        "- MA100 只由真实逐 Episode 值计算，未插值。",
        "- 原 Event、Checkpoint、training_history 均未修改。",
        "- 缺失指标见 `LOAD_REPORT.json` 的 `missing_metrics`。",
        "",
        "## TensorBoard 采样设置",
        "",
        "- 启动参数为 `--samples_per_plugin scalars=5000`：最长序列 3500 点，"
        "因此所有真实点都会返回，没有降采样。",
        "- **不能使用 `scalars=0`**：TensorBoard 会把它解释为 0 个样本，"
        "Tag 列表仍在但所有曲线为空。",
        "- 数据级检查（不只是 HTTP 200）：",
        "",
        "```bash",
        "python v83_spatial_stage2/verify_tensorboard_dashboard.py \\",
        "  --url http://127.0.0.1:%d" % port,
        "```",
        "",
        "## 离线图片",
        "",
        "```bash",
        "python v83_spatial_stage2/stage2l_tensorboard_dashboard.py --figures-only",
        "```",
        "",
        "图片（PNG 300 dpi + PDF 矢量 + CSV 表格）写入 "
        "`runs/tensorboard/rl_stage2k_stage2l_dashboard/figures/`；"
        "索引为 `figures/FIGURE_INDEX.md`。",
        "",
        "## 启动与访问",
        "",
        "服务已由 tmux 在服务器 `127.0.0.1:%d` 启动；VS Code Remote SSH 中"
        "打开 Ports 面板，Forward a Port 输入 `%d`，然后在本机浏览器打开 "
        "`http://localhost:%d/`。" % (port, port, port),
        "",
        "服务器端自测：",
        "",
        "```bash",
        f"curl -I http://127.0.0.1:{port}/",
        f"python v83_spatial_stage2/verify_tensorboard_dashboard.py --url "
        f"http://127.0.0.1:{port}",
        "```",
        "",
    ]
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def _expected_point_count(spec):
    if spec.kind == "validation":
        return EXPECTED_VALIDATION_POINTS
    if spec.tag in MA100_TAGS:
        return EXPECTED_TRAIN_EPISODES - 99
    if spec.kind == "update":
        return None
    return EXPECTED_TRAIN_EPISODES


def _load_dashboard_events():
    """Read the six unified event directories without touching originals."""
    if not EVENTS_ROOT.is_dir():
        raise FileNotFoundError(
            f"{EVENTS_ROOT} does not exist; build the dashboard first"
        )
    event_data = {}
    for run in RUNS:
        event_dir = EVENTS_ROOT / run["key"]
        if not event_dir.is_dir():
            raise FileNotFoundError(
                f"missing dashboard events for {run['key']}: {event_dir}"
            )
        events = _read_tb_events(event_dir)
        if "__error__" in events:
            raise RuntimeError(
                f"cannot read {event_dir}: {events['__error__']}"
            )
        train_points = events.get("train/jointr", [])
        validation_points = events.get("validation/jointr", [])
        if len(train_points) != EXPECTED_TRAIN_EPISODES:
            raise RuntimeError(
                f"{run['key']}: train/jointr has {len(train_points)} points, "
                f"expected {EXPECTED_TRAIN_EPISODES}"
            )
        if len(validation_points) != EXPECTED_VALIDATION_POINTS:
            raise RuntimeError(
                f"{run['key']}: validation/jointr has "
                f"{len(validation_points)} points, expected "
                f"{EXPECTED_VALIDATION_POINTS}"
            )
        event_data[run["key"]] = events
    return event_data


def _collect_figure_series(event_data):
    """Map every figure spec to its real Event series, validating point counts."""
    collected = {}
    for spec in FIGURE_SPECS:
        series = []
        for run in RUNS:
            if spec.scope != "all" and run["algorithm"] != spec.scope:
                continue
            points = event_data[run["key"]].get(spec.tag, [])
            if not points:
                raise RuntimeError(
                    f"{spec.key}: {run['key']} has no points for {spec.tag}"
                )
            expected = _expected_point_count(spec)
            if expected is not None and len(points) != expected:
                raise RuntimeError(
                    f"{spec.key}: {run['key']} has {len(points)} points for "
                    f"{spec.tag}, expected {expected}"
                )
            if expected is None and len(points) < 2:
                raise RuntimeError(
                    f"{spec.key}: {run['key']} has too few update points for "
                    f"{spec.tag}"
                )
            points = sorted(
                (int(step), float(value)) for step, value in points
            )
            if not np.all(np.isfinite([value for _, value in points])):
                raise RuntimeError(
                    f"{spec.key}: non-finite value in {run['key']} {spec.tag}"
                )
            series.append((run, points))
        if not series:
            raise RuntimeError(f"{spec.key}: no runs matched scope {spec.scope}")
        collected[spec.key] = series
    return collected


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 10.3,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.6,
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.6,
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "text.color": "#0b0b0b",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "legend.frameon": False,
    })
    return plt


def _figure_subtitle(spec, series):
    counts = sorted({len(points) for _, points in series})
    if spec.kind == "validation":
        point_text = "18 Dev validation points per run"
    elif spec.kind == "ma100":
        point_text = "3,401 points from the real 100-episode moving average"
    elif spec.kind == "update":
        point_text = (
            "/".join(str(count) for count in counts)
            + " PPO update points per run"
        )
    else:
        point_text = "3,500 per-episode training points per run"
    if spec.scope == "all":
        scope_text = "6 runs (4 DQN + 2 PPO)"
    elif spec.scope == "dqn":
        scope_text = "4 DQN runs"
    else:
        scope_text = "2 PPO runs"
    return (
        f"{scope_text}; {point_text}; no interpolation. "
        "2k-V2 = regenerated city; 2l = fixed city."
    )


def _plotted_y(spec, points):
    x = np.asarray([step for step, _ in points], dtype=np.float64)
    y = np.asarray([value for _, value in points], dtype=np.float64)
    if spec.percent:
        y = y * 100.0
    return x, y


def _series_stats(run, points, spec):
    x, y = _plotted_y(spec, points)
    return {
        "run": run["key"],
        "label": RUN_LEGEND_LABELS[run["key"]],
        "algorithm": run["algorithm"],
        "points": len(points),
        "first_episode": int(x[0]),
        "last_episode": int(x[-1]),
        "last_value": float(y[-1]),
        "min_value": float(np.min(y)),
        "max_value": float(np.max(y)),
    }


def _write_figure_csv(path, spec, series):
    path.parent.mkdir(parents=True, exist_ok=True)
    x_unit = "validation_episode" if spec.kind == "validation" else (
        "training_episode"
    )
    value_unit = "percent" if spec.percent else "raw"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "figure", "tag", "run", "algorithm", "episode", "value",
            "x_unit", "value_unit",
        ])
        for run, points in series:
            x, y = _plotted_y(spec, points)
            for episode, value in zip(x, y):
                writer.writerow([
                    spec.key, spec.tag, run["key"], run["algorithm"],
                    int(episode), f"{value:.12g}", x_unit, value_unit,
                ])


def _draw_figure(plt, spec, series, png_path, pdf_path, dpi):
    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.15)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    sparse = spec.kind in ("validation", "update")
    raw_alpha = {6: 0.34, 4: 0.50, 2: 0.78}.get(len(series), 0.55)
    for run, points in series:
        x, y = _plotted_y(spec, points)
        is_dqn = run["algorithm"] == "dqn"
        ax.plot(
            x,
            y,
            color=RUN_COLORS[run["key"]],
            linestyle="-" if is_dqn else (0, (4, 2)),
            linewidth=0.52 if spec.kind == "train" else 1.05,
            alpha=raw_alpha if spec.kind == "train" else 0.96,
            marker=("o" if is_dqn else "s") if sparse else None,
            markersize=3.0 if spec.kind == "validation" else 2.2,
            markeredgecolor="#fcfcfb",
            markeredgewidth=0.35,
            markevery=1,
            label=RUN_LEGEND_LABELS[run["key"]],
            zorder=3,
        )

    if spec.symlog:
        ax.set_yscale("symlog", linthresh=spec.symlog)
    if spec.percent:
        ax.set_ylim(-2, 102)
    if spec.tag == "dqn/epsilon":
        ax.set_ylim(0, 1.05)

    all_y = np.concatenate([_plotted_y(spec, points)[1] for _, points in series])
    if not spec.symlog and np.min(all_y) < 0 < np.max(all_y):
        ax.axhline(0, color="#c3c2b7", linewidth=0.6, zorder=1)

    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    fig.text(
        0.085,
        0.968,
        spec.title,
        fontsize=10.5,
        fontweight="bold",
        color="#0b0b0b",
        ha="left",
        va="top",
    )
    fig.text(
        0.085,
        0.918,
        _figure_subtitle(spec, series),
        fontsize=6.4,
        color="#52514e",
        ha="left",
        va="top",
    )
    if spec.note:
        ax.text(
            0.995,
            0.02,
            spec.note,
            transform=ax.transAxes,
            fontsize=5.9,
            color="#52514e",
            ha="right",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "#fcfcfb",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=5,
        )

    ncol = 3 if len(series) >= 6 else (2 if len(series) >= 2 else 1)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.535, 0.790),
        ncol=ncol,
        fontsize=6.5,
        handlelength=2.1,
        columnspacing=1.0,
        labelspacing=0.35,
        borderaxespad=0.0,
        frameon=False,
    )
    ax.grid(True, axis="y", which="major", color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.0, width=0.5, colors="#52514e")
    ax.margins(x=0.008, y=0.08)

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor=fig.get_facecolor(),
    )
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.035,
        facecolor=fig.get_facecolor(),
        metadata={
            "Title": spec.title,
            "Creator": "MEC-V8.2 Stage-2k/2l dashboard",
        },
    )
    plt.close(fig)


def _write_figure_index(path, manifest):
    lines = [
        "# Stage-2k / Stage-2l Offline Figures",
        "",
        f"Generated: {manifest['created_utc']}  ",
        f"Source events: `{manifest['source_events']}`  ",
        f"Formats: PNG ({manifest['dpi']} dpi) + PDF vector  ",
        "",
        "所有训练图使用 Event 中的全部 3,500 个 Episode 点；验证图使用全部 18 个点；"
        "MA100 图使用 3,401 个真实移动平均点；PPO 更新图使用 A 222 / C 218 个"
        "更新点。没有插值或平滑。",
        "",
        "DQN 为实线，PPO 为虚线。每个图的完整绘图数据在 `data/<figure>.csv`，"
        "作为颜色对比度的表格视图；`series_summary.csv` 给出每条曲线的统计摘要。",
        "",
        "Q 值指标同时提供线性图和 symlog 图，便于同时观察视觉 DQN 的异常量级"
        "与其他运行。",
        "",
        "| Figure | Tag | Series | Points/series | PNG | PDF | Data |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for item in manifest["figures"]:
        counts = sorted({series["points"] for series in item["series"]})
        count_text = " / ".join(f"{count:,}" for count in counts)
        lines.append(
            f"| `{item['key']}` | `{item['tag']}` | {len(item['series'])} | "
            f"{count_text} | [PNG](./{item['png']}) | "
            f"[PDF](./{item['pdf']}) | [CSV](./{item['data']}) |"
        )
    lines += [
        "",
        "`figure_manifest.json` 记录每个图的 source tag、每条曲线的点数、"
        "首末 Episode、末值和极值。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _export_figures(figures_dir=FIGURES_ROOT, dpi=300):
    figures_dir = Path(figures_dir)
    if not figures_dir.is_absolute():
        figures_dir = PROJECT_ROOT / figures_dir
    data_dir = figures_dir / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    event_data = _load_dashboard_events()
    collected = _collect_figure_series(event_data)
    plt = _setup_matplotlib()

    figures = []
    summary_rows = []
    for spec in FIGURE_SPECS:
        series = collected[spec.key]
        png_path = figures_dir / f"{spec.key}.png"
        pdf_path = figures_dir / f"{spec.key}.pdf"
        data_path = data_dir / f"{spec.key}.csv"
        _draw_figure(plt, spec, series, png_path, pdf_path, dpi)
        _write_figure_csv(data_path, spec, series)
        series_info = [
            _series_stats(run, points, spec) for run, points in series
        ]
        for item in series_info:
            summary_rows.append({
                "figure": spec.key,
                "tag": spec.tag,
                "value_unit": "percent" if spec.percent else "raw",
                **item,
            })
        figures.append({
            "key": spec.key,
            "title": spec.title,
            "tag": spec.tag,
            "scope": spec.scope,
            "kind": spec.kind,
            "xlabel": spec.xlabel,
            "ylabel": spec.ylabel,
            "percent": spec.percent,
            "symlog_linthresh": spec.symlog,
            "png": png_path.name,
            "pdf": pdf_path.name,
            "data": f"data/{data_path.name}",
            "series": series_info,
        })
        print(
            f"[figures] {spec.key}: {len(series)} series, "
            f"png+pdf+csv written",
            flush=True,
        )

    manifest = {
        "version": "stage2k-stage2l-figure-export-v1",
        "created_utc": utc_now(),
        "source_events": str(EVENTS_ROOT),
        "dpi": int(dpi),
        "formats": ["png", "pdf"],
        "point_policy": (
            "all real Event points; 3500 training, 3401 MA100, 18 validation, "
            "PPO updates 222/218; no interpolation"
        ),
        "palette": RUN_COLORS,
        "line_styles": {"dqn": "solid", "ppo": "dashed"},
        "figures": figures,
    }
    manifest_path = figures_dir / "figure_manifest.json"
    index_path = figures_dir / "FIGURE_INDEX.md"
    summary_path = figures_dir / "series_summary.csv"
    atomic_write_json(manifest_path, manifest)
    _write_figure_index(index_path, manifest)

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "figure", "tag", "run", "algorithm", "points",
            "first_episode", "last_episode", "last_value",
            "min_value", "max_value", "value_unit",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                key: (
                    f"{row[key]:.12g}"
                    if key in ("last_value", "min_value", "max_value")
                    else row[key]
                )
                for key in fieldnames
            })

    print(
        f"[figures] wrote {len(figures)} figures to {figures_dir} "
        f"(PNG {dpi} dpi, PDF vector, CSV table view)",
        flush=True,
    )
    print(f"[figures] manifest: {manifest_path}", flush=True)
    print(f"[figures] index: {index_path}", flush=True)
    return manifest


def _build_dashboard():
    DASHBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    # Build beside the live directory and swap atomically at the end.  The
    # TensorBoard service can keep serving the old directory while we write.
    build_root = DASHBOARD_ROOT / ".events_build"
    previous_root = DASHBOARD_ROOT / ".events_previous"
    for path in (build_root, previous_root):
        if path.exists():
            shutil.rmtree(path)
    build_root.mkdir(parents=True, exist_ok=True)

    reports = []
    for run in RUNS:
        history, train, validation = _read_history(run)
        tb_events = _read_tb_events(run["tb_dir"])
        event_dir = EVENTS_ROOT / run["key"]
        writer = SummaryWriter(
            str(build_root / run["key"]), flush_secs=5
        )
        _write_run_events(run, writer, history, train, validation)
        ppo_logged, ppo_missing = _write_ppo_updates(
            run, writer, tb_events, validation
        )
        writer.flush()
        writer.close()

        expected = [
            "train/jointr",
            "train/jointr_ma100",
            "train/navigation_reward_ma100",
            "train/latency_penalty_ma100",
        ]
        if run["algorithm"] == "dqn":
            expected += [
                "dqn/td_loss", "dqn/q_mean", "dqn/q_max",
                "dqn/grad_norm", "dqn/epsilon",
            ]
        else:
            expected += ["ppo/policy_entropy"]
            expected += [item["tag"] for item in ppo_logged]
        missing = list(ppo_missing)
        if not tb_events:
            missing.append(
                f"original TB events unavailable at {run['tb_dir']}"
            )
        reports.append({
            "key": run["key"],
            "label": run["label"],
            "algorithm": run["algorithm"],
            "state": run["state"],
            "environment": run["environment"],
            "topology": run["topology"],
            "validation": run["validation"],
            "protocol": run["protocol"],
            "history": str(run["history"] / "training_history.json"),
            "original_tensorboard": str(run["tb_dir"]),
            "dashboard_events": str(event_dir),
            "train_episodes": len(train),
            "validation_points": len(validation),
            "metrics_expected": expected,
            "ppo_update_sources": ppo_logged,
            "missing_metrics": missing,
            "original_tb_tags": sorted(
                tag for tag in tb_events if tag != "__error__"
            ),
        })
        print(
            f"[dashboard] {run['label']}: {len(train)} episodes, "
            f"{len(validation)} validations, events={event_dir}",
            flush=True,
        )

    # Swap only after all six writers are closed; old events stay intact if
    # any writer fails.
    if EVENTS_ROOT.exists():
        EVENTS_ROOT.rename(previous_root)
    build_root.rename(EVENTS_ROOT)
    if previous_root.exists():
        shutil.rmtree(previous_root)

    atomic_write_json(REPORT_PATH, {
        "version": "stage2k-stage2l-tensorboard-dashboard-v1",
        "created_utc": utc_now(),
        "dashboard_root": str(DASHBOARD_ROOT),
        "events_root": str(EVENTS_ROOT),
        "runs": reports,
    })
    # Port/session are filled by the start script; read them if already set.
    port = 6006
    session = "tb_stage2l_dashboard"
    _write_readme(reports, port, session, LOG_PATH)
    print(f"[dashboard] wrote {REPORT_PATH}", flush=True)
    print(f"[dashboard] wrote {README_PATH}", flush=True)
    print("STAGE2K_STAGE2L_TENSORBOARD_DASHBOARD_COMPLETE", flush=True)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build the unified TensorBoard events and optionally export "
            "offline PNG/PDF figures."
        )
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        help="export PNG/PDF figures after rebuilding the dashboard events",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="skip the event rebuild and export figures from existing events",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_ROOT,
        help=f"figure output directory (default: {FIGURES_ROOT})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300)",
    )
    args = parser.parse_args(argv)

    if not args.figures_only:
        _build_dashboard()
    if args.figures or args.figures_only:
        _export_figures(args.figures_dir, args.dpi)
    elif args.figures_dir != FIGURES_ROOT:
        parser.error("--figures-dir requires --figures or --figures-only")


if __name__ == "__main__":
    main()
