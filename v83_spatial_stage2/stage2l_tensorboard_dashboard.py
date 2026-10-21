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
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
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


PPO_UPDATE_TAGS = {
    "train/update_policy_loss": "ppo/policy_loss",
    "train/update_value_loss": "ppo/value_loss",
    "train/update_approx_kl": "ppo/approx_kl",
    "train/update_clip_fraction": "ppo/clip_fraction",
    "train/update_explained_variance": "ppo/explained_variance",
    "train/update_entropy": "ppo/update_entropy",
}

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
            writer.add_scalar("train/navigation_reward", float(nav), episode)
        if pen is not None:
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
        "- `train/navigation_reward`、`train/latency_penalty`",
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
        "```",
        "",
    ]
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    DASHBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    # Only the dashboard's own generated event directory is replaced.
    if EVENTS_ROOT.exists():
        shutil.rmtree(EVENTS_ROOT)
    EVENTS_ROOT.mkdir(parents=True, exist_ok=True)

    reports = []
    for run in RUNS:
        history, train, validation = _read_history(run)
        tb_events = _read_tb_events(run["tb_dir"])
        event_dir = EVENTS_ROOT / run["key"]
        writer = SummaryWriter(str(event_dir), flush_secs=5)
        _write_run_events(run, writer, history, train, validation)
        ppo_logged, ppo_missing = _write_ppo_updates(
            run, writer, tb_events, validation
        )
        writer.flush()
        writer.close()

        expected = ["train/jointr", "train/jointr_ma100"]
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


if __name__ == "__main__":
    main()
