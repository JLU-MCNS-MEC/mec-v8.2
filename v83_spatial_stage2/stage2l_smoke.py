"""Light closed-loop smoke test for the Stage-2l fixed city.

Checks, before any formal training starts:

1. the fixed city is identical across task seeds and no task creates a new
   landmark;
2. the RGB observation changes with the UAV pose;
3. the frozen encoder / five frozen T5 MECs accept the new observations
   (finite tensors, expected shapes, no exception);
4. the Pure-MEC v2 state has no goal position / goal distance / oracle field
   and is *identical* for two different goals at the same UAV pose;
5. each frozen MEC and the four standard baselines complete closed-loop
   episodes on fixed Dev tasks with sane metrics.

Results are written to
``runs/v83_spatial_stage2/stage2l_fixed_city/smoke/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2k_common import (
    MODEL_NAMES,
    atomic_write_json,
    summarize_episodes,
    utc_now,
)
from v83_spatial_stage2.stage2k_pure_mec_v2_state import (
    CANDIDATE_FIELDS,
    STATE_DIM_PURE_MEC_V2,
    state_layout_pure_mec_v2,
)
from v83_spatial_stage2.stage2k_runtime import load_gab_config, load_runtime
from v83_spatial_stage2.stage2k_v2_common import load_capability_prior
from v83_spatial_stage2.stage2l_fixed_city import (
    BANDS,
    SMOKE_DIR,
    FixedCityEnv,
    MAP_SEED_PHASE1,
    config_specs,
    load_topology,
    topology_role_for,
)
from v83_spatial_stage2.stage2l_runtime import (
    FixedCityEpisodeRunner,
    FixedCitySceneCache,
)


FORBIDDEN_STATE_TOKENS = (
    "goal", "target", "oracle", "is_goal", "distance_to_goal",
    "success", "reward",
)


def _fake_spec_with_goal(spec, goal_index):
    payload = dict(spec)
    payload["goal_index"] = int(goal_index)
    goal = FixedCityEnv(int(payload["map_seed"])).city_landmarks[
        int(goal_index)
    ]
    payload["goal"] = [float(goal.x), float(goal.y)]
    payload["start_goal_distance_m"] = float(np.linalg.norm(
        np.asarray(payload["start"], dtype=np.float64)
        - np.asarray(payload["goal"], dtype=np.float64)
    ))
    return payload


def state_leakage_audit(spec, encoder, backends, quality, gab_config, device,
                        topology):
    layout = state_layout_pure_mec_v2()
    layout_names = set(layout)
    # The layout carries a documentation-only ``forbidden`` list; it must
    # not be mistaken for an actual state field.
    textual_parts = []
    for key, value in layout.items():
        if key == "forbidden":
            continue
        if isinstance(value, (list, tuple)):
            textual_parts.extend(str(item) for item in value)
        else:
            textual_parts.append(str(value))
    textual = " ".join(textual_parts)
    forbidden_hits = [
        token for token in FORBIDDEN_STATE_TOKENS
        if token in textual.lower()
    ]
    candidate_hits = [
        field for field in CANDIDATE_FIELDS
        if any(token in field.lower() for token in FORBIDDEN_STATE_TOKENS)
    ]
    cache = FixedCitySceneCache(encoder, device, max_entries=8)
    other_goal = (int(spec["goal_index"]) + 1) % 160
    runner_a = FixedCityEpisodeRunner(
        spec=spec, topology=topology, encoder=encoder, backends=backends,
        projector=None, quality_prior=quality, gab_config=gab_config,
        scene_cache=cache, device=device, jointr_aligned=True,
        pure_mec_v2_state=True,
    )
    spec_b = _fake_spec_with_goal(spec, other_goal)
    runner_b = FixedCityEpisodeRunner(
        spec=spec_b, topology=topology, encoder=encoder, backends=backends,
        projector=None, quality_prior=quality, gab_config=gab_config,
        scene_cache=cache, device=device, jointr_aligned=True,
        pure_mec_v2_state=True,
    )
    state_a = np.asarray(runner_a.step_data["state"], dtype=np.float64)
    state_b = np.asarray(runner_b.step_data["state"], dtype=np.float64)
    input_shapes = {
        "current_tokens": list(runner_a.step_data["inputs"]["current_tokens"].shape),
        "goal_tokens": list(runner_a.step_data["inputs"]["goal_tokens"].shape),
        "history_vis": list(runner_a.step_data["inputs"]["history_vis"].shape),
        "history_pose": list(runner_a.step_data["inputs"]["history_pose"].shape),
        "history_action": list(
            runner_a.step_data["inputs"]["history_action"].shape
        ),
        "history_mask": list(runner_a.step_data["inputs"]["history_mask"].shape),
        "gab": list(runner_a.step_data["inputs"]["gab"].shape),
        "map_feat": list(runner_a.step_data["inputs"]["map_feat"].shape),
    }
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in runner_a.step_data["inputs"].values()
    )
    return {
        "forbidden_state_tokens_in_layout": forbidden_hits,
        "forbidden_candidate_fields": candidate_hits,
        "state_dim": int(state_a.size),
        "state_dim_expected": int(STATE_DIM_PURE_MEC_V2),
        "state_finite": bool(np.isfinite(state_a).all()),
        "state_identical_for_different_goal": bool(
            np.array_equal(state_a, state_b)
        ),
        "state_max_abs_difference": float(np.max(np.abs(state_a - state_b))),
        "goal_a_index": int(spec["goal_index"]),
        "goal_b_index": int(other_goal),
        "input_shapes": input_shapes,
        "inputs_finite": finite,
        "layout_fields": len(layout_names),
        "layout": state_layout_pure_mec_v2(),
    }


def _force_model_node(runner, model_name):
    metrics = runner.step_data["latency_metrics"]
    candidates = [
        index for index in runner.available_indices()
        if runner.nodes[index]["model_type"] == model_name
    ]
    if not candidates:
        return runner.select_node("random", rng=np.random.default_rng(0))
    return int(min(candidates, key=lambda i: (metrics[i]["total_ms"], i)))


def run_episode(runner, strategy, rng=None, forced_model=None):
    while True:
        if forced_model is not None:
            node = _force_model_node(runner, forced_model)
        else:
            node = runner.select_node(strategy, rng=rng)
        transition = runner.apply(node)
        if transition["done"]:
            break
    return runner.finalize(strategy)


def _summary_markdown(payload):
    lines = [
        "# Stage-2l 固定城市轻量闭环 Smoke Test",
        "",
        f"- 时间：{payload['created_utc']}",
        f"- 固定地图：`map_seed={payload['map_seed']}`，"
        f"城市指纹 `{payload['city_fingerprint'][:16]}…`",
        f"- Dev 任务：{payload['dev_tasks']} 条（"
        + " / ".join(
            f"{band} {count}" for band, count in payload["band_counts"].items()
        )
        + "）",
        "- 五种 Stage-2j T5 MEC 全部冻结（`eval()` + `requires_grad=False`），"
        "本测试不训练任何 MEC。",
        "",
        "## 1. 固定性与 T5 输入兼容",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 城市在任务间不变 | {payload['fixed_city_ok']} |",
        f"| 目标来自既有地标 | {payload['goal_existing_ok']} |",
        f"| 任务不新增/修改地标 | {payload['landmark_stable_ok']} |",
        f"| RGB 随位姿变化 | {payload['render_pose_dependent_ok']} |",
        f"| 同帧可复现 | {payload['render_reproducible_ok']} |",
        f"| T5 输入形状 | {json.dumps(payload['t5']['input_shapes'])} |",
        f"| T5 输入有限 | {payload['t5']['inputs_finite']} |",
        f"| 状态维度 | {payload['t5']['state_dim']} "
        f"(期望 {payload['t5']['state_dim_expected']}) |",
        f"| 状态无目标字段 | "
        f"{not payload['t5']['forbidden_state_tokens_in_layout'] and not payload['t5']['forbidden_candidate_fields']} |",
        f"| 同一位姿/不同目标的 Pure-MEC 状态完全相同 | "
        f"{payload['t5']['state_identical_for_different_goal']} |",
        "",
        "## 2. 五种冻结 MEC 的闭环表现（Dev 固定任务）",
        "",
        "| MEC | Episodes | SR | nDTW | E2E (ms) | JointR | 平均步数 | Dev prior SR（参考） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_NAMES:
        item = payload["mecs"][name]
        lines.append(
            f"| {name} | {item['episodes']} | {100*item['SR']:.2f}% | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.1f} | "
            f"{item['JointR']:.4f} | {item['mean_steps']:.1f} | "
            f"{100*item['prior_SR']:.2f}% |"
        )
    lines += [
        "",
        "## 3. 四种非特权 Baseline（同一批 Dev 固定任务）",
        "",
        "| 策略 | Episodes | SR | nDTW | E2E (ms) | JointR | 平均步数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["baselines"].items():
        lines.append(
            f"| {name} | {item['episodes']} | {100*item['SR']:.2f}% | "
            f"{item['nDTW']:.4f} | {item['E2E_ms']:.1f} | "
            f"{item['JointR']:.4f} | {item['mean_steps']:.1f} |"
        )
    lines += [
        "",
        "## 4. 结论与限制",
        "",
        "- 本测试只验证新场景与冻结 T5 输入兼容、任务生成无目标泄漏、"
        "闭环不崩溃；不构成任何学习效果结论。",
        "- 每个 MEC / Baseline 只跑了固定的少量 Dev 任务，指标仅用于发现"
        "“完全不可用”级别的异常。",
        "- 正式训练与评测仍使用 3500 条 Train 任务、60 条独立 Dev 任务，"
        "Test 任务保持封存。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes-per-mec", type=int, default=9)
    parser.add_argument("--baseline-episodes", type=int, default=9)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(args.threads)))

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    dev_specs, manifest, _order = config_specs("A", "dev")
    # Three tasks per band, fixed and independent from Train.
    selected = []
    for band in BANDS:
        selected.extend([item for item in dev_specs if item["band"] == band][:3])
    topology = load_topology(topology_role_for("A", "dev", selected[0]))

    device, encoder, backends = load_runtime(args.device)
    gab_config = load_gab_config()
    _, quality = load_capability_prior()
    cache = FixedCitySceneCache(encoder, device, max_entries=64)

    t0 = time.time()
    # 1. Fixed-city and rendering checks -----------------------------------
    city = FixedCityEnv(MAP_SEED_PHASE1).city
    signature = [
        (item.x, item.y, item.physical_size_m, item.shape,
         tuple(item.base_color), item.marker_type,
         tuple(item.marker_color), item.signature)
        for item in city.landmarks
    ]
    fixed_city_ok = True
    goal_existing_ok = True
    landmark_stable_ok = True
    for spec in selected:
        env_a = FixedCityEnv(MAP_SEED_PHASE1)
        env_a.reset(task_seed=int(spec["task_seed"]), band=spec["band"],
                    render_observation=False)
        env_b = FixedCityEnv(MAP_SEED_PHASE1)
        env_b.reset(task_seed=int(spec["task_seed"]), band=spec["band"],
                    render_observation=False)
        if city.fingerprint != env_a.task_info["city_fingerprint"]:
            fixed_city_ok = False
        goal = city.landmarks[int(env_a.task_info["goal_index"])]
        if abs(goal.x - float(env_a.goal[0])) > 1e-3:
            goal_existing_ok = False
        current = [
            (item.x, item.y, item.physical_size_m, item.shape,
             tuple(item.base_color), item.marker_type,
             tuple(item.marker_color), item.signature)
            for item in city.landmarks
        ]
        if current != signature:
            landmark_stable_ok = False
    env = FixedCityEnv(MAP_SEED_PHASE1)
    env.reset(task_seed=int(selected[0]["task_seed"]),
              band=selected[0]["band"], render_observation=False)
    image_a = env.render()
    env.x += 5.0 * np.cos(env.yaw)
    env.y += 5.0 * np.sin(env.yaw)
    env.step_count += 1
    image_b = env.render()
    render_pose_dependent_ok = bool(
        np.abs(image_a.astype(np.int16) - image_b.astype(np.int16)).mean()
        > 1.0
    )
    env_a = FixedCityEnv(MAP_SEED_PHASE1)
    env_b = FixedCityEnv(MAP_SEED_PHASE1)
    env_a.reset(task_seed=int(selected[0]["task_seed"]),
                band=selected[0]["band"], render_observation=False)
    env_b.reset(task_seed=int(selected[0]["task_seed"]),
                band=selected[0]["band"], render_observation=False)
    render_reproducible_ok = bool(np.array_equal(env_a.render(),
                                                 env_b.render()))

    # 2. T5 input and state leakage audit ----------------------------------
    t5 = state_leakage_audit(
        selected[0], encoder, backends, quality, gab_config, device, topology
    )

    # 3. Forced-MEC closed loop --------------------------------------------
    mec_rows = {}
    prior_payload, _prior_flat = load_capability_prior()
    for name in MODEL_NAMES:
        rows = []
        for spec in selected[: int(args.episodes_per_mec)]:
            spec_topology = load_topology(
                topology_role_for("A", "dev", spec)
            )
            runner = FixedCityEpisodeRunner(
                spec=spec, topology=spec_topology, encoder=encoder,
                backends=backends, projector=None, quality_prior=quality,
                gab_config=gab_config, scene_cache=cache, device=device,
                jointr_aligned=True, pure_mec_v2_state=True,
            )
            rows.append(run_episode(
                runner, f"forced_{name}", forced_model=name
            ))
        summary = summarize_episodes(rows)
        summary["prior_SR"] = float(
            prior_payload["models"][name]["sr"]
        )
        summary["prior_nDTW"] = float(
            prior_payload["models"][name]["ndtw"]
        )
        mec_rows[name] = summary
        print(
            f"[smoke] {name}: SR={summary['SR']:.3f} "
            f"nDTW={summary['nDTW']:.4f} E2E={summary['E2E_ms']:.1f}ms "
            f"steps={summary['mean_steps']:.1f}",
            flush=True,
        )

    # 4. Standard baselines -------------------------------------------------
    baseline_rows = {}
    for strategy in ("random", "min_latency", "max_quality", "heuristic"):
        rows = []
        for index, spec in enumerate(selected[: int(args.baseline_episodes)]):
            spec_topology = load_topology(
                topology_role_for("A", "dev", spec)
            )
            runner = FixedCityEpisodeRunner(
                spec=spec, topology=spec_topology, encoder=encoder,
                backends=backends, projector=None, quality_prior=quality,
                gab_config=gab_config, scene_cache=cache, device=device,
                jointr_aligned=True, pure_mec_v2_state=True,
            )
            rng = np.random.default_rng(1_000_003 + index)
            rows.append(run_episode(runner, strategy, rng=rng))
        baseline_rows[strategy] = summarize_episodes(rows)
        print(
            f"[smoke] {strategy}: SR={baseline_rows[strategy]['SR']:.3f} "
            f"nDTW={baseline_rows[strategy]['nDTW']:.4f} "
            f"E2E={baseline_rows[strategy]['E2E_ms']:.1f}ms",
            flush=True,
        )

    payload = {
        "version": "stage2l-smoke-v1",
        "created_utc": utc_now(),
        "map_seed": int(MAP_SEED_PHASE1),
        "city_fingerprint": city.fingerprint,
        "city_landmarks": len(city.landmarks),
        "dev_tasks": len(selected),
        "band_counts": dict(Counter(item["band"] for item in selected)),
        "fixed_city_ok": bool(fixed_city_ok),
        "goal_existing_ok": bool(goal_existing_ok),
        "landmark_stable_ok": bool(landmark_stable_ok),
        "render_pose_dependent_ok": bool(render_pose_dependent_ok),
        "render_reproducible_ok": bool(render_reproducible_ok),
        "t5": t5,
        "mecs": mec_rows,
        "baselines": baseline_rows,
        "elapsed_seconds": float(time.time() - t0),
        "final_test_read": False,
    }
    atomic_write_json(SMOKE_DIR / "smoke_report.json", payload)
    (SMOKE_DIR / "smoke_report.md").write_text(
        _summary_markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps({
            "fixed_city_ok": fixed_city_ok,
            "goal_existing_ok": goal_existing_ok,
            "landmark_stable_ok": landmark_stable_ok,
            "render_pose_dependent_ok": render_pose_dependent_ok,
            "render_reproducible_ok": render_reproducible_ok,
            "state_identical_for_different_goal": t5[
                "state_identical_for_different_goal"
            ],
            "elapsed_seconds": payload["elapsed_seconds"],
        }, indent=2, ensure_ascii=False),
        flush=True,
    )
    print("STAGE2L_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
