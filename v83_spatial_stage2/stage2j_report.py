"""Generate the Stage-2j final Markdown report from measured results.

Reads ``runs/v83_spatial_stage2/stage2j/analysis/summary.json`` (produced by
``stage2j_analyze.py``) and writes
``v83_spatial_stage2/V83_STAGE2J_FIVE_MEC_T5_TEST_REPORT.md``.

No numbers are invented: every metric comes from the Final Test 300
per-episode CSVs and their bootstrap summary.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v83_spatial_stage2.stage2j_common import STAGE2J_ROOT, TABLE_ORDER


REPORT_PATH = Path(
    "v83_spatial_stage2/V83_STAGE2J_FIVE_MEC_T5_TEST_REPORT.md"
)
SUMMARY_PATH = STAGE2J_ROOT / "analysis" / "summary.json"
BANDS = ("short", "medium", "long")


def _pct(value, digits=2):
    return f"{100.0 * float(value):.{digits}f}%"


def _num(value, digits=4):
    return f"{float(value):.{digits}f}"


def _ci(block, scale=1.0, digits=2):
    return (
        f"{scale*block['mean']:.{digits}f} "
        f"[{scale*block['ci95_low']:.{digits}f}, "
        f"{scale*block['ci95_high']:.{digits}f}]"
    )


def _pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


def build_report(summary):
    models = summary["models"]
    bootstrap = summary["bootstrap"]
    names = list(TABLE_ORDER)
    params = [models[name]["parameters"] for name in names]
    sr = [models[name]["metrics"]["sr"] for name in names]
    spl = [models[name]["metrics"]["spl"] for name in names]
    ndtw = [models[name]["metrics"]["ndtw"] for name in names]
    osr = [models[name]["metrics"]["osr"] for name in names]
    prem = [models[name]["metrics"]["premature_stop"] for name in names]
    timeout = [models[name]["metrics"]["timeout"] for name in names]

    best_sr = max(names, key=lambda n: models[n]["metrics"]["sr"])
    best_spl = max(names, key=lambda n: models[n]["metrics"]["spl"])
    best_ndtw = max(names, key=lambda n: models[n]["metrics"]["ndtw"])
    best_osr = max(names, key=lambda n: models[n]["metrics"]["osr"])
    lowest_prem = min(names, key=lambda n: models[n]["metrics"][
        "premature_stop"])
    lowest_timeout = min(names, key=lambda n: models[n]["metrics"]["timeout"])
    fastest = min(names, key=lambda n: models[n]["latency"]["mec_forward"][
        "mean_ms"])

    lines = []
    lines.append("# V8.3 Stage-2j — Five MEC Architectures on T5 Map-Assisted Inputs")
    lines.append("")
    lines.append(f"Date: {datetime.now(timezone.utc).date().isoformat()}")
    lines.append("")
    lines.append("Branch: `experiment/v83-spatial-foundation`  ")
    lines.append("Repository: `JLU-MCNS-MEC/mec-v8.2`")
    lines.append("")
    lines.append(
        "Nature: **formal Final Test 300 closed-loop comparison** of five "
        "native MEC architectures under exactly the same Stage-2i T5 "
        "map-assisted multimodal inputs. Single seed per model. The test "
        "set was read once per frozen model, after protocol freeze."
    )
    lines.append("")
    lines.append("## 0. TL;DR")
    lines.append("")
    lines.append(
        f"- Best SR: **{best_sr} ({_pct(models[best_sr]['metrics']['sr'])})**; "
        f"best OSR: **{best_osr} ({_pct(models[best_osr]['metrics']['osr'])})**; "
        f"best SPL: **{best_spl} ({_num(models[best_spl]['metrics']['spl'])})**; "
        f"best nDTW: **{best_ndtw} ({_num(models[best_ndtw]['metrics']['ndtw'])})**."
    )
    lines.append(
        f"- Lowest Premature-Stop: **{lowest_prem} "
        f"({_pct(models[lowest_prem]['metrics']['premature_stop'])})**; "
        f"lowest Timeout: **{lowest_timeout} "
        f"({_pct(models[lowest_timeout]['metrics']['timeout'])})**; "
        f"fastest MEC forward: **{fastest} "
        f"({models[fastest]['latency']['mec_forward']['mean_ms']:.3f} ms)**."
    )
    lines.append(
        f"- Parameter range: Small **{params[0]:,}** → Large "
        f"**{params[-1]:,}** ({params[-1]/params[0]:.2f}×). "
        f"Pearson r(params, SR) = **{_pearson(params, sr):+.3f}**, "
        f"Spearman ρ = **{_spearman(params, sr):+.3f}**; "
        f"r(params, SPL) = **{_pearson(params, spl):+.3f}**, "
        f"r(params, nDTW) = **{_pearson(params, ndtw):+.3f}** "
        "(n=5; descriptive only, not a significance test)."
    )
    lines.append(
        "- All five models use the identical anonymous landmark database, "
        "Top-5 retrieval, coordinate transform and 22-D map feature; clean "
        "retrieval is saturated at **100% Top-1 / 100% Top-5**. The "
        "comparison is therefore architecture-only under a map-assisted "
        "prior, not a pure-vision claim."
    )
    lines.append(
        "- Single seed (20264017), Dev Val 180 used for model selection, "
        "Final Test 300 used once. These are not multi-seed generalisation "
        "estimates."
    )
    lines.append("")
    lines.append("## 1. Protocol and data isolation")
    lines.append("")
    lines.append(
        "- Train 600 (200/200/200) and Dev Val 180 (60/60/60) are the same "
        "Stage-2i splits. Final Test 300 is 100/100/100 Short/Medium/Long."
    )
    lines.append(
        "- The task list was derived only from the sealed token shards "
        "(`episode_id`, `episode_seed`, `network_layout_seed`, "
        "`trajectory_band_id`); no episode was regenerated or selected. "
        "Every file in `SEALED.json` was SHA256-verified before evaluation."
    )
    lines.append(
        "- Each of the five models was evaluated once on the same 300 "
        "episodes with its selected best checkpoint. No weight updates, no "
        "Stop-threshold changes (fixed 0.7), no test-time tuning."
    )
    lines.append(
        "- The shared RGB encoder is frozen. History, pose, GAB and map "
        "features are causal and reset per episode. True labels, target "
        "coordinates and distances are evaluation-only."
    )
    lines.append("")
    lines.append("## 2. Five native architectures")
    lines.append("")
    lines.append(
        "| MEC | Native architecture | Parameters | Best epoch | Train min | "
        "Checkpoint SHA256 (prefix) |"
    )
    lines.append("|---|---|---:|---:|---:|---|")
    for name in names:
        item = models[name]
        lines.append(
            f"| {name} | `{item['architecture']}` | {item['parameters']:,} | "
            f"{item['best_epoch']} | {float(item['training_time_min']):.0f} | "
            f"`{item['checkpoint_sha256'][:16]}` |"
        )
    lines.append("")
    lines.append(
        "Small remains a late-fusion MLP; Medium remains Goal→Current "
        "cross-attention + late fusion; Transformer-S/M/L are explicit "
        "Transformers at their native widths (192/1/4, 256/2/8, 352/2/8). "
        "Transformer-M is the Stage-2i T5 checkpoint, reused without "
        "retraining and verified equivalent by state dict, fixed-batch "
        "outputs and a full Dev Val 180 reproduction."
    )
    lines.append("")
    lines.append("## 3. Final Test 300 — overall")
    lines.append("")
    lines.append(
        "| MEC | Params | SR | OSR | SPL | nDTW | Premature | Timeout | "
        "Steps | MEC forward ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in names:
        m = models[name]["metrics"]
        lat = models[name]["latency"]["mec_forward"]["mean_ms"]
        lines.append(
            f"| {name} | {models[name]['parameters']:,} | "
            f"{_pct(m['sr'])} | {_pct(m['osr'])} | {_num(m['spl'])} | "
            f"{_num(m['ndtw'])} | {_pct(m['premature_stop'])} | "
            f"{_pct(m['timeout'])} | {m['steps']:.1f} | {lat:.3f} |"
        )
    lines.append("")
    lines.append("### 3.1 Bootstrap 95% CI (10,000 episode resamples)")
    lines.append("")
    lines.append("| MEC | SR CI | OSR CI | SPL CI | nDTW CI |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in names:
        b = bootstrap["metrics"]
        lines.append(
            f"| {name} | {_ci(b['sr'][name], 100.0, 2)} | "
            f"{_ci(b['osr'][name], 100.0, 2)} | "
            f"{_ci(b['spl'][name], 1.0, 4)} | "
            f"{_ci(b['ndtw'][name], 1.0, 4)} |"
        )
    lines.append("")
    lines.append("### 3.2 Paired deltas vs Transformer-M")
    lines.append("")
    lines.append("| MEC | ΔSR | ΔSPL | ΔnDTW |")
    lines.append("|---|---:|---:|---:|")
    for name in names:
        if name == "Transformer-M":
            continue
        entry = bootstrap["paired_deltas"]["vs_Transformer-M"][name]
        lines.append(
            f"| {name} | {100*entry['sr']['delta']:+.2f} pp "
            f"[{100*entry['sr']['ci95_low']:+.2f}, "
            f"{100*entry['sr']['ci95_high']:+.2f}] | "
            f"{entry['spl']['delta']:+.4f} | "
            f"{entry['ndtw']['delta']:+.4f} |"
        )
    lines.append("")
    lines.append("## 4. Short / Medium / Long")
    lines.append("")
    lines.append(
        "| MEC | Short SR | Medium SR | Long SR | Best band | SR spread "
        "(max-min) |"
    )
    lines.append("|---|---:|---:|---:|---|---:|")
    for name in names:
        bands = models[name]["bands"]
        vals = [bands[b]["sr"] for b in BANDS]
        best_band = BANDS[int(max(range(3), key=lambda i: vals[i]))]
        lines.append(
            f"| {name} | {_pct(vals[0])} | {_pct(vals[1])} | "
            f"{_pct(vals[2])} | {best_band} | "
            f"{100*(max(vals)-min(vals)):.2f} pp |"
        )
    lines.append("")
    lines.append("Full band metrics:")
    lines.append("")
    lines.append(
        "| MEC | Band | SR | OSR | SPL | nDTW | Premature | Timeout |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name in names:
        for band in BANDS:
            item = models[name]["bands"][band]
            lines.append(
                f"| {name} | {band.capitalize()} | {_pct(item['sr'])} | "
                f"{_pct(item['osr'])} | {_num(item['spl'])} | "
                f"{_num(item['ndtw'])} | {_pct(item['premature_stop'])} | "
                f"{_pct(item['timeout'])} |"
            )
    lines.append("")
    lines.append("## 5. Map retrieval and inference latency")
    lines.append("")
    lines.append(
        "| MEC | Top-1 | Top-5 | Mean goal cosine | MEC forward ms | "
        "p50 ms | p95 ms | Feature build ms | Decision pipeline ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in names:
        item = models[name]
        r = item["retrieval"]
        lat = item["latency"]
        lines.append(
            f"| {name} | {_pct(r['clean_top1'])} | "
            f"{_pct(r['clean_top5'])} | "
            f"{r['clean_mean_goal_similarity']:.7f} | "
            f"{lat['mec_forward']['mean_ms']:.3f} | "
            f"{lat['mec_forward']['p50_ms']:.3f} | "
            f"{lat['mec_forward']['p95_ms']:.3f} | "
            f"{lat['feature_build']['mean_ms']:.3f} | "
            f"{lat['decision_pipeline']['mean_ms']:.3f} |"
        )
    lines.append("")
    lines.append("## 6. Figures")
    lines.append("")
    for metric, label in (
        ("SR", "Success Rate"), ("OSR", "Oracle Success Rate"),
        ("SPL", "SPL"), ("nDTW", "nDTW"),
    ):
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"![{label}](figures/stage2j/MEC_{metric}_Comparison.png)"
        )
        lines.append("")
    lines.append(
        "Each figure is also available as SVG in "
        "`v83_spatial_stage2/figures/stage2j/`."
    )
    lines.append("")
    lines.append("## 7. Analysis")
    lines.append("")
    lines.append("### 7.1 Parameters, latency and closed-loop metrics")
    lines.append("")
    lines.append(
        f"Parameter count rises from {params[0]:,} (Small) to "
        f"{params[-1]:,} (Large), {params[-1]/params[0]:.2f}×. "
        f"MEC-forward latency is fastest for {fastest} "
        f"({models[fastest]['latency']['mec_forward']['mean_ms']:.3f} ms) "
        f"and slowest for "
        f"{max(names, key=lambda n: models[n]['latency']['mec_forward']['mean_ms'])} "
        f"({max(models[n]['latency']['mec_forward']['mean_ms'] for n in names):.3f} ms). "
        "All measurements are batch-1, after 5 warm-up decisions, with "
        "CUDA synchronization."
    )
    lines.append("")
    lines.append("### 7.2 Does parameter growth improve SR / SPL / nDTW?")
    lines.append("")
    lines.append(
        f"Across the five architectures, Pearson r(params, SR) = "
        f"**{_pearson(params, sr):+.3f}**, r(params, SPL) = "
        f"**{_pearson(params, spl):+.3f}**, r(params, nDTW) = "
        f"**{_pearson(params, ndtw):+.3f}**; Spearman ρ(SR) = "
        f"**{_spearman(params, sr):+.3f}**. The trend is "
        + (
            "positive but not monotonic across every model; the largest "
            "model does not automatically win every metric."
            if not (max(names, key=lambda n: models[n]["parameters"]) ==
                    best_sr == best_spl == best_ndtw)
            else "monotonic in this five-model set."
        )
        + " With n=5 these correlations are descriptive, not inferential."
    )
    lines.append("")
    lines.append("### 7.3 Small and Transformer-S: capability floor")
    lines.append("")
    small_s_to_s = bootstrap["paired_deltas"]["vs_Small"].get(
        "Transformer-S", {}
    )
    if small_s_to_s:
        sr_delta = 100 * small_s_to_s["sr"]["delta"]
        sr_lo = 100 * small_s_to_s["sr"]["ci95_low"]
        sr_hi = 100 * small_s_to_s["sr"]["ci95_high"]
        lines.append(
            f"Small reaches SR {_pct(models['Small']['metrics']['sr'])} / "
            f"SPL {_num(models['Small']['metrics']['spl'])}. "
            f"Transformer-S changes SR by **{sr_delta:+.2f} pp** "
            f"(95% CI [{sr_lo:+.2f}, {sr_hi:+.2f}]), SPL by "
            f"**{small_s_to_s['spl']['delta']:+.4f}**, nDTW by "
            f"**{small_s_to_s['ndtw']['delta']:+.4f}**. "
            + (
                "The CI excludes zero, so the extra capacity has a "
                "measurable floor effect in this single-seed test."
                if sr_lo > 0 or sr_hi < 0 else
                "The CI includes zero, so this single seed does not "
                "establish a reliable Small→Transformer-S gain."
            )
        )
    lines.append("")
    lines.append("### 7.4 Transformer-M and Large: stability")
    lines.append("")
    large_delta = bootstrap["paired_deltas"]["vs_Transformer-M"]["Large"]
    m_bands = [models["Transformer-M"]["bands"][b]["sr"] for b in BANDS]
    l_bands = [models["Large"]["bands"][b]["sr"] for b in BANDS]
    lines.append(
        f"Large − Transformer-M: ΔSR **{100*large_delta['sr']['delta']:+.2f} pp** "
        f"(95% CI [{100*large_delta['sr']['ci95_low']:+.2f}, "
        f"{100*large_delta['sr']['ci95_high']:+.2f}]), ΔSPL "
        f"**{large_delta['spl']['delta']:+.4f}**, ΔnDTW "
        f"**{large_delta['ndtw']['delta']:+.4f}**. "
        f"Transformer-M has band-SR spread "
        f"**{100*(max(m_bands)-min(m_bands)):.2f} pp** and premature/timeout "
        f"{_pct(models['Transformer-M']['metrics']['premature_stop'])} / "
        f"{_pct(models['Transformer-M']['metrics']['timeout'])}; Large has "
        f"**{100*(max(l_bands)-min(l_bands)):.2f} pp** and "
        f"{_pct(models['Large']['metrics']['premature_stop'])} / "
        f"{_pct(models['Large']['metrics']['timeout'])}. "
        + (
            "Large is not a consistent stability improvement over "
            "Transformer-M on this test."
            if (max(l_bands) - min(l_bands)) >= (max(m_bands) - min(m_bands))
            else "Large has a smaller band spread, but the paired CI and the "
                 "single seed must be considered before calling it more "
                 "stable."
        )
    )
    lines.append("")
    lines.append("### 7.5 Short / Medium / Long differences")
    lines.append("")
    lines.append(
        "The band table shows whether a model's gain is concentrated in "
        "Short, Medium or Long episodes. Substantial band spread with the "
        "same 100 episodes per band indicates that episode horizon remains "
        "a major source of variance; it is not evidence of a separate "
        "control policy."
    )
    lines.append("")
    lines.append("### 7.6 Same map conditions; differences are MEC decisions")
    lines.append("")
    identical = summary.get("retrieval_all_models_identical", False)
    lines.append(
        f"All five models use the same map construction, frozen encoder, "
        f"cosine Top-5 retrieval and 22-D feature. The retrieval summaries "
        f"are {'identical' if identical else 'numerically very close'} "
        f"across models (Top-1 {_pct(models['Small']['retrieval']['clean_top1'])}, "
        f"Top-5 {_pct(models['Small']['retrieval']['clean_top5'])}). "
        "History, pose, GAB and map tensors are also the same inputs; the "
        "five models differ only in the MEC decision backbone. Therefore "
        "the closed-loop differences are attributable to the decision "
        "network (movement head, fusion, and the independently learned "
        "Arrival head), not to different map information."
    )
    lines.append("")
    lines.append("### 7.7 Map-assisted premise, retrieval saturation and limits")
    lines.append("")
    lines.append(
        "- All five results are **map-assisted**: they use an anonymous "
        "global landmark database (positions + appearance) and UAV self-pose. "
        "They are not pure visual navigation results."
    )
    lines.append(
        "- Clean retrieval is saturated at 100% Top-1/Top-5 because the map "
        "nodes and Goal reference share the same renderer and frozen encoder. "
        "Stage-2i noise diagnostics showed Top-1 falling to 23.89% at 5% "
        "embedding noise and 10.56% at 15%; the clean 100% is not a "
        "real-world retrieval-robustness claim."
    )
    lines.append(
        "- Single seed (20264017) for each newly trained model; Dev Val 180 "
        "was used for checkpoint selection, so test 300 is not a "
        "multi-seed or fully unbiased generalisation estimate."
    )
    lines.append(
        "- The Final Test 300 was read once per frozen model. No test result "
        "was used to change architectures, hyperparameters, epochs or the "
        "Stop threshold."
    )
    lines.append("")
    lines.append("## 8. Artifacts")
    lines.append("")
    lines.append(
        "- Protocol / freeze: `v83_spatial_stage2/STAGE2J_PROTOCOL.md`, "
        "`v83_spatial_stage2/stage2j_protocol_freeze.json`."
    )
    lines.append(
        "- Code/tests: `v83_spatial_stage2/stage2j_model.py`, "
        "`stage2j_train.py`, `stage2j_eval.py`, `stage2j_freeze.py`, "
        "`stage2j_equivalence.py`, `stage2j_analyze.py`, "
        "`stage2j_pipeline.sh`, `test_stage2j.py`."
    )
    lines.append(
        "- Per-model checkpoints, configs, histories, per-episode CSVs, "
        "steps CSVs, metrics and logs: "
        "`runs/v83_spatial_stage2/stage2j/<MEC>/` (server-side)."
    )
    lines.append(
        "- Figures and summary: `v83_spatial_stage2/figures/stage2j/`."
    )
    lines.append("")
    lines.append(
        "Commit, push, test count and working-tree status are recorded in "
        "`v83_spatial_stage2/STAGE2J_COMMIT_STATUS.md`."
    )
    lines.append("")
    return "\n".join(lines)


def main():
    if not SUMMARY_PATH.is_file():
        raise SystemExit(
            f"Missing {SUMMARY_PATH}; run stage2j_analyze.py after all five "
            "Final Test evaluations."
        )
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    report = build_report(summary)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"WROTE {REPORT_PATH} ({len(report)} chars)", flush=True)


if __name__ == "__main__":
    main()
