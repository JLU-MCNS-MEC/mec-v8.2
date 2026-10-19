# V8.3 Stage-2g.2 — P1-2 Near-Target Recovery Closed-Loop Smoke

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Status: **EXECUTED after approval — STOP.** The P1-2 probe was re-run with
  saved weights and reproduced the old Dev-Val predictions exactly
  (max |Δp| = 0.0). The 36-episode R1 smoke showed **no SR/SPL/nDTW
  improvement**; 2/2 triggers were far-invisible (166 m / 335 m), 4/4
  recovery actions were wrong, with 0 re-visible and 0 enter-20 m. See §7.
- Scope: data supplement + probe + 36-episode closed-loop smoke; no MEC
  training, no controller integration, Final Test 300 sealed. The
  pre-execution blocker audit is retained in §§1–6 for provenance.

> **Execution update (2026-10-19, after approval):** the blocker described in
> §§1–6 was resolved by adding `--save-models`, re-running the same P1-2
> configuration, and verifying exact prediction reproduction. The 36-episode
> R1 smoke was then executed; the result is **STOP** (§7).

## 结论摘要 (pre-execution audit, historical; see §7 for executed result)

1. **闭环 smoke 没有执行。** 阻塞点不是阈值、数据或代码配置，而是
   **P1-2 探针的模型权重没有保存**。`stage2g1_probe.py` 只保存了 OOF/Dev Val
   的预测概率和指标，没有保存 `state_dict`；`stage2g1_probe.py`、
   `stage2g_probe.py`、`stage2g1_dataset.py`、`stage2g1_analyze.py` 中均没有
   `torch.save` / `state_dict` 调用。
2. **已做全盘搜索。** `runs/` 与 `/tmp` 中 12:00 之后没有任何新的
   `.pt` / `.pth` / `.ckpt` / `.bin`；Probe 目录只有 `probe_results.json`、
   `analysis.*` 和 `predictions_v*_*.npz`。预测文件的唯一键是
   `probability`，不是权重。
3. **阈值是可恢复的，且来源正确。** `probe_results.json` 的 external
   `thresholds` 由 **Train OOF** 的 `select_thresholds(train_data, oof_score)`
   选出；Dev Val 只用于应用阈值和报告指标。P1-2 near-inv 的 f1/far-vis-5%
   阈值为 **0.13694876571430528**；recall-0.5 阈值为
   **0.9666095860450129**；precision-first 为 **0.9999919652208562**。
   `trigger_oracle_diagnostic`（Dev Val 自选阈值）存在，但**没有**用于 gate。
4. **可恢复的是“配置”，不可恢复的是“权重”。** P1-2 的输入维度（3,016 =
   2,858 当前 + 2×79 历史）、MLP 结构（128 hidden、dropout 0.2）、训练配置
   （30 epochs、Adam 1e-3、weight decay 1e-4、batch 8,192、class-weighted BCE +
   prior correction、external seed 1000）和合并数据路径都可从代码/报告恢复；
   但训练后的 MLP 参数（external 模型和 5 个 OOF fold 模型）没有落盘。
5. **不能用保存的预测代替权重。** 保存的 `predictions_*.npz` 只覆盖固定的
   离线状态（Dev Val 17,494 个），闭环会访问新的状态；离线预测不能作为
   运行时 gate，也不能反推出唯一权重。
6. **按规则不重新训练。** 重新训练整个探针不是“恢复权重”；GPU/dropout
   非确定性还会使新权重与旧阈值不完全匹配。若重新训练，必须先验证新模型
   能复现旧的 Dev Val 预测/指标，否则应视为一个新探针并在 **Train OOF** 上
   重新选阈值。
7. **解除阻塞的最小方案：**
   - 给 `stage2g1_probe.py` 增加 `--save-models`/provenance：保存 external
     模型和 fold 模型的 `state_dict`、模型配置、seed、数据路径/hash、git
     commit、阈值；
   - 用同一份 combined Train/Val、同一配置和 seed 重新生成权重（约 2.5 min，
     不训练新 MEC）；
   - 验证新权重在 Dev Val 上的预测与旧 `predictions_v2_near_target_invisible_val.npz`
     一致（概率/AUC/阈值指标在容差内）；一致才做 36-episode R0/R1 smoke；
     不一致则按新探针处理，阈值只能在 Train OOF 重选，不能用 Dev Val 调参；
   - 然后执行本轮原本设计的 R0/R1 smoke。
8. **本轮没有 R0/R1 指标、没有 closed-loop 结论。** R0 的 Stage-2f 参考基线
   是 36.1% SR / 0.361 SPL / 0.409 nDTW（12 episodes/band），但本轮没有重新
   运行；在没有 P1-2 权重的情况下，R1 无法构造。

## 1. Audit evidence

### 1.1 Git and workspace

- Branch: `experiment/v83-spatial-foundation`
- HEAD: `363f15d` (`Add Stage-2g.1 near-target supplement and history probe`)
- `git status`: clean; local == origin.
- Disk: 34 GB free; GPU idle at audit time.

### 1.2 Probe artifacts

`runs/v83_spatial_stage2/stage2g1/probe_20261019T123000Z/` contains:

```
probe_results.json
analysis.json
analysis.md
predictions_v0..v3_<head>_oof.npz
predictions_v0..v3_<head>_val.npz
```

There is **no** `.pt`, `.pth`, `.ckpt` or `.bin` file. The prediction files
contain only a `probability` array (e.g. `(17494,) float64`), not model
parameters.

Search results:

| Check | Result |
|---|---|
| `find runs ... -newermt '2026-10-19 12:00'` for weight extensions | no files |
| `find /tmp ...` for weight extensions | no files |
| `grep -rn torch.save/state_dict/pickle.dump` in Stage-2g/2g.1 probe/dataset/analyze scripts | none |
| `stage2g_probe.train_and_predict` return value | `(probabilities, {"final_train_loss": ...})` — no model |
| `stage2g1_probe.run_variant` external output | `probabilities`, `train_info`, `pos_weight` — no state dict |

The probe was trained in memory and discarded after predictions were written.
The same limitation applies to the 5-fold OOF models.

### 1.3 Threshold provenance

`stage2g1_probe.py` calls:

```python
evaluate_result(valid_data, external[head]["probabilities"], head, args,
                train_data=train_data,
                train_probabilities=variant_result["oof"][head])
```

Inside `evaluate_result`:

```python
train_score = score_for_head(train_probabilities, head)
result["thresholds"] = select_thresholds(train_data, train_score)
for name, entry in result["thresholds"].items():
    val_metrics = trigger_metrics(data, score, entry["threshold"])
```

So the external `thresholds` are selected on **Train OOF**; Dev Val is used only
to apply them. `trigger_oracle_diagnostic` is a separate Dev-Val-selected
diagnostic and is not used for the gate. This matches the report.

Recoverable P1-2 (variant 2) near-inv thresholds:

| Operating point | Threshold (Train OOF) | Dev Val precision | Dev Val recall | Dev Val far-vis FT |
|---|---:|---:|---:|---:|
| `f1_far_visible_5pct` | **0.13694876571430528** | 0.914 | 0.813 | 0.0007 |
| `recall_0.5` | **0.9666095860450129** | 1.000 | 0.529 | 0.0000 |
| `precision_far_visible_5pct` | **0.9999919652208562** | 1.000 | 0.125 | 0.0000 |
| `precision_0.8` | 0.9999996663423116 | 0.000 (0 flags) | 0.000 | 0.0000 |

The f1/far-vis-5% threshold is the one used in the Stage-2g.1 report and the
intended closed-loop gate. The threshold is recoverable; the model it applies
to is not.

### 1.4 Recoverable vs missing

| Item | Status |
|---|---|
| Combined Train/Dev-Val data (`combined_*.npz`) | **available** |
| Supplement and hard-history artifacts | **available** |
| P1-2 input definition (current 2,858 + 2×79 history) | **available** (code/report) |
| MLP architecture / training config | **available** (code/report) |
| Train OOF-selected thresholds | **available** (`probe_results.json`) |
| External P1-2 MLP state_dict | **MISSING** |
| 5-fold OOF P1-2 MLP state_dicts | **MISSING** |
| Exact training-command provenance JSON (seed/hidden/lr/epochs/data hash) | **not saved**; code defaults + report only |
| Offline predictions/metrics | **available**, but not a runtime model |

The exact run command was:

```bash
python v83_spatial_stage2/stage2g1_probe.py \
  --train runs/v83_spatial_stage2/stage2g1/combined_train_20261019T121500Z/combined_train.npz \
  --val runs/v83_spatial_stage2/stage2g1/combined_val_20261019T121500Z/combined_val.npz \
  --output-dir runs/v83_spatial_stage2/stage2g1/probe_20261019T123000Z \
  --epochs 30 --bootstrap 200
```

The script defaults are `hidden=128`, `batch_size=8192`, `lr=1e-3`,
`weight_decay=1e-4`, `seed=0` (external model seed 1000), 5 folds. The command
was not persisted by the probe script itself; the report and shell session are
the only provenance.

## 2. Why the smoke cannot run

The closed loop needs a per-state probability from the P1-2 near-inv MLP:

- The saved Dev-Val predictions are for the fixed offline states only. Closed
  loop revisits new observations, so they cannot be indexed or interpolated.
- The threshold 0.1369487657 is tied to the probability scale of the exact
  trained model. Applying it to a different model (even a re-run with the same
  seed) requires reproducing that model; otherwise the threshold is not
  guaranteed to have the same precision/recall.
- The OOF fold models are also missing, so we cannot use a fold ensemble.
- The user explicitly instructed: if weights cannot be reliably recovered,
  report the missing content and do not retrain the whole probe. Therefore no
  R0/R1 closed-loop run was started.

## 3. Unblock plan (proposed, not executed)

### 3.1 Save the probe

Add to `stage2g1_probe.py`:

- `--save-models` option;
- save `state_dict` for the external model and each fold model;
- save a `probe_provenance.json` with model config, seed, git commit,
  data paths/hashes, threshold table and label definition;
- optionally save a `probe_spec.json` (input dim, feature names, history length,
  action one-hot layout).

This is a code/provenance fix, not a new probe design.

### 3.2 Re-run and verify

Re-run the same command with `--save-models` on the same combined data and same
config/seed (estimated ~2.5 min on the RTX 5090; no MEC training). Before use:

1. compute Dev Val probabilities from the saved external model;
2. compare with `predictions_v2_near_target_invisible_val.npz`:
   - max absolute probability difference should be small (e.g. <1e-5–1e-4);
   - the saved threshold metrics should match within rounding;
3. if the comparison matches, use the saved model and the existing
   Train-OOF-selected threshold;
4. if it does not match (GPU/dropout nondeterminism), treat the re-run as a
   **new probe**: re-select the threshold on its own Train OOF predictions only;
   do not tune on Dev Val. If the new probe does not reproduce the offline gate,
   stop the recovery line.

### 3.3 Then run the 36-episode R0/R1 smoke

The original smoke design (unchanged):

- same 36 Dev-Val episodes (12 per band; same seeds/network layouts as the
  Stage-2f smoke);
- **R0:** frozen M0@0.7, confirm every per-episode outcome against the saved
  Stage-2f baseline;
- **R1:** M0@0.7 + P1-2 near-inv gate at the Train-OOF-selected threshold
  (0.13694876571430528 for the f1/far-vis-5% point);
- trigger uses only the P1-2 current+2-frame observable history; no true
  distance/visibility/heading/oracle action;
- if a recent high-confidence target bearing exists, make a short correction
  toward it; otherwise apply one bounded single-step correction;
- each trigger overwrites at most 2–3 Movement steps, then exits recovery;
- cooldown to prevent repeated triggering; no long in-place scan; **Forward is
  allowed** during recovery;
- no change to the M0 Arrival head, Stop threshold, Movement weights or
  checkpoint.

Metrics: Overall/Short/Medium/Long SR/OSR/SPL/nDTW, Premature-Stop, Timeout,
steps; trigger count; re-visible rate; enter-20m-within-40; near-target-invisible
precision/recall; far-visible/far-invisible false-trigger; paired W/L/T.

Stop conditions: if R1 does not improve SR/SPL/nDTW, stop the Recovery
direction; if Timeout rises or repeated/wrong redirection appears, stop the
strategy immediately and report; do not expand to Dev Val 180 this round;
Final Test 300 stays sealed.

## 4. What was and was not done

- **Done:** audit of git, probe directory, weight files, code save paths and
  threshold provenance; verification that the threshold is Train-OOF-selected;
  tests (32 pass).
- **Not done:** probe retraining; R0/R1 closed-loop smoke; R0 baseline rerun;
  full Dev Val 180; controller integration; MEC training; Final Test read.
- **No results claimed:** there are no R0/R1 SR/SPL/nDTW or trigger metrics in
  this report because R1 could not be constructed.

## 5. Tests and status

```
python -m unittest v83_spatial_stage2.test_stage2f \
  v83_spatial_stage2.test_stage2f_supplement \
  v83_spatial_stage2.test_stage2g \
  v83_spatial_stage2.test_stage2g1
# 32 tests OK
```

No source code was modified in this blocked round; the worktree is clean.

## 6. Artifacts

Existing probe evidence (read-only):

```
runs/v83_spatial_stage2/stage2g1/probe_20261019T123000Z/
  probe_results.json
  analysis.json
  analysis.md
  predictions_v2_near_target_invisible_{oof,val}.npz
  ... (other heads/variants)
```

Data references:

```
runs/v83_spatial_stage2/stage2g1/combined_train_20261019T121500Z/combined_train.npz
runs/v83_spatial_stage2/stage2g1/combined_val_20261019T121500Z/combined_val.npz
runs/v83_spatial_stage2/stage2g1/supplement_train_20261019T120000Z/
runs/v83_spatial_stage2/stage2g1/supplement_val_20261019T120500Z/
runs/v83_spatial_stage2/stage2g1/hard_history_train_20261019T121000Z/
runs/v83_spatial_stage2/stage2g1/hard_history_val_20261019T121200Z/
```

Reference R0 baseline (Stage-2f 36-episode smoke, not rerun this round):
SR 36.1%, OSR 38.9%, SPL 0.361, nDTW 0.409, Premature 50.0%, Timeout 13.9%,
steps 129.0. It is context only; the new R0 must still be re-confirmed in the
new harness before R1 comparison.

## Appendix A. P1-2 configuration recovered from code/report

| Item | Value |
|---|---|
| Variant | P1-2 (current + 2 previous frames) |
| Current features | 2,858 |
| History per frame | 74 compact visual + 4 action one-hot + 1 dt = 79 |
| Input dim | 3,016 |
| Model | MLP 128, ReLU, dropout 0.2 |
| Training | 30 epochs, Adam lr 1e-3, weight decay 1e-4, batch 8,192 |
| Loss | class-weighted BCE + prior correction |
| Folds | 5-fold episode-level OOF on Train 600 |
| External model seed | 1000 (fold seeds 0–4) |
| Threshold selection | Train OOF only |
| f1/far-vis-5% threshold | 0.13694876571430528 |
| recall-0.5 threshold | 0.9666095860450129 |

## Appendix B. Historical note

This appendix was written when the report was blocked. It is superseded by §7:
the P1-2 weights were saved and reproduced exactly (max |Δp| = 0.0), the
36-episode R1 smoke was executed, and the result is **STOP**. Retained for
provenance; the current status is not “blocked.”


## 7. Execution results after approval (2026-10-19)

### 7.1 Weights, provenance and exact reproduction

The P1-2 probe was re-run with the same combined Train/Dev-Val data, input
definition, MLP (128 hidden, dropout 0.2), 30 epochs, Adam (lr 1e-3, weight
decay 1e-4), batch 8,192, 5-fold episode OOF and external seed 1000. The new
run saved:

- 72 model checkpoints (`models/v{0..3}_{head}_{fold|external}.pt`) including
  `state_dict`, standardization mean/std, `pos_weight`, input dimension,
  feature names, compact history names, history order, action encoding, clip,
  seed, config and git commit;
- `probe_provenance.json` with data SHA256, thresholds, config and model list.

New probe directory:
`runs/v83_spatial_stage2/stage2g1/probe_20261019T124500Z/`.
`reproduction_check.json` compares the new external P1-2 near-inv Dev-Val
predictions with the old
`probe_20261019T123000Z/predictions_v2_near_target_invisible_val.npz`:

| Check | Result |
|---|---|
| States compared | 17,494 |
| Max absolute probability difference | **0.0** |
| Mean absolute probability difference | **0.0** |
| Old Train-OOF threshold | 0.13694876571430528 |
| New Train-OOF threshold | 0.13694876571430528 |
| Reproduction verdict | **REPRODUCED** |
| Offline gate at selected threshold | **PASS** (precision 0.914, far-visible FT 0.0007, far-invisible FT 0.027) |

The old Train-OOF threshold was therefore reused, as permitted by the
pre-agreed `max |Δp| <= 1e-4` criterion. No Dev-Val threshold tuning was done.

### 7.2 R0 baseline verification

The 36 selected Dev-Val episodes (12/band, same `select_specs` even selection)
match the Stage-2f smoke baseline exactly on `episode_index`, `seed` and
`band`; R0 outcomes were reused from
`smoke_budget12_20261019T031000Z/baseline_per_episode.csv`. Episodes without a
trigger in R1 reproduced R0 exactly (`no_trigger_r0_mismatches = []`).

### 7.3 R0 vs R1

Table: SR/OSR/SPL/nDTW/Pre-Premature/Timeout/Steps.

| Split | Policy | SR | OSR | SPL | nDTW | Premature | Timeout | Steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | R0 | 36.11% | 38.89% | 0.361 | 0.4086 | 50.00% | 13.89% | 129.03 |
| Overall | R1 | 36.11% | 38.89% | 0.361 | 0.4086 | 52.78% | 11.11% | 125.61 |
| Short | R0 | 50.00% | 58.33% | 0.500 | 0.5456 | 41.67% | 8.33% | 79.58 |
| Short | R1 | 50.00% | 58.33% | 0.500 | 0.5456 | 41.67% | 8.33% | 78.00 |
| Medium | R0 | 33.33% | 33.33% | 0.333 | 0.4002 | 50.00% | 16.67% | 166.67 |
| Medium | R1 | 33.33% | 33.33% | 0.333 | 0.4002 | 58.33% | 8.33% | 158.00 |
| Long | R0 | 25.00% | 25.00% | 0.250 | 0.2800 | 58.33% | 16.67% | 140.83 |
| Long | R1 | 25.00% | 25.00% | 0.250 | 0.2800 | 58.33% | 16.67% | 140.83 |

Paired per episode (R1 − R0):

| Metric | W/L/T | Mean Δ | 95% CI |
|---|---|---:|---|
| Success | **0W / 0L / 36T** | 0.00 | — |
| SPL | 0W / 0L / 36T | 0.0000 | [0.0000, 0.0000] |
| nDTW | 2W / 0L / 34T | **+3.48e-7** | [0, 1.04e-6] |
| Steps | 0W / 2L / 34T | −3.42 | [−9.74, 0] |
| Timeout | 0W / 1L / 35T | −2.78 pp | [−8.33, 0] |
| Premature Stop | 1W / 0L / 35T | +2.78 pp | [0, 8.33] |

The nDTW “win” is +3.5e-7 and the CI upper bound is 1e-6; this is numerically
zero, not a meaningful improvement. The only outcome-level change is episode
142 switching from timeout to premature Stop; episode 174 kept its premature
Stop but ended earlier. No episode changed success.

### 7.4 Triggers and closed-loop probe behaviour

Only **2 triggers** occurred, in 2 of 36 episodes:

| Episode | Band | Trigger step | p_near | True distance | Goal visible | Last high side | Steps since high | Recovery actions | Wrong | Re-visible ≤10 | Enter <20 m ≤40 | Exit |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 174 | short | 24 | 0.1822 | **166.2 m** | 0 | unknown | 999 | 2 | 2 | 0 | 0 | budget |
| 142 | medium | 118 | 0.2023 | **335.1 m** | 0 | unknown | 999 | 2 | 2 | 0 | 0 | budget |

Both triggers are **far-invisible**, not near-target-invisible. The trigger
aggregate is:

| Trigger metric | Value |
|---|---:|
| Triggers / episodes | 2 / 2 |
| Trigger true goal-invisible rate | 100% |
| Trigger distance | 166.2 m / 335.1 m |
| Re-visible ≤5 / ≤10 steps | 0% / 0% |
| Enter <20 m within 40 steps | 0% |
| Recovery actions / wrong vs heading oracle | 4 / 4 (100%) |
| Repeated triggers per episode | max 1; 0 episodes with ≥2 |

The closed-loop probe step metrics (4,418 observable states where the probe
was evaluated at the fixed threshold 0.13694876571430528):

| Subset | n | Probe flags | Rate / precision / recall |
|---|---:|---:|---|
| Near-target-invisible (<100 m, invisible) | 50 | **0** | precision 0.000, recall 0.000 |
| Far-visible (≥100 m, visible) | 2,386 | 0 | false-trigger 0.000 |
| Far-invisible (≥100 m, invisible) | 1,568 | **2** | false-trigger 0.00128 |
| Near-visible (<100 m, visible) | 414 | 0 | false-trigger 0.000 |

Overall probe metrics on these closed-loop states: AUROC **0.632**
[0.465, 0.967], AUPRC **0.043**, Brier 0.0113, ECE 0.0112. The near-inv
probability maximum on real closed-loop near-inv states was **0.0412**, far
below the fixed threshold; the two flags are from the far-inv tail (max
0.2023). The offline Dev-Val performance (AUROC 0.985, AP 0.904) did not
transfer to the M0 closed-loop distribution.

### 7.5 Failure analysis

1. **Distribution shift is the binding failure.** The synthetic supplement's
   near-invisible positives were generated by turning away after a
   high-confidence view. Real M0 near-invisible states in this smoke are
   sustained/far invisible with no recent high-confidence observation
   (`steps_since_high_conf = 999` at both triggers). The probe is
   under-confident on the real near-inv states (max p 0.0412) and fires on the
   far-inv tail instead.
2. **The gate fired in the wrong regime.** Both triggers were at 166 m and
   335 m, not <100 m; the closed-loop near-inv precision/recall is 0/0. This
   repeats the Stage-2f far-trigger failure, though with only 2 events.
3. **The bounded recovery action did not re-acquire.** Both triggers had no
   recent high-confidence side, so the policy used the fallback single-turn +
   forward correction; all 4 actions were wrong against the heading oracle,
   with 0 re-visible and 0 enter-20 m.
4. **No performance benefit.** SR and SPL are exactly unchanged
   (0W/0L/36T), nDTW is +3.5e-7 (numerically zero), premature Stop rises by
   2.78 pp, Timeout falls by 2.78 pp, and two episodes changed their paths.
   The 36-episode smoke is a mechanism test, not a performance result.
5. **The offline gate was necessary but not sufficient.** Exact weight
   reproduction (max |Δp| = 0.0) rules out a weight-loading error; the failure
   is the real closed-loop state distribution, not the saved checkpoint.

### 7.6 Decision and next step

**Decision: STOP the P1-2 near-target recovery direction.**

- SR, SPL and nDTW show no meaningful improvement (nDTW +3.5e-7; all other
  success/SPL changes zero).
- The gate triggered only on far-invisible states; closed-loop near-inv
  precision/recall is 0/0.
- 100% of recovery actions were wrong; 0% re-visible; 0% enter-20 m; one
  episode changed timeout to premature Stop.
- Do not expand to Dev Val 180; do not integrate the policy; do not train a
  new MEC.

Recommended alternative research (future, not started):

- Collect **real M0 closed-loop near-target-invisible token states** (not
  synthetic turn-away states) and retrain a presence/localization signal on
  that distribution; the synthetic supplement did not transfer.
- Or treat far invisibility as a map/coarse-navigation problem rather than a
  local scan/recovery problem.
- Do not continue hand-written scan/step-correction recovery without a
  closed-loop-validated target-presence signal.

### 7.7 Limitations

- 36 episodes (12/band), 2 triggers; trigger-level estimates are unstable and
  the paired CIs are wide.
- R0 was reused from the Stage-2f smoke after exact episode/seed/band
  verification; it was not re-run.
- The P1-2 probe is a single seed; the supplement contains synthetic
  near-invisible states and synthetic hard-state histories.
- True distance/visibility/heading/oracle action were used only for offline
  scoring; the runtime policy used only the probe probability, M0 base action
  and observable high-confidence side/recency.
- No MEC training, no checkpoint modification, no controller integration;
  Final Test 300 remains sealed.

### 7.8 Artifacts

```
runs/v83_spatial_stage2/stage2g1/probe_20261019T124500Z/
  models/ (72 checkpoints)
  probe_provenance.json
  reproduction_check.json
  probe_results.json
  predictions_v2_near_target_invisible_val.npz

runs/v83_spatial_stage2/stage2g2/recovery_20261019T130000Z/
  r1_per_episode.csv
  steps/r1.csv
  events/r1.csv
  probe_steps.csv
  metrics.json
  provenance.json
```

Code: `stage2g2_policy.py`, `stage2g2_recovery_eval.py`,
`stage2g2_reproduce.py`, `test_stage2g2.py`; `stage2g1_probe.py` now supports
`--save-models`; `stage2g_probe.py` exposes optional stats/model returns.

Tests: **38 pass** (32 prior + 6 Stage-2g.2 policy/input tests).
