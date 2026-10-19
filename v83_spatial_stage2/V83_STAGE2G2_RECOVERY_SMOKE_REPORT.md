# V8.3 Stage-2g.2 — P1-2 Near-Target Recovery Closed-Loop Smoke

- Date: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Status: **BLOCKED before execution.** The P1-2 probe weights were never
  saved, so the closed-loop smoke cannot run without retraining the probe.
  Per the round's instruction, the probe was **not retrained**, and no R0/R1
  outcomes were fabricated.
- Scope: audit of the existing Stage-2g.1 artifacts; no MEC training, no probe
  retraining, no closed-loop run, no controller integration, Final Test 300
  sealed.

## 结论摘要 (Chinese TL;DR)

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

## Appendix B. What would make this report obsolete

A future round with a saved P1-2 external `state_dict` (and provenance) can
replace §2–§3 with the actual 36-episode R0/R1 smoke results. Until then, the
correct status is **blocked**, not “negative” and not “positive.”
