# V8.3 Stage-2j Protocol — Five MEC Architectures on T5 Inputs

- Date frozen: 2026-10-19
- Branch: `experiment/v83-spatial-foundation`
- Frozen code commit: `3663c7f53b9b834e0924e8bc4f929a1f33be6bf4`
  (original freeze `81b7552db23c7aac7268b724953c3fcbd95bb0f2`;
  pre-test amendment 1 is recorded in §9)
- Machine-readable freeze: `v83_spatial_stage2/stage2j_protocol_freeze.json`
  (status `frozen`, 21 critical source hashes, model parameter counts,
  training/evaluation protocol, SEALED manifest hash)
- Frozen protocol file: this document

This protocol was written **before any model was evaluated on Final Test
300**. It fixes the five architectures, the shared input set, the training
budget, the checkpoint-selection rule, the evaluation entry point, the
Stop threshold and the test-reading procedure.

---

## 1. Scope

This round compares five existing MEC architectures under the **same**
Stage-2i T5 map-assisted multimodal input:

1. Small — native `late_fusion_mlp`
2. Transformer-S — native `explicit_transformer` (192D, 1 layer, 4 heads)
3. Medium — native `goal_cross_attention` (384D, 6 heads)
4. Transformer-M — native `explicit_transformer` (256D, 2 layers, 8 heads),
   reusing the existing Stage-2i T5 checkpoint
5. Large — native `explicit_transformer` (352D, 2 layers, 8 heads)

Not in scope:

- no re-test of Stage-2i T0–T4;
- no new input-modality ablation;
- no change to the T5 multimodal design, map construction, retrieval,
  coordinate transform or 22-D map representation;
- no Stop calibration experiment, no OSR–SR gap optimisation;
- no test-set hyperparameter search.

---

## 2. Frozen model table

| MEC | Architecture | Width / depth / heads | Parameters | Initialisation |
|---|---|---|---:|---|
| Small | `late_fusion_mlp` | 256 hidden | 611,716 | seed 20264017 |
| Transformer-S | `explicit_transformer` | 192D / 1 layer / 4 heads / ff×3 | 1,332,868 | seed 20264017 |
| Medium | `goal_cross_attention` | 384D / 6 heads | 1,991,044 | seed 20264017 |
| Transformer-M | `explicit_transformer` | 256D / 2 layers / 8 heads / ff×4 | 3,033,732 | Stage-2i T5 checkpoint |
| Large | `explicit_transformer` | 352D / 2 layers / 8 heads / ff×4 | 5,248,932 | seed 20264017 |

The architecture identities are checked against `config.GBS_MODEL_CONFIGS` in
`test_stage2j.py`. Small and Medium keep their native movement backbones:
mean-pooled late fusion, and Goal→Current cross-attention + late fusion,
respectively. They are not converted into Transformers. The shared
16-step history encoder and the 8-feature observable Arrival head are
protocol-level input/inference components, initialised from the same fixed
seeds for every architecture; they are not part of the architecture
difference under test.

Transformer-M is the Stage-2i T5 module with only an input-shape validation
wrapper and **no new parameters**. Its checkpoint is:

```
runs/v83_spatial_stage2/stage2i/T5_map_assisted/best.pt
SHA256 57c88b7d71e1a7f17862aaa8429afa8a041cba5e747f744f3a7febf47f273c6d
best epoch 24, checkpoint score 0.7798980641233373
```

Equivalence is verified by `stage2j_equivalence.py`: state-dict keys/shapes/
values, fixed-batch outputs, Dev Val 180 per-episode outcomes, step actions,
distances, arrival probabilities and aggregate metrics.

---

## 3. Frozen input set

All five models receive exactly:

| Input | Shape | Notes |
|---|---|---|
| Current RGB tokens | 576 × 384 | frozen shared encoder, precomputed |
| Goal RGB tokens | 3 × 384 | 14/28/56 px scales |
| Visual history | 16 × 768 | mean+max pooled past Current tokens |
| UAV relative pose history | 16 × 10 | relative motion/pose + executed action |
| Executed-action history | 16 × 4 | one-hot |
| History mask | 16 | causal, reset per episode |
| GAB | 20 | causal 8-bin bearing belief + confidence/entropy/age/valid |
| Anonymous landmark graph | 22 | Top-5 retrieval + 2 summaries |

No 3×8×8 heatmap branch and no network-state branch enter any model. GAB is
computed from observable Current/Goal similarity but only its 20-D belief is
fed to the MEC. History, pose, GAB and map features are causal, reset at
episode boundaries, and use the existing Stage-2i auxiliary sidecars
(`runs/v83_spatial_stage2/stage2i/aux/`). The map database is the same
map-assisted privileged prior for all five models: anonymous nodes, frozen
encoder, cosine retrieval to the Goal RGB token, Top-5, 22-D features. No
goal identity, `is_goal`, `is_distractor`, true target coordinate, true
distance or oracle action is read.

Output definition: movement Forward/Left/Right; binary Arrival gates Stop at
**0.7**. Same supervision: movement CE (soft class weights) + Arrival BCE,
no `pos_weight`.

---

## 4. Frozen training protocol

| Item | Value |
|---|---|
| Train set | `runs/v83_spatial_stage2/datasets/train_600/` (200/200/200) |
| Model selection | Dev Val 180 offline (`dev_val_180/`), 60/60/60 |
| Max epochs | 50 |
| Early-stopping patience | 8 |
| Batch size | 48 (validation 96) |
| Learning rate | 6e-5 |
| Optimizer | AdamW, weight decay 1e-4 |
| LR schedule | ReduceLROnPlateau factor 0.5, patience 3, min 1e-6 |
| Seed | 20264017 |
| AMP | enabled |
| Shared RGB encoder | frozen; tokens precomputed |
| Loss | movement CE (soft class weights) + Arrival BCE; no pos_weight |
| Checkpoint score | 0.7 × Movement Macro-F1 + 0.3 × mean(natural, hard) Arrival-F1 |
| Offline metric threshold | 0.5 (selection only) |
| Closed-loop Stop threshold | 0.7 |
| Test-time weight updates | none |

The four new MECs are trained independently with this exact budget. No model
gets extra epochs or a different learning rate because of its test result.
Transformer-M is not retrained; its original optimisation history is reused.

---

## 5. Frozen Final Test 300 protocol

### 5.1 Authorisation and freeze

- Formal reading requires `--split final_test --allow-final-test` and the
  frozen JSON path.
- `stage2j_eval.py` refuses the formal test unless every critical source hash
  and the model parameter count match the freeze.
- The SEALED manifest hash is checked against the freeze.

### 5.2 Seal verification

- `SEALED.json` lists 265 files: 264 token shards + `dataset_stats.json`.
- Every listed SHA256 is recomputed and must match; the actual shard set must
  equal the listed shard set. Sealed files are read-only and are never
  modified or regenerated.

### 5.3 Task list

The exact task list is derived **only** from the sealed token shards:
`episode_id`, `episode_seed`, `network_layout_seed`, `trajectory_band_id`.
No episode is regenerated and no seed is invented. Every row for an episode
must agree on its metadata; step ids must be contiguous from zero. The
loader requires 300 episodes, ids 0–299, and 100/100/100 Short/Medium/Long.
It cross-checks the sealed `dataset_stats.json` trajectory records (seed,
band, network layout seed, start-goal distance range) as a consistency check
only. The derived list is stored at
`runs/v83_spatial_stage2/stage2j/specs/final_test_300_spec.json` with the
SEALED manifest SHA256 and its own SHA256 in each `metrics.json`.

### 5.4 Evaluation

- Each model is evaluated once on the same 300 episodes, using its selected
  best checkpoint. No weight updates, no threshold changes, no early stop
  based on test performance.
- True labels, target coordinates and true distances are used only for
  evaluation metrics (SPL/nDTW oracle reference), never as model input.
- Metrics: SR, OSR, SPL, nDTW, Premature-Stop, Timeout, average decision
  steps; overall and Short/Medium/Long; map retrieval Top-1/Top-5; measured
  MEC-forward latency (batch 1, mean/p50/p95) plus feature-build and full
  decision-pipeline latency.
- Per-episode `per_episode.csv`, per-step `steps.csv`, `metrics.json`,
  retrieval CSV and logs are saved under each model's
  `eval_final_test_300/`.

### 5.5 Bug policy

Only engineering fixes independent of test outcomes are allowed. Any such
fix must be recorded, the critical-file hash change documented, and the
evaluation consistency re-verified. No change may be motivated by improving
a test score. If a formal test is interrupted, `--resume` continues from
completed episodes; completed episodes are not rerun.

---

## 6. Execution order and reporting

Execution order:

```
Small → Transformer-S → Medium → Large → Transformer-M
```

For each model: train (except Transformer-M, which is registered from the
existing checkpoint) → select the best Dev Val checkpoint → run Final Test
300 → display the full result block and cumulative table → append the
snapshot to `RESULTS.md` → start the next model. Models not yet tested show
`Pending`.

Outputs:

- per-model: checkpoint, config, training history, per-episode CSV, steps
  CSV, `metrics.json`, training/evaluation logs;
- `runs/v83_spatial_stage2/stage2j/progress.json` with independent training
  and test states;
- `runs/v83_spatial_stage2/stage2j/RESULTS.md` and the committed
  `v83_spatial_stage2/STAGE2J_RESULTS.md`;
- four bar charts (PNG 300 dpi + SVG) in
  `v83_spatial_stage2/figures/stage2j/`;
- final report `v83_spatial_stage2/V83_STAGE2J_FIVE_MEC_T5_TEST_REPORT.md`.

---

## 7. Pre-freeze inspection disclosure

Before this protocol was frozen, the following Final Test information was
inspected:

- the `final_test_300` directory listing;
- `SEALED.json` manifest fields and its file-hash list;
- metadata arrays only from `shard_0000.pt` (`episode_ids`,
  `episode_seeds`, `network_layout_seeds`, `trajectory_band_ids`,
  `step_ids`) to confirm the schema;
- top-level aggregate fields of `dataset_stats.json`.

Not read: Final Test token tensors, labels/arrival targets, any model
prediction, or any aggregate test metric. Zero models had been evaluated on
Final Test. These inspections did not select models, seeds, bands,
hyperparameters or thresholds; the task list used for evaluation is derived
from the sealed shards at test time under §5.3.

---

## 8. Reproduce

```bash
# tests
python -m unittest -v v83_spatial_stage2.test_stage2j

# protocol freeze (once, before any test evaluation)
python v83_spatial_stage2/stage2j_freeze.py

# train one new MEC
python v83_spatial_stage2/stage2j_train.py --model Small \
  --output-dir runs/v83_spatial_stage2/stage2j/Small

# register the existing Transformer-M checkpoint
python v83_spatial_stage2/stage2j_train.py --model Transformer-M \
  --reuse-stage2i-t5 --output-dir runs/v83_spatial_stage2/stage2j/Transformer-M

# Dev Val equivalence (Transformer-M)
python v83_spatial_stage2/stage2j_eval.py --model Transformer-M \
  --split dev_val --output-dir runs/v83_spatial_stage2/stage2j/Transformer-M/eval_dev_val_180
python v83_spatial_stage2/stage2j_equivalence.py

# formal Final Test 300 (one model, once)
python v83_spatial_stage2/stage2j_eval.py --model Small \
  --split final_test --allow-final-test \
  --freeze-json runs/v83_spatial_stage2/stage2j/protocol_freeze.json

# aggregate + figures after all five
python v83_spatial_stage2/stage2j_analyze.py
```

---

## 9. Freeze amendment 1 (pre-test, 2026-10-19)

- **Trigger:** engineering audit of task-list provenance before any model
  had been evaluated on Final Test 300.
- **Change:** `stage2j_eval.py` now writes the derived spec file once and
  reuses it on later models, validating a canonical content hash
  (`spec_content_sha256`) over `(sealed_manifest_sha256, specs)`. This makes
  the task-list provenance hash identical for all five models; it does not
  change seeds, bands, episodes, metrics or model code.
- **Original freeze:** `81b7552` (protocol document `dc51ff0`).
- **Amended freeze:** `3663c7f`; `stage2j_protocol_freeze.json` was
  regenerated with the new critical-file hashes.
- **Unchanged:** all five model structures and parameter counts, the shared
  T5 input contract, the training protocol, checkpoint selection, Stop
  threshold 0.7 and the Final Test procedure.
- **Test status at amendment:** zero models evaluated on Final Test 300;
  no test token tensor, label or metric had been read.
