# Stage-2j T5 input examples (three real Train-600 samples)

Date: 2026-10-20

Branch: `experiment/v83-spatial-foundation`

This directory contains **three real, unmodified Stage-2j T5 input samples**
exported from Train 600, their visualisations, and a lightweight reproduction
test.  No model was trained, fine-tuned or evaluated; no MEC or checkpoint was
modified.  Final Test 300 was not read.

The eight NPZ arrays are exactly the tensors that `stage2j_model.py` receives
for one decision.  All three NPZ files pass
`stage2j_model._validate_common_inputs` when batch-wrapped with
`unsqueeze(0)`, with `heatmap=None` and `network_state=None`.

## 1. Data isolation and provenance

- Source split: `runs/v83_spatial_stage2/datasets/train_600` (600 episodes,
  200/200/200 Short/Medium/Long; 51,965 natural samples).
- Sidecars: `runs/v83_spatial_stage2/stage2i/aux/train` (the existing Stage-2i
  history/GAB/map caches).  They are read-only; no file under `runs/` is
  written or regenerated.
- Hard states (`sample_type` 1/2/3: hard-positive, boundary-negative,
  distractor-negative) are excluded from selection.
- Only natural samples (`sample_type == 0`) are eligible.
- Final Test 300 is rejected by path guard in the exporter and test.
- The visual observation is reconstructed by replaying the archived expert
  actions with the stored episode seed and band; GAB and map features are
  reproduced from the archived heatmaps / map database.  See
  `diagnostics.json` for the replay check.

## 2. The three cases

The rules below were fixed before exporting and were applied once to the full
natural pool.  They use observable quantities only; navigation success, true
distance, goal coordinates and oracle actions were not selection criteria.
The full rule text, pool sizes and percentiles are in `selection_rules.json`.

| Case | Episode | Step | Band | Selection evidence | What it shows |
|---|---:|---:|---|---|---|
| `case_01_clear_match` | 143 | 19 | short | current confidence **0.9917** (100th pct); 16/16 history slots; GAB conf 0.9811, valid | A sharp, confident local Goal-Current match |
| `case_02_weak_match_history` | 116 | 105 | long | current confidence **0.0217** (0.78th pct); 16/16 history; GAB conf 0.1267 ≥ threshold 0.0929, valid | Weak current evidence but a valid 16-step history and a reliable GAB branch |
| `case_03_ambiguous_map` | 204 | 8 | long | map Top-5 spread **0.002053** (0.23rd pct); top-1 = 0.99999988, top-5 = 0.997946; 8/8 history | Five anonymous candidates are nearly tied by Goal-RGB cosine |

### Why case 2 is genuinely "weak current evidence"

`current confidence` is the **peakiness** of the 3×8×8 Goal-Current heatmap
after max-over-scales and softmax at temperature 0.10, not the absolute cosine
value.  In case 2 the 24×24 cosine map is uniformly high (mean 0.88, max
0.964), so the softmax over 64 cells is nearly uniform and the confidence is
only 0.0217.  The GAB still carries useful information: its last-reliable
8-bin branch peaks at bin 3 (0.407) and the state is valid (age 1/16).  This is
exactly the "low current confidence, valid history/belief" condition.

Case 3's map candidates are selected by visual similarity only; the goal node
is included in the database as an anonymous node like every other node.

## 3. NPZ contents

Each case has `t5_inputs.npz` (compressed, ~425–435 KB) with exactly these
eight arrays and dtypes:

| Key | Shape | dtype | Source | Meaning |
|---|---|---|---|---|
| `current_tokens` | 576 × 384 | float16 | archived train shard | frozen encoder tokens of the current 336×336 RGB, 24×24 patches |
| `goal_tokens` | 3 × 384 | float16 | archived train shard | Goal reference tokens at 14/28/56 px object width |
| `history_vis` | 16 × 768 | float16 | aux/train sidecar | mean+max pooled past Current tokens, slot 0 = t-1 |
| `history_pose` | 16 × 10 | float32 | aux/train sidecar | 6 relative motion/pose dims + 4 executed-action one-hot |
| `history_action` | 16 × 4 | float16 | aux/train sidecar | executed-action one-hot, slot 0 = t-1 |
| `history_mask` | 16 | uint8 | aux/train sidecar | causal validity mask |
| `gab` | 20 | float32 | aux/train sidecar | 8 belief + 8 last-reliable + confidence/entropy/age/valid |
| `map_feat` | 22 | float32 | aux/train sidecar | Top-5 × (sim, relative forward, relative right, normalized distance) + top-1 + margin |

`history_pose` contains 4 action one-hot dims **and** `history_action` stores
the same 4-D action sequence.  This redundancy is the current code's actual
design (`stage2i_data.pose_action_feature` and `history_arrays`); it is
documented in `../T5_INPUT_MODALITIES_AND_ARCHITECTURES.md` and was not changed
in this round.

The NPZ contains **no** goal coordinates, true distance, oracle action,
success label, goal identity, `is_goal`, `goal_index` or retrieval-correctness
label.  Those appear only in `diagnostics.json`, which is explicitly marked
`offline_only: true` and `not_model_input: true`.

## 4. Files per case

| File | Purpose |
|---|---|
| `t5_inputs.npz` | the eight real model-input arrays |
| `case_metadata.json` | episode/step/band, source shard + SHA256, array table, selection values, episode action provenance |
| `input_summary.json` | per-array ranges/means/std, validity checks, local-match, history, GAB and map statistics |
| `diagnostics.json` | offline-only replay/retrieval correctness, true distance, absolute pose, observation re-encode check |
| `current_rgb.png` | reconstructed current 336×336 observation (same scene as the archived tokens) |
| `goal_reference_scales.png` | the three Goal RGB reference scales (14/28/56 px) |
| `local_similarity_24x24.png` | current 576 tokens × 3 Goal tokens cosine similarity, 24×24; GAB confidence annotated |
| `gab_bearing_belief.png` | 8-bin current belief, 8-bin last-reliable belief, confidence/entropy/age/valid |
| `landmark_top5_map.png` | UAV-centric anonymous-map view with the actual Top-5 retrieved candidates and a side table |
| `landmark_top5.csv` | rank, similarity, relative_forward, relative_right, normalized_distance for the five candidates |

## 5. Run the export / tests

From the repository root:

```bash
# Re-export (refuses to overwrite existing case directories unless --force)
python3 v83_spatial_stage2/t5_input_examples/export_t5_input_examples.py

# Lightweight checks (no training, no closed-loop evaluation)
python3 -m unittest -v v83_spatial_stage2.t5_input_examples.test_t5_input_examples
```

The test suite checks: all three NPZ files load; exact shapes/dtypes; Stage-2j
shape validation; 16-step history reconstruction from the archived episode;
GAB reproduction from archived heatmaps; 22-D map-feature reproduction from the
map database; Top-5 CSV consistency; absence of privileged labels; metadata
hashes; and that the three selection rules were met.

## 6. Verification result (2026-10-20)

```
Ran 10 tests in 1.5s
OK
```

- `history_vis` recomputation matches the archived sidecar exactly except for
  at most one float16 ULP (6.1e-5) caused by CPU-vs-GPU pooling order; the test
  allows 2e-4 absolute tolerance.
- GAB matches the archived sidecar at `atol=1e-7`.
- `map_feat` matches at `atol=1e-5`; the Top-5 CSV reproduces the same
  similarity/forward/right/distance values.
- The reconstructed current observation re-encodes to cosine ≥ 0.999997
  against the archived tokens (max absolute difference < 0.001).

## 7. Important interpretation limits

- **This is not a pure-vision result.**  The anonymous landmark map and UAV
  self-pose provide a strong map-assisted positional prior.  In this simulator
  the clean retrieval is saturated at **100% Top-1 / 100% Top-5** because the
  map nodes and Goal reference share the same renderer and frozen encoder.
  Stage-2i noise diagnostics reduced Top-1 to 23.89% at 5% embedding noise and
  10.56% at 15%; the clean saturation is not a real-world robustness claim.
- **The 24×24 local-similarity image is not a T5 input.**  It is an observable
  intermediate.  The only belief input to the MEC is the 20-D GAB; the 3×8×8
  heatmap is an online intermediate used to update GAB, not a separate model
  branch.
- **The map visualisation never uses goal identity.**  Top-5 is computed by
  cosine similarity to the Goal RGB token; the goal rank is reported only in
  `diagnostics.json` after the fact.
- These are three illustrative samples from the **training** split, not a
  benchmark, not a held-out evaluation, and not evidence of model quality.
  Single simulator seed, single feature pipeline.
