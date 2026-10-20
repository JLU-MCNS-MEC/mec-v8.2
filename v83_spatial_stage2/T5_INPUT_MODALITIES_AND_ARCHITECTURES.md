# Stage-2j T5 Input Modalities and Five MEC Architectures

Date: 2026-10-20

Branch: `experiment/v83-spatial-foundation`

Companion examples: `v83_spatial_stage2/t5_input_examples/`

This document describes exactly what the Stage-2j models receive, how each
tensor is produced, how the five MECs fuse it, and which quantities are real
runtime inputs versus offline diagnostics.  It is based on the frozen code:
`stage2i_data.py`, `stage2i_prepare.py`, `stage2h_gab.py`, `stage2i_model.py`,
`stage2j_model.py` and `stage2j_eval.py`.

The three exported examples in `t5_input_examples/` contain the eight arrays
below for one real decision each.  No model was trained or evaluated to
produce them.

---

## 1. The eight runtime inputs

| Input | Shape for one decision | dtype in the archive | Runtime source |
|---|---|---|---|
| Current RGB tokens | 576 × 384 | float16 | frozen shared RGB encoder on the current 336×336 observation |
| Goal RGB tokens | 3 × 384 | float16 | frozen encoder on 14/28/56-px Goal reference canvases |
| Visual history | 16 × 768 | float16 | mean+max pooled past Current tokens |
| UAV relative-pose history | 16 × 10 | float32 | replay/online pose deltas + executed-action one-hot |
| Executed-action history | 16 × 4 | float16 | executed-action one-hot |
| History mask | 16 | uint8 | causal validity |
| GAB | 20 | float32 | causal 8-bin bearing belief + confidence/entropy/age/valid |
| Anonymous landmark graph | 22 | float32 | Top-5 map retrieval + 2 summaries |

There is **no** 3×8×8 heatmap branch and **no** network-state branch in
Stage-2j (`stage2j_model._validate_common_inputs` rejects both).  The heatmap
is still computed online as an observable intermediate to update GAB, but it
is not passed to the MEC.

No simulator ground truth enters these eight arrays: no goal coordinate, true
distance, true bearing, visibility flag, oracle action, success label, node
identity or `is_goal` flag.

---

## 2. Current RGB tokens

### 2.1 Encoder

`encoder.py:VisualTokenEncoder` is a small ViT-style patch encoder:

- input: 336×336×3 RGB in `[0,1]`;
- patch embedding: `Conv2d(3, 384, kernel_size=14, stride=14)`;
- 24×24 = 576 patch tokens, 384-D each;
- 2 Transformer layers, 6 heads, MLP ratio 4, dropout 0.10;
- positional embedding over the 24×24 grid.

The encoder is **frozen** (`shared_encoder.pt`).  It is used identically for
Current observations, Goal references and anonymous map-node canvases.

### 2.2 What the 576 tokens mean

Each token describes a 14×14-pixel patch of the current image.  The token grid
is row-major: token index `r*24 + c` is image row `r`, column `c` (top-left
origin).  The archived Train-600 tokens are float16; the model casts them to
float32 before use.  At runtime `stage2j_eval.py` computes the same tokens
online with the frozen encoder.

---

## 3. Goal RGB tokens

The Goal reference is **not one image**.  The simulator renders the same Goal
landmark at three object widths on a 336×336 canvas:

| Goal scale | Object width | Purpose |
|---|---:|---|
| scale 0 | 14 px | far / small projection |
| scale 1 | 28 px | medium projection |
| scale 2 | 56 px | near / large projection |

Each canvas goes through the frozen encoder to 576 tokens, then is reduced to
one token by **foreground-weighted pooling**:

1. background colour is read from the top-left pixel;
2. foreground weight = `max_channel |pixel - background| > 8/255`;
3. the weight mask is area-pooled to the 24×24 token grid;
4. the 576 tokens are averaged with those weights;
5. if the mask is empty, the centre token is used.

Result: a 3×384 tensor.  Three scales make the match less sensitive to how
large the Goal appears in the Current image.

---

## 4. Current–Goal local matching

### 4.1 Cosine similarity on the 24×24 grid

For each Goal scale `s` and Current patch `i`:

```
sim[s, i] = cosine( normalize(goal_token[s]), normalize(current_token[i]) )
```

The implementation is `gbs_model_v3.goal_current_similarity`
(`F.normalize` + `einsum`), producing 3×576 values.  Reshaped to 3×24×24, this
is exactly what `local_similarity_24x24.png` visualises.

### 4.2 The 3×8×8 heatmap (online intermediate only)

`token_data.goal_conditioned_spatial_heatmap` reshapes the 3×576 similarity to
3×24×24 and applies `adaptive_avg_pool2d` to 3×8×8.  This heatmap is used by
GAB (Section 6) and is stored in the original token shards, but **Stage-2j does
not feed it to any MEC**.  It is not a separate T5 branch.

### 4.3 Arrival-head match features

The shared binary Arrival head independently reduces the 3×576 similarity to
8 observables:

1. `max_match`: maximum patch-best similarity;
2. `topk_mean`: mean of the 10 largest patch-best similarities;
3. `mean_match`: mean patch-best similarity;
4. `entropy`: entropy of the softmax over patches at temperature 0.10;
5. `weighted_x`, `weighted_y`: attention-weighted patch coordinates;
6. `best_x`, `best_y`: coordinates of the best patch.

"patch-best" means max over the three Goal scales for each Current patch.  These
8 features feed the Arrival head, not the movement backbone.

---

## 5. 16-step visual and relative-pose history

### 5.1 Causal ordering

For the decision state at episode step `t`:

- history slot `0` = state `t-1`;
- slot `1` = state `t-2`;
- ...
- slot `15` = state `t-16`.

At `t=0` all slots are zero and masked.  Padding never contains future states.
The history is reset at every episode boundary.

### 5.2 `history_vis` (16×768)

For each past state, the Current 576×384 tokens are reduced to:

```
visual_descriptor = concat(mean over 576 tokens, max over 576 tokens)  # 768
```

The descriptor is stored as float16, matching the archived training sidecar.
It is **not** the raw 576 tokens; it is a compact past-scene summary.

### 5.3 `history_pose` (16×10)

For a history slot, let `previous_pose = (p_x, p_y, p_yaw)` be the past state,
`next_pose` the state reached after its executed action, and
`reference_pose = (r_x, r_y, r_yaw)` the query state `t`.  The 10 features are:

| Index | Feature | Normalisation |
|---|---|---|
| 0 | motion forward `dx cos(p_yaw) + dy sin(p_yaw)` | ÷ 5 m (`FORWARD_STEP_M`) |
| 1 | motion right `-dx sin(p_yaw) + dy cos(p_yaw)` | ÷ 5 m |
| 2 | motion yaw `wrap(next_yaw - p_yaw)` | ÷ 15° (`TURN_STEP_RAD`) |
| 3 | relative forward `(p_x-r_x) cos(r_yaw) + (p_y-r_y) sin(r_yaw)` | ÷ 100 m |
| 4 | relative right `-(p_x-r_x) sin(r_yaw) + (p_y-r_y) cos(r_yaw)` | ÷ 100 m |
| 5 | relative yaw `wrap(p_yaw - r_yaw)` | ÷ π |
| 6–9 | executed action one-hot `[Forward, Left, Right, Stop]` | 0/1 |

The motion block (0–2) says what the UAV did between the past and next states.
The relative block (3–5) says where the past state was in the query state's
frame.  Neither uses the goal position or true distance.

### 5.4 `history_action` (16×4) and the documented redundancy

`history_action[slot]` is the same executed-action one-hot as
`history_pose[slot, 6:10]`.  The current code therefore stores the action
twice:

- inside `history_pose` as 4 of its 10 dims;
- as a separate 16×4 tensor.

`HistoryEncoder` projects both through **separate** linear layers and adds the
results:

```
x = visual_proj(history_vis) + pose_proj(history_pose)
    + action_proj(history_action)
```

So the action signal enters twice through different learnt projections.  This
is an actual design redundancy, not a documentation artefact.  It is recorded
here for transparency; **this round does not modify it**.  A future ablation
could remove one copy, but that would change the frozen input contract.

### 5.5 `history_mask` (16)

`1` for a real past slot, `0` for padding.  The mask depends only on `t`:
`min(t, 16)` leading ones.  The three exported examples have 16, 16 and 8 valid
slots respectively.

### 5.6 History encoder

`HistoryEncoder` (shared by all five MECs, 128-D, 2 layers, 4 heads):

1. project 768→128, 10→128, 4→128;
2. sum the three projections and multiply by the mask;
3. add a learnt per-slot position embedding (also masked);
4. prepend a CLS token;
5. run a 2-layer, 4-head Transformer encoder (GELU, pre-norm, ff 2×);
6. return the normalised CLS vector (128-D).

The five MECs then project this 128-D vector to their own width.

---

## 6. Goal-Anchor Belief (GAB), 20-D

### 6.1 Evidence from the observable heatmap

For the current state, GAB takes the archived/online 3×8×8 heatmap:

```
heat[r, c] = max over the 3 Goal scales of heatmap[s, r, c]
logits = heat.flatten() / 0.10
p = softmax(logits)                       # 64 cells
bins[c] = sum over rows r of p[r, c]      # 8 horizontal bearing bins
confidence = max(p)                       # peakiness of the 8x8 map
```

The 8 bins are **image columns**, i.e. egocentric horizontal direction, not
depth.  Bin 0 is the left side of the image and bin 7 the right side.

### 6.2 Causal update

State: `belief` (8), `last_reliable` (8), `confidence`, `last_seen_age`,
`valid`.  Constants: α=0.50, β=0.25, forward β=0.15, max age 16, FOV 90°,
turn 15°, 8 bins (bin width 11.25°).

1. **Rotate the prior** if the previously executed action was LEFT or RIGHT:
   `±15° / 11.25° = ±1.333 bins` (LEFT shifts positive, RIGHT negative);
   FORWARD is a no-op.  There is no fixed-pixel translation warp.
2. Compute `evidence` and `confidence` as above.
3. If `confidence >= reliability_threshold`:
   ```
   belief       = 0.5 * prior + 0.5 * evidence
   last_reliable = evidence
   confidence   = 0.5 * confidence + 0.5 * evidence_confidence
   age = 0; valid = 1
   ```
4. Otherwise decay toward uniform:
   ```
   decay  = 0.15 if previous_action == FORWARD else 0.25
   belief = (1-decay) * prior + decay * (1/8)
   confidence *= (1-decay)
   age += 1; valid = 0 if age >= 16
   ```
5. Emit 20 features:
   `[belief(8), last_reliable(8), confidence, entropy(belief), age/16, valid]`.

The reliability threshold is fitted on **Train natural heatmaps only** (60th
percentile); the Stage-2i aux manifest value is
`0.09289754098983038`.  The belief and state reset at every episode boundary.

The exported `case_02_weak_match_history` illustrates the useful case: current
confidence 0.0217, but `last_reliable` peaks at bin 3 (0.407), age 1/16 and
`valid=1`.

---

## 7. Anonymous landmark map, 22-D

### 7.1 Map database

For an episode, the map is built from **all** environment landmarks:

- 160 ordinary landmarks + the Goal object = 161 nodes;
- the list is permuted with a seed-derived RNG;
- every node is rendered into a canonical 336×336 survey canvas with the same
  shape/marker drawing primitives and a 56-px object width;
- every canvas goes through the same frozen encoder and the same
  foreground-weighted pooling as the Goal references, producing one 384-D
  embedding;
- node positions are stored in world coordinates.

No node carries `is_goal`, `is_distractor`, a Goal ID or an identity flag into
the feature builder.  `build_landmark_map` returns `goal_index` **for
diagnostics only**; it is never passed to `map_retrieval_features`.

### 7.2 Retrieval

```
sims = cosine(node_embeddings, goal_token_scale2)
order = argsort(-sims)[:5]
```

`goal_token_scale2` is the 56-px Goal token (index 2), because the map
canvases use 56-px objects.  Retrieval uses appearance only.

### 7.3 The 22-D feature

Relative coordinates are in the UAV's current frame:

```
forward = (nx - x) cos(yaw) + (ny - y) sin(yaw)
right   = -(nx - x) sin(yaw) + (ny - y) cos(yaw)
distance = hypot(nx - x, ny - y)
```

All three are divided by `MAP_POSITION_SCALE_M = 500 m`.

For each of the 5 candidates, in rank order:

```
[similarity, forward/500, right/500, distance/500]      # 20 dims
```

Then two summaries:

- `top1 = sims[order[0]]`;
- `margin = sims[order[0]] - sims[order[1]]`.

Total 5×4 + 2 = 22.  The exported `landmark_top5.csv` contains exactly the five
per-candidate rows; the two summaries are in `map_feat[20:22]`.

### 7.4 Map-assisted premise and saturation

In the current simulator, clean retrieval is **100% Top-1 / 100% Top-5**:
the map nodes and the Goal reference are rendered by the same renderer and
encoded by the same frozen encoder.  The 22-D feature therefore provides a
strong map-assisted positional prior.  This is **not** a pure-visual navigation
result and should not be presented as one.

Stage-2i noise diagnostics show the retrieval degrading to 23.89% Top-1 at 5%
embedding noise and 10.56% at 15%; the clean 100% is a simulator property, not
a real-world robustness claim.

---

## 8. The five MEC architectures and how they fuse the inputs

All five share the same **HistoryEncoder** and the same **VariantArrivalHead**
(initialised from the same fixed seeds), so the architecture comparison changes
the decision backbone, not the input encoding.  The parameter counts are from
the frozen Stage-2j protocol.

| MEC | Movement backbone | Width / depth / heads | Parameters |
|---|---|---|---:|
| Small | `late_fusion_mlp` | 256 hidden | 611,716 |
| Transformer-S | `explicit_transformer` | 192 / 1 layer / 4 heads / ff×3 | 1,332,868 |
| Medium | `goal_cross_attention` | 384 / 6 heads | 1,991,044 |
| Transformer-M | `explicit_transformer` | 256 / 2 layers / 8 heads / ff×4 | 3,033,732 |
| Large | `explicit_transformer` | 352 / 2 layers / 8 heads / ff×4 | 5,248,932 |

### 8.1 Small — late-fusion MLP

1. mean-pool the 576 Current tokens → 384;
2. mean-pool the 3 Goal tokens → 384;
3. encode history → 128;
4. take GAB 20 and map 22 as-is;
5. concatenate `[384, 384, 128, 20, 22]` → 938;
6. `Linear(938→256) → GELU → Dropout(0.1) → Linear(256→3)`.

No cross-attention; the Current/Goal interaction is only through the pooled
vectors and the shared Arrival head.

### 8.2 Transformer-S — explicit transformer, 192-D

1. project all 384-D tokens to 192-D;
2. add learnt Current/Goal position embeddings;
3. run Goal→Current multi-head attention (3 queries, 576+1 keys/values; the
   extra key is a zero "no-match" token for Transformer variants);
4. build the token sequence:
   `[CLS, 576 current tokens, 3 matched goal tokens, history token, GAB token,
   map token]` = 583 tokens;
5. 1-layer, 4-head Transformer encoder (GELU, pre-norm, ff 3×);
6. take the CLS output → `Linear(192→3)` movement logits.

### 8.3 Medium — Goal→Current cross-attention + late fusion, 384-D

1. Current tokens + learnt position; Goal tokens + learnt scale position;
2. 6-head cross-attention: 3 Goal queries attend to 576 Current tokens plus a
   learnt `no_match` token;
3. residual + LayerNorm, then a 384→384 FFN with residual/norm;
4. concatenate:
   `[mean Current(384), mean Goal(384), mean matched(384), history(128),
   GAB(20), map(22)]` = 1322;
5. `Linear(1322→384) → GELU → Dropout → Linear(384→3)`.

This is the native Medium movement backbone, not a Transformer conversion.

### 8.4 Transformer-M — exactly the Stage-2i T5 module

`Stage2JTransformerMEC` subclasses `Stage2ITransformer("T5")` and adds only the
Stage-2j shape validation.  No parameters are added.  The state dict, fixed
batch outputs and Dev Val 180 outcomes are verified equivalent by
`stage2j_equivalence.py`; the checkpoint is the Stage-2i T5 `best.pt`
(SHA256 `57c88b7d…`).  Fusion is the same 583-token explicit Transformer as
Transformer-S but at 256-D, 2 layers, 8 heads, ff×4.

### 8.5 Large — explicit transformer, 352-D

Same 583-token explicit-Transformer fusion as Transformer-M, at 352-D width,
2 layers, 8 heads, ff×4.  It is a wider, not deeper, model.

### 8.6 What is common

For all five:

- the movement head is a linear map to **3** classes: Forward, Left, Right;
- Stop is **not** a movement class;
- the Arrival head is a shared 8→64→32→1 MLP over the 8 match features,
  followed by a sigmoid;
- a decision is:
  ```
  move = argmax(move_logits)
  stop = sigmoid(arrival_logit) >= 0.7
  action = Stop if stop else move
  ```
- the shared HistoryEncoder and Arrival head are protocol-level components,
  not part of the architecture difference under test.

### 8.7 Movement Head vs Arrival Head

| Head | Question | Inputs | Output | Role |
|---|---|---|---|---|
| Movement Head | "Which direction should I move?" | full multimodal fusion (tokens + history + GAB + map) | 3 logits | selects Forward/Left/Right every step |
| Arrival Head | "Am I at the goal now?" | only the 8 Current/Goal match features | 1 logit → sigmoid | gates Stop at threshold 0.7 |

The Arrival head never sees map geometry, GAB, executed actions, the true
distance or target coordinates.  It is deliberately a visual
Current/Goal-match detector.  The movement head never emits Stop.

---

## 9. Runtime inputs versus offline diagnostics

### 9.1 Real runtime model inputs

Exactly the eight arrays in Section 1, in the Stage-2j shapes.

### 9.2 Computed online but not fed to the MEC

- the 3×8×8 Goal-Current similarity heatmap (used to update GAB);
- the 8 Arrival match features (used only by the Arrival head);
- the pooled history embedding (produced inside the model);
- the anonymous map's node embeddings and similarities (used to build the
  22-D feature).

### 9.3 Offline diagnostics only

These appear in `t5_input_examples/*/diagnostics.json` and in the plots, but
are **never** model inputs:

- the 24×24 Current×Goal cosine panels;
- the 8-bin current evidence and GAB belief plots;
- the anonymous-map Top-5 map/CSV and relative bearings;
- goal rank / Top-1 / Top-5 retrieval correctness;
- absolute UAV pose, true distance to goal, goal visibility;
- oracle actions, success labels, trajectory outcome;
- the observation re-encode check.

`diagnostics.json` carries `"offline_only": true` and
`"not_model_input": true`.

---

## 10. The `history_pose` / `history_action` redundancy

This is worth stating explicitly because it is easy to miss:

- `history_pose` is **10-D**, not 6-D: 6 relative motion/pose features plus a
  **4-D executed-action one-hot**;
- `history_action` is a separate **4-D** one-hot of the same action.

Therefore `history_pose[:, 6:10] == history_action` exactly (verified by the
test suite).  `HistoryEncoder` projects both, so the action is represented
twice through separate `pose_proj` and `action_proj` weights.  The current
Stage-2j protocol keeps this design unchanged.  It is a candidate for a future
ablation, not a bug fixed in this round.

---

## 11. Scope and interpretation

- These inputs are for **Stage-2j** (five MEC architectures) and come from the
  Stage-2i T5 feature pipeline.  Earlier variants T0–T4 are out of scope.
- The three examples are Training-split samples used for illustration; they
  are not a held-out evaluation and do not measure model quality.
- The map retrieval is saturated in the clean simulator; the results are
  map-assisted, not pure visual navigation.
- Selection of the three examples used observable features only; success,
  true distance, goal coordinates and oracle actions were not selection
  criteria.  The fixed rules and candidate pools are in
  `t5_input_examples/selection_rules.json`.
- No training, fine-tuning, closed-loop evaluation or checkpoint change was
  performed for this document or the examples.
- Final Test 300 was not read.
