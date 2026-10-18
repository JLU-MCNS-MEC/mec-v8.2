# V8.3 Visual-Spatial Grounding Diagnostic Report

- Date: 2026-10-18
- Branch: `experiment/v83-spatial-foundation`
- Code HEAD at run time: `339ee5a` (visual grounding / geometry prototypes and tests)
- Frozen: shared RGB encoder + Transformer-M Best-50
- Data: 60 V8.3 **development validation** episodes (seeds 900000–900077), policies 0.6 and 0.9
- Steps: 30,715; consecutive transitions: 30,595
- Runtime: **236.8 s** (~3.9 min)
- No MEC training; no Depth; no model/checkpoint changes. Goal projection/silhouette used only for offline metrics; never a model input.

## 1. Code/test audit

Run before diagnostics:

- `test_visual_grounding_v83.py` — 5/5 OK
- `test_mec_geometry_v83.py` — 3/3 OK
- Full related CPU suite (including depth, TensorBoard, threshold, probe tests): **31/31 OK**
- `compileall` for the grounding and geometry modules: OK

No defects were found in `visual_grounding_v83.py` or `mec_geometry_v83.py`; they were not modified. The new diagnostic script had two local issues fixed during smoke testing (checkpoint path and two test assertions); the prototype modules remain unchanged.

## 2. Localization metrics

Maps:

- `old8_up`: existing 3×8×8 heatmap, max over scales, nearest-upsampled to 24×24 (the actual model heatmap resolution is 8×8).
- `baseline24`: old per-patch 3-scale MAX cosine on the 24×24 token grid.
- `coherent24`: new multiscale local-consistency map.
- `temporal24`: coherent + yaw-registered previous map; equals coherent on non-turn steps.

Metric definition: R@1 = top-1 patch contains the Goal center; R@4 = top-4 candidate patches contain the Goal center; IoU = top-4 patch mask vs offline Goal silhouette mask. Only steps with the Goal projection center inside the image are included. Goal projection/silhouette are offline diagnostics only.

### 2.1 Overall (visible/center-inside subset)

| Map | n | R@1 | R@4 | IoU |
|---|---:|---:|---:|---:|
| old8_up (actual 8×8 model input) | 13,876 | 5.15% | 20.69% | 0.1337 |
| baseline24 | 13,876 | 17.53% | 48.95% | 0.2960 |
| coherent24 | 13,876 | **23.90%** | **63.32%** | 0.3120 |
| temporal24 | 13,876 | **30.15%** | **65.60%** | **0.3268** |

Overall visible rate: **45.23%**; center-inside rate: **45.18%**. More than half of all states have no Goal projection; all localization metrics are conditional on visibility.

On the **turn-valid subset** (n=15,809; temporal map is actually used):

| Map | n | R@1 | R@4 | IoU |
|---|---:|---:|---:|---:|
| baseline24 | 9,116 | 15.03% | 48.57% | 0.3052 |
| coherent24 | 9,116 | 25.75% | 69.83% | 0.3314 |
| temporal24 | 9,116 | **35.25%** | **73.30%** | **0.3539** |

- Coherent improves R@1 by +6.4 points, R@4 by +14.4 points, IoU by +1.6 points over baseline24 overall.
- Temporal improves R@1 by +9.5 points and R@4 by +3.5 points over coherent on turn-valid states.
- `old8_up` is much weaker than both 24×24 maps: the 8×8 model heatmap loses spatial precision.

### 2.2 By trajectory band

| Band | Map | n | R@1 | R@4 | IoU |
|---|---|---:|---:|---:|---:|
| short | baseline24 | 4,671 | 15.05% | 44.23% | 0.3292 |
| short | coherent24 | 4,671 | **33.57%** | **78.59%** | **0.3612** |
| medium | baseline24 | 5,098 | 15.69% | 40.41% | 0.2776 |
| medium | coherent24 | 5,098 | **23.07%** | **51.18%** | **0.2873** |
| long | baseline24 | 4,107 | **22.64%** | **64.91%** | 0.2810 |
| long | coherent24 | 4,107 | 13.95% | 61.02% | **0.2868** |

- Coherent clearly improves short and medium bands.
- **Long band is a regression for R@1/R@4** (22.64→13.95, 64.91→61.02); IoU is slightly better. The local-consistency prior can favor broader coherent regions at long range.
- Goal visibility by band: short 47.5%, medium 34.1%, long 69.6%.

### 2.3 By current distance

| Distance | Map | n | R@1 | R@4 | IoU |
|---|---:|---:|---:|---:|---:|
| 0–20 m | baseline24 | 54 | 1.85% | 5.56% | 0.1437 |
| 0–20 m | coherent24 | 54 | 7.41% | 31.48% | 0.1626 |
| 0–20 m | temporal24 | 54 | 7.41% | **35.19%** | **0.1681** |
| 20–40 m | baseline24 | 536 | 0.93% | 7.65% | 0.1527 |
| 20–40 m | coherent24 | 536 | 6.16% | 22.76% | 0.1536 |
| 40–60 m | baseline24 | 1,947 | 9.45% | 28.30% | 0.1864 |
| 40–60 m | coherent24 | 1,947 | **27.27%** | **55.16%** | 0.1772 |
| 60–100 m | baseline24 | 1,181 | 6.01% | 26.16% | 0.2634 |
| 60–100 m | coherent24 | 1,181 | **28.96%** | **66.13%** | **0.2809** |
| 100–200 m | baseline24 | 5,685 | 21.76% | 56.20% | 0.3634 |
| 100–200 m | coherent24 | 5,685 | 21.72% | **73.26%** | **0.4124** |
| 200+ m | baseline24 | 4,473 | 20.90% | **60.21%** | **0.2856** |
| 200+ m | coherent24 | 4,473 | **26.20%** | 58.73% | 0.2721 |

- Coherent gives the largest gains at 40–200 m (R@4 +27–35 points; IoU +0.05 at 100–200 m).
- At 0–20 m, top-1 remains poor for every map; the sample is only 54 center-inside steps. Coherent/temporal improve R@4 but not R@1. On the turn-valid near-goal subset (n=10), R@1=0 for all maps; temporal R@4=44.4% vs coherent 22.2%.
- At 200+ m, coherent improves R@1 but slightly reduces R@4/IoU relative to baseline.

## 3. Yaw temporal alignment

Transition consistency is measured between the previous coherent map and the current coherent map, before/after yaw warping using the **actually executed previous action**. FORWARD is deliberately invalid for a rigid yaw warp (no translation flow model).

| Policy | Action | n | Unwarped Pearson | Warped Pearson | Δ Pearson | Win rate | Unwarped MSE | Warped MSE | Unwarped top-10 | Warped top-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6 | Forward | 4,494 | 0.8567 | — (invalid) | — | — | 0.1367 | — | 0.4361 | — |
| 0.6 | Left | 2,901 | 0.2568 | 0.6526 | +0.3959 | 91.8% | 0.7728 | 0.3213 | 0.0337 | 0.4549 |
| 0.6 | Right | 2,936 | 0.2580 | 0.6686 | +0.4106 | 95.0% | 0.7720 | 0.3024 | 0.0359 | 0.4724 |
| 0.9 | Forward | 10,292 | 0.8437 | — (invalid) | — | — | 0.1315 | — | 0.3934 | — |
| 0.9 | Left | 4,961 | 0.2482 | 0.6447 | +0.3965 | 91.1% | 0.8133 | 0.3394 | 0.0532 | 0.3925 |
| 0.9 | Right | 5,011 | 0.2489 | 0.6702 | +0.4212 | 93.0% | 0.8124 | 0.3073 | 0.0546 | 0.4074 |

**Conclusion:** yaw registration strongly improves visual-map consistency across LEFT/RIGHT turns (Pearson +0.40; win rate 91–95%; top-10 overlap increases ~10×). Forward transitions have high unwarped consistency (0.84–0.86) but no valid warp; the module correctly invalidates the temporal signal rather than pretending a fixed pixel shift is a translation model.

## 4. MEC BEV / Network 5D leakage audit

Static audit:

- `mec_geometry_v83.py` uses node `x/y/bandwidth_hz`, UAV-relative position/yaw, and previous/current network trends only. No Goal coordinate, true goal distance, oracle mask, or Goal ID.
- The BEV output is egocentric; the shift-invariance test passes (`mec_relative_geometry` returns identical maps after shifting both nodes and UAV by the same offset).
- `global_network_state()` returns only: availability ratio, nearest relative MEC distance, best-link SNR, best-link bandwidth, best-link visual token transmission latency. No Goal, model type/capacity, or absolute UAV coordinate.
- `build_random_gbs_layout()` takes only a layout seed; Goal is not an input.

Empirical ridge probe on held-out episodes (even episodes train, odd episodes test; n_train=2,974, n_test=3,190):

| Feature set | Target | R² | MAE | Permutation p |
|---|---|---:|---:|---:|
| network5 | bearing sin | -0.194 | 0.395 | 1.000 |
| network5 | bearing cos | -0.281 | 0.925 | 1.000 |
| network5 | goal distance | -0.333 | 194.1 m | 0.549 |
| bev192 | bearing sin | -1.184 | 0.514 | 0.0196 |
| bev192 | bearing cos | -1.071 | 1.027 | 0.0196 |
| bev192 | goal distance | -1.026 | 253.3 m | 0.980 |

All R² values are **negative** (worse than predicting the mean). The low permutation p-values for BEV bearing only mean the fitted mapping is less bad under the true target ordering than under permuted targets; they are not evidence of positive predictive power. The 192-D BEV ridge is also overfit/high-variance.

**Audit conclusion:** no explicit Goal-position leakage in the MEC BEV or Network 5D implementations, and **no evidence that Network 5D/BEV provide independent Goal navigation information** in this validation. Caveat: layout and Goal both derive from the episode seed; an indirect seed-level correlation cannot be fully excluded, but the held-out probe finds no positive signal.

## 5. Error analysis

1. **Visibility is the first constraint.** Only 45.2% of steps have the Goal projection visible (medium band only 34.1%). Localization metrics are conditional on visibility; a movement policy that frequently faces away limits any static visual grounding feature.
2. **Near-goal top-1 is still weak.** At 0–20 m, baseline/coherent R@1 is 1.9%/7.4% (n=54); on turn-valid near-goal states R@1=0. Coherent/temporal improve R@4, but the top-1 patch is still unreliable for a direct Stop decision.
3. **Long-range regression.** Coherent improves short/medium and 40–200 m, but long-band R@1/R@4 are worse than baseline. A blind replacement of the old map would hurt long-range localization; a gated/combined map is safer.
4. **8×8 heatmap resolution is a real bottleneck.** The actual model heatmap (`old8_up`) has much lower R@1/R@4 than either 24×24 map. Increasing the grounding resolution is a likely win, but requires a Movement Head input-shape change and retraining.
5. **Temporal gain is turn-only.** Yaw alignment is strongly beneficial on LEFT/RIGHT transitions, but FORWARD translation has no valid warp. The temporal channel must carry an explicit valid mask; first-frame and forward transitions must fall back to the static coherent map.
6. **No occlusion modeling.** The Goal projection may be visible but occluded by a nearer landmark; this diagnostic uses projection visibility only and does not use Depth. Near-goal localization may be optimistic/pessimistic depending on occlusion.
7. **Development-set caveat.** These 60 episodes were already used for threshold experiments; they are a development validation set, not an untouched final test.

## 6. Which features improve localization, which do not

| Feature | Localization effect | Verdict |
|---|---|---|
| `old8_up` existing heatmap | R@1 5.2%, R@4 20.7%, IoU 0.134 | Poor at 24×24 precision; resolution-limited |
| `baseline24` patchwise max cosine | R@1 17.5%, R@4 49.0%, IoU 0.296 | Fair baseline; weak at 0–40 m and medium band |
| `coherent24` local consistency | R@1 23.9%, R@4 63.3%, IoU 0.312; strong short/medium and 40–200 m | **Improves localization overall**, but **degrades long-band R@1/R@4**; needs gating/combination |
| `temporal24` yaw-aligned turn map | Turn-valid R@1 35.3%, R@4 73.3%, IoU 0.354; Pearson +0.40 on turns | **Strong on turns**, invalid on FORWARD; turn-only feature with valid mask |
| Network 5D | Ridge R² negative; no positive Goal-bearing/distance prediction | No evidence of new independent Goal information |
| MEC BEV | Ridge R² negative; no positive Goal-bearing/distance prediction | No evidence of new independent Goal information; use only as geometry/context |

## 7. Recommendation for Movement Head integration

Do **not** run full MEC training yet. The validation supports a staged movement-head ablation, not a blind replacement:

1. **M1 candidate — coherent local consistency:** feed the 24×24 coherent map (or a learned combination of baseline24 + coherent24) as an additional Movement Head input. Because coherent regresses on long-band R@1/R@4, a gated/learned combination with baseline is preferable to replacing baseline.
2. **M2 candidate — turn-valid temporal map:** add the yaw-aligned temporal channel only for LEFT/RIGHT transitions, with an explicit valid mask; fall back to coherent for FORWARD and first frames. Preserve previous-action provenance and reset per episode; never teacher-force a previous action from a different state.
3. **Network 5D / BEV:** do not add them as “new Goal information” based on this audit. Keep the existing 5D as-is; if BEV is tested later, it should be framed as geometry/context, not target localization.
4. **Next controlled experiment:** on Small and Transformer-M, single seed, 50–80 epochs, M0 baseline / M1 coherent / M2 temporal-turn-valid, with independent closed-loop validation and pre-registered SR/OSR/SPL. Do not train all five models or use Depth until the localization gains translate into closed-loop improvement.
5. **Near-goal Stop:** static top-1 remains weak at 0–20 m; a distance/visibility-aware or hysteresis Stop gate is likely needed in addition to improved grounding.

## 8. Artifacts

Results (Git-ignored):

`runs/v83_spatial_prevalidation/20261018T021943534425Z/`

- `step_metrics.csv` (30,715 rows)
- `transition_metrics.csv` (30,595 rows)
- `localization_overall.csv`
- `localization_by_band.csv`
- `localization_by_distance.csv`
- `temporal_summary.csv`
- `leakage_probe.json`
- `metadata.json`

Full stdout: `runs/v83_spatial_prevalidation/full_20261018T021942Z.log`

Code/tests/report: `v83_spatial_prevalidation/` (committed separately; no large artifacts).

No existing model, checkpoint, dataset, or historical result was modified. No Depth was used. No MEC training was started.
