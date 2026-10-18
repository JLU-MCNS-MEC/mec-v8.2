# V8.3 Visual-Spatial Pre-Validation

Status: diagnostics complete (2026-10-18). No MEC training, no Depth, no checkpoint/dataset changes.

## Files

- `visual_grounding_diagnostic.py` — offline diagnostics for baseline/coherent/temporal grounding, yaw-turn consistency, and MEC BEV/Network leakage.
- `test_visual_grounding_diagnostic.py` — CPU tests for localization metrics, IoU, consistency metrics, heatmap upsampling, distance bins and R².
- `VISUAL_GROUNDING_DIAGNOSTIC_REPORT.md` — full report with comparison tables, error analysis, leakage audit and integration recommendation.

## Results

`runs/v83_spatial_prevalidation/20261018T021943534425Z/` (Git-ignored):

- `step_metrics.csv` (30,715 steps, policies 0.6/0.9)
- `transition_metrics.csv` (30,595 transitions)
- `localization_overall.csv`, `localization_by_band.csv`, `localization_by_distance.csv`
- `temporal_summary.csv`
- `leakage_probe.json`
- `metadata.json`

Full stdout: `runs/v83_spatial_prevalidation/full_20261018T021942Z.log`

## Headline results

- Coherent 24×24 local-consistency map vs 24×24 baseline: R@1 23.90% vs 17.53%, R@4 63.32% vs 48.95%, IoU 0.312 vs 0.296.
- Coherent improves short/medium and 40–200 m, but **regresses long-band R@1/R@4**; a gated/combined map is safer than a blind replacement.
- Actual 8×8 model heatmap: R@1 5.15%, R@4 20.69%, IoU 0.134; resolution is a real bottleneck.
- Turn-valid yaw temporal map: R@1 35.25%, R@4 73.30%, IoU 0.354; turn Pearson improves +0.40 (win rate 91–95%). FORWARD is correctly invalid; no fake translation warp.
- Network 5D and BEV: no explicit Goal leakage; held-out ridge probe R² is negative for Goal bearing/distance, so no evidence of independent Goal navigation information.
- Near-goal (0–20 m) top-1 remains weak; static top-1 alone is insufficient for Stop.

## Safety

- No changes to `visual_grounding_v83.py` or `mec_geometry_v83.py`; all 31 related CPU tests pass.
- Goal projection/silhouette used only for offline metrics; never a model input.
- No Depth used; no MEC retraining.
- All large results are under Git-ignored `runs/`.
