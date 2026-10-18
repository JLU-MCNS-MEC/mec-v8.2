# V8.3 Depth Pre-Validation

Status: **diagnostics and lightweight probe training complete.** P0/P1/P2 were trained on 2026-10-18; P1 did not show a robust gain, so Stop-gate development validation was not run. See `PROBE_EXPERIMENT_REPORT.md`.

## Files

- `trajectory_depth_audit.py` — offline audit script for true Goal-region vs visual-match-region Depth coverage.
- `TRAJECTORY_DEPTH_AUDIT_REPORT.md` — full-trajectory audit on 60 held-out episodes under Stop thresholds 0.6 and 0.9.
- `PROBE_DESIGN.md` — RGB-only vs RGB+Depth Arrival probe design, dimensions, split isolation, leakage checks, and budget.
- `PROBE_EXPERIMENT_REPORT.md` — executed P0/P1/P2 results, alignment checks, deltas, error analysis, and decision.
- `depth_probe_experiment.py` — probe data builder, P0/P1/P2 training, metrics, caches, curves.
- `test_depth_probe_pipeline.py` — CPU tests for probe dimensions, metrics, ECE and P2 shuffle control.
- `runs/v83_depth_prevalidation/20261018T015027817453Z/` — raw trajectory audit outputs.
- `runs/v83_depth_probe/20261018T020614605676Z/` — probe caches, weights, curves, metrics, alignment checks.
- `runs/v83_depth_probe/full_20261018T020613Z.log` — full probe stdout.
- `runs/v83_depth_prevalidation/cpu_tests_after_depth_fix.log` — 13/13 existing CPU tests.

## Headline results

- Latest Depth fixes (`b1f250f`, `fc79dba`) and probe tests pass: **19/19 CPU tests OK**.
- True Goal-region Depth is available and correct when the Goal is visible: 100% at 40 m/20 m; 79.2%/82.1% at 60 m (Goal not visible in the rest).
- Visual-match Depth is usually available (94–100%), but at 20 m only ~50% of episodes have a visual region dominated by Goal Depth; top-1 patch contains the Goal center in **0%** of 20 m cases.
- Probe results (single seed):
  - P0 RGB-only: Natural Val AUPRC 0.8266, Hard Val 0.8396, Brier 0.0680/0.0661.
  - P1 RGB+Depth: Natural Val AUPRC 0.8370 (+0.0104 vs P0), Hard Val 0.8407 (+0.0010), but Brier/ECE worse on both.
  - P2 Shuffled Depth: Hard Val AUPRC 0.8402, essentially tied with P1 and better calibrated.
- **No robust P1 gain; Stop-gate validation was not run.** Visual matching remains the limiting factor, not the Depth sensor.

## Safety

- No existing model weights, datasets, historical results, or training hyperparameters modified.
- No five-MEC retraining. Only lightweight probes were trained; weights/caches are under Git-ignored `runs/`.
- The only new source/docs are in this `v83_depth_prevalidation/` directory.
