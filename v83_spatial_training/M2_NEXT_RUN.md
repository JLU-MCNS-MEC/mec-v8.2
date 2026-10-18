# M2 temporal visual reasoning — approved next experiment

M0/M1 trained under `runs/v83_spatial_training/run_20261018T023830Z/`.
M1 vs M0: 60 development episodes SR 10→13, OSR 10→15, paired OSR gain 5 and losses 0 (p=0.0625). These data justify an M2 hypothesis test, **not** a claim of significance.

Only Transformer-M M2 is authorized next. No new multi-seed runs, no full 5-MEC retraining, and no Depth or MEC BEV.

## Code correction

`eval_closedloop_spatial.py` now:
- accepts `--reference-model-dir` to evaluate M0/M1 from the existing immutable run and M2 from a new run;
- reports M1 vs M0, M2 vs M0, M2 vs M1;
- treats *fewer* premature stops/timeouts as favorable in paired counts. Raw rates and `delta_rate` remain unchanged.

This is a reporting correction, not a model/metric-definition change.

## Commands (Ubuntu Workspace, DRL)

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2
git status --short
git pull --ff-only origin experiment/v83-spatial-foundation
conda activate DRL

python -m unittest -v v83_spatial_training.test_spatial_training_pipeline
python -m compileall -q v83_spatial_training

CACHE=runs/v83_spatial_training/cache_20261018T023637Z
M2RUN=runs/v83_spatial_training/m2_run_20261018
BASE=runs/v83_spatial_training/run_20261018T023830Z

# Before any training, verify files exist:
test -f "$CACHE/train.npz" && test -f "$CACHE/val.npz" && test -f "$CACHE/val_hard.npz"
test -f "$BASE/Transformer-M/M0/best.pt" && test -f "$BASE/Transformer-M/M1/best.pt"
test ! -e "$M2RUN" || { echo "M2 run directory already exists; choose a NEW output dir"; exit 1; }

# One-batch smoke (isolated, inexpensive)
python v83_spatial_training/train_spatial_mec.py \
  --models M2 --smoke --cache-dir "$CACHE" \
  --output-dir runs/v83_spatial_training/m2_smoke_20261018

# Train only M2, same seed/hyperparameters as M0/M1
python v83_spatial_training/train_spatial_mec.py \
  --models M2 --cache-dir "$CACHE" --output-dir "$M2RUN" \
  --seed 20264017 --max-epochs 80 --patience 10

# Paired 60-episode development comparison; no checkpoint copying/overwrites.
python v83_spatial_training/eval_closedloop_spatial.py \
  --model-dir "$M2RUN" --reference-model-dir "$BASE" \
  --modes M0 M1 M2 --max-episodes 60 --threshold 0.5
```

**Important:** check actual output dir names before execution. For `--smoke`, use a fresh output path. If `test ! -e` fails, stop; do NOT delete anything. `set -e` outside a persistent shell is not needed; Claude should halt on any failed command.

## Protocol integrity

* Archived and new M0/M1 checkpoints untouched.
* Cached temporal data uses previous expert action on training sequences and actual previous model action during closed-loop evaluation.
* Evaluate every model on the same 60 episode seeds with one fixed Stop threshold 0.5. Do not tune threshold on the evaluation set.
* Additional train parameters and distribution shift matter. Log M2 parameter count and % over M1, best epoch, movement Macro-F1 and directional balance, natural/hard Arrival-F1, SR/OSR/SPL/nDTW, early-stop/timeout, and paired win/loss/ties.
* **Use per-episode paired confidence intervals**, and describe n=60 as exploratory development evidence. Correlated timesteps do not count as independent trials.
* Report latency cautiously: original estimator uses `time.perf_counter` without CUDA synchronization. Accurate inference latency requires `torch.cuda.synchronize()` before/after timed segments, warmup and representative runs; don't overstate M2 faster/slower from the previous unsynchronized 1.2 ms numbers.
* If M2 fails to improve OSR/SR relative to M1, do not expand to all models. Focus on long-distance / invisible-goal navigation and off-policy recovery data.

## Next decision

A meaningful M2 improvement is paired positive movement reachability (OSR) without material SR regression. If M2's offline score improves but 60-episode closed-loop OSR stays flat, temporal correlation gains alone are not enough. Keep network 5D and Arrival Head fixed across modes.
