# V8.3 phase 1 — held-out Stop/Movement diagnostics (no retraining)

## Why this phase

Offline checkpoint score is weakly aligned with full closed-loop SR. Previous
best single-MEC OSR was <=25%, and premature stops terminate rollout.
We first test whether changing Stop decision alone can improve the reachable
success region, WITHOUT adding Depth or changing trained models.

## Sources and isolation

Source checkpoints: `runs/archive_pycharm_v82_20261018/gbs_v82_epoch_compare_results/`
(best50 and best100). Source shared encoder from the same archive. Leave the old
200-epoch and test checkpoints untouched.

New closed-loop validation spec: `runs/v83_validation/closedloop_val_specs.json`.
Created using `seed_start=900000`, excluding episode seeds in archived
train/natural-val/hard-val/final-test. This does **not** use archived final-test
episodes to choose a threshold.

Evaluation output (timestamped):
`runs/v83_validation/threshold_diagnostics/<UTC-run>/`.
All paths under `runs/` are Git-ignored; never upload large checkpoints to Git.

## Commands on the Ubuntu server (not in Claude chat)

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2
git status --short
git pull --ff-only origin experiment/v83-spatial-foundation
source /home/ubuntu5/miniconda3/etc/profile.d/conda.sh
conda activate DRL
python -m pip check
python -m unittest -v test_tensorboard_logging.py test_mec_geometry_v83.py test_v83_threshold_pipeline.py
python -m compileall -q prepare_closedloop_validation_v83.py eval_stop_threshold_v83.py
python prepare_closedloop_validation_v83.py --num-episodes 60
python eval_stop_threshold_v83.py --models Transformer-M --budgets 50 --thresholds 0.5 0.7 never --max-episodes 2
```

The last command uses just 2 tasks and 3 policies (6 rollouts), no training.
Output is printed per policy and written to a new `runs/` folder.
Review Python test results, CPU/GPU errors, dataset path and metadata first.

If smoke succeeds, run held-out validation (5 models **only after** cheap
diagnosis; run two initially):

```bash
python eval_stop_threshold_v83.py --models Transformer-M Small --budgets 50 --thresholds 0.5 0.6 0.7 0.8 0.9 never --max-episodes 60
```

This reruns visual encoder and simulates trajectories; it uses GPU, can take
substantial time. Run only when you explicitly approve the total experiment
budget. `never` is **not a deployable policy**, it is a diagnostic
counterfactual, forced movement even if Arrival probability is high.

Choose a threshold from this **independent validation set**, not the prior
60-episode final test set. On a new final test split, freeze threshold and
model before reporting SR/OSR/SPL.

### What the outcomes mean

* With higher Stop threshold: OSR rises, SR rises -> early stopping was a major blocker
* With higher threshold: OSR rises, SR stays low -> arrival calibration is still needed
* With `never`: OSR remains < 50% -> movement / grounding / recovery remains the dominant limitation
* Premature-Stop drops but timeout goes up -> the policy is delaying termination, not necessarily navigating better

## Claude Code + DeepSeek (already configured by user)

Inside the Workspace on the server, launch `claude` and verify its model is
the configured DeepSeek endpoint, not an unavailable `opus` alias.
Use this prompt INSIDE the Claude Code interaction:

> You are working on branch experiment/v83-spatial-foundation. First check
> git status and current commit. Without modifying existing checkpoints,
> token datasets or results, run the unit tests and Python compileall above.
> Audit new validation seeds against all existing split statistics, inspect
> threshold override and the never-stop action path. Only then run
> Transformer-M best50 for two held-out episodes at stop thresholds
> 0.5, 0.7 and never. Record errors and summarize results. Do not start
> full training or modify GitHub automatically. If a bug is found,
> propose the minimal change and wait for approval.

Claude cannot push stdout directly into the current ChatGPT conversation
without a separately connected integration. Copy the concise summary back
here for diagnosis/approval.

## Next stage: multimodal geometric model

The existing `mec_geometry_v83.py` is a **separate prototype**, not wired into
train/eval. It creates UAV-relative topology maps and a 5D network delta.
Separate physical MEC BEV geometry from Current/Goal **camera** matching.
Next controlled A0/A1/A2/A3 ablations will study visual matching heatmap,
encoded network state, and dynamic MEC geometry. A topology independent of
Goal is not automatically navigation-useful. Depth stays disabled.
Early stopping and 50–80 epochs will be introduced in a future dedicated
training entry point, NOT retroactively into archived checkpoints.
