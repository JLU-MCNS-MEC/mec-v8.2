# V8.3 Spatial Reasoning — stage gate before GPU training

## Evidence that motivates the change

* Existing RGB+Depth Arrival probe gains: Natural Val AUPRC +0.0104, Hard Val +0.0010; Brier and ECE worse than RGB-only. Depth is NOT enabled by default.
* Independent held-out 60-episode Transformer-M rollout: no-stop OSR 19/60=31.67%; 41/60 unreachable under the frozen movement policy.
* Near-goal diagnostic: top-1 goal/current visual patch did not contain Goal center and visual candidate IoU was approximately 0.32-0.34. This suggests improving grounding before training with Depth.
* MEC layout is drawn independently of Goal. UAV-MEC geometry is a hypothesis to test, not a known source of target localization signal.

## Existing branches / safety

Work only on `experiment/v83-spatial-foundation`. Leave:
* `runs/archive_pycharm_v82_20261018/` unchanged;
* old checkpoint and token directories unchanged;
* all original V8.2 training/eval scripts unchanged until an independent V8.3 implementation passes validation;
* the historical 60 episode final test out of model selection.

The 60 episodes with seeds ~900000 have already been used for threshold experiments. They are now a DEVELOPMENT validation set, NOT an untouched final test.

## Prototype modules now implemented

1. `visual_grounding_v83.py`: static multiscale coherent goal matching with neighborhood support, separate from baseline patchwise max cosine matching.
2. `visual_grounding_v83.py`: previous-frame yaw reprojection using known LAST ACTION only for pure LEFT/RIGHT turns. For FORWARD the temporal signal is invalidated rather than pretending a rigid 2D warp is correct. Explicit tracker reset per episode prevents leakage.
3. `mec_geometry_v83.py`: UAV-relative MEC BEV occupancy, bandwidth-weighted occupancy, and network trends; no Goal location or MEC model capacity/type.

**None is yet connected to MEC training or used to claim SR/OSR improvement.**

## Near-term checks before integration

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2
git status --short
git pull --ff-only origin experiment/v83-spatial-foundation
source /home/ubuntu5/miniconda3/etc/profile.d/conda.sh
conda activate DRL
python -m unittest -v test_visual_grounding_v83.py test_mec_geometry_v83.py
python -m compileall -q visual_grounding_v83.py mec_geometry_v83.py
```

### Diagnostic design (no retraining)

A. Using frozen shared encoder and *fresh episode-split VAL trajectories*, compare old goal cosine heatmap to new local-coherence map on goal-object localization. Oracle Goal projection/soft mask ONLY for metric computation; never feed it into MEC. Report R@1 patch (containing Goal center), R@4 or region IoU, visible-only subsets, by distance band; explicitly check projection/Occlusion caveats.

B. On consecutive frames with a known turn action, check whether yaw-aligned previous map improves consistency against unwarped maps. Separate turn-only and forward-motion cases; current module does not claim translation flow modeling. Record no-goal/invisible rates and calibration.

C. Only after A/B yields evidence, build a new isolated training/eval entry point for a shared MOVEMENT branch (Stop fixed initially), on a single seed. Stage ablations:

- M0 current/goal V8.2 frozen visual baseline
- M1 + coherent local goal matching
- M2 + yaw-aligned temporal map (only for turn transitions)
- M3 + network 5D
- M4 + network 5D plus independent MEC BEV geometry

Ablations must retrain each affected predictor on the same episode train/validation splits, distinguish information gain from model-size/optimizer confounds and verify no future-action leakage. Do not claim preexisting 5D network state has new information beyond the old baseline. Raw UAV pose is not an MEC input: geometry should be expressed in a local reference frame.

**Further concern:** previous-action under *oracle rollout* is teacher-forced; a closed-loop policy produces its own preceding actions and visits out-of-distribution states. For training, choose explicit previous-action provenance, mask first-frame and hard independent samples, and never attach unrelated shuffled records as 'previous' states. For evaluation, tracker.reset() each episode and update only with the actually executed action.

### Training gate

Try Small and Transformer-M before expanding all five MEC models. Single deterministic seed as requested. Budget 50-80 epochs + separate early stopping; go beyond that only if val continues improving. Use an **independent closed-loop validation split** for checkpoint selection; final test is never used for iteration. Objective: improve paired OSR and SR with robust localization gain; no arbitrary guarantee of 50%.

### Agent task protocol

Claude Code / DeepSeek may run CPU smoke tests immediately, inspect code, and propose fixes. Do not run expensive GPU retraining until the baseline-vs-new map localization and action-conditioned temporal-alignment diagnostic reports are inspected. Push text source + tests only; big artifacts remain in ignored runs/.
