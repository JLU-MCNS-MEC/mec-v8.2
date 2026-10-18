# V8.3 Spatial MEC Training (M0/M1/M2)

Status: Transformer-M **M0/M1 trained and evaluated**; M2 not started (awaiting decision). See `V83_SPATIAL_TRAINING_REPORT.md`.

Independent V8.3 training/evaluation entry point. It does **not** modify the V8.2 training entry or the five original model architectures.

## Ablations

- **M0**: original V8.2 model and inputs.
- **M1**: M0 + learnable residual fusion of `baseline24` and `coherent24` into the Movement Head.
- **M2**: M1 + `temporal24` and `innovation24` with an explicit temporal-validity mask.

Network 5D is retained unchanged. MEC BEV and Depth are not used.

## Files

- `spatial_cache.py` — build Train/Val/Val_hard spatial feature caches from archived token shards.
- `spatial_data.py` — `(episode_id, step_id)` cache lookup and shard loader.
- `spatial_mec.py` — `SpatialFusionHead` and `SpatialAwareMEC` M0/M1/M2 wrappers.
- `train_spatial_mec.py` — deterministic training entry, based on the V8.2 epoch-compare logic.
- `eval_closedloop_spatial.py` — 60-episode development-validation closed-loop evaluator with per-episode paired comparison.
- `test_spatial_training_pipeline.py` — CPU tests for wrappers, validity masking, cache lookup and base-initialization identity.

## Alignment rules for the temporal cache

- Keys: archived `(episode_id, step_id)`.
- Previous action: archived label at `step_id-1` of the same episode.
- First frame, hard independent samples, missing previous state, and FORWARD previous actions: `valid=0`.
- LEFT/RIGHT previous actions: yaw warp valid; `temporal=(1-carry)*coherent+carry*warped`, `innovation=coherent-warped`.
- No future action and no Goal truth/position/mask is used.

Cache alignment counts (full Train/Val/Val_hard) are in
`runs/v83_spatial_training/cache_20261018T023637Z/cache_manifest.json`:
no missing previous samples, no duplicate keys; first-frame/forward/hard samples are invalid.

## Commands

```bash
# build cache
python v83_spatial_training/spatial_cache.py --output-dir runs/v83_spatial_training/cache_<timestamp>

# single-batch / closed-loop smoke
python v83_spatial_training/train_spatial_mec.py --smoke --models M0 M1 --cache-dir <cache> --output-dir runs/v83_spatial_training/smoke_<timestamp>
python v83_spatial_training/eval_closedloop_spatial.py --smoke --model-dir runs/v83_spatial_training/smoke_<timestamp> --max-episodes 2

# first authorized Transformer-M M0/M1 run
python v83_spatial_training/train_spatial_mec.py --models M0 M1 --cache-dir <cache> --output-dir runs/v83_spatial_training/run_<timestamp> --max-epochs 80 --patience 10

# 60-episode development-validation closed-loop evaluation
python v83_spatial_training/eval_closedloop_spatial.py --model-dir runs/v83_spatial_training/run_<timestamp> --modes M0 M1 --max-episodes 60
```

## Fairness

- Same seed (20264017), shard/loader order, batch size, optimizer, scheduler, loss and 80-epoch/patience-10 budget for M0/M1.
- M1 base initialization is identical to M0; the fusion head is zero-initialized so epoch-0 behavior matches M0 exactly.
- Extra fusion parameters: M0 2,722,164; M1 2,724,839 (+2,675).
- The 60 V8.3 episodes are a development-validation set, not an untouched final test.
