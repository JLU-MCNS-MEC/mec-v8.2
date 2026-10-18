# V8.3 Stage-2 Preparation

Status: Stage-2a Transformer-M M0/M1 training and 180-episode closed-loop evaluation complete; **M1 was not a robust improvement**. Stage-2b Never-Stop and Arrival-only A0/A1 are also complete: A1 improves A0 but remains far below the original M0; no Small/M1-Dir runs were started. See `STAGE2A_REPORT.md` and `V83_STAGE2B_REPORT.md`.

## Data design

| Split | Episodes | Short / Medium / Long | Seeds | Notes |
|---|---:|---:|---|---|
| Train | 600 | 200 / 200 / 200 | 1,000,000+ | natural + hard arrival states |
| Dev Val | 180 | 60 / 60 / 60 | 2,000,000+ | natural + hard states; development only |
| Final Test | 300 | 100 / 100 / 100 | 3,000,000+ | natural only; sealed with SHA256/read-only |

Network layout seeds use `NETWORK.layout_seed_base + seed_start*17 + episode_id`,
so the new ranges (≈17.82M / 34.82M / 51.82M) are disjoint from all old splits
(old layouts ≈1.16M / 1.50M / 1.67M / 2.01M; V8.3 dev ≈16.12M).

The existing V8.2 data/training protocol is not modified. Equal band weights are
applied only inside the Stage-2 generation process.

## Commands

```bash
# smoke generation (6 episodes per split, separate seeds)
python v83_spatial_stage2/generate_stage2_datasets.py --smoke

# full generation (guards against overwriting)
python v83_spatial_stage2/generate_stage2_datasets.py \
  --output-root runs/v83_spatial_stage2/datasets

# audit: split counts/seeds, far-negative retention, M1 Long failures, head direction
python v83_spatial_stage2/audit_stage2.py \
  --data-root runs/v83_spatial_stage2/datasets \
  --output-dir runs/v83_spatial_stage2/prep_audit_<timestamp>

# Stage-2 spatial cache (Train + Dev Val only; Final Test not cached)
python v83_spatial_stage2/build_stage2_cache.py \
  --data-root runs/v83_spatial_stage2/datasets \
  --output-dir runs/v83_spatial_stage2/cache_<timestamp>

# single-batch M1 smoke on new Train
python v83_spatial_training/train_spatial_mec.py --models M1 --smoke \
  --cache-dir <stage2-cache> --train-dir <stage2-train> \
  --val-dir <stage2-dev-val> --val-hard-dir <stage2-dev-val> \
  --output-dir runs/v83_spatial_stage2/smoke_<timestamp>

# 2-episode closed-loop M1 smoke on new Dev Val (existing M1 checkpoint; no training)
python v83_spatial_training/eval_closedloop_spatial.py \
  --model-dir runs/v83_spatial_training/run_20261018T023830Z \
  --modes M1 --max-episodes 2 \
  --val-specs <prep-audit>/dev_val_180_spec.json \
  --output-dir runs/v83_spatial_stage2/closedloop_smoke_<timestamp>
```

## Planned Stage-2 experiments (paused for approval)

- Backbones: Transformer-M and Small only.
- Within each backbone, M0/M1 use identical seed/data order/LR/batch/80-epoch/patience-10; across backbones use the original reasonable settings:
  - Transformer-M: lr 6e-5, batch 48.
  - Small: lr 1e-4, batch 64.
- First compare M0/M1 on Train 600 and Dev Val 180 (natural/hard + closed-loop).
- Then, if warranted, controlled Arrival far-negative ablations A0=3% (old), A1=10%, A2=25%, A3=distance-stratified 10%, using the same tokens and training budget; no old protocol changes.
- Then a minimal M1-Dir head variant (2×2 pooled spatial summary) if the baseline confirms direction loss matters.
- Final Test 300 is sealed and never used for selection/tuning.

See `STAGE2_PREP_REPORT.md` for diagnostics, data statistics, and time/disk estimates.
