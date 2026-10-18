# V8.3 Spatial Foundation — runbook (no Depth)

This is a **diagnostic/prototype branch** derived from `feat/tensorboard-v82-20261017`. It is not a trained V8.3 model. Nothing here replaces V8.2 architectures or their old results. Do not merge without results.

## Key evidence (60 paired episodes)

* V8.2 200-epoch checkpoints SR: Small 10.00, Transformer-S 13.33, Medium 21.67, Transformer-M 20.00, Large 13.33 (%).
* 100-epoch protocol, best100 SR: Small 10.00, Transformer-S 8.33, Medium 8.33, Transformer-M 21.67, Large 10.00 (%).
* Highest OSR = 23.33–25.00%, depending on checkpoint group; closing OSR-SR alone cannot meet 50% SR. No Depth is added.
* Offline 70/30 composite Score correlates imperfectly with long-horizon SR. Do NOT select the final model solely by the composite score or on the final test set.

## Phase 0: safeguard historical artifacts (on Ubuntu only)

From an SSH shell, NOT from the Claude interactive prompt:

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2
git status --short
git fetch origin
git switch --track -c experiment/v83-spatial-foundation origin/experiment/v83-spatial-foundation
source /home/ubuntu5/miniconda3/etc/profile.d/conda.sh
conda activate DRL
python migrate_v82_workspace.py --results-only
```

If local branch already exists: `git switch experiment/v83-spatial-foundation` then `git pull --ff-only`. If git status shows edits, stop and reconcile rather than reset them.

If dry run shows correct paths, no file conflicts, enough disk and **all old jobs have ended**:

```bash
python migrate_v82_workspace.py --apply --results-only
python migrate_v82_workspace.py
python migrate_v82_workspace.py --apply
```

The source files remain in /tmp. Destination existing artifacts are compared by SHA256 and are NEVER overwritten. The migration manifest is saved in `logs/migrations`. The dataset copy may be large and take time; it is required to run old V8.2 evaluations in Workspace. Verify critical files first: `shared_encoder.pt`, `token_dataset_test/dataset_stats.json`, 5 best checkpoints per experiment, and the CSV and plots. The `*.pt` and dataset paths are Git-ignored.

**Do not run** `generate_token_dataset.py` merely for migration: that script deletes its target output directory before regenerating, by design.

## Phase 1: TensorBoard validation and historical curves

Install TensorBoard into DRL **only after finishing earlier active work**, without upgrading Torch/CUDA:

```bash
python -m pip show tensorboard || python -m pip install tensorboard
python -m unittest -v test_tensorboard_logging.py test_mec_geometry_v83.py
python csv_to_tensorboard.py
find logs/tensorboard -name 'events.out.tfevents.*' | head
```

On Windows CMD/PowerShell open SSH port forwarding:

```powershell
ssh -p 2203 -L 6006:127.0.0.1:6006 ubuntu5@YOUR_SERVER_HOST
```

On the server:

```bash
tensorboard --logdir logs/tensorboard --host 127.0.0.1 --port 6006
```

Open http://localhost:6006 in the Windows browser.

## Phase 2: spatial and scoring diagnosis (NO new training in this branch)

`mec_geometry_v83.py` supplies **3 distinct UAV-relative ground-plane MEC geometry maps**:

1. coverage/occupancy distribution;
2. normalized bandwidth-weighted spatial distribution;
3. range-approach/recede trend relative to previous position;

plus delta of the existing 5 network-observable scalars. The input uses UAV position and yaw **only to calculate relative quantities**; it does not give the model absolute coordinates, Goal coordinates, oracle distance, MEC model type or capacity.

**Warning:** this ground-plane map is NOT pixel-aligned with the 8x8 Current/Goal camera matching heatmap. Do not sum them channel-wise. In a future trainable fusion architecture, separately encode vision matching + egocentric MEC geometry + 5D network state; then combine embeddings for Movement and Arrival, with reliability gating. A moving snapshot requires a previous UAV pose, valid previous network state, a same-episode time order and an explicit reset at episode boundaries. Independent shuffled token records cannot recover the temporal context reliably.

Recommended controlled ablations:
  A0 Current+Goal; A1 A0+visual heatmap; A2 A1+network 5D; A3 A2+dynamic MEC geometry.
Do not add Depth until closed-loop OSR has improved and a separate experiment justifies arrival-distance evidence.
Use the 100-epoch script's per-shard deterministic sampling, natural+hard validation, a single fixed seed per model as requested, initial budget 50–80 with early stopping, and a completely separate closed-loop validation split to choose checkpoints. Never choose a checkpoint on the final test split.

**P1 evidence thresholds (proposals, not findings):** score each input ablation using paired episode SR/OSR/SPL and confidence intervals; measure image ROI matching accuracy separately; log Movement recall and failure categories. Expand training only when a new hypothesis improves validation, rather than assuming 100+ epochs fixes input ambiguity.

## Phase 3: AI agent operation

From Workspace, launch `claude --model opus` only after checking its configured provider (previously verified using DeepSeek). Ask Claude Code to inspect status and run only CPU tests first. Treat source edits, expensive training and checkpoint overwrites as approval-required. Send the command output and changed file diff to ChatGPT for collaborative decisions.

**Avoid** executing `train_gbs_v3.py` or `train_gbs_v3_epoch_compare.py` in the migrated result directory: their original checkpoint output paths may overwrite existing artifacts or explicitly refuse to run.
