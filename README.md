# MEC V8.2 — TensorBoard monitoring

This branch adds TensorBoard scalar logging to two existing training scripts without changing model architectures, training protocols, seeds, losses, optimizers, datasets, output directories, or checkpoint selection. Original CSV, PNG and checkpoint files are preserved.

## Protect the currently running PyCharm jobs

Existing training under /tmp/pycharm_project_479/test_v8.2 is not changed by a GitHub branch update. Wait for in-flight jobs to complete before installing TensorBoard in the shared DRL environment, as pip or conda may update dependencies. Never run setup or branch changes against the PyCharm /tmp working directory.

## Install in DRL (after current training completes)

```bash
cd /home/ubuntu5/Workspace/MEC-V8.2
source /home/ubuntu5/miniconda3/etc/profile.d/conda.sh
conda activate DRL
python -m pip show tensorboard
# If missing, after current training is complete:
python -m pip install tensorboard
tensorboard --version
python -c "from torch.utils.tensorboard import SummaryWriter; print('TensorBoard OK')"
```

## CPU-only tests: no dataset, GPU or checkpoints

```bash
python -m unittest -v test_tensorboard_logging.py
```

The tests use temporary folders, create events and read scalar values back via EventAccumulator. They verify metric mapping, isolated run names, historical import idempotence and preservation of the original CSV.

## Convert existing CSV without retraining

After previous training finishes, copy any output directories from /tmp/pycharm_project_479/test_v8.2 into the stable workspace, preserving original files. Do not delete or move active results.

```bash
python csv_to_tensorboard.py --source-dirs gbs_v82_input_redesign_results gbs_v82_epoch_compare_results
```

Historical events are stored under logs/tensorboard/historical/<experiment>/<model>/<CSV_SHA256_prefix>/. Re-importing identical CSV content skips the existing event directory; changed CSV content creates a new snapshot.

## Future training runs

The two existing entry points additionally write TensorBoard Events for every epoch:

```bash
python train_gbs_v3.py
python train_gbs_v3_epoch_compare.py
```

Important: these commands run full training; do not execute them for a TensorBoard setup check. They still use existing token datasets and output paths and can overwrite or reject prior checkpoints according to their original code. Use an isolated experiment directory before new long runs.

Event paths: logs/tensorboard/<experiment>/<UTC_run_id>-pid<PID>/<model>/.

Recorded scalar namespaces:

- train/*: existing per-epoch total, movement and arrival Loss, training accuracy
- val/*: existing metrics including Macro-F1, natural/hard Arrival-F1 when available, false/missed Stop
- selection/checkpoint_score and optimizer/lr

Values are unchanged from original CSV (F1/Accuracy rates use 0..1 units, not percent).

## Remote browser access

On **Windows CMD/PowerShell**, establish the tunnel (use your existing SSH host):

```powershell
ssh -p 2203 -L 6006:127.0.0.1:6006 ubuntu5@YOUR_SERVER_HOST
```

In that server SSH shell:

```bash
source /home/ubuntu5/miniconda3/etc/profile.d/conda.sh
conda activate DRL
cd /home/ubuntu5/Workspace/MEC-V8.2
tensorboard --logdir logs/tensorboard --host 127.0.0.1 --port 6006
```

Open http://localhost:6006 on Windows. Keep the tunnel and TensorBoard process open. Avoid binding TensorBoard directly to the public network.

## Experiment files and Git

Git ignores logs/ and *_results/, including event files, CSV, PNG and model checkpoints. They are NOT automatically backed up to GitHub. Copy critical training results from /tmp after the old jobs finish and back them up separately.

## Stage-2n: fixed-city latency-only MEC association

Stage-2n is an independent experiment under `v83_spatial_stage2/`:

- protocol: `STAGE2N_PROTOCOL.md`
- setup / pre-training audit: `STAGE2N_SETUP.md`
- task builder: `stage2n_build.py`
- trainer (DQN/PPO): `stage2n_train.py`
- launcher (6 runs, never stops old jobs): `stage2n_launch.py`
- final evaluator: `stage2n_eval.py`
- analysis/figures: `stage2n_analyze.py`
- status: `stage2n_status.py`
- tests: `test_stage2n.py`

Quick start:

```bash
python v83_spatial_stage2/stage2n_build.py
python v83_spatial_stage2/test_stage2n.py
python v83_spatial_stage2/stage2n_consistency.py --tasks 6
python v83_spatial_stage2/stage2n_launch.py --max-parallel 6 --threads 2
python v83_spatial_stage2/stage2n_eval.py --split dev
python v83_spatial_stage2/stage2n_eval.py --split test --allow-final-test
python v83_spatial_stage2/stage2n_analyze.py
```

Raw checkpoints, TensorBoard events and per-episode logs stay in the ignored
`runs/v83_spatial_stage2/stage2n_latency_only/` directory.  Tracked results,
figures and the report are under `v83_spatial_stage2/stage2n_evidence/`.

### Stage-2n inference-latency calibration

The calibration measures the frozen Stage-2j `Small` and `Transformer-M`
checkpoints on real training inputs (batch 1, FP32), compares them with the
unchanged 2 TFLOPS analytical model, and saves results under
`v83_spatial_stage2/stage2n_evidence/latency_calibration/`.

```bash
python v83_spatial_stage2/stage2n_latency_calibrate.py
python v83_spatial_stage2/test_stage2n_latency_calibrate.py
```

Key outputs: `latency_calibration_report.md`, `latency_measurements.csv`,
`latency_summary.csv`, `latency_calibration.json`, and the 300 dpi figure
`figures/latency_comparison.png`.  This is a server RTX 5090 proxy
measurement, not UAV/MEC target-hardware data.  It does not train or read the
sealed Test set.
