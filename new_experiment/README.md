# Gradient conflict in multitask CIFAR-100

This independent experiment compares scalar joint training and PCGrad on twenty
conditional five-class CIFAR-100 tasks. Each superclass is one task. A shared CNN
feeds twenty private heads; the superclass is supplied at inference. Accuracy is
therefore **not ordinary 100-class CIFAR-100 accuracy**. Chance accuracy is 20%.
This is a simple shared-CNN adaptation of the PCGrad benchmark, not a reproduction
of the paper's routing-network architecture or its reported accuracy.

## Setup

Use Python 3.10+ and preferably a CUDA GPU. The default CNN has three convolutional
blocks (160 channels each), ReLU and max pooling, two shared 320-unit fully
connected layers, and twenty five-class linear heads. It trains from scratch.
There is no batch normalization or dropout, keeping task-order state out of the
comparison. PCGrad needs additional memory for task gradients.

Run from `new_experiment/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python 1_prepare_assets.py
```

For Google Colab, upload/open [ESTR3114_CIFAR100_Colab.ipynb](ESTR3114_CIFAR100_Colab.ipynb),
choose a GPU runtime, and run cells in order. Push this folder to your GitHub
repository first so the notebook can clone it. **Assets stay on Colab's local
disk**. Main outputs go directly to `MyDrive/ESTR3114_CIFAR100/runs/<condition>/`,
with analysis and smoke runs separate. No parent-project code or assets are used.
The notebook includes a final checkpoint-download cell, excluding dataset files.

## How to use

Prepare assets first, then choose one condition or the sweep. Step 3 calls step 2
internally; running both is not required. Model saving is optional in the CLI and
enabled by default in the notebook to preserve final and best-validation states.

### 1. Prepare assets: `1_prepare_assets.py`

```powershell
python 1_prepare_assets.py --validation-per-class 50 --split-seed 123
```

This downloads the official CIFAR-100 Python archive, verifies its published MD5,
reads the supplied coarse/fine labels, and verifies the twenty-by-five mapping.
The raw archive and prepared arrays remain under `resources/`. No arbitrary tar
members are extracted. Optional `--archive` uses an already-downloaded archive.

A fixed seed reserves 50 of the 500 training examples per fine class: 45,000 train,
5,000 validation, and the untouched official 10,000 test examples. Each task thus
has 2,250 training, 250 validation, and 500 test examples. Split indices, labels,
class names, training-only channel mean/std, source checksum, and prepared file
hashes are retained. Changing split settings requires a fresh resources folder.

Training uses four-pixel zero padding, a random 32x32 crop, and horizontal flips.
Validation/test use only fixed channel normalization. There is no pretraining.
Complete preparation is safely reused after hash verification. If preparation
fails after writing partial arrays, keep them and choose a new resources path.

### 2. Train one condition: `2_train.py`

```powershell
python 2_train.py --method joint --steps 10000 --seed 42 --device cuda --save-model
python 2_train.py --method pcgrad --steps 10000 --seed 42 --device cuda --save-model
```

Each update uses `--batch-per-task 8` images from each task: 160 images total.
Both methods use identical sampling and augmentation RNG streams, separate from
PCGrad's projection RNG. Batch hashes in the histories allow paired analysis to
verify this equality. Task samplers cycle shuffled permutations without dropping
tail examples. Exposure is logged as image count and equivalent training epochs.

Defaults: SGD with learning rate 0.01, momentum 0.9, weight decay 0.0005, and a
cosine schedule defined relative to the complete step budget. Gradient clipping
is disabled (`--max-grad-norm 0`); a positive value enables it. Float32 and
deterministic PyTorch operations are used. These are starting settings, not a
claim of optimal hyperparameters. Longer budgets change the full schedule.

Validation metrics print **every 100 steps and at the final step** with
`flush=True`. `--eval-every` changes that interval. Every evaluation retains all
task/class metrics and confusion counts; `--diagnostics-every 100` samples gradient
diagnostics. Training loss, per-task losses/accuracies, batch hash, LR, gradient
norm, image exposure, and elapsed time are recorded every step.

Before a main run, use a smoke test in a separate folder:

```powershell
python 2_train.py --method pcgrad --steps 2 --batch-per-task 1 --eval-every 2 --diagnostics-every 1 --device cuda --output results_smoke/pcgrad
```

`--output` is an exact run folder; by default it is
`results/<method>_step<steps>_seed<seed>/`. Existing completed compatible runs are
reused. Settings, code hashes, device type/name, versions, and data hashes must
match. Incompatible or incomplete outputs are protected from overwriting.

With `--save-model`, checkpoints are:

- `best_validation.pt`: highest mean validation task accuracy; ties use lower loss.
- `final.pt`: the final training state.
- `last.pt`: resumable state saved at each evaluation interval.

They include weights, optimizer momentum, task mapping, normalization, settings,
sampler permutations/positions, and sampling/augmentation/projection/torch RNG
states. Use the same command with `--resume --save-model` and the same output
folder after interruption. Completed runs are skipped; incomplete runs resume
from the last committed checkpoint. Any newer log rows are truncated to that
step before continuing. This is exact continuation with unchanged settings, not
an option to change the total training budget. CUDA bitwise reproducibility across
different software/hardware is not promised.

`--test` optionally evaluates both final and best-validation states after training
(only final when weights are not saved), retaining separate metrics and per-image
probabilities. Test scores never select checkpoints. Alternatively evaluate saved
weights later with step 5, as the notebook does after validation inspection.

### 3. Run matched experiments: `3_run_experiments.py`

```powershell
python 3_run_experiments.py --seeds 42 43 44 --steps 10000 --device cuda --save-model --resume
```

This runs **six conditions**: joint/PCGrad for each of three seeds. Duplicate
seeds/methods run once. `--methods joint` or `--methods pcgrad` restricts the sweep.
Other training flags pass through to step 2. Step 3's `--output` is the experiment
root, unlike step 2's exact run directory. Child trainers run unbuffered, printing
their metrics every 100 steps. `main_summary.csv` updates after each completion.
An error stops the sweep rather than silently skipping a failed condition.

At default batch size, 10,000 updates process 1.6M images, about 35.56 training
epochs. Measure a pilot before choosing the final budget and learning rate.
Use validation and equal tuning budgets for both methods. Change the output root
when changing other settings at the same steps/seed; names alone are not a full
configuration identifier. No independent-per-task baseline is trained by default.

### 4. Analyze results: `4_analyze.py`

```powershell
python 4_analyze.py --results results --output analysis
```

Analysis needs no GPU or dataset. `--steps` filters a training budget. Outputs:

- `report.md` and `analysis_summary.json`: per-setting mean and sample SD across seeds.
- `paired_differences.csv`: PCGrad-minus-joint metrics and runtime ratios for common seeds.
- `per_task_metrics.csv`, `per_class_metrics.csv`: accuracy, F1, precision, recall, support.
- `confusions_*.json`: raw counts summed over available seeds with task/class names.
- `summary_convergence.png`, `summary_tasks.png`, `summary_runtime.png`.
- `gradient_cosines_*.png` and JSON: mean sampled pre-projection task-pair cosines.

Different settings use separate signatures; duplicate seeds in a group raise an
error. Matched pairs must have identical sampling/augmentation hashes. One seed
cannot estimate seed SD. Raw matrices sum counts across seeds, not averaged F1.
If `evaluation_test_final.json` / `evaluation_test_best_validation.json` exist,
analysis verifies their checkpoint hashes and incorporates them without changing
the original training summary. The training CSV indexes original summaries.

### 5. Inspect or evaluate a checkpoint: `5_predict.py`

```powershell
python 5_predict.py --checkpoint results/joint_step10000_seed42/checkpoints/final.pt --image example.png --task fish
```

This prints five class probabilities using a supplied task ID/name. Image
inspection needs no dataset; normalization and labels are in the checkpoint.
RGB images are resized to 32x32. The superclass is not inferred.

After selecting the experiment using validation, evaluate saved weights:

```powershell
python 5_predict.py --checkpoint results/joint_step10000_seed42/checkpoints/final.pt --evaluate-split test --resources resources --output results/joint_step10000_seed42/evaluation_test_final.json --device cuda
```

Use the corresponding best-validation checkpoint and output basename for that
state. These files contain per-image identifiers, targets, predictions and all
five probabilities, as well as class metrics and confusion matrices. This allows
error analysis without rerunning inference. Do not select methods on test scores.

## What is measured

The scalar objective is `sum_t w_t * CE_t`, with twenty fixed weights `w_t=1/20`.
Optional `--task-weights` accepts twenty positive values normalized to sum one.
These are task-importance weights, not PCGrad's projection coefficients.

PCGrad collects original shared gradients, then sequentially projects each current
task gradient against original peer gradients in a random order, only when their
dot product is negative. It combines the results with the same task weights.
Private heads retain their own weighted gradients. For efficiency, projections
use a Gram matrix and coefficient rows; tests compare this to direct vector
projection. The peer-order RNG is independent and checkpointed. Later projections
can reintroduce conflicts with earlier peers; all final pairwise cosines need not
be nonnegative. This procedure does not guarantee superior accuracy.

Evaluation reports unweighted means across tasks: accuracy, macro-F1, cross-entropy,
and worst-task accuracy. Each of the twenty raw 5x5 confusion matrices has rows
as true class and columns as predicted class. Per-class precision, recall, F1 and
support are retained at **every validation evaluation**, and at test evaluation
when requested. Undefined precision/recall/F1 use zero; support exposes absent
classes. Confusions are within-task, not an unconstrained 100-way classifier.

Diagnostics retain all twenty shared gradient norms, 20x20 raw/post cosine matrices,
conflict fraction over nonzero off-diagonal pairs, projection-relative norm, and
the sampled peer orders. Joint computes individual gradients only on diagnostic
steps; PCGrad does so every update. Timing includes training, validation, logging,
and checkpoint I/O, excluding loading and final test evaluation. Training-only
seconds and peak CUDA allocated memory are also recorded. Compare runtime on
matched hardware/cadence; no fixed runtime/memory guarantee is made.

## Folder structure

```text
new_experiment/
    1_prepare_assets.py
    2_train.py
    3_run_experiments.py
    4_analyze.py
    5_predict.py
    ESTR3114_CIFAR100_Colab.ipynb
    README.md
    requirements.txt
    configs/default.json           # Reference defaults; not auto-loaded
    cifar_mtl/                    # Independent named package avoids parent src collisions
        __init__.py
        data.py
        model.py
        gradients.py
        evaluation.py
        results.py
    tests/
        conftest.py
        test_experiment.py
    resources/                    # Generated local assets
        cifar-100-python.tar.gz
        dataset.npz
        split_indices.npz
        task_mapping.json
        manifest.json
    results/                      # Generated run folders and main_summary.csv
        joint_step10000_seed42/
            config.json
            history.jsonl
            validation.jsonl
            summary.json
            checkpoints/          # Optional final.pt, best_validation.pt, last.pt
            evaluation_test_*.json # Optional later evaluation
    analysis/                     # Generated reports, CSV/JSON and figures
```

Run offline checks with `python -m pytest tests -q`. Tiny synthetic data still
covers all twenty tasks. Checks cover split/mapping correctness, matched batches,
Gram/direct PCGrad equivalence, ordinary joint gradients, private-head isolation,
checkpoint interruption/resume, analysis and prediction. No data download occurs.

## Root Python scripts

- **`1_prepare_assets.py`**
  - **Purpose:** Verify/download CIFAR-100 and prepare fixed independent splits.
  - **Dependencies:** Internet once (or `--archive`), PyTorch/NumPy, data/result modules.
  - **Output:** Local `resources/` archive, arrays, mapping, and manifest.
- **`2_train.py`**
  - **Purpose:** Train one scalar-joint or PCGrad condition, optionally resume/save/test.
  - **Dependencies:** Prepared assets and `cifar_mtl/`; PyTorch/NumPy.
  - **Output:** Configuration, every-step history, detailed validation, summary, optional checkpoints/test predictions.
- **`3_run_experiments.py`**
  - **Purpose:** Run both methods for all matched seeds.
  - **Dependencies:** Calls step 2; inherits its requirements.
  - **Output:** Flat condition folders and aggregate CSV.
- **`4_analyze.py`**
  - **Purpose:** Compare learning, task/class errors, gradients, variability and runtime.
  - **Dependencies:** Completed runs; NumPy/Matplotlib; no GPU.
  - **Output:** Reports, CSV/JSON, learning curves, task plots, gradient heatmaps.
- **`5_predict.py`**
  - **Purpose:** Conditional image classification or later checkpoint evaluation.
  - **Dependencies:** Saved weights; Pillow/PyTorch/NumPy; assets only for split evaluation.
  - **Output:** Printed predictions/metrics and optional evaluation JSON.

References: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html),
[Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782).
