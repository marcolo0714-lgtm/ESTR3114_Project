# Gradient conflict in token prediction and sentiment classification

This repository implements the experiment in `estr3114_proposal.pdf`. A GPT-2
Small backbone is trained on TweetEval sentiment with two losses: next-token
prediction and negative/neutral/positive classification. The comparison covers
token-only, sentiment-only, ordinary weighted AdamW, and PCGrad with AdamW.

## Setup

Use Python 3.10 or newer and preferably a CUDA GPU with ample memory. GPT-2
Small has about 124 million shared parameters, plus an **untied** language head;
PCGrad holds separate task gradients, so CPU runs are very slow and GPU memory
needs are substantial. Reduce `--batch-size` and `--max-length` when needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python 1_prepare_assets.py
```

For a GPU, install a CUDA-enabled PyTorch build using the instructions for your
driver at https://pytorch.org/get-started/locally/ before installing the remaining
requirements. Check `python -c "import torch; print(torch.cuda.is_available())"`.

`1_prepare_assets.py` requires internet **once**. It writes the GPT-2 checkpoint,
tokenizer, TweetEval's original train/validation/test splits, download cache, and
revision metadata under `resources/`. After that, training reads local files
only. The dataset and model files are too large to include in a source checkout;
run the preparation step before training. Sources are listed in
`resources/REFERENCES.md`. Do not tune on the test split.



## How to use

Run commands from the project root after installing the requirements. Prepare
assets first, then choose either an individual training run or the full
experiment. Step 3 calls step 2 internally; you do not need to run both.

### 1. Prepare assets: `1_prepare_assets.py`

```powershell
python 1_prepare_assets.py
```

This downloads GPT-2, its tokenizer, and TweetEval sentiment into `resources/`
with revision metadata. Internet access is required for preparation; subsequent
training and prediction use local files. Skip this step if assets are already
prepared.

### 2. Train one condition: `2_train.py`

Use this script to train and evaluate one method:

```powershell
python 2_train.py --method joint --weight 0.3 --seed 42 --steps 1000
python 2_train.py --method pcgrad --weight 0.3 --seed 42 --steps 1000
```

`--method` accepts `token`, `sentiment`, `joint`, and `pcgrad`. Use the same
learning rate, schedule, steps, batch size, sequence length, and seed for a
matched comparison. `--output` sets a separate run folder. Nonempty run folders are protected from overwriting. `--test` evaluates
the held-out split at the end. `--save-model` saves final weights and tokenizer
and can consume substantial disk space; metrics are always saved. A smaller
development run, once the resources are prepared:

```powershell
python 2_train.py --method pcgrad --steps 2 --eval-every 2 `
  --diagnostics-every 1 --batch-size 1 --train-limit 8 `
  --validation-limit 8 --max-length 32 --device cpu
```

Each run saves `config.json`, `history.jsonl`, and `summary.json` under
`results/weight<weight>_<method>_step<steps>_seed<seed>/` by default. For joint and PCGrad,
`--weight` controls the sentiment task's contribution to the update.

### 3. Run the full weight sweep: `3_run_experiments.py`

```powershell
python 3_run_experiments.py --weights 0.1 0.3 1.0 --seeds 42 43 44 `
  --main-steps 1000
```

Every supplied weight is used for full joint and PCGrad training with every
seed. Token-only and sentiment-only controls run once per seed because their
updates do not depend on the sentiment weight. Duplicate weights and seeds
are run once. Weights must be finite and positive.

There is no pilot or automatic weight selection. The old `--pilot-steps`,
`--pilot-train-limit`, `--pilot-validation-limit`, and `--max-token-loss-gap`
options have been removed. With three weights and three seeds, this runs
**24 main conditions**: `(2 controls + 2 methods * 3 weights) * 3 seeds`.
All main conditions evaluate validation during training and test at the end.
Use validation for future weight selection; test scores are for reporting.

Outputs under the chosen experiment folder:

- `weight1.0_token_step1000_seed42/` and
  `weight1.0_sentiment_step1000_seed42/`: controls (weight metadata
  is 1.0 and is unused by these single-task updates).
- `weight0.1_joint_step1000_seed42/`,
  `weight0.1_pcgrad_step1000_seed42/`, etc.: separate
  configuration, history, and summary for each method/weight/seed.
- `main_summary.csv`: one row per run, including its method, weight, and seed.

The default `results/` folder can hold multiple experiments.
Each condition uses `weight[value]_[method]_step[number]_seed[number]`.
Different weights or step budgets get different folders. Completed runs with
matching training settings are reused, while incomplete runs or incompatible
settings at the same path raise an error. For example, changing the batch size
without changing weight, method, steps, or seed requires a separate `--output`.
`main_summary.csv` includes completed test runs in that folder and is updated after each condition. Equivalent single-task controls are counted once per seed. It saves metrics but no model checkpoints. Run step 4 to create
figures and summaries; each weight remains separate in the analysis.

For Colab after cloning the updated code:

```python
!python 3_run_experiments.py --weights 0.1 0.3 1.0 --seeds 42 43 44 --main-steps 1000 --device cuda
!python 4_analyze.py --results results
```

### 4. Analyze results: `4_analyze.py`

After the full experiment completes, run:

```powershell
python 4_analyze.py --results results
```

`--results` selects the flat result folder (default: `results/`). The script
reads each condition's `config.json`, `history.jsonl`, and `summary.json`.
It rebuilds `main_summary.csv`, even if a Colab session ended before the sweep
finished. Only completed runs with test metrics enter the combined main results.
Validation-only runs are retained; short weight-check points are plotted
separately. Equivalent single-task controls with identical settings and metrics
are counted once per seed, preferring the weight-1.0 placeholder. Conflicting
repeated runs raise an error instead of being silently combined.

Five figures are written directly under the selected folder:

- `summary_convergence.png`: all weights on shared validation learning curves.
- `summary_tradeoff.png`: all validation trade-offs, with sample-SD error bars.
- `summary_layer_conflicts.png`: a line plot of all method/weight layer conflicts.
- `summary_runtime.png`: recorded training-plus-validation seconds per step.
- `summary_weight_effects.png`: all weights, paired seed differences and runtime.

`analysis_summary.json` records means, sample SDs and diagnostics;
`summary_inventory.json` identifies included, duplicate and validation-only runs.
No `main/` or `figures/` subfolder is created. GPU/model loading is not needed.

### 5. Inspect a checkpoint: `5_predict.py`

To inspect a trained checkpoint, use `--save-model` while training:

```powershell
python 2_train.py --method pcgrad --weight 0.3 --seed 42 --steps 1000 --save-model
```

Then provide the saved weights and a text:

```powershell
python 5_predict.py --checkpoint results/weight0.3_pcgrad_step1000_seed42/final_model.pt `
  --text "I love this!"
```

The script requires the local model/tokenizer assets from step 1. It prints
JSON containing the predicted sentiment, probabilities for all three labels,
token loss, and perplexity. It creates no files.

## What is measured

- Token loss averages over **real predicted tokens**, then perplexity is its
  exponential. Padding is ignored. The tokenizer appends EOS and pads on the
  right; sentiment uses the final EOS representation.
- Accuracy and macro-F1 are computed over all three sentiment labels.
- At the sampled diagnostic steps, `history.jsonl` contains global and per-layer
  **pre-projection** gradient cosine, global conflict, both shared gradient
  norms, and (for PCGrad) the post-projection global cosine. The analysis summary
  includes global conflict frequency and gradient norm ratio. The layer plot
  shows the fraction of sampled steps with negative per-layer cosine. PCGrad
  projects against the **whole shared backbone gradient**, so individual
  layers may still have negative post-projection cosine.
- The two heads are private. The pretrained language head is copied from GPT-2
  embeddings initially but no longer tied to them, permitting separate
  head-only gradients. The token head receives `g_token`; the sentiment head
  receives `lambda * g_sentiment`. The backbone receives their weighted sum
  for joint AdamW or two-task PCGrad's projected weighted sum.
- For fair wall-clock interpretation, joint AdamW computes two task gradients
  on diagnostic steps, whereas PCGrad does so on every step. The timer includes
  evaluation and logging. Compare runtime under the same cadence and machine.

The fixed step budget and matched initialization/batch order permit controlled
comparisons. These runs do not prove Pareto dominance or causal benefit of
conflict removal: losses, gradients, and optimization trajectories change with
lambda. Report means and variability across seeds and discuss negative findings
as well as improvements.

## Folder structure

The map shows the main project files and generated output locations. Results
appear as training and analysis complete; model checkpoints are optional.

```
├── 1_prepare_assets.py                 # Download local model and data
├── 2_train.py                          # Train one condition
├── 3_run_experiments.py                # Run all weights and matched seeds
├── 4_analyze.py                        # Plot and aggregate results
├── 5_predict.py                        # Inspect a saved checkpoint
├── src/                               # Reusable implementation
│   ├── __init__.py                     # Package marker
│   ├── data.py                         # Tokenization, EOS, and padding
│   ├── model.py                        # Shared backbone, heads, and losses
│   ├── gradients.py                    # PCGrad and gradient diagnostics
│   └── evaluation.py                   # Validation/test metrics
├── tests/                             # Small automated checks
│   ├── test_data_and_losses.py         # Padding and task losses
│   ├── test_gradients.py               # Gradient combination and PCGrad
│   ├── test_experiment_sweep.py        # Weight/seed coverage and output safety
│   ├── test_training_smoke.py          # Training-loop integration
│   └── test_analysis_smoke.py          # Analysis output generation
├── resources/                         # Prepared assets and references
│   ├── REFERENCES.md                   # Dataset, model, and paper links
│   ├── manifest.json                   # Asset revisions and dataset metadata
│   ├── data/
│   │   ├── train.jsonl                 # Training tweets and labels
│   │   ├── validation.jsonl            # Validation tweets and labels
│   │   └── test.jsonl                  # Held-out tweets and labels
│   ├── model/                         # GPT-2 weights, config, and tokenizer
│   └── hf_cache/                      # Hugging Face dataset cache
├── results/                           # Flat runs, original ZIPs and summaries
│   ├── *.zip                          # Original downloaded archives
│   ├── weight<weight>_<method>_step<steps>_seed<seed>/
│   │   ├── config.json
│   │   ├── history.jsonl
│   │   ├── summary.json
│   │   └── final_model.pt             # Optional checkpoint
│   ├── main_summary.csv
│   ├── analysis_summary.json
│   ├── summary_inventory.json
│   ├── summary_layout_migration.json
│   ├── summary_*.png                  # Five combined figures
│   └── ESTR3114_Combined_Experiment_Summary.pdf
├── output/previous_analysis/          # Preserved superseded output
├── estr3114_proposal.pdf               # Original project proposal
├── estr3114_Project_Resources.pdf      # Project resource document
├── README.md                          # Setup and experiment documentation
└── requirements.txt                   # Python dependencies
```

The root scripts are numbered by workflow: **1** prepare assets, **2** train one
condition, **3** run the full weight sweep, **4** analyze results, and **5** inspect
a saved checkpoint. Step 3 calls step 2 internally; step 5 requires a checkpoint
saved with `--save-model`. The numbers are a guide, not a requirement to run
every script in sequence.

Run local checks with `python -m pytest -q`. These tests use tiny tensors and
do not require model downloads. To verify full integration, run the two-step
development command above after preparing the resources.

## Root Python scripts

Install the packages in `requirements.txt` first. Paths below use the default
output locations.

- **`1_prepare_assets.py`**
  - **Purpose:** Download GPT-2 and TweetEval sentiment for later offline use.
  - **Dependencies:** Internet access; `datasets` and `huggingface_hub`
    (installed through the requirements). No earlier script is required.
  - **Output:** `resources/model/` (weights and tokenizer),
    `resources/data/{train,validation,test}.jsonl`, `resources/hf_cache/`, and
    `resources/manifest.json` (revisions and dataset metadata).
- **`2_train.py`**
  - **Purpose:** Train and evaluate one token-only, sentiment-only, joint, or
    PCGrad condition.
  - **Dependencies:** Assets from step 1; `src/data.py`, `src/model.py`,
    `src/gradients.py`, and `src/evaluation.py`; PyTorch, Transformers, NumPy,
    and scikit-learn.
  - **Output:** `results/weight<weight>_<method>_step<steps>_seed<seed>/` containing
    `config.json`, `history.jsonl`, and `summary.json`. `--test` adds test
    metrics; `--save-model` also writes `final_model.pt` and `tokenizer/`.
- **`3_run_experiments.py`**
  - **Purpose:** Run joint and PCGrad at every supplied weight and seed,
    with token-only and sentiment-only controls once per seed.
  - **Dependencies:** Calls `2_train.py` internally and inherits its
    dependencies and prepared assets. No separate step 2 run is needed.
  - **Output:** `<output>/weight<weight>_<method>_step<steps>_seed<seed>/` with configuration, history, and
    summary files; `<output>/main_summary.csv`. Defaults: 24 main runs.
    No pilot, weight selection, or model checkpoints.
- **`4_analyze.py`**
  - **Purpose:** Plot results and aggregate performance, gradient conflicts,
    and runtime.
  - **Dependencies:** Flat condition folders with settings, histories and summaries;
    `src/results.py`, Matplotlib and NumPy. No GPU required.
  - **Output:** `main_summary.csv`, `analysis_summary.json`, `summary_inventory.json`,
    and five `summary_*.png` figures directly under `results/`.
- **`5_predict.py`**
  - **Purpose:** Inspect one text's sentiment, sentiment probabilities, token
    loss, and perplexity.
  - **Dependencies:** `final_model.pt` from step 2 with `--save-model`;
    step 1's local model/tokenizer assets; `src/model.py`, PyTorch, and
    Transformers.
  - **Output:** Prints a JSON result to the terminal. Creates no files.

Local results are consolidated directly under `results/`. Original ZIP files
are unchanged. `summary_layout_migration.json` records moves and checksums.
Superseded derived reports/plots and one incomplete colliding run are preserved
outside results under `output/previous_analysis/`. All complete condition folders
remain available, including validation-only checks and duplicate raw controls.

Steps 2 and 3 accept `--output` to change their destination; step 4 accepts
`--results` to choose the experiment folder to analyze.
