# Gradient conflict in next-token and next-concept prediction

This independent experiment compares ordinary weighted AdamW and PCGrad with
AdamW in a small ConceptLM-style GPT-2 language model. It jointly learns next-token
and next-concept prediction, with token-only and architecture-only controls.
All code, prepared assets, and results belong to this folder; nothing is imported
from the original sentiment experiment.

## Setup

Use Python 3.10 or newer and preferably a CUDA GPU with ample memory. This model
adds a concept module to pretrained GPT-2 Small. PCGrad keeps separate task
gradients, so memory use is substantial. Reduce `--batch-size` and
`--eval-batch-size` if needed. Sequence length is chosen during asset preparation.

Run these commands from `another_experiment/`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python 1_prepare_assets.py
```

For Google Colab, open [ESTR3114_NCP_Colab.ipynb](ESTR3114_NCP_Colab.ipynb)
and select a GPU runtime. The notebook uses the same clone/Drive/run workflow as
the original experiment, with independent assets and outputs under
`MyDrive/ESTR3114_NCP/`. Push these new files to GitHub before cloning in Colab.

`1_prepare_assets.py` requires internet once. It downloads this experiment's own
GPT-2 weights, tokenizer, and WikiText-2 raw dataset. Hugging Face revisions resolve
to immutable commits, recorded in `resources/manifest.json`. Subsequent training
uses local files. Assets are too large for the source checkout and are gitignored.
Do not tune on the test split.

## How to use

Run commands from this folder after installing requirements. Prepare assets first,
then run one condition or a full sweep. Step 3 calls step 2 internally; you do not
need both. The scripts are independent of the parent project's numbered scripts.

### 1. Prepare assets: `1_prepare_assets.py`

```powershell
python 1_prepare_assets.py --sequence-length 256
```

This writes the converted pretrained checkpoint, tokenizer, documents, packed
token blocks, cache, and checksummed manifest under `resources/`. WikiText-2 is a
small development corpus; it is not the paper's original large-scale training data.
Published splits are retained; articles are assembled using headings, and exact
duplicate documents are removed across splits. Splits are independently packed
with EOS separators and masked right padding. Cross-document transitions within a
split are trained normally. No chunk crosses split boundaries.

Optional `--local-text corpus.jsonl` accepts one `{ "text": "..." }` document
per line; TXT uses blank-line-separated documents. At least 20 distinct documents
are required. They are deduplicated and assigned a seeded 80/10/10 split before
packing. `--resources` selects a separate asset folder. Nonempty folders are
protected: reuse complete assets or choose a fresh folder when changing settings.

### 2. Train one condition: `2_train.py`

```powershell
python 2_train.py --method joint --weight 1.0 --beta 0.1 --seed 42 --steps 1000
python 2_train.py --method pcgrad --weight 1.0 --beta 0.1 --seed 42 --steps 1000
```

`--method` accepts `token_baseline`, `concept_token_only`, `joint`, and `pcgrad`.
`--weight` (alias `--lambda`) scales concept prediction loss; `--beta` scales
quantizer loss. Controls use zero for both. Match initialization, batch order,
learning rate, schedule, batch size, sequence length, and seed when comparing.

Each run saves `config.json`, `history.jsonl`, `validation.jsonl`, and
`summary.json` under `results/<method>_lambda<weight>_beta<beta>_step<steps>_seed<seed>/`.
`--output` is an exact run directory. `--test` evaluates the held-out split once
at the final step. `--save-model` saves final and best-validation model weights;
checkpoints can use substantial disk space and are optional.

Before a large sweep, run a two-step smoke test (not a performance conclusion):

```powershell
python 2_train.py --method pcgrad --weight 1 --beta 0.1 --steps 2 --eval-every 2 --diagnostics-every 1 --batch-size 1 --eval-batch-size 1 --device cuda
```

Completed runs are reused only when configurations, implementation hashes, and
asset hashes match. Nonempty incomplete or incompatible folders raise an error.
There is no mid-run resume: model checkpoints do not store optimizer/RNG state.
Use a new output root when changing settings, including test/checkpoint flags.

### 3. Run the full weight sweep: `3_run_experiments.py`

```powershell
python 3_run_experiments.py --weights 0.3 1.0 3.0 --betas 0.1 --seeds 42 43 44 --steps 1000 --batch-size 2 --eval-batch-size 2 --device cuda
```

Every weight/beta pair runs joint and PCGrad with every seed. Controls run once
per seed, because their updates do not use lambda or beta. The example runs
**24 conditions**: `(2 controls + 2 methods * 3 weights * 1 beta) * 3 seeds`.
Duplicate supplied weights, betas, and seeds run once. Full multitask weights must
be finite and positive. `--betas 0.01 0.1 1.0` expands the rectangular grid.

`--output` sets the experiment root (unlike step 2's exact run directory).
`main_summary.csv` is updated after each completed condition. Training options
such as `--resources`, `--steps`, `--save-model`, and `--test` pass through to step 2.
Select settings on validation first; use a separate output root with `--test`
for final reporting. With `--save-model`, step 5 can inspect saved checkpoints.

The Colab notebook defaults to 1,000 steps, batch size 2, three weights, one beta,
and three seeds. Its smoke run is separate from main results. A reporting switch
enables a fresh matched comparison with test evaluation after validation selection;
it is disabled by default. Edit the settings cell for a 6,000-step budget.

### 4. Analyze results: `4_analyze.py`

```powershell
python 4_analyze.py --results results --output analysis
```

This reads completed run folders and rebuilds `main_summary.csv`. It writes:

- `summary_convergence.png`: validation token loss and concept-error curves.
- `summary_tradeoff.png`: validation token/concept diagnostics with sample-SD bars.
- `summary_runtime.png`: recorded seconds per step.
- `analysis_summary.json`: per-setting means, sample SDs, and gradient diagnostics.
- `paired_differences.json`: matched-seed PCGrad-minus-joint token metrics.
- `report.md`: readable metric tables and paired differences.

Different training settings and budgets are separated by settings signatures;
duplicate seed observations are rejected. `--steps 6000` filters one budget.
Analysis needs no GPU or model download. In Colab, the same explicit results and
analysis paths are used for generating figures and displaying them.

### 5. Inspect a checkpoint: `5_predict.py`

Use `--save-model` in step 2 or 3, then provide the final or best-validation file:

```powershell
python 5_predict.py --checkpoint results/joint_lambda1.0_beta0.1_step1000_seed42/checkpoints/final.pt --text "The scientist entered the laboratory" --max-new-tokens 20 --device cuda
```

The script uses the local tokenizer and prints next-token probabilities, token
loss, concept diagnostics, and optional greedy continuation. `--resources` must
match the assets used for training. Concept codes are latent indices, not readable
semantic labels. Generation recomputes the sequence each step (no KV cache) and
must fit positional capacity. Prediction creates no files.

## What is measured

This is an adaptation of Liu et al., *Next Concept Prediction in Discrete Latent
Space Leads to Stronger Language Models* (https://arxiv.org/abs/2602.08984).
Official reference: https://github.com/LUMIA-Group/ConceptLM.
PCGrad reference: https://arxiv.org/abs/2001.06782.

Pretrained GPT-2 Small supplies embeddings and 12 token layers. After the first
layer, adjacent groups of four states are mean pooled. A new two-layer causal
Transformer predicts a soft combination of codebook entries per segment. Twelve
segments each have 64 entries. Predicted concepts are added to token states before
the remaining token layers. The token output head is tied to input embeddings.

At token position 3, the first complete block (positions 0..3) can guide the
prediction of token 4. Token positions 0..2 receive no concept guidance. All
concept attention is causal. Future representations are supervision only.

The objective is `token_CE + lambda * concept_MSE + beta * VQ_MSE`.
Concept MSE averages features and valid next-block pairs; token CE averages valid
predicted tokens. VQ MSE averages features and valid complete blocks. Thus numeric
weights depend on these explicitly defined reductions.

Important adaptation choices: concept targets are detached; task predictions use
detached codebook embeddings. Only VQ loss trains the quantizer, using nearest
entries against detached encoder states. The codebook has a two-layer ReLU MLP.
There is **no encoder commitment penalty**, so no internal commitment coefficient
is confused with the outer beta. These choices isolate quantizer updates but are
not a claim of exact reproduction of the paper's gradient routing or results.
Token prediction also trains the concept predictor. Changing concept loss is not
equivalent to adding an independent private head as in a sentiment experiment.

### Conditions

| Method | Concept guidance | Auxiliary losses |
|---|---|---|
| `token_baseline` | Disabled | None |
| `concept_token_only` | Enabled | None; frozen randomly initialized codebooks |
| `joint` | Enabled | Token + weighted concept + weighted VQ |
| `pcgrad` | Enabled | Same losses; token/concept gradient projection |

PCGrad projects both raw task gradients against their original peer when their
dot product is negative, globally over the intersection of their parameter
supports. It sums the projected token gradient and lambda times the projected
concept gradient. Private gradients are preserved. Beta times the VQ gradient is
added normally. Codebook gradients are never included in task projection.
Gradient clipping happens after combination. AdamW applies to trainable parameters.
Baselines freeze inactive modules. The architecture-only control's codebooks stay
fixed so it contains no hidden auxiliary objective; interpret that control with
this limitation in mind.

Primary metric: held-out token NLL/perplexity. Concept MSE is a diagnostic with
moving targets, not a directly comparable semantic quality score across models.
Monitor representation variance, codebook entropy/usage, gradient norm imbalance,
cosines, and projection size to detect collapse or negligible gradient surgery.
Codebook usage is aggregated over the entire evaluated split. Representation
variance is explicitly a mean of batch-level variances, not whole-split variance.
Analysis separates different settings and steps using a settings signature, rejects
duplicate seed observations, and reports paired PCGrad-minus-joint token metrics.
One-seed sample SD is unavailable rather than zero. No significance is claimed.

Training uses float32 for transparent gradient arithmetic; lower batch size if
memory is limited. PCGrad stores separate task gradients and takes more memory
and time. Joint computes separate task gradients only at diagnostic steps;
PCGrad does so every step. Runtime includes validation, logging, and optional
checkpoint I/O, excludes final test and model/data loading, and requires matched
hardware/cadence for interpretation. CUDA peak memory includes final validation
and model/optimizer allocations. No fixed runtime or GPU-memory guarantee is made.
No full-size training or download occurs when running the synthetic tests.


## Folder structure

The map shows source files and generated output locations. Paths are relative to
`another_experiment/`; prepared data, results, and analysis are gitignored.

```text
├── 1_prepare_assets.py                 # Download independent model and data
├── 2_train.py                          # Train one condition
├── 3_run_experiments.py                # Run all lambda/beta/seed conditions
├── 4_analyze.py                        # Aggregate and plot results
├── 5_predict.py                        # Inspect saved weights
├── ESTR3114_NCP_Colab.ipynb             # Google Colab workflow
├── src/                               # Independent implementation
│   ├── __init__.py
│   ├── data.py                        # Documents, packing, masks, hashes
│   ├── model.py                       # Token encoder, concepts, decoder
│   ├── quantizer.py                   # Learned product codebooks
│   ├── losses.py                      # Masked reductions
│   ├── optimization.py                # Joint/PCGrad gradient routing
│   ├── evaluation.py                  # Count-weighted metrics
│   └── results.py                     # Naming, JSON I/O, run indexing
├── tests/                             # Synthetic offline checks
│   ├── conftest.py
│   └── test_experiment.py
├── configs/default.json               # Reference defaults, not auto-loaded
├── resources/                         # Generated independent assets
│   ├── initial_model.pt
│   ├── manifest.json
│   ├── tokenizer/
│   ├── cache/
│   └── data/                          # Document JSONL and block NPZ per split
├── results/                           # Generated runs and main_summary.csv
│   └── <method>_lambda<w>_beta<b>_step<n>_seed<s>/
│       ├── config.json
│       ├── history.jsonl
│       ├── validation.jsonl
│       ├── summary.json
│       └── checkpoints/               # Optional best_validation.pt, final.pt
├── analysis/                          # Generated figures, JSON, report.md
├── README.md
└── requirements.txt
```

The root scripts are numbered by workflow: **1** prepare assets, **2** train one
condition, **3** run the full sweep, **4** analyze results, **5** inspect a saved
checkpoint. Step 3 calls step 2 internally; step 5 needs a saved checkpoint.

Run local checks from this folder with `python -m pytest tests -q`. They use tiny
models and synthetic data without model downloads. Checks cover causality, padding,
gradient routing, PCGrad mathematics, checkpoint reload, safe reuse, and analysis.
Pretrained-weight conversion parity uses a tiny random HF model when Transformers
is available. The Colab notebook runs these checks before preparing assets.

## Root Python scripts

Install `requirements.txt` first. Paths below use this experiment's defaults.

- **`1_prepare_assets.py`**
  - **Purpose:** Download and prepare GPT-2, tokenizer, and text splits.
  - **Dependencies:** Internet, Transformers, Hugging Face Hub, datasets, PyTorch,
    NumPy; `src/data.py`, `src/model.py`, and related modules. No parent assets.
  - **Output:** `resources/initial_model.pt`, `tokenizer/`, `cache/`, `data/`,
    and `manifest.json`.
- **`2_train.py`**
  - **Purpose:** Train/evaluate one baseline, joint, or PCGrad condition.
  - **Dependencies:** Step 1 assets; model, data, optimization, evaluation, and
    result modules under `src/`; PyTorch and NumPy.
  - **Output:** One run folder with settings, histories, summary, and optional
    `checkpoints/best_validation.pt` and `checkpoints/final.pt`.
- **`3_run_experiments.py`**
  - **Purpose:** Run joint/PCGrad across all lambda/beta pairs and matched seeds,
    with each control once per seed.
  - **Dependencies:** Calls `2_train.py` and inherits its assets/dependencies.
  - **Output:** Run folders plus `main_summary.csv` under the selected root.
- **`4_analyze.py`**
  - **Purpose:** Compare metrics, variability, paired differences, and runtime.
  - **Dependencies:** Completed runs; `src/results.py`, Matplotlib; no GPU.
  - **Output:** Updated CSV, three PNG figures, two JSON summaries, and `report.md`.
- **`5_predict.py`**
  - **Purpose:** Inspect predictions and optionally generate a short continuation.
  - **Dependencies:** Saved checkpoint from step 2/3, matching local tokenizer,
    model modules, PyTorch, and Transformers.
  - **Output:** JSON printed to the terminal; no files.
