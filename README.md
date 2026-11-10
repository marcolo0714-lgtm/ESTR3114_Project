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



## Run one condition

```powershell
python 2_train.py --method joint --weight 0.3 --seed 42 --steps 1000
python 2_train.py --method pcgrad --weight 0.3 --seed 42 --steps 1000
```

`--method` accepts `token`, `sentiment`, `joint`, and `pcgrad`. Use the same
learning rate, schedule, steps, batch size, sequence length, and seed for a
matched comparison. `--output` sets a separate run folder. `--test` evaluates
the held-out split at the end. `--save-model` saves final weights and tokenizer
and can consume substantial disk space; metrics are always saved. A smaller
development run, once the resources are prepared:

```powershell
python 2_train.py --method pcgrad --steps 2 --eval-every 2 `
  --diagnostics-every 1 --batch-size 1 --train-limit 8 `
  --validation-limit 8 --max-length 32 --device cpu
```

To inspect a trained checkpoint, use `--save-model` while training and then:

```powershell
python 5_predict.py --checkpoint results/pcgrad_w0.3_seed42/final_model.pt `
  --text "I love this!"
```

## Full experiment

```powershell
python 3_run_experiments.py --weights 0.1 0.3 1.0 --seeds 42 43 44 `
  --pilot-steps 100 --main-steps 1000
python 4_analyze.py --results results/comparison
```

The pilot compares weighted joint AdamW candidates to a token-only control on
the **same subset and seed**. It selects the best macro-F1 among candidates
whose token loss is at most 5% above the control's, falling back to the lowest
token loss if none qualifies. The selected weight is then held fixed across
three seeds and all four main conditions. The main runs evaluate the test split
only after training. `main_summary.csv`, each run's `config.json`, `history.jsonl`,
and `summary.json`, plus four figures and `analysis_summary.json` provide the
project's results. Runtime varies substantially by device.

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
├── 3_run_experiments.py                # Run pilot and full comparison
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
│   ├── test_selection.py               # Pilot weight selection
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
├── results/                           # Generated experiment outputs
│   ├── <method>_w<weight>_seed<seed>/   # Individual step 2 run
│   │   ├── config.json                # Run settings and metadata
│   │   ├── history.jsonl              # Training and validation history
│   │   ├── summary.json               # Final metrics
│   │   ├── final_model.pt             # Only with --save-model
│   │   └── tokenizer/                 # Only with --save-model
│   └── comparison/                    # Full experiment and analysis
│       ├── pilot/                     # Pilot runs, each with settings/metrics
│       ├── main/                      # Method/seed runs with settings/metrics
│       ├── pilot_selection.json       # Selected weight and pilot results
│       ├── main_summary.csv           # All main-run results
│       ├── analysis_summary.json       # Aggregate statistics from step 4
│       └── figures/                   # Plots generated by step 4
│           ├── convergence.png
│           ├── tradeoff.png
│           ├── runtime.png
│           └── layer_conflicts.png     # Requires gradient diagnostics
├── estr3114_proposal.pdf               # Original project proposal
├── estr3114_Project_Resources.pdf      # Project resource document
├── README.md                          # Setup and experiment documentation
└── requirements.txt                   # Python dependencies
```

The root scripts are numbered by workflow: **1** prepare assets, **2** train one
condition, **3** run the full experiment, **4** analyze results, and **5** inspect
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
  - **Output:** `results/<method>_w<weight>_seed<seed>/` containing
    `config.json`, `history.jsonl`, and `summary.json`. `--test` adds test
    metrics; `--save-model` also writes `final_model.pt` and `tokenizer/`.
- **`3_run_experiments.py`**
  - **Purpose:** Select the sentiment weight using a validation pilot, then
    compare all four methods across seeds.
  - **Dependencies:** Calls `2_train.py` internally and inherits its
    dependencies and prepared assets. No separate step 2 run is needed.
  - **Output:** `results/comparison/pilot/` and `main/` with per-run
    configuration, history, and summary files; `pilot_selection.json` and
    `main_summary.csv`. Defaults: 4 pilot runs and 12 main runs. Does not save
    model checkpoints.
- **`4_analyze.py`**
  - **Purpose:** Plot results and aggregate performance, gradient conflicts,
    and runtime.
  - **Dependencies:** Step 3's `main/*/history.jsonl` and `main_summary.csv`;
    optionally `pilot_selection.json`; Matplotlib and NumPy.
  - **Output:** Under `results/comparison/`, `analysis_summary.json` and
    `figures/convergence.png`, `tradeoff.png`, and `runtime.png`, plus
    `layer_conflicts.png` when gradient diagnostics exist.
- **`5_predict.py`**
  - **Purpose:** Inspect one text's sentiment, sentiment probabilities, token
    loss, and perplexity.
  - **Dependencies:** `final_model.pt` from step 2 with `--save-model`;
    step 1's local model/tokenizer assets; `src/model.py`, PyTorch, and
    Transformers.
  - **Output:** Prints a JSON result to the terminal. Creates no files.

Steps 2 and 3 accept `--output` to change their destination; step 4 accepts
`--results` to choose the experiment folder to analyze.
