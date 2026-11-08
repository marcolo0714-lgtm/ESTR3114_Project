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
python prepare_assets.py
```

For a GPU, install a CUDA-enabled PyTorch build using the instructions for your
driver at https://pytorch.org/get-started/locally/ before installing the remaining
requirements. Check `python -c "import torch; print(torch.cuda.is_available())"`.

`prepare_assets.py` requires internet **once**. It writes the GPT-2 checkpoint,
tokenizer, TweetEval's original train/validation/test splits, download cache, and
revision metadata under `resources/`. After that, training reads local files
only. The dataset and model files are too large to include in a source checkout;
run the preparation step before training. Sources are listed in
`resources/REFERENCES.md`. Do not tune on the test split.

## Run one condition

```powershell
python train.py --method joint --weight 0.3 --seed 42 --steps 1000
python train.py --method pcgrad --weight 0.3 --seed 42 --steps 1000
```

`--method` accepts `token`, `sentiment`, `joint`, and `pcgrad`. Use the same
learning rate, schedule, steps, batch size, sequence length, and seed for a
matched comparison. `--output` sets a separate run folder. `--test` evaluates
the held-out split at the end. `--save-model` saves final weights and tokenizer
and can consume substantial disk space; metrics are always saved. A smaller
development run, once the resources are prepared:

```powershell
python train.py --method pcgrad --steps 2 --eval-every 2 `
  --diagnostics-every 1 --batch-size 1 --train-limit 8 `
  --validation-limit 8 --max-length 32 --device cpu
```

To inspect a trained checkpoint, use `--save-model` while training and then:

```powershell
python predict.py --checkpoint results/pcgrad_w0.3_seed42/final_model.pt `
  --text "I love this!"
```

## Full experiment

```powershell
python run_experiments.py --weights 0.1 0.3 1.0 --seeds 42 43 44 `
  --pilot-steps 100 --main-steps 1000
python analyze.py --results results/comparison
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

## Code map and checks

| File | Purpose |
| --- | --- |
| `prepare_assets.py` | Download and record local model/data resources |
| `src/data.py` | Tokenization, EOS, and right padding |
| `src/model.py` | Two heads and masked task losses |
| `src/gradients.py` | Gradient extraction, PCGrad, and diagnostics |
| `src/evaluation.py` | Aggregate validation/test metrics |
| `train.py` | Reproducible single-condition training |
| `run_experiments.py` | Validation pilot and matched three-seed comparison |
| `analyze.py` | Figures and aggregate summary |
| `predict.py` | Inspect a saved checkpoint on one text |

Run local checks with `python -m pytest -q`. These tests use tiny tensors and
do not require model downloads. To verify full integration, run the two-step
development command above after preparing the resources.
