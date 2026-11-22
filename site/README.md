# ESTR3114 model comparison website

A standalone, local Python website with four pages:

1. **Try text models**: actual next-token and sentiment predictions from the four
   seed-44, 6,000-step checkpoints. Joint and PCGrad use sentiment weight 3.
2. **Try image models**: an empty space reserved for experiment 2.
3. **Experiment 1**: settings, methods, metrics, five result tables, graphs,
   and interpretation of the token/sentiment experiment.
4. **Experiment 2**: an empty space reserved for the unfinished experiment.

No image inference or experiment-2 integration has been implemented.

## Run on this computer

From the project root in PowerShell:

```powershell
cd site
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:8000**. Stop with Ctrl+C. The first prediction loads
the selected checkpoints; subsequent predictions reuse them. Models run
sequentially to keep CPU and memory usage predictable. Selecting fewer methods
reduces loading time and memory. All four float32 checkpoints occupy about
2.6 GB on disk; allow additional RAM for the application and PyTorch.

You can also double-click `start_site.cmd` on Windows to start the server using
the prepared environment, then open the address above in your browser.

Optional arguments:

```powershell
python app.py --port 8001 --threads 4
python app.py --device cuda
```

CPU is the default. CUDA requires a compatible GPU and PyTorch installation.
Use one server process; multiple workers would duplicate model memory.

## Move to another computer

Copy the **entire `site` folder including `models/`**. Do not copy the virtual
environment; recreate it with Python 3.10 or newer:

```powershell
cd site
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

On macOS/Linux use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.
Internet is needed to install dependencies, but model loading, prediction,
templates, JavaScript, CSS, report data, and figures work offline. There are no
CDNs, APIs, training-dataset reads, parent-project imports, or external symlinks.
The bundled development environment may reuse installed system packages; it is
not a portable environment and is not needed on the destination computer.

This is a local application, not a publicly hosted URL. Sharing the local URL
does not make it accessible from other computers.

## Inputs and outputs

- Text is tokenized with the saved GPT-2 tokenizer. Up to 95 text tokens are
  retained and EOS is appended, matching the 96-token training limit.
- One forward pass per model supplies next-token probabilities **at the final
  text position** and sentiment **at the appended EOS position**.
- The next-token panel lists the top five probabilities from the full
  vocabulary. Leading spaces are preserved in quoted token strings.
- Token-only's sentiment output is withheld because its head is untrained.
  Sentiment-only still has a token head, but no token-loss supervision.
- The interface reports input truncation and separate loading/inference times.
  Sentences are neither persisted nor logged. Image uploads are not enabled.
- Example predictions do not establish superiority. Report tables summarize
  three seeds, whereas the demonstration uses one seed. Demo rerun summaries
  are shown separately from the original evaluations.

## Folder contents

```text
site/
├── app.py                     # Local FastAPI server
├── requirements.txt
├── requirements-dev.txt
├── inference/                 # Local GPT-2 architecture and inference
├── templates/                 # Four pages and shared layout
├── static/                    # Local CSS, JavaScript, and figures
├── models/
│   ├── registry.json          # Checkpoint metadata and SHA-256 hashes
│   └── text/
│       ├── backbone_config.json
│       ├── token/             # Weights, tokenizer, config, test summary
│       ├── sentiment/
│       ├── joint/
│       └── pcgrad/
├── experiment_data/           # Copied metrics and generated report data
├── tools/                     # Optional model import and report rebuild
└── tests/
```

Weights and local environments are ignored by Git. Copy `models/` explicitly
when transferring the site; cloning source alone will not include checkpoints.
The model registry contains hashes so transferred weights can be verified.

`tools/import_models.py --archives <four ZIP paths> --backbone <JSON path>`
imports the trusted checkpoint archives. Only required members are copied.
This optional setup tool takes explicit input paths; the running application
uses only files inside this folder. Existing checkpoints are already imported.

`python tools/build_report.py` rebuilds the report JSON from the copies of
metrics and demo summaries already inside `site`. There is no automatic link
to changing experiment outputs.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp=.test_tmp
```

Tests cover navigation, all five report conditions, local figures, invalid
inputs, truncation, private checkpoint paths, and correct next-token/EOS
positions using an actual tiny GPT-2 model. Real checkpoint loading and browser
prediction are also checked during implementation.
