# ESTR3114 model comparison website

For Vercel hosting with a separate model server, see [DEPLOYMENT.md](DEPLOYMENT.md).
The module exports `app` for Vercel/Uvicorn; local execution still uses the CLI.

A standalone, local Python website with four pages:

1. **Try text models**: actual next-token and sentiment predictions from the four
   seed-44, 6,000-step checkpoints. Joint and PCGrad use sentiment weight 3.
2. **Try image models**: upload an image, supply its superclass, and compare joint
   and PCGrad predictions using best-validation or final CNN checkpoints.
3. **Experiment 1**: settings, methods, metrics, five result tables, graphs,
   and interpretation of the token/sentiment experiment.
4. **Experiment 2**: the completed 10,000-step, seed-42 CIFAR-100 comparison:
   settings, validation and held-out test metrics, per-task/class errors, confusion matrices,
   learning curves, gradient diagnostics, runtime and limitations.

For the presentation QR-code entrance and one-file cloudflared URL update, see
[DEMO.md](DEMO.md). Edit `demo_config.json` after starting your tunnel. Vercel
shows first-visit guidance and replaces prediction forms with links to the live
demo; local/cloudflared mode keeps the working forms.

Experiment 2 has one matched seed. Each best-validation checkpoint has also been
evaluated on the 10,000 held-out test images: joint 67.07%, PCGrad 68.80% accuracy.
These are conditional five-class scores with the superclass supplied. No estimate
of variation across training seeds is available.

Both experiment reports keep a persistent sidebar on desktop. On smaller
screens, a top-left **Show sections** button opens the navigation only when
needed. Selecting a section, pressing Escape, or clicking outside closes it.

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
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe app.py
```

On macOS/Linux use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.
Internet is needed to install dependencies, but model loading, prediction,
templates, JavaScript, CSS, report data, and figures work offline in local mode.
There are no CDNs, external API calls, training-dataset reads, parent-project
imports, or external symlinks in that mode. Vercel frontend mode forwards
prediction inputs to the configured model server instead.
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
  Sentences are neither persisted nor logged. Image uploads are handled by the
  separate image page described below.
- Example predictions do not establish superiority. Experiment 1 tables summarize
  three seeds, whereas its demonstration uses one seed. Demo rerun summaries
  are shown separately from the original evaluations.

## Image predictions and Experiment 2

Choose an image file or paste a copied image with Ctrl+V (Cmd+V on Mac) while
the image page is focused. Both input methods share the preview, size limits
and prediction flow. Clear removes the selected or pasted image. Clipboard
contents are read only from an explicit paste event; no clipboard permission
or background access is needed. Pasting an image URL alone does not fetch it.

Three clickable examples (aquarium fish, sunflower and pickup truck) load a
bundled image and automatically select its correct superclass. Select Predict
after choosing an example. These are fixed CIFAR-100 training images, selected
as the first retained training image of each class without consulting model
predictions; they illustrate the interface, not held-out performance.
`experiment_data/image_examples.json` records their labels, original indices
and source, and `static/examples/` holds the original 32 x 32 PNG images.

The image page requires a supplied superclass and chooses among its five fine
classes. It does not infer the superclass or perform unrestricted 100-class
recognition. Select Joint, PCGrad or both. Best-validation checkpoints are the
default: joint step 9,500 and PCGrad step 9,200. Final states are both at 10,000.
All four states come from the seed-42, 10,000-step training runs.

Images are decoded in memory (PNG/JPEG/WebP, up to 8 MB locally, 4 MB through
Vercel, and 20 million pixels),
camera orientation is honored, and RGB pixels are bilinearly resized to 32x32.
The training-only channel normalization is read from bundled model metadata.
No uploaded image or filename is saved. The API accepts raw image bytes, so
there is no multipart-upload dependency or temporary upload file.

`inference/image_model.py` contains the local CNN architecture;
`inference/images.py` loads and verifies checkpoints lazily, caches them, and
serializes image inference. `models/images/registry.json` records task/class
mapping, preprocessing and checkpoint hashes. Only weights and inference
metadata are bundled; optimizer and RNG states remain in the training outputs.
The four inference checkpoints total about 23 MB.

`experiment_data/images/` holds compact copies of settings, summaries, curves
and diagnostics. `experiment_data/image_results.json` supplies the report, and
`static/figures/cifar/` contains its local figures. These files have no runtime
dependency on `../new_experiment` or the CIFAR dataset. The report distinguishes
validation-selected states from final states, and labels validation and test
scores separately. `experiment_data/images/test_results.json` bundles test metrics,
confusion counts, checkpoint/evaluation hashes and local CPU evaluation timings;
the full per-image prediction files remain in the experiment results directory.

Optional explicit import/rebuild commands, run from `site`:

```powershell
python tools/import_image_models.py --results ../new_experiment/results --analysis ../new_experiment/analysis
python tools/import_image_test_results.py --experiment ../new_experiment
python tools/build_image_report.py
```

The importer reads the training folder only during preparation; the report
builder uses only the copies already bundled inside `site`. When moving the
site, include `models/images/` as well as `models/text/`; model weights are not
included in a source-only Git clone. The import scripts do not train models.

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
