# Deploy the website on Vercel with a separate model server

The website now exports a top-level `app = create_app()` from `app.py`.
`pyproject.toml` explicitly identifies `app:app` as the Vercel entry point.
The same code supports a lightweight frontend and a full inference server.

## Why there are two services

The checkpoints total about 2.63 GB before PyTorch and other dependencies.
Vercel's standard Python function bundle limit is 500 MB. The Vercel deployment
therefore contains pages, report data, figures and an API proxy, but no model
weights or machine-learning packages. The separate server runs the saved models.
Local use still runs everything together and works offline after installation.

Request path: browser -> Vercel `/api/...` -> model server `/api/...` -> Vercel
-> browser. The browser keeps same-origin API calls, so no browser CORS setup or
public exposure of a server token is required. Submitted text/images are sent
to the configured model server for prediction; neither application saves them.

## 1. Start the model server

Use the entire `site` folder, including `models/text/` and `models/images/`.
Checkpoint files are ignored by Git: a source-only clone does not contain them.
Transfer these folders explicitly to the server, or build the Docker image from
the local folder that already has the weights. Retain the registry JSON files.

Docker commands, run from `site`:

```sh
docker build -t estr3114-model-server .
docker run --rm -p 8000:8000 -e MODEL_SERVER_TOKEN=YOUR_PRIVATE_TOKEN estr3114-model-server
```

The Dockerfile installs CPU PyTorch and the full inference dependencies. It sets
`SITE_MODE=local`, starts one process, and respects a provider-supplied `PORT`
(default 8000). Persistent model files must be present in the image or mounted
at `/app/models`. Allow enough RAM for the four GPT-2 checkpoints and PyTorch;
8 GB is a practical starting allocation for keeping all four text models cached.
No GPU or retraining is required. The first request loads the selected models.

Without Docker:

```sh
python -m pip install -r requirements-local.txt
python app.py --host 0.0.0.0 --port 8000 --threads 4
```

Set `SITE_MODE=local` on the model host. Optionally set `MODEL_SERVER_TOKEN` to a
private token and use the same value on Vercel as described below. Configure the
host/provider's HTTPS URL, for example `https://your-model-server.example`.
That is a placeholder: use your actual running server's base URL, without `/api`.
A local `127.0.0.1` URL is not reachable from Vercel.

Check `/api/health`: it should return `mode: "local"`. Checkpoint availability
can be checked at `/api/models` and `/api/image-models` (include the Bearer token
when configured). This repository does not provision or pay for a model host.

## 2. Configure Vercel

- **Root Directory:** `site` when deploying this project repository. If uploading
  only the contents of `site`, use that uploaded folder as the project root.
- **Framework Preset:** FastAPI. Leave custom build/install/output overrides
  unset unless your existing project requires them.
- **Python:** 3.12, pinned by `.python-version`.
- **Environment variable:** `SITE_MODE=frontend` (Vercel also selects this
  automatically through its `VERCEL` environment variable).
- **Environment variable:** `INFERENCE_API_URL=https://YOUR-ACTUAL-MODEL-SERVER`.
- **Optional secret:** `INFERENCE_API_TOKEN`, matching `MODEL_SERVER_TOKEN` on
  the model server. Store secrets in the provider settings, not source files.

Redeploy after changing environment variables. `vercel.json` excludes model
files, inference code and development files from the function. `.vercelignore`
also excludes weights and local environments from CLI uploads. The default
`requirements.txt` and `pyproject.toml` install only lightweight web/proxy
dependencies. Full local/server requirements are in `requirements-local.txt`.

If no inference URL is configured, all four pages still render; prediction
controls explain that a model server must be configured, and prediction API
requests return a clear 503. A website deployment alone does not deploy models.

The proxy permits images up to **4 MiB**, safely below Vercel's 4.5 MB request
limit. Local inference permits **8 MiB**. The page, clipboard input and file
picker use the applicable limit. Smaller CIFAR-sized images need far less.

## 3. Verify the deployed pair

1. Visit `/api/health` on the website: expect `mode: "frontend"` and
   `inference_configured: true`.
2. Visit all four pages and open the report figures.
3. On each prediction page, verify models are available **on the model server**.
4. Predict a short sentence, then upload/paste an image and select its superclass.
5. If the site reports 401, check that the two server-token values match. A 502
   usually means the backend URL/service is unreachable or is returning HTML
   instead of the API. A 503 from the backend can mean checkpoint files are absent.

No live Vercel deployment or model-host provisioning is performed by these files.
The URL and hosting account must refer to a real model server before remote
prediction works. Both services should use the same code and model registries.

## References

- https://vercel.com/docs/frameworks/backend/fastapi
- https://vercel.com/docs/functions/runtimes/python
- https://vercel.com/docs/functions/limitations
