# Run the live presentation demo

The QR code points to the permanent Vercel site. Vercel shows the experiment
reports and a first-visit popup linking to the temporary demo. Its two model
pages are guide pages, with no prediction controls. The local/cloudflared site
keeps the working forms and does not show the popup or redirect back to itself.

## 1. Start the website on your computer

Open PowerShell in the **project root** and keep this terminal open:

```powershell
$env:SITE_MODE = "local"
.\site\.venv\Scripts\python.exe site/app.py --host 127.0.0.1 --port 8000 --threads 4
```

Open `http://127.0.0.1:8000/text` and try a prediction. Image models are at
`http://127.0.0.1:8000/images`. This uses the already installed local environment
and saved checkpoints; there is no training.

## 2. Start the tunnel

In a **second PowerShell terminal**, run:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Keep both terminals running. Copy the generated `https://...trycloudflare.com`
base address. Open its `/text` and `/images` pages and check a prediction before
publishing the link. Keep the computer awake and connected during the session.

## 3. Edit ONE file and push

Edit **`site/demo_config.json`**. Replace the empty string with the actual
generated URL, preserving the double quotes:

```json
{
  "url": "https://YOUR-GENERATED-ADDRESS.trycloudflare.com"
}
```

Use only the base address, without `/text` or `/images`. Every popup, banner
and model-page button uses this file. No Vercel environment-variable change or
report rebuild is required. The URL is a public link, not a secret.

In a third terminal at the **project root**:

```powershell
git add site
git commit -m "Publish live presentation demo link"
git push
```

Wait for Vercel's Git-triggered deployment to finish. Open the permanent site
in a private/incognito window: verify the popup and both demo buttons. The QR
code and permanent address do not change.

The popup appears once per browser tab/session after dismissal. A new tunnel
URL triggers a fresh announcement. Browsers blocking session storage may show
the popup again. The demo link also stays visible on the permanent site.

## If you restart the tunnel or finish the demo

A new quick tunnel usually gets a different address. Update the same JSON
file and push again. Do not stop/restart a working tunnel just to update Git.

After the session, stop both terminals with Ctrl+C. Set `"url": ""` in the
same file and push to remove the expired link. Visitors will see a clear
message that no live link is published; reports remain accessible.

The URL only links browsers to the live website. It does not configure the API
proxy. The optional `INFERENCE_API_URL` proxy remains available for other uses,
but the permanent site's model pages deliberately guide presentation visitors
to the external demo.
