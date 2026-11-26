# The API on Vercel

The backend runs on Vercel as a **container image**: `Dockerfile.vercel` at the
repository root, built by Vercel, pushed to the Vercel Container Registry and
served by Vercel Functions. No tunnel, no laptop.

It is a second Vercel project on this repository:

| Project | Root Directory | Framework preset | What it serves |
| --- | --- | --- | --- |
| `absa-sentiment` | `frontend/` | Vite | the UI |
| `absa-api` | `.` (repository root) | **Container** | the API |

Production URL: `https://absa-api-pravesh-aryas-projects.vercel.app`

## Why a container and not a plain Python function

A function is the better fit on paper — no image to build, faster cold start —
and the code for it is still here and still works: `server.py` exports the
ASGI app from a filename Vercel's Python runtime looks for, and
`requirements.txt` is the dependency set. It got as far as a successful build
and was refused at the last step.

The bundle is ~1.5GB (`VERCEL_ANALYZE_BUILD_OUTPUT=1` prints the breakdown):

| | |
| --- | --- |
| torch's shared libraries (`torch/lib`) | 454MB |
| the artifact | 430MB |
| transformers, pandas, numpy, torch's own tests and headers | ~480MB |
| the rest | ~210MB |
| **total** | **1575MB** |

Python's standard limit is 500MB and the large-functions beta is documented as
raising it to 5GB. The build *does* opt in:

```
Function "server.py" exceeds the standard size limit; enabling large functions (beta).
```

and the deployment is then refused anyway:

```
Error: [NOW_SANDBOX_WORKER_MAX_LAMBDA_SIZE] 1 function exceeded the uncompressed maximum size of 250 MB.
```

250MB, not the 500MB the docs give for Python, and the 5GB ceiling the beta
promises never applied — with `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` set on
Production and Preview and Fluid Compute on. The docs do not explain the
mismatch; the likeliest reading is that the beta is not honoured on a Hobby
account. If that changes, the function path needs only a `vercel.json` with
`buildCommand: "python deploy/vercel/fetch_artifact.py"` and the framework
preset moved back to FastAPI.

A container is not subject to that limit. The registry's ceiling is 15GB.

## One-time setup

1. **Create the project.** Import the repository, name it `absa-api`, leave
   Root Directory at the repository root.
2. **Set the framework preset to Container.** This one is load-bearing:

   ```sh
   vercel project update --framework container --yes
   ```

   Without it there are two wrong outcomes rather than one. With the FastAPI
   preset (which is what gets auto-detected from `requirements.txt`) Vercel
   builds the oversized function and the deployment fails. With no preset and
   no build command it deploys the repository as **static files** — the build
   log shows nothing but "Downloading 126 deployment files", every route
   answers 404, and nothing says a Dockerfile was ignored.
3. **Set the environment variables** (all environments):

   | Variable | Value | Why |
   | --- | --- | --- |
   | `ALLOW_DEGRADED_START` | `false` | the weights are in the image, so a load failure means the image is broken. Crash and say so in the logs rather than serve a `degraded` /health that reads like a transient fault |

   Nothing else is needed. `ARTIFACT_DIR`, `DEVICE` and `ALLOW_DEGRADED_START`
   are also set in the Dockerfile; `CORS_ORIGINS` already defaults to a list
   containing `https://absa-sentiment.vercel.app`. `MODEL_REPO` and
   `ARTIFACT_DIR` are both read by the build script, so a rebuild can serve a
   different artifact without a code change.
4. **Turn Vercel Authentication down to previews only.** Settings →
   Deployment Protection → Vercel Authentication → *Only Preview
   Deployments*. It defaults to protecting production too, which makes the API
   answer `302` to a Vercel login page — a browser on the deployed frontend
   cannot get past that, and `curl` sees "Protected by Vercel Authentication".
   Preview deployments can stay protected; `vercel curl <url>` reaches them.
5. **Point the frontend at it.** In `absa-sentiment`, set
   `VITE_API_URL=https://absa-api-pravesh-aryas-projects.vercel.app` and
   redeploy. Vite substitutes it at build time, so that redeploy is required;
   after it, `?api=` is a debugging affordance rather than how the site is
   wired.

## What the image does

Sixteen steps, and the two that matter are the install and the fetch.
`requirements.txt` pins `torch==2.14.1+cpu` against the PyTorch CPU index, so
the CUDA build (~2.5GB of `nvidia-*` wheels that never load) cannot be
resolved by accident, and `transformers==5.19.0` matches what the artifact
records in its own `config.json` — a drift there rebuilds the backbone
slightly differently and the saved `state_dict` stops matching.

`deploy/vercel/fetch_artifact.py` then pulls the 450MB artifact from the Hub
*into the image*. Fetching it at boot instead would add the download to every
cold start, and the container has no persistent disk to cache it on.

The last line is `uvicorn server:app --host 0.0.0.0 --port ${PORT:-80}` in
shell form, because Vercel routes to port 80 unless the project's `PORT` says
otherwise and that has to be read at boot, not baked in.

## Measured

- **Image build**: ~2 minutes, including the Hub pull (450MB in ~2s on
  Vercel's network).
- **Cold start**: ~12s to a `200` from `/api/v1/health`. The function scales
  to zero after 5 minutes idle in production (30s in preview), so the first
  request after a quiet spell pays for loading 450MB of weights. Hit
  `/api/v1/health` before a demo.
- **Memory**: peak resident set while loading is ~1.3GB against the 2GB a
  Hobby function gets. Both the freshly built model and the state dict exist
  at once inside `load_state_dict`, so the peak is at startup, not per
  request. That is the number to watch if the artifact grows.

## The one thing that only showed up in a deployment

The first container booted and died on `ModuleNotFoundError: No module named
'sklearn'`. Importing `training.src.artifact` executes
`training/src/__init__.py`, which re-exports the whole training package —
including `training.src.evaluation`, which imports `sklearn.metrics` at module
scope. It is in the local venv through `training/requirements.txt`, so nothing
local can catch it. `scikit-learn` is now in both deployment requirement
files; `deploy/hf-space/requirements.txt` had the same hole and would have
failed the same way on its first boot.

## Keeping it in step with the code

A push to `main` redeploys. `vercel link` connected the GitHub repository when
it created the project, so this behaves like `absa-sentiment`: `main` builds
production, any other branch builds a preview. `vercel deploy --prod` from the
repository root does the same thing without a commit.

The build re-pulls the artifact from the Hub, so a deployment carries the
current state of both the code and the weights. To serve different weights,
change `MODEL_REPO` in the project's environment variables and redeploy.

One consequence worth knowing: because the framework preset is Container and
the whole repository is the build context, a commit that moves or renames
`Dockerfile.vercel` deploys the repository as **static files** instead of
failing. Every route then answers 404, and the build log says nothing beyond
"Downloading N deployment files".

`deploy/hf-space/` is the same container shaped for a Hugging Face Space. It
is kept separate because it clones the repository from GitHub (a Space's own
repo holds three files), while here the build context *is* the repository.
