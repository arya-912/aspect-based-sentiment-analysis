# Deploying

## What is live

| Piece | Where | State |
| --- | --- | --- |
| `frontend/` | Vercel, project `absa-sentiment` — <https://absa-sentiment.vercel.app> | deployed |
| `backend/` | Vercel, project `absa-api` — <https://absa-api-pravesh-aryas-projects.vercel.app> | deployed, see [deploy/vercel](vercel/README.md) |
| `training/artifacts/absa-joint-v1` | the Hub at <https://huggingface.co/arya912/absa-joint-v1>, pulled into the API's image at build time | served model, 450MB |

Both halves are hosted, so there is nothing to start before a demo. The one
thing to know is the cold start: the API scales to zero after five idle
minutes, and the instance that answers the next request has to load 450MB of
weights first -- about 12 seconds. Open
<https://absa-api-pravesh-aryas-projects.vercel.app/api/v1/health> a minute
before you present.

`./deploy/demo.sh` still works and is still the fallback: it boots the API
locally, opens a Cloudflare quick tunnel, and prints a
`https://absa-sentiment.vercel.app/?api=https://<random>.trycloudflare.com`
link that points the deployed UI at your machine. Reach for it if a
deployment is broken mid-demo, or to show a model that is not the deployed
one.

## How the backend is hosted

A container image on Vercel: `Dockerfile.vercel` at the repository root, built
by Vercel and served by Vercel Functions, with the artifact pulled from the
Hub into the image at build time. The setup, the environment variables and the
numbers behind it are in [deploy/vercel/README.md](vercel/README.md).

This used to say the backend could not be hosted. One of the two walls it
described has moved:

**Vercel.** The blocker was a 250MB unzipped function with no persistent disk,
which torch alone overran. That is still the limit a *function* gets here: the
bundle is ~1.5GB, and although the build opts into the large-functions beta
that is meant to allow 5GB, the deployment is refused at 250MB anyway. What
does work is a container, which goes through the Container Registry instead --
15GB -- and holds the same FastAPI app. A Hobby function gets 2GB of memory
and peak usage while loading is ~1.3GB, which is the figure to watch if the
artifact ever grows.

**Hugging Face Spaces.** Unchanged: a Space that runs compute still needs PRO,
and `create_repo` returns `402 Payment Required` without it. The container in
`deploy/hf-space/` is written and ready for the day that is worth $9/mo, or
for any other host that runs containers.

## Which model is served

`Settings.ARTIFACT_DIR` defaults to `training/artifacts/absa-joint-v1`, the
pooled laptop+restaurant model. It beats each specialist on that specialist's
own gold test set, so one backend covers both domains; the UI reads `corpus`
from `GET /model`, sees the `Joint_Train` stem, and says "Laptop and
restaurant reviews" with examples from both. See the README's
[Two domains, one model](../README.md#two-domains-one-model).

Note that `metrics.json` is scored on whatever test set the run used, so the
headline numbers are not comparable across artifacts: `absa-v1`'s are the 638
laptop rows, `absa-joint-v1`'s the 1,758 pooled rows. The per-domain
comparison in the README is the one to quote.

Serve a different artifact without editing anything:

```sh
ARTIFACT_DIR=training/artifacts/absa-rest-v1 uvicorn app.main:app
```

`absa-joint-v1` and `absa-v1` are both on the Hub. The deployed API pulls
`arya912/absa-joint-v1`, so a retrained artifact has to be pushed there before
a redeploy can serve it:

```sh
.venv/bin/hf upload arya912/absa-joint-v1 training/artifacts/absa-joint-v1 . --repo-type=model
```

To serve a different repo without touching the code, set `MODEL_REPO` (and
`ARTIFACT_DIR`, which the build script and the service read from the same
variable) in the `absa-api` project's environment variables.

## The `?api=` override

`VITE_API_URL` is substituted by Vite at **build** time, so a URL that
changes cannot come from it — and a quick tunnel's hostname is new on every
run. `frontend/src/services/api.ts` therefore resolves the API origin at
runtime, in this order:

1. `?api=<url>` in the query string, which is remembered in `localStorage`,
   so a reload or an in-app navigation keeps working once it drops off. An
   empty `?api=` forgets it.
2. `VITE_API_URL`, compiled in at build time.
3. `http://localhost:8000`.

Non-http(s) values are ignored rather than thrown, so a mistyped URL leaves
the previous default in place instead of breaking the page.

Step 2 is now the deployed API, so the live site needs no parameter at all.
The override earns its keep in the other direction: append
`?api=http://localhost:8000` and the deployed page talks to a backend on this
machine instead. Browsers allow that particular mixed-content case because
localhost counts as a trustworthy origin, and the backend's CORS list carries
the Vercel origin, so nothing else has to be configured. A tunnel is only for
reaching a local API from someone else's device.

The trade-off worth knowing: anyone can point the deployed page at a host of
their choosing with `?api=`. That is fine for a demo, and is the reason the
override is not wired to anything more sensitive than the API origin.

## CORS

`Settings.CORS_ORIGINS` defaults to the two dev-server origins plus
`https://absa-sentiment.vercel.app`, so the live site works with no
environment set. Override it for a different frontend; both forms parse:

```sh
CORS_ORIGINS=https://a.example,https://b.example
CORS_ORIGINS='["https://a.example"]'
```

The field is annotated `NoDecode` because pydantic-settings JSON-decodes a
tuple-typed setting *before* validation, which rejected the comma form
without the validator ever seeing it. That needs pydantic-settings >= 2.7.0.

A tunnel hostname does **not** need adding here. CORS is enforced on the
`Origin` header, which is the frontend's origin, not the API's.

## Redeploying

Pushing to `main` redeploys `absa-sentiment`: the GitHub repo is connected to
it and `main` is the production branch.

`absa-api` is **not** connected to the repo, so it only deploys when asked to.
Run `vercel git connect` once `main` carries `Dockerfile.vercel` and it will
behave like the other one; until then, and after any change to the image:

```sh
vercel deploy --prod
```

Mind which project the directory is linked to. `.vercel/project.json` holds
one link per directory, the repository root is `absa-api`'s Root Directory as
well as the place `absa-sentiment` is linked from, and `vercel deploy` sends
whichever of the two the link names. Run `vercel link` if you are not sure.

`.vercelignore` keeps `.venv`, `training/artifacts` and `training/data` out of
the upload — together ~3.7GB, and any one of the five `model.pt` files exceeds
Vercel's 100MB per-file limit. It must *not* exclude `backend/` or `training/`
as it once did: the API's image is built from them.

## Moving the API somewhere else

`deploy/hf-space/` is a Docker container for the API: it clones this repo and
pulls the weights from the Hub at build time, so it carries no code of its
own. It is the way off Vercel if the function's cold start or its 2GB of
memory ever stops being enough — HF Spaces with PRO, Google Cloud Run,
Fly.io, or anything else that runs containers. For a non-HF host, change
`EXPOSE`/`--port` from 7860 to whatever that platform expects, and set
`CORS_ORIGINS` there.
