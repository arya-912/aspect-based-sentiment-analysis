# Deploying

## What is live

| Piece | Where | State |
| --- | --- | --- |
| `frontend/` | Vercel — <https://absa-sentiment.vercel.app> | deployed |
| `training/artifacts/absa-joint-v1` | local; `absa-v1` is on the Hub at <https://huggingface.co/arya912/absa-v1> | served model, 450MB |
| `backend/` | your machine, exposed by a tunnel when demoing | run on demand |

To demo:

```sh
./deploy/demo.sh
```

It boots the API, opens a Cloudflare quick tunnel, waits for both to answer,
and prints a link like

```
https://absa-sentiment.vercel.app/?api=https://<random>.trycloudflare.com
```

Open that and the deployed UI talks to your machine. Ctrl-C stops both.

## Why the backend is not hosted

Two separate walls, in the order they were hit.

**Vercel.** A serverless function is capped at 250MB unzipped with no
persistent disk. The CPU build of torch is ~200MB installed and `model.pt` is
another 450MB, so the function is over the cap before any of our code counts,
and there is nowhere to keep weights a cold start could reuse. No
configuration changes this.

**Hugging Face Spaces.** A Docker Space would fit comfortably, but Spaces
running on compute now require PRO ($9/mo) — free accounts can only create
*static* Spaces. The container is written and ready in `deploy/hf-space/`; it
needs a paid account to run, and `create_repo` returns `402 Payment Required`
without one.

So the backend runs where the weights already are: locally.

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

Only `absa-v1` is on the Hub. If you ever use the container in
`deploy/hf-space/`, push the joint weights first -- its `MODEL_REPO` now
points at `arya912/absa-joint-v1`:

```sh
.venv/bin/hf upload arya912/absa-joint-v1 training/artifacts/absa-joint-v1 . --repo-type=model
```

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

Because step 3 is the deployed default and the backend's CORS list carries the
Vercel origin, **no tunnel is needed when you demo from this machine**: open
the deployed site, run `uvicorn app.main:app`, and the HTTPS page will call
`http://localhost:8000` directly. Browsers allow that particular
mixed-content case because localhost counts as a trustworthy origin. The
tunnel is only for reaching the API from someone else's device.

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

## Redeploying the frontend

The Vercel project is linked at the repo root with Root Directory
`frontend`, so deploy from the root, not from inside `frontend/`:

```sh
npx vercel deploy --prod
```

Pushing to `main` also deploys, since the GitHub repo is connected and
`main` is the production branch. `.vercelignore` keeps `.venv` and
`training/` out of the upload — together ~3.7GB, and either of the five
`model.pt` files alone exceeds Vercel's 100MB per-file limit.

## If you ever want it always-on

`deploy/hf-space/` is a working Docker container for the API: it clones this
repo and pulls the weights from the Hub at build time, so it carries no code
of its own. It needs a host that runs containers — HF Spaces with PRO, Google
Cloud Run, Fly.io, or similar. For a non-HF host, change `EXPOSE`/`--port`
from 7860 to whatever that platform expects, and set `CORS_ORIGINS` there.
