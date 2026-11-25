---
title: ABSA API
emoji: 💬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Aspect-based sentiment analysis over laptop reviews.
---

# ABSA API

Inference backend for the aspect-based sentiment analysis project. A
fine-tuned `bert-base-uncased` with an aspect-attention head: send a sentence
and an aspect term occurring inside it, get the sentiment expressed about that
aspect, plus the per-token attention the decision used.

The frontend is deployed separately on Vercel; this Space serves only the API.

- `GET /api/v1/health` — readiness, model version, device
- `GET /api/v1/model` — artifact metadata and evaluation metrics
- `POST /api/v1/predict` — `{"text": "...", "aspect": "..."}`
- `GET /docs` — generated OpenAPI docs

## Why this Space holds no application code

The `Dockerfile` clones the public GitHub repository and pulls the trained
artifact from a Hub model repo at build time. The weights are ~450MB and are
gitignored upstream, so they cannot travel with the code; keeping both out of
this repo means a rebuild always takes the current state of each, and this
repo stays three small files.

The trade-off: pushing to GitHub does **not** redeploy this Space. After an
upstream change, trigger a rebuild here (Settings → Factory rebuild).

## Variables to set

Settings → Variables and secrets. None are secret.

| Variable | Value | Why |
| --- | --- | --- |
| `CORS_ORIGINS` | `https://<your-app>.vercel.app` | Without the deployed frontend's exact origin, every browser request is blocked by CORS. Comma-separate to allow more than one. |

Both `GIT_REF` (default `main`) and `MODEL_REPO` are build arguments with
defaults in the `Dockerfile`; set them here only to pin a different code
revision or weights repo.

## Note on cold starts

A free Space sleeps after inactivity. The first request after it wakes pays
the container start plus the model load, which together run to roughly a
minute — the frontend's health banner will report the API as unreachable
until that finishes. Hitting `/api/v1/health` once before a demo warms it.
