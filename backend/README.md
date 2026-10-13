# ABSA inference API

FastAPI service wrapping the trained artifact from `../training/`. Send a
sentence and an aspect term inside it; get the sentiment expressed about that
aspect.

```
POST /api/v1/predict   {"text": "...", "aspect": "..."}  -> sentiment + confidence
GET  /api/v1/health                                      -> liveness + model readiness
GET  /api/v1/model                                       -> which model, and how it scored
```

## Layout

```
backend/
├── app/
│   ├── main.py                 app factory, lifespan, CORS, router mounting
│   ├── core/
│   │   ├── config.py           Settings (env-overridable) + get_settings()
│   │   └── errors.py           ModelNotReadyError + the error handlers
│   ├── schemas/
│   │   ├── request.py          PredictRequest
│   │   └── response.py         PredictResponse, HealthResponse, ModelMetaResponse
│   ├── services/
│   │   ├── model_registry.py   singleton holding the loaded artifact
│   │   └── predictor.py        encode -> forward -> softmax -> label
│   └── routes/sentiment.py     the three endpoints
├── tests/                      39 tests, no artifact needed
└── requirements.txt
```

## Running

```bash
pip install -r backend/requirements.txt     # torch etc. come from training/
cd backend
uvicorn app.main:app --reload --port 8000
```

Then <http://localhost:8000/docs>.

The artifact location and device are read from the environment:

```bash
ARTIFACT_DIR=training/artifacts/absa-v1 DEVICE=cpu uvicorn app.main:app
```

`ARTIFACT_DIR` is resolved against the **repository root** when relative, so
the service starts the same way from any directory.

## Design notes

**The model loads once, at startup.** `lifespan` calls `registry.load()` before
the first request. The artifact is a few hundred megabytes, so loading it
per-request is not an option, and `ModelRegistry` is a singleton holding it for
the process.

**A missing model degrades, it does not crash.** If the artifact cannot be
loaded the service still boots: `/health` returns 200 with
`{"status": "degraded", "model_loaded": false}`, and `/predict` answers 503.
A crash-looping container tells an operator far less than a running one that
says exactly what is wrong. Set `ALLOW_DEGRADED_START=false` to make a load
failure fatal instead.

**`/health` is always 200.** A caller needs to distinguish "process up, model
still loading" from "process unreachable". Answering 503 on an unready model
collapses those two cases into one.

**The training package is imported, not vendored.** `ModelRegistry` puts the
repo root on `sys.path` and imports `training.src.artifact` (`training/`
resolves as a namespace package). The model architecture lives in exactly one
place, so the serving code cannot drift from what was trained. torch and
transformers are imported inside `load()` rather than at module scope, so
importing the app stays fast.

**Status codes are chosen to tell a client what to do:**

| Code | When | Why that code |
| --- | --- | --- |
| 422 | body failed validation — missing, blank, too long | before the model is touched |
| 400 | the encoder rejected the input, e.g. an aspect that tokenises to nothing | the request is malformed in a way only the tokenizer can see |
| 503 | model not loaded | retrying later is reasonable |
| 500 | anything unexpected | the client can do nothing |

Whitespace-only input is rejected as 422, not 400: `min_length=1` alone accepts
`" "`, which would otherwise reach the tokenizer and surface as a different
status for what is really the same mistake.

**Every error has the same shape** — `{"error": "<message>", "status": <code>}`
— including the 404 and 405 the router raises. That needs a handler on
*Starlette's* `HTTPException`, not FastAPI's: FastAPI's subclasses it, so a
handler registered on the subclass alone silently misses router errors and
they come back as `{"detail": ...}`.

**Internals never reach the client.** The 500 handler logs the exception with
its stack trace and returns a fixed `"internal server error"`. The 503 message
likewise omits the load failure's cause, which can carry filesystem paths;
it is logged at ERROR instead. Both have tests asserting nothing leaks.

## Tests

```bash
cd backend && python -m pytest tests/ -q
```

39 tests, and **none of them need the artifact**. `conftest.py` provides a fake
model (fixed logits `[0.1, 0.2, 0.7]`), a fake encoder returning correctly
shaped zero tensors, and a fake label map, injected into the registry
singleton and restored afterwards so no test leaks state into the next.

Covered: the happy path and exact confidence arithmetic, every label decoding,
all validation rejections including boundary lengths and whitespace, 400/503/500
paths, both leak assertions, CORS for the two dev origins and its rejection of
others, the versioned prefix, and the OpenAPI schema.

The doubles honour the same contracts the service depends on — the encoder's
tensor dict, the model's `{"logits": ...}`, the label map's `id2label` — so the
routes, schemas and error handling are exercised for real while inference stays
deterministic.
