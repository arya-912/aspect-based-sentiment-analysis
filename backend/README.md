# ABSA inference API

FastAPI service wrapping the trained artifact from `../training/`. Send a
sentence and an aspect term inside it; get the sentiment expressed about that
aspect.

```
POST /api/v1/predict   {"text": "...", "aspect": "..."}  -> sentiment, scores, attention, aspect_in_text
                       (422 if the aspect is not in the text — the model cannot abstain)
GET  /api/v1/health                                      -> liveness + model readiness
GET  /api/v1/model                                       -> which model, which corpus, and how it scored
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
│   │   └── response.py         TokenAttention, PredictResponse, Health/ModelMeta
│   ├── services/
│   │   ├── aspect_match.py     is the aspect actually in the review? (the 422 gate)
│   │   ├── model_registry.py   singleton holding the loaded artifact
│   │   └── predictor.py        encode -> forward -> softmax -> label, attention, aspect check
│   └── routes/sentiment.py     the three endpoints
├── tests/                      88 tests, no artifact needed
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
| 422 | the aspect does not occur in the review text | the model cannot abstain, so the question is refused rather than answered |
| 400 | the encoder rejected the input, e.g. an aspect that tokenises to nothing | the request is malformed in a way only the tokenizer can see |
| 503 | model not loaded | retrying later is reasonable |
| 500 | anything unexpected | the client can do nothing |

Whitespace-only input is rejected as 422, not 400: `min_length=1` alone accepts
`" "`, which would otherwise reach the tokenizer and surface as a different
status for what is really the same mistake.

**`/model` reports the corpus, not just the version.** `corpus` is the stem of
the artifact's recorded `paths.raw_train` (`Laptop_Train_v2`,
`Restaurants_Train_v2`), so a client can label the domain and offer examples
that belong to it. Read from the path rather than the version string because
the version is a name someone chose and the path is what the trainer actually
read. It is *not* normalised to a domain word here: this service has no list of
domains, and inventing one would make the endpoint wrong the first time a
corpus is added, so the mapping lives in the client. `null` when the artifact
records no paths.

**An absent aspect is refused, not answered.** The head is a softmax over three
polarities with no way to abstain — nothing in training taught it one, since
every SemEval aspect term is a literal span of its sentence — so an aspect the
review never mentions still gets a confident polarity, and it is the sentence's
overall sentiment wearing the aspect's name: `positive` at 0.9803 for
`keyboard` against "The battery life is great but the screen is dim.". `confidence` is a softmax over
the three options, not a probability the aspect is there, so no threshold
catches it.

`services/aspect_match.py` decides presence on **whole words, matched on
stems**, and the route turns a `False` into a 422 before the model is touched —
a string comparison needs no artifact, and a caller fixing a typo should not be
told to retry later. Stems rather than substrings because rejecting a real
aspect over an `-s` (`batteries` against a review saying `battery`) would be
worse than the failure being fixed; whole words because a substring test finds
`OS` inside `most`; contiguous because `hard drive` is not present in a review
about a hard case and a test drive. The stemmer handles plurals and possessives
only — no `-ing`/`-ed`, because over-stemming fails in the direction that lets
an aspect match a word that is not it, while under-stemming only costs a
rejection the caller can see and correct.

It is a gate on *presence*, not on aspecthood: `"aspect": "the"` is in the text,
so it is answered. Distinguishing "mentioned" from "evaluated" needs a fourth
`not_mentioned` class and a negative-sampling set built by hand, which changes
the label space and ends comparability with SemEval-2014 Task 4 and with this
project's own no-attention ablation. `REQUIRE_ASPECT_IN_TEXT=false` restores
the old answer-anyway behaviour, which is what the figure above was measured
with.

The frontend applies the same rule in `lib/words.ts` before it sends anything,
so the two never contradict each other; a parity check over 35 words and 14
phrase pairs found no disagreement.

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

88 tests, and **none of them need the artifact**. `conftest.py` provides a fake
model (fixed logits `[0.1, 0.2, 0.7]` and a deliberately uneven attention row),
a fake encoder laying out `[CLS]` / review / aspect slots with a tiny fake
tokenizer, and a fake label map, injected into the registry singleton and
restored afterwards so no test leaks state into the next. The fake encoder's
aspect slots repeat ids from its review slots by default, as a real aspect term
does; `FakeEncoder(aspect_in_context=False)` gives the off-distribution case.

Covered: the happy path and exact confidence arithmetic, every label decoding,
all validation rejections including boundary lengths and whitespace, 400/503/500
paths, both leak assertions, CORS for the two dev origins and its rejection of
others, the versioned prefix, and the OpenAPI schema.

Also covered, since `/predict` now returns them: the full label distribution
summing to 1 and agreeing with the chosen label, attention sorted heaviest
first with `##` pieces merged into whole words, attention restricted to the
review tokens, the cap on how many are returned, and the two degraded paths —
an encoder with no tokenizer and a model that reports no attention both yield
`attention: null` with the prediction intact.

`test_aspect_match.py` covers the gate on strings alone, with no fixtures at
all: the stem rules and their length guards (`OS`, `bus` and `this` keep their
`s`), the inflected forms that must be *accepted*, contiguity, whole-word
matching, a synonym still counting as absent, and an aspect of no words at all
being left to the checks that can say something specific about it.

On the route: an absent aspect is a 422 carrying the documented error shape and
no prediction beside it; it is still a 422 when the model never loaded, rather
than a 503; an inflected aspect is a 200; and with `REQUIRE_ASPECT_IN_TEXT`
off, the old 200-with-a-flag response comes back unchanged.

`aspect_in_text` keeps its own cases, now narrowed to what the gate lets
through: an aspect present in the review, one the model did not see verbatim
(still a 200, prediction intact — the stem-match and truncation cases), one
whose tokens all appear but never adjacently — a span, not a bag of words, so
this is `false` — and a missing `aspect_mask`, which yields `null` rather than
a guess.

The doubles honour the same contracts the service depends on — the encoder's
tensor dict, the model's `{"logits": ..., "attention": ...}`, the label map's
`id2label` — so the routes, schemas and error handling are exercised for real
while inference stays deterministic.
