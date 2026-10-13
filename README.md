# ABSA — Aspect-Based Sentiment Analysis (SemEval laptops)

Given a review sentence **and an aspect term inside it**, predict the sentiment
expressed *about that aspect*:

> "Battery life is amazing but the **screen** is dim."
> → `battery life` → **positive**, `screen` → **negative**

The same sentence therefore yields several labelled examples, one per aspect.
That is what separates ABSA from plain sentence-level sentiment, and it drives
the design of both the encoder and the train/val split.

## Layout

```
absa-project/
├── training/            model training, evaluation, artifact packaging
│   ├── src/
│   │   ├── data/           dataset.py — loading, cleaning, splitting
│   │   ├── preprocessing/  tokenizer.py — sentence-pair encoding + masks
│   │   ├── models/         absa_model.py — BERT + aspect attention
│   │   ├── evaluation/     evaluate.py (metrics), reporting.py (rendering)
│   │   ├── config.py       YAML config loading + validation
│   │   ├── trainer.py      training loop, clipping, best checkpoint
│   │   └── artifact.py     build/load a self-contained run bundle
│   ├── data/raw/           the four SemEval CSVs (gitignored)
│   ├── data/processed/     generated splits (gitignored)
│   ├── artifacts/          run outputs: weights, metrics, reports (gitignored)
│   ├── configs/            default.yaml (+ quick.yaml for smoke tests)
│   ├── notebooks/          data exploration
│   ├── tests/              pytest suite
│   └── train.py            CLI: prepare | train | evaluate
├── backend/             FastAPI inference service
│   ├── app/
│   │   ├── main.py         app factory, lifespan, CORS
│   │   ├── core/           config.py (Settings), errors.py (handlers)
│   │   ├── schemas/        request.py, response.py
│   │   ├── services/       model_registry.py, predictor.py
│   │   └── routes/         sentiment.py — predict | health | model
│   ├── tests/              39 tests, no artifact needed
│   └── requirements.txt
└── frontend/            (not started)
```

## Data

Four CSVs from the SemEval-2014 Task 4 laptop domain live in
`training/data/raw/`:

| File | Rows | Usable | Sentences | Contents |
| --- | --- | --- | --- | --- |
| `Laptop_Train_v2.csv` | 2358 | 2313 | 1456 | training set, with aspects + polarity |
| `Laptops_Test_Gold.csv` | 654 | 638 | 411 | official test set, with gold polarity |
| `Laptops_Test_Data_PhaseA.csv` | 800 | — | 800 | **unlabelled** — `id, Sentence` only |
| `Laptops_Test_Data_PhaseB.csv` | 800 | — | 800 | **unlabelled** — `id, Sentence` only |

"Usable" is after dropping `conflict` (45 train / 16 test rows).

Schema: `id, Sentence, Aspect Term, polarity, from, to`, where `from`/`to` are
character offsets such that `Sentence[from:to] == Aspect Term`.

Labels: `negative=0`, `neutral=1`, `positive=2`. Rows marked `conflict`
(an aspect praised and criticised at once) are dropped, which is standard
practice for this benchmark.

Two quirks the loader handles, both verified against the real files:

- **`conflict` rows** — dropped (45 train, 16 test), and counted in the log line.
- **Column-shifted rows** — 7 rows of `Laptop_Train_v2.csv` carry a stray empty
  field right after `Sentence`, e.g.
  `3074,The screen is bright and the keyboard is nice,,screen,positive,4,10`.
  These parse with a blank aspect and every later field one column to the
  right. `load_raw_csv` detects them (unknown polarity **and** blank aspect,
  with a trailing extra column to pull from) and realigns them rather than
  discarding real data. Every row's offsets are then validated, so a silent
  misalignment cannot slip through.

PhaseA/PhaseB have no aspect annotations, so `load_raw_csv` rejects them with a
clear error — they are inputs for the aspect-*extraction* subtask, not this one.

### Splitting

`split_train_val` groups by **sentence text**, never by row. A sentence with
three aspects lands wholly in train or wholly in validation. Splitting by row
would put the same sentence on both sides and inflate validation scores, since
the model would have already seen the context at training time.

This matters more than it sounds: sentences average 1.59 aspects (max 13), and
**11.3% carry more than one polarity** — those are exactly the examples a
sentence-level sentiment model cannot get right, and the ones a leaky split
would quietly hand the model for free.

## Encoding

`ABSAEncoder` builds a BERT sentence pair:

```
[CLS] the screen is bright and crisp [SEP] screen [SEP]
```

with `truncation="only_first"`, so a long review is clipped but the aspect
phrase always survives — it is what the prediction is about. Alongside the
usual tensors it emits two boolean masks, derived from `token_type_ids` and the
special-tokens mask:

- `context_mask` — the review's own tokens only (no `[CLS]`/`[SEP]`/`[PAD]`)
- `aspect_mask` — the aspect segment's tokens only

They are disjoint by construction, and the model pools over them.

## Model — `BertAspectAttention`

The sentence-pair encoding hands the model the aspect, but nothing tells it
*which context words matter for that aspect*. This architecture makes that
explicit:

```
H     = BERT([CLS] text [SEP] aspect [SEP])        (B, T, d)
a     = mean(H[aspect_mask])                       aspect representation
e_i   = vᵀ tanh(W_h h_i + W_a a)                   additive attention scores
alpha = softmax(e)  masked to context_mask         (B, T), zero off-mask
c     = Σ_i alpha_i h_i                            attended context vector
logits = MLP([c ; a ; h_CLS])                      3d → d → num_labels
```

So "Battery life is amazing but the screen is dim" can attend to *amazing* for
`battery life` and *dim* for `screen`, from one shared pass over the sentence.
`W_a a` does not depend on `i`, so it is computed once and broadcast.

The head sees all three of `[c ; a ; h_CLS]`: keeping the `[CLS]` pair summary
means the model can fall back on it when the attention is uninformative,
rather than depending on attention alone. The classifier is
`Dropout → Linear(3d, d) → GELU → Dropout → Linear(d, num_labels)`.

Two numerical details that matter in practice:

- **Masking uses `finfo.min`, not `-inf`.** A row whose `context_mask` is
  entirely empty would make `softmax` return NaN and poison the whole batch's
  gradient. With a finite floor the row degrades to uniform, and the weights
  are then re-zeroed outside the mask so no attention leaks onto padding.
- **`masked_mean` clamps its denominator**, for the same reason on the aspect
  side.

Class weights are a non-persistent buffer: they follow `.to(device)` but stay
out of the `state_dict`, because they describe the *training split's* class
balance rather than the model. That keeps `load_artifact` a strict, exact load.

The pretrained encoder and the randomly-initialised head get different
learning rates (`training.lr` vs `optim.head_lr`), biases and LayerNorm
weights are exempt from weight decay, and the backbone's pooler is excluded
from the optimizer — this model pools `last_hidden_state` itself, so no
gradient ever reaches the pooler.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
```

## Usage

Three subcommands, each usable on its own:

```bash
cd training

python train.py prepare     # clean + split the corpus -> train.csv / val.csv
python train.py train       # fine-tune, then write the artifact
python train.py evaluate    # score the saved artifact on the gold test set
```

Config comes from `training/configs/default.yaml`. `--config` picks another
file and per-command flags override individual values:

```bash
python train.py --config configs/quick.yaml train   # 1 epoch, max_length 64
python train.py train --epochs 2 --batch-size 32 --version absa-v2
python train.py train --model-name roberta-base --lr 2e-5
python train.py evaluate --show-attention           # most-attended tokens
python train.py evaluate --artifact artifacts/absa-v1 --out-dir /tmp/report
```

`train` prepares the splits automatically if they are missing; `--prepare`
forces a re-split. Device is `cuda` when available, else `cpu`; `--device`
overrides. Config paths are written repo-relative, so every command behaves
the same whether run from the repo root or from `training/`.

### Config

```yaml
seed: 42
model:    {name: bert-base-uncased, max_length: 128, dropout: 0.1, num_labels: 3}
training: {epochs: 4, batch_size: 16, lr: 3e-5, weight_decay: 0.01,
           warmup_ratio: 0.1, max_grad_norm: 1.0, val_fraction: 0.1}
paths:    {raw_train: ..., official_test: ..., processed_dir: ..., artifacts_dir: ...}
artifact: {version: absa-v1}
```

The loader coerces numeric keys after parsing. PyYAML only recognises a float
when it sees *both* a decimal point and a signed exponent, so the entirely
reasonable `lr: 3e-5` loads as the **string** `"3e-5"` and would otherwise
reach AdamW untouched and fail a long way from the cause.

### The artifact

A run writes `training/artifacts/<version>/`:

| Entry | Contents |
| --- | --- |
| `model.pt` | best checkpoint's `state_dict` (weights only) |
| `encoder/` | the base transformer's **config** — enough to rebuild the shape |
| `tokenizer/` | saved tokenizer files |
| `config.json` | every hyperparameter, plus run id, best epoch, library versions |
| `label_map.json` | `label2id` / `id2label` |
| `metrics.json` | accuracy, macro F1, per-class, confusion matrix |
| `best.pt` | the trainer's running best checkpoint |
| `validation_metrics.{json,txt,md}` | best-epoch validation report |

`encoder/` holds the config and **not** a second copy of the weights: every
weight is already in `model.pt`, so reconstruction is `AutoModel.from_config`
followed by a strict `load_state_dict`. That makes `load_artifact` work with
no network and without duplicating ~440MB of parameters — there is a test that
blocks `socket.connect` for the duration of a load to keep it that way.

The checkpoint kept is the best epoch by **validation macro F1**, not the last
and not accuracy: `neutral` is the minority class, so accuracy can climb while
the model quietly gives up on it.

## Inference

An artifact is self-contained and loads offline:

```python
from src.artifact import load_artifact, predict_pairs

model, encoder, label_map, config = load_artifact("artifacts/absa-v1")

predict_pairs(model, encoder, label_map, [
    ("Battery life is amazing but the screen is dim.", "screen"),
])
# [{'label': 'negative', 'confidence': 0.91, 'scores': {...},
#   'attention': [{'token': 'dim', 'weight': 0.31}, ...]}]
```

`load_artifact` returns the model already in `eval()` mode. Each prediction
carries its attention over the context tokens — the main reason this
architecture is worth its extra parameters, and the thing a UI can show.

This is the seam the `backend/` service will sit on.

## Tests

```bash
cd training && python -m pytest tests/ -q
```

138 tests, all on CPU and all offline. Model and artifact tests run against a
tiny randomly-initialised BERT (2 layers, hidden size 32) built from a config,
so nothing is downloaded and the suite finishes in seconds — the architecture
under test is the attention and the head, not the pretrained weights.

| File | Covers |
| --- | --- |
| `test_dataset.py` | `LabelMap`, the CSV quirks above, offset validation, split integrity (no sentence leakage, determinism per seed) |
| `test_tokenizer.py` | mask shapes and semantics (disjoint, no specials or padding, correct multi-word spans), truncation, empty-aspect/text errors |
| `test_model.py` | attention normalisation and masking, degenerate rows, backward reach, optimizer groups, metrics |
| `test_trainer.py` | epoch metrics, gradient clipping, best-checkpoint selection and restore, scheduler stepping, overfitting one batch |
| `test_artifact.py` | every required file written, exact offline round-trip, each missing entry rejected, config loading |

Tests needing the corpus skip cleanly when it is absent.

Two behaviours are worth calling out, because both are the kind of thing that
only surfaces end to end:

- **`load_artifact` is offline.** One test patches `socket.connect` to raise
  for the duration of a load, so a future change that reaches for the hub
  fails the suite rather than a user's air-gapped machine.
- **A weighted model must reload.** `nn.CrossEntropyLoss(weight=...)` is itself
  an `nn.Module`, so assigning it to the model registers it as a submodule and
  pushes `loss_fn.weight` into the `state_dict`. A model trained *with* class
  weights then cannot be loaded strictly into one built without them. The
  unweighted test fixture does not exercise this, so there is an explicit
  weighted round-trip test.

## Baseline

`bert-base-uncased` + aspect attention, `max_length=64`, batch size 16,
lr 3e-5, on the official gold test set (638 examples after dropping
`conflict`):

| epochs | accuracy | macro F1 | weighted F1 |
| --- | --- | --- | --- |
| 1 | 0.7727 | 0.7255 | 0.7697 |
| 2 | 0.7821 | 0.7447 | 0.7821 |

Per class, after 2 epochs:

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| negative | 0.6395 | 0.8594 | 0.7333 | 128 |
| neutral | 0.6736 | 0.5740 | 0.6198 | 169 |
| positive | 0.9068 | 0.8563 | 0.8808 | 341 |

### What the attention learned

`python train.py evaluate --show-attention`, after one epoch — the top-weighted
context tokens for each aspect:

| aspect | prediction | top attended context tokens |
| --- | --- | --- |
| `Boot time` | positive (0.81) | **super** 0.23, is 0.19, **fast** 0.16 |
| `tech support` | negative (0.62) | **not** 0.15, would 0.12, **fix** 0.06 |
| `Set up` | positive (0.79) | was 0.51, **easy** 0.37 |

The weights land on the sentiment-bearing words rather than spreading evenly,
which is the whole reason the attention is there.

One thing that looks like a discrepancy but is not: `evaluate` reports a lower
loss than the training run did (0.5482 vs 0.6302) while every metric matches to
four decimals. Training weights the loss by inverse class frequency; a reloaded
artifact has no class weights, because they describe the training split rather
than the model and are deliberately not saved. The *metrics* matching exactly is
what confirms the reload is faithful.

`neutral` remains the weak class — it is both the vaguest category and
under-represented, and most of its errors go to `negative`. These are
short smoke runs at `max_length=64`; the default config (4 epochs,
`max_length=128`) should be run before drawing conclusions.

## Serving

The backend wraps a trained artifact behind three endpoints:

```bash
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload --port 8000
```

```
POST /api/v1/predict   {"text": "...", "aspect": "..."}
GET  /api/v1/health
GET  /api/v1/model
```

```console
$ curl -s -X POST localhost:8000/api/v1/predict \
    -H 'Content-Type: application/json' \
    -d '{"text":"The battery life is great but the screen is dim.","aspect":"screen"}'
{"aspect":"screen","sentiment":"negative","confidence":0.9112,"model_version":"absa-v1"}
```

It imports the training package rather than vendoring a copy of the
architecture, so the serving path cannot drift from what was trained. If the
artifact is missing the service still boots and reports
`{"status":"degraded","model_loaded":false}` on `/health`, with `/predict`
answering 503 — easier to diagnose than a crash-looping container. See
[`backend/README.md`](backend/README.md) for the full design notes.

## Next

`frontend/` is intentionally empty. The artifact format and the API above are
the foundation it will build on.
