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
│   ├── tests/              49 tests, no artifact needed
│   └── requirements.txt
└── frontend/            React + TypeScript UI (Vite)
    └── src/
        ├── types/          shapes mirroring the backend schemas
        ├── services/       api.ts — typed axios client
        ├── components/     SentimentBadge, ResultCard, AttentionBars, HealthBanner, ReviewForm
        ├── pages/          AnalysisPage, PerformancePage
        └── App.tsx         tab toggle
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
# [{'aspect': 'screen', 'label': 'negative', 'confidence': 0.7228,
#   'scores': {'negative': 0.7228, 'neutral': 0.0039, 'positive': 0.2732},
#   'attention': [{'token': 'amazing', 'weight': 0.285}, ...]}]
```

`load_artifact` returns the model already in `eval()` mode. Each prediction
carries its attention over the context tokens — the main reason this
architecture is worth its extra parameters, and the thing the UI shows: the
served `/predict` passes it through, merged back into whole words.

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

## Results

`bert-base-uncased` + aspect attention, trained with `configs/default.yaml`
(4 epochs, `max_length=128`, batch size 16, lr 3e-5 encoder / 1e-4 head,
class-weighted loss) and scored on the official gold test set (638 examples
after dropping `conflict`). This is the run packaged as `artifacts/absa-v1`.

| metric | value |
| --- | --- |
| accuracy | **0.7884** |
| macro F1 | **0.7474** |
| weighted F1 | 0.7870 |

Per class:

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| negative | 0.6450 | 0.8516 | 0.7340 | 128 |
| neutral | 0.6835 | 0.5621 | 0.6169 | 169 |
| positive | 0.9061 | 0.8768 | 0.8912 | 341 |

Confusion matrix, rows = gold, columns = predicted:

```
           pred ->
           negative   neutral  positive
negative |      109        14         5
 neutral |       48        95        26
positive |       12        30       299
```

### Training history

Per epoch, from `artifacts/absa-v1/config.json`:

| epoch | train loss | train acc | val loss | val acc | val macro F1 | neutral F1 | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8374 | 0.6228 | 0.8133 | 0.6897 | 0.6284 | 0.3579 | 586 |
| 2 | 0.4630 | 0.8313 | 0.7772 | 0.7241 | 0.6612 | 0.4146 | 655 |
| **3** | 0.2298 | 0.9178 | 0.9108 | 0.7802 | **0.7067** | 0.4267 | 658 |
| 4 | 0.1153 | 0.9625 | 1.0943 | 0.7716 | 0.6836 | 0.3611 | 651 |

Training loss falls monotonically while validation loss turns upward after
epoch 2 — overfitting, unsurprising with 2081 training rows against 112M
parameters. Validation macro F1 still peaks at epoch 3, so that is the
checkpoint kept and packaged. Selecting on validation loss would have picked
epoch 2; keeping the last epoch would have given the worst of the four. About
42.5 minutes total on CPU, and `training.epochs: 4` is enough to find the peak
and see it pass — there is no early stopping, so every configured epoch runs.

### What the attention learned

`python train.py evaluate --show-attention` prints the top-weighted context
tokens for the first few gold-test examples. Against `absa-v1`:

| aspect | prediction | top attended context tokens |
| --- | --- | --- |
| `Boot time` | positive (0.992) | **super** 0.18, is 0.16, **fast** 0.13 |
| `tech support` | negative (0.997) | would 0.25, **not** 0.23, support 0.06 |
| `Set up` | positive (0.999) | **easy** 0.35, was 0.31, up 0.20 |

The weights concentrate rather than spreading evenly, and the sentiment-bearing
word is in the top three every time. They are not purely lexical, though:
`would` outranks `not` on the second example and `was` takes 0.31 on the third,
so tokens carrying clause structure get real weight too. The honest claim is
that the attention gives the model a path to aspect-specific evidence and
visibly concentrates on opinion words — not that it cleanly isolates one word
per aspect. No ablation has been run, so its contribution is not yet measured.

One thing that looks like a discrepancy but is not: `evaluate` reports a lower
loss than the training run did (0.7117 vs 0.8510) while every metric matches to
four decimals. Training weights the loss by inverse class frequency; a reloaded
artifact has no class weights, because they describe the training split rather
than the model and are deliberately not saved. The *metrics* matching exactly is
what confirms the reload is faithful.

`neutral` remains the weak class — both the vaguest category (SemEval uses it
for an aspect mentioned without evaluation) and the most under-represented
(421 of 2081 training rows). Most of its errors go to `negative`: 48 of the 74
missed neutrals, which is also why `negative` precision sits at 0.6450 despite
0.8516 recall.

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
{
  "aspect": "screen",
  "sentiment": "negative",
  "confidence": 0.6317,
  "scores": {"negative": 0.6317, "neutral": 0.0044, "positive": 0.3638},
  "attention": [
    {"token": "great", "weight": 0.2499},
    {"token": "is", "weight": 0.2201},
    {"token": "dim", "weight": 0.0567}
  ],
  "aspect_in_text": true,
  "model_version": "absa-v1"
}
```

`scores` is the whole softmax distribution, and `attention` is the review's
most-weighted words for this aspect, heaviest first. WordPiece pieces are
folded back into whole words before they are returned — `unresponsive` arrives
from BERT as `un ##res ##pon ##sive`, and four fragments would render as noise.
`attention` is `null` rather than absent when the served model cannot report
it, so a client always knows which case it is in.

`aspect_in_text` says whether the aspect occurs verbatim in the review. It
matters because the model answers just as confidently either way: send the
sentence above with `"aspect":"keyboard"` and it returns **positive at 0.9803**
about a keyboard the sentence never mentions. Every training example took its
aspect from a literal span of its sentence, so an absent aspect is outside the
regime the model learned, and a UI that shows the confidence without saying so
is overclaiming. The check compares token ids rather than raw strings, so it
agrees with what the model actually consumed — casing, punctuation and
possessives are already normalised by the tokenizer (`OS` matches `os`,
`hard drive` matches `hard drive's`), and an occurrence that truncation cut is
correctly reported as absent. The frontend turns a `false` into a warning
above the confidence figure.

It imports the training package rather than vendoring a copy of the
architecture, so the serving path cannot drift from what was trained. If the
artifact is missing the service still boots and reports
`{"status":"degraded","model_loaded":false}` on `/health`, with `/predict`
answering 503 — easier to diagnose than a crash-looping container. See
[`backend/README.md`](backend/README.md) for the full design notes.

## Frontend

Two screens: analyze a review, and view the served model's metrics.

```bash
cd backend && uvicorn app.main:app --port 8000   # must be up first
cd frontend && npm install && npm run dev        # http://localhost:5173
```

`VITE_API_URL` (see `frontend/.env.example`) overrides the backend URL; it
defaults to `http://localhost:8000`, and the backend's CORS policy already
allows the Vite dev origin. Functional rather than polished — plain Tailwind,
no animations, no responsive work. See
[`frontend/README.md`](frontend/README.md).

## Running the whole stack

```bash
# 1. train a model (writes training/artifacts/absa-v1/)
cd training && python train.py train

# 2. serve it
cd ../backend && uvicorn app.main:app --port 8000

# 3. the UI
cd ../frontend && npm run dev
```

## Tests at a glance

| Suite | Count | Needs an artifact? |
| --- | --- | --- |
| `training/tests` | 138 | no — tiny encoder built from config |
| `backend/tests` | 45 | no — mocked registry |
| `frontend` | `tsc -b` + oxlint | no |
