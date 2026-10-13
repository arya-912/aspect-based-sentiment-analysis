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
│   │   ├── models/         absa_model.py — transformer + aspect pooling
│   │   ├── evaluation/     evaluate.py (metrics), reporting.py (rendering)
│   │   ├── config.py       typed, YAML-backed configuration
│   │   ├── trainer.py      training loop, scheduling, best checkpoint
│   │   ├── artifact.py     save/load a self-contained run bundle
│   │   └── experiment.py   end-to-end orchestration
│   ├── data/raw/           the four SemEval CSVs (gitignored)
│   ├── data/processed/     generated splits (gitignored)
│   ├── artifacts/          run outputs: weights, metrics, reports (gitignored)
│   ├── configs/            YAML run definitions
│   ├── notebooks/          data exploration
│   ├── tests/              pytest suite
│   └── train.py            CLI entry point
├── backend/             (not started)
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

## Model

`ABSAModel` = pretrained encoder → pooled representation → classifier.
`pooling` chooses what the head sees:

| `pooling` | Representation |
| --- | --- |
| `cls` | `[CLS]` alone — the plain sentence-pair baseline |
| `aspect` | mean over the aspect tokens |
| `aspect_context` | `[CLS] ‖ mean(aspect) ‖ mean(context)` (default) |

Pooling is masked-mean with a clamped denominator, so a degenerate example
pools to zeros instead of NaN and cannot poison a batch's gradient.

The randomly-initialised head gets a higher learning rate than the pretrained
encoder (`head_learning_rate` vs `learning_rate`), and biases/LayerNorm weights
are exempt from weight decay — both standard for transformer fine-tuning.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
```

## Training

```bash
cd training

python train.py                                  # defaults
python train.py --config configs/bert_base.yaml  # from YAML
python train.py --config configs/quick.yaml      # 1 epoch, smoke test
python train.py --epochs 2 --batch-size 16 --name my-run
python train.py --pooling cls --name cls-baseline # ablation
python train.py --dump-config /tmp/resolved.yaml  # inspect, don't train
```

CLI flags override the YAML; anything left unset keeps its config value. Run
`python train.py --help` for the full list.

A run writes to `training/artifacts/<name>/`:

| File | Contents |
| --- | --- |
| `model.pt` | best checkpoint's state dict |
| `config.yaml` | the exact resolved config |
| `metadata.json` | labels, metrics, history, library versions |
| `tokenizer/` | saved tokenizer, so inference needs no download |
| `validation.{json,txt,md}` | best-epoch validation report |
| `test.{json,txt,md}` | official gold-test report |
| `*_predictions.csv` | per-example gold/predicted/probabilities |
| `summary.json` | headline numbers |

The checkpoint kept is the best epoch by `train.monitor` (default `macro_f1`),
not the last — macro F1 rather than accuracy because `neutral` is the minority
class and the one the model finds hardest.

## Inference

An artifact is self-contained:

```python
from src.artifact import load_artifact

run = load_artifact("artifacts/bert-base-aspect-context")

run.predict("Battery life is amazing but the screen is dim.", "screen")
# {'label': 'negative', 'confidence': 0.91, 'scores': {...}, ...}

run.predict_batch([(sentence, aspect), ...])   # one forward pass
```

This is the seam the `backend/` service will sit on.

## Tests

```bash
cd training && python -m pytest tests/ -q
```

65 tests covering the label map, the CSV quirks above, offset validation,
split integrity (no sentence leakage, determinism per seed), mask semantics
(disjoint, no specials or padding, correct multi-word spans), truncation,
config round-tripping and metric edge cases. Tests needing the corpus or a
cached tokenizer skip cleanly when either is absent.

## Baseline

One epoch of `bert-base-uncased` with `aspect_context` pooling at
`max_length=64`, on the official gold test set (638 examples after dropping
`conflict`):

| metric | value |
| --- | --- |
| accuracy | 0.7618 |
| macro F1 | 0.7220 |
| weighted F1 | 0.7641 |

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| negative | 0.5989 | 0.8281 | 0.6951 | 128 |
| neutral | 0.6419 | 0.5621 | 0.5994 | 169 |
| positive | 0.9105 | 0.8358 | 0.8716 | 341 |

`neutral` is clearly the weak class — it is both the vaguest category and
under-represented, and most of its errors go to `negative`. The default config
(4 epochs, `max_length=128`) should be run before drawing conclusions; the
above is a smoke-test baseline, not a tuned result.

## Next

`backend/` and `frontend/` are intentionally empty — the training package and
its artifact format are the foundation they will build on.
