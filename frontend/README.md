# ABSA frontend

React + TypeScript (Vite) UI over the ABSA backend. Two screens: analyze a
review, and view the served model's evaluation metrics.

## Running

The backend must be up first (it serves on `:8000`):

```bash
cd backend && uvicorn app.main:app --port 8000
```

Then:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

`npm run build` type-checks (`tsc -b`) and bundles; `npm run lint` runs oxlint.

### Configuration

```bash
cp .env.example .env     # then edit if the backend is not on :8000
```

`VITE_API_URL` defaults to `http://localhost:8000`. The backend's CORS policy
allows `localhost:5173` and `localhost:3000`, so pointing a dev server at a
different origin needs that list widened in `backend/app/core/config.py`.
`vite preview` defaults to `:4173` and is blocked by that policy — run it with
`--port 5173` if you need to check a production build against the API.

## Layout

```
src/
├── types/index.ts            shapes mirroring the backend schemas
├── lib/
│   ├── theme.ts              light / dark / system resolution
│   ├── format.ts             percent and count formatting
│   └── words.ts              word spans, attention-to-word mapping
├── services/api.ts           axios client; throws a typed error
├── components/
│   ├── Card.tsx              the one raised surface
│   ├── icons.tsx             the six inline glyphs
│   ├── SentimentBadge.tsx    colored dot + the word, never color alone
│   ├── BarRow.tsx            one labelled horizontal bar
│   ├── AttentionBars.tsx     exact attention weights
│   ├── HighlightedSentence.tsx  attention painted onto the review
│   ├── AspectPicker.tsx      click words to build a verbatim aspect span
│   ├── HealthBanner.tsx      warns when the API is down or has no model
│   ├── ResultCard.tsx        one prediction + distribution + attention
│   ├── ReviewForm.tsx        examples, text + aspect input, submit
│   ├── MetricTile.tsx        one headline number
│   ├── PerClassTable.tsx     per-class F1 bars + the exact table
│   ├── ConfusionMatrix.tsx   where the errors go
│   └── ThemeToggle.tsx       light / dark / system
├── pages/
│   ├── AnalysisPage.tsx      composer + the results for one sentence
│   └── PerformancePage.tsx   metrics, per-class scores, confusion matrix
├── App.tsx                   hash-routed sections
└── main.tsx                  mount
```

## The Analysis screen

**Results belong to a sentence, not to the app.** Analyzing a second aspect of
the same review adds to the list rather than replacing it, which is the only
way the UI can show the thing that distinguishes this task: one sentence, a
different verdict per aspect. Changing the review starts a new list, because an
attention overlay and a mixed-polarity claim are both statements about a
specific sentence. Re-analyzing an aspect replaces its entry instead of
stacking a duplicate.

**The mixed-polarity callout is asserted by the data, not by the prose.** It
appears only when the model has actually returned different polarities for
different aspects of the sentence on screen.

**The aspect can be clicked out of the review.** Every training example took
its aspect from a literal span of its sentence, so a typed aspect that is
absent is outside the regime the model learned — and the model answers just as
confidently anyway. Building the aspect by slicing the review's own characters
makes that mistake unreachable rather than merely warning about it afterwards.
Click a word, shift-click a second to span several. The picker is one tab stop
with arrow-key movement inside it, since a 30-word review would otherwise be 30
tab stops. It renders the review's own punctuation and spacing, so what you
click is what you submit.

**Attention is drawn on the sentence, not beside it.** The weights are computed
over the review's own tokens, so a detached bar chart of token names throws
away where in the sentence the weight fell. The exact figures stay a click away
under "Exact weights", which is also what lets the overlay use color at all.

One honest limit, stated in the UI: **the API reports weights per word, not per
position.** When a word occurs twice ("is" twice is entirely normal) two
entries arrive with the same token and different weights, and nothing says
which occurrence each belongs to, so every occurrence is shown at the higher
figure.

## Visualization

Colors are assigned by the job they do, following the project's data-viz
reference, and the palette was validated with its checker rather than eyeballed.

**Sentiment is a diverging scale**, not three categorical hues: negative and
positive are the two poles and `neutral` is the gray midpoint. Yellow in the
middle was wrong — in SemEval, `neutral` means the aspect was *mentioned
without being evaluated*, which is an absence of polarity, not a caution state.

The red/green poles measure CVD ΔE 7.2 in light mode (8.6 in dark), inside the
band that is legal **only** when something other than color also carries the
meaning. Every sentiment in this UI is spelled out in text beside its swatch,
and the badge keeps its label in primary ink with a colored dot alongside —
which is also why the label can never be dropped in favour of the dot.

**The confusion matrix** uses the documented single-hue sequential ramp, shaded
by each cell's share of its own row: the gold classes are nowhere near balanced
(341 positive against 128 negative), so shading raw counts would just redraw
the class distribution. Its dark ramp is *reversed*, not brightened — "near
zero" has to recede toward the dark surface too. Every cell prints its count.

**Attention bars all share one hue**: bar length already encodes magnitude, and
coloring them by their own value would spend the identity channel re-encoding
it.

## Notes

**`strict` had to be turned on.** The Vite 8 `react-ts` template ships
`tsconfig.app.json` *without* `"strict": true`. It is enabled now, and the
build type-checks under it. There is no `any` and no non-null assertion in
`src/`.

**Errors are one type.** `api.ts` converts everything — a non-2xx body, a
network failure, CORS — into `ApiRequestError`, a real `Error` that satisfies
the `ApiError` interface. `isApiError()` narrows it, since TypeScript types a
`catch` binding as `unknown`. A request that never reached the server gets
`status: 0`, which keeps "backend is down" distinguishable from any real HTTP
status. The backend answers `{"error": ..., "status": n}` on every failure, so
that is the field `api.ts` reads — but it falls back through `detail` and a raw
string body, because a proxy or an upstream crash can return something else.

**Tailwind only emits class names it finds written out.** Every color lookup
table holds complete class strings, and computed widths are inline styles, for
that reason. The one exception is the sequential ramp, which lives outside
`@theme` as plain custom properties: it is read only through an inline `var()`,
and Tailwind can drop `@theme` variables no utility references.

**Dark mode is a real third choice.** `<html data-theme>` is stamped before the
first paint by an inline script in `index.html`, so a dark-mode user never sees
a white flash; the dark steps are selected for the dark surface rather than
flipped. `system` keeps following the OS live, not only at load. Every
`localStorage` access is wrapped, because it *throws* in some privacy modes.

**The section lives in the URL fragment**, so back works, a reload stays put,
and `#performance` can be linked to. The sections are real navigation, so they
are anchors rather than buttons dressed as tabs. Each page still gets a `key`,
so opening Performance refetches rather than showing a stale snapshot; the
fetch is aborted on unmount.

**Per-class rows iterate a fixed label order** (negative, neutral, positive)
rather than `Object.keys(per_class)`, so row order does not depend on JSON key
order, and a label the report omits still shows a row with dashes.

**The health banner separates two failures that look identical from in here.**
The API being unreachable and the API being up with no model loaded both leave
the app unable to predict, but they need different fixes. `axios` gives the
first a `status` of 0; the second is a 200 with `{"status": "degraded"}`, which
the backend answers deliberately instead of a 503 so the two stay
distinguishable. It renders nothing while healthy — including during its first
request, so a working load never flashes a banner — and re-checks every 5s only
while unhealthy, stopping for good on the first success.

**`prefers-reduced-motion` is honoured**, and focus is a single `:focus-visible`
treatment visible on both surfaces.

## Verified

- `npm run build` clean under `strict`; `npm run lint` clean.
- Against a live backend on the real artifact: all three endpoints, the exact
  `PredictResponse` and `ModelMetaResponse` key sets, `confidence` within
  [0, 1], `per_class` covering all three labels, and `confusion_matrix`
  arriving as `[[109,14,5],[48,95,26],[12,30,299]]`.
- Rendered in a headless browser at 1280px and 390px, light and dark: no
  horizontal overflow at phone width on either screen.
- `aspect_in_text === false` renders a warning *above* the confidence figure,
  since the model is just as confident about an absent aspect as a present one
  — "keyboard" against a sentence about battery and screen returns positive at
  0.9803. Only an explicit `false` warns: `null` means the backend could not
  tell, and a caveat built from "unknown" would cry wolf.
- `scores` and `attention` are both nullable, so `ResultCard` renders those
  sections only when they arrive.
