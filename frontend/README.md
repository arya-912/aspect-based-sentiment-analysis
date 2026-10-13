# ABSA frontend

React + TypeScript (Vite) UI over the ABSA backend. Two screens: analyze a
review, and view the served model's evaluation metrics.

Functional, not polished — layout is plain Tailwind, with no animations or
responsive work. That is deliberate at this stage.

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

## Layout

```
src/
├── types/index.ts            shapes mirroring the backend schemas
├── services/api.ts           axios client; throws a typed error
├── components/
│   ├── SentimentBadge.tsx    coloured pill, always with its text label
│   ├── ResultCard.tsx        one prediction
│   └── ReviewForm.tsx        text + aspect input, validation, submit
├── pages/
│   ├── AnalysisPage.tsx      form + latest result
│   └── PerformancePage.tsx   model metadata + per-class metrics table
├── App.tsx                   tab toggle (useState, no router)
└── main.tsx                  mount
```

## Notes

**`strict` had to be turned on.** The Vite 8 `react-ts` template ships
`tsconfig.app.json` *without* `"strict": true` — only `noUnusedLocals` and
`noUnusedParameters`. It is enabled now, and the build type-checks under it.
There is no `any` and no non-null assertion in `src/`.

**Errors are one type.** `api.ts` converts everything — a non-2xx body, a
network failure, CORS — into `ApiRequestError`, a real `Error` that satisfies
the `ApiError` interface so a `catch` gets both a stack and typed fields.
`isApiError()` narrows it, since TypeScript types a `catch` binding as
`unknown`. A request that never reached the server gets `status: 0`, which
keeps "backend is down" distinguishable from any real HTTP status.

The backend answers `{"error": "...", "status": n}` on every failure, so that
is the field `api.ts` reads — but it falls back through `detail` and a raw
string body, because a proxy or an upstream crash can return something else
entirely.

**Badge colours are written out, not interpolated.** Tailwind scans source for
literal class names, so `bg-${color}-100` compiles to nothing and renders
unstyled. The lookup table in `SentimentBadge.tsx` holds complete class
strings for that reason.

**The sentiment is always spelled out**, never conveyed by colour alone.

**Per-class rows iterate a fixed label order** (negative, neutral, positive)
rather than `Object.keys(per_class)`, so row order does not depend on JSON key
order, and a label the report happens to omit still shows a row with dashes.

**The Performance page refetches when its tab is opened.** `App.tsx` gives each
page a `key`, so switching tabs remounts rather than showing a stale snapshot.
The fetch is aborted on unmount, so a mid-flight request cannot set state after
the component is gone.

## Verified

- `npm run build` — clean under `strict`
- `npm run lint` — clean
- Against a live backend: all three endpoints, the exact `PredictResponse` and
  `ModelMetaResponse` key sets, `confidence` within [0, 1], `per_class` covering
  all three labels, the 422 and 404 error bodies `api.ts` parses, and the CORS
  header for `localhost:5173`.
