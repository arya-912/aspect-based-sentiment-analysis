import { useEffect, useState } from 'react'

import Card from '../components/Card'
import ConfusionMatrix from '../components/ConfusionMatrix'
import MetricTile from '../components/MetricTile'
import PerClassTable from '../components/PerClassTable'
import { OfflineIcon } from '../components/icons'
import { asCount, asPercent } from '../lib/format'
import { getModelMeta, isApiError } from '../services/api'
import type { ModelMetaResponse, PerClassMetrics } from '../types'

/** Placeholder tiles while the request is in flight. */
function LoadingSkeleton() {
  return (
    <div className="space-y-6" role="status" aria-label="Loading model performance">
      <div className="grid gap-4 sm:grid-cols-3">
        {[0, 1, 2].map((key) => (
          <Card key={key} className="p-4">
            <div className="h-3 w-20 rounded bg-grid" />
            <div className="mt-3 h-7 w-24 rounded bg-grid" />
          </Card>
        ))}
      </div>
      <Card className="p-5">
        <div className="h-3 w-28 rounded bg-grid" />
        <div className="mt-4 space-y-3">
          {[0, 1, 2].map((key) => (
            <div key={key} className="h-1.5 w-full rounded bg-grid" />
          ))}
        </div>
      </Card>
    </div>
  )
}

/** Model metadata and evaluation metrics, fetched once on mount. */
export default function PerformancePage() {
  const [meta, setMeta] = useState<ModelMetaResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Aborted on unmount so switching tabs mid-request cannot set state on an
    // unmounted component, and so a stale response cannot overwrite a newer one.
    const controller = new AbortController()

    getModelMeta(controller.signal)
      .then((data) => {
        setMeta(data)
        setError(null)
      })
      .catch((caught: unknown) => {
        if (controller.signal.aborted) return
        setError(
          isApiError(caught) ? caught.message : 'Could not load model performance.',
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })

    return () => controller.abort()
  }, [])

  if (loading) return <LoadingSkeleton />

  if (error !== null) {
    return (
      <Card
        role="alert"
        className="flex gap-3 border-critical/40 bg-critical/10 p-4 text-sm"
      >
        <span className="mt-0.5 text-critical">
          <OfflineIcon />
        </span>
        <div>
          <p className="font-medium text-ink">Could not load model performance</p>
          <p className="mt-1 text-ink-2">{error}</p>
        </div>
      </Card>
    )
  }

  if (meta === null) {
    return <p className="text-sm text-ink-2">No model metadata available.</p>
  }

  const { evaluation } = meta
  // The backend sends per_class keyed by label name.
  const perClass: Record<string, PerClassMetrics> = evaluation?.per_class ?? {}
  const split =
    evaluation?.evaluation_split === undefined
      ? undefined
      : `${evaluation.evaluation_split}${
          evaluation.n === undefined ? '' : `, ${asCount(evaluation.n)} examples`
        }`

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-base font-semibold text-ink">Model performance</h2>
        <p className="mt-1 text-sm text-ink-2">
          Scores from the artifact&rsquo;s held-out evaluation
          {split === undefined ? '' : ' '}
          {split !== undefined && <span className="font-mono text-xs">({split})</span>}.
        </p>
      </section>

      {/*
        Macro F1 leads, and is the emphasized figure. It is the metric this
        model was actually selected on, and the honest one here: `neutral` is
        the minority class, so accuracy can climb while the model quietly gives
        up on it.
      */}
      <div className="grid gap-4 sm:grid-cols-3">
        <MetricTile
          label="Macro F1"
          value={asPercent(evaluation?.macro_f1)}
          detail="Unweighted mean across classes"
          emphasis
        />
        <MetricTile label="Accuracy" value={asPercent(evaluation?.accuracy)} />
        <MetricTile
          label="Weighted F1"
          value={asPercent(evaluation?.weighted_f1)}
          detail="Weighted by class support"
        />
      </div>

      <Card className="p-5">
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Serving
        </p>
        <dl className="mt-3 grid gap-4 text-sm sm:grid-cols-3">
          <div>
            <dt className="text-muted">Artifact</dt>
            <dd className="mt-0.5 font-mono text-ink">{meta.version}</dd>
          </div>
          <div>
            <dt className="text-muted">Base model</dt>
            <dd className="mt-0.5 font-mono text-ink">{meta.base_model}</dd>
          </div>
          <div>
            <dt className="text-muted">Task</dt>
            <dd className="mt-0.5 text-ink">{meta.task}</dd>
          </div>
        </dl>
      </Card>

      {evaluation === null ? (
        <Card className="p-5">
          <p className="text-sm text-ink-2">
            This artifact ships without an evaluation report, so per-class scores are
            unavailable.
          </p>
        </Card>
      ) : (
        <>
          <Card className="p-5">
            <h3 className="mb-4 text-sm font-semibold text-ink">Per-class scores</h3>
            <PerClassTable perClass={perClass} />
          </Card>

          {evaluation.confusion_matrix !== undefined && (
            <Card className="p-5">
              <h3 className="text-sm font-semibold text-ink">Where the errors go</h3>
              <p className="mt-1 mb-4 text-sm text-ink-2">
                Accuracy says how often the model is wrong. This says what it is wrong
                about.
              </p>
              <ConfusionMatrix matrix={evaluation.confusion_matrix} />
            </Card>
          )}
        </>
      )}
    </div>
  )
}
