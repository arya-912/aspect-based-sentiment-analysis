import { useEffect, useState } from 'react'

import Card from '../components/Card'
import ConfusionMatrix from '../components/ConfusionMatrix'
import EmptyState from '../components/EmptyState'
import MetricTile from '../components/MetricTile'
import PerClassTable from '../components/PerClassTable'
import SectionHeader from '../components/SectionHeader'
import SentimentLegend from '../components/SentimentLegend'
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
      <div role="alert">
        <EmptyState
          title="Could not load model performance"
          tone="critical"
          icon={<OfflineIcon />}
        >
          {error}
        </EmptyState>
      </div>
    )
  }

  if (meta === null) {
    return (
      <EmptyState title="No model metadata available">
        The server answered, but without the fields this page reads. Check that
        an artifact is loaded.
      </EmptyState>
    )
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
      <SectionHeader
        as="h2"
        title="Model performance"
        description={
          <>
            Scores from the artifact&rsquo;s held-out evaluation
            {split === undefined ? '' : ' '}
            {split !== undefined && (
              <span className="font-mono text-xs">({split})</span>
            )}
            . Served exactly as the artifact recorded them.
          </>
        }
      />

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
        <SectionHeader title="Serving" />
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
        <EmptyState title="This artifact ships without an evaluation report">
          Per-class scores and the confusion matrix come from the artifact&rsquo;s
          own <span className="font-mono">metrics.json</span>. Run{' '}
          <span className="font-mono">python train.py evaluate</span> to produce one.
        </EmptyState>
      ) : (
        <>
          <Card className="p-5">
            <SectionHeader
              title="Per-class scores"
              description="neutral is the weakest class in every run here, and the one macro F1 exists to keep visible."
              aside={<SentimentLegend />}
            />
            <div className="mt-4">
              <PerClassTable perClass={perClass} />
            </div>
          </Card>

          {evaluation.confusion_matrix !== undefined && (
            <Card className="p-5">
              <SectionHeader
                title="Where the errors go"
                description="Accuracy says how often the model is wrong. This says what it is wrong about."
              />
              <div className="mt-4">
                <ConfusionMatrix matrix={evaluation.confusion_matrix} />
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
