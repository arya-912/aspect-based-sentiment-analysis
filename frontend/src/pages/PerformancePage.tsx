import { useEffect, useState } from 'react'

import { getModelMeta, isApiError } from '../services/api'
import { SENTIMENTS, type ModelMetaResponse, type PerClassMetrics } from '../types'

/** Format a 0-1 metric as a percentage, or a dash when absent. */
function asPercent(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(1)}%`
}

interface MetricProps {
  label: string
  value: string
}

function Metric({ label, value }: MetricProps) {
  return (
    <div className="rounded-lg border border-gray-300 bg-white p-4">
      <p className="text-xs uppercase tracking-wide text-gray-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-gray-900">{value}</p>
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

  if (loading) {
    return <p className="text-sm text-gray-600">Loading model performance…</p>
  }

  if (error !== null) {
    return (
      <div
        role="alert"
        className="rounded-md border border-red-300 bg-red-50 p-4 text-sm text-red-800"
      >
        <p className="font-medium">Could not load model performance</p>
        <p className="mt-1">{error}</p>
      </div>
    )
  }

  if (meta === null) {
    return <p className="text-sm text-gray-500">No model metadata available.</p>
  }

  const { evaluation } = meta
  // The backend sends per_class keyed by label name. Iterating SENTIMENTS
  // rather than Object.keys keeps the row order fixed (negative, neutral,
  // positive) regardless of JSON key order, and still shows a label that the
  // report happens to omit.
  const perClass: Record<string, PerClassMetrics> = evaluation?.per_class ?? {}

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-semibold text-gray-900">Model performance</h2>
        <p className="mt-1 text-sm text-gray-600">
          Scores from the artifact's held-out evaluation
          {evaluation?.evaluation_split !== undefined && (
            <>
              {' '}
              (<span className="font-mono">{evaluation.evaluation_split}</span>
              {evaluation.n !== undefined && `, ${evaluation.n} examples`})
            </>
          )}
          .
        </p>
      </section>

      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div className="rounded-lg border border-gray-300 bg-white p-4">
          <p className="text-xs uppercase tracking-wide text-gray-500">Version</p>
          <p className="mt-1 font-mono text-sm text-gray-900">{meta.version}</p>
          <p className="mt-1 text-xs text-gray-500">{meta.base_model}</p>
        </div>
        <Metric label="Accuracy" value={asPercent(evaluation?.accuracy)} />
        <Metric label="Macro F1" value={asPercent(evaluation?.macro_f1)} />
      </dl>

      {evaluation === null ? (
        <p className="text-sm text-gray-500">
          This artifact ships without an evaluation report, so per-class scores
          are unavailable.
        </p>
      ) : (
        <section>
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
            Per-class scores
          </h3>
          <table className="w-full border border-gray-300 bg-white text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="border-b border-gray-300 p-2 font-medium">Label</th>
                <th className="border-b border-gray-300 p-2 font-medium">F1</th>
                <th className="border-b border-gray-300 p-2 font-medium">Precision</th>
                <th className="border-b border-gray-300 p-2 font-medium">Recall</th>
                <th className="border-b border-gray-300 p-2 font-medium">Support</th>
              </tr>
            </thead>
            <tbody>
              {SENTIMENTS.map((label) => {
                const scores = perClass[label]
                return (
                  <tr key={label} className="border-b border-gray-200 last:border-b-0">
                    <td className="p-2 capitalize text-gray-900">{label}</td>
                    <td className="p-2 font-medium">{asPercent(scores?.f1)}</td>
                    <td className="p-2 text-gray-700">{asPercent(scores?.precision)}</td>
                    <td className="p-2 text-gray-700">{asPercent(scores?.recall)}</td>
                    <td className="p-2 text-gray-700">{scores?.support ?? '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}
