import { useState } from 'react'

import ResultCard from '../components/ResultCard'
import ReviewForm from '../components/ReviewForm'
import type { PredictResponse } from '../types'

/**
 * The predict screen: a form, and the latest result beneath it.
 *
 * Validation and API errors are surfaced by {@link ReviewForm} itself, next to
 * the fields that caused them. This page owns only the result.
 */
export default function AnalysisPage() {
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-lg font-semibold text-gray-900">Analyze a review</h2>
        <p className="mt-1 text-sm text-gray-600">
          Enter a sentence and an aspect term inside it. The model predicts the
          sentiment expressed about that aspect specifically — the same sentence
          can be positive about one aspect and negative about another.
        </p>
      </section>

      <ReviewForm onResult={setResult} onLoadingChange={setLoading} />

      <section className="border-t border-gray-200 pt-6">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Result
        </h3>

        {loading ? (
          <p className="text-sm text-gray-600">Analyzing…</p>
        ) : result === null ? (
          <p className="text-sm text-gray-500">
            No result yet. Submit a review above to see a prediction.
          </p>
        ) : (
          <ResultCard result={result} />
        )}
      </section>
    </div>
  )
}
