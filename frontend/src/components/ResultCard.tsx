import type { PredictResponse } from '../types'
import SentimentBadge from './SentimentBadge'

interface ResultCardProps {
  result: PredictResponse
}

/** Format a 0-1 probability as a percentage, or a dash when absent. */
function formatConfidence(confidence: number | null): string {
  if (confidence === null || Number.isNaN(confidence)) return '—'
  return `${(confidence * 100).toFixed(1)}%`
}

/** One prediction, laid out as a bordered card. */
export default function ResultCard({ result }: ResultCardProps) {
  const { aspect, sentiment, confidence, model_version } = result

  return (
    <div className="rounded-lg border border-gray-300 bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-gray-500">Aspect</p>
          <p className="text-lg font-semibold text-gray-900">{aspect}</p>
        </div>
        <SentimentBadge sentiment={sentiment} />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-4 border-t border-gray-200 pt-4 text-sm">
        <div>
          <dt className="text-gray-500">Confidence</dt>
          <dd className="font-medium text-gray-900">{formatConfidence(confidence)}</dd>
        </div>
        <div>
          <dt className="text-gray-500">Model</dt>
          <dd className="font-mono text-gray-900">{model_version}</dd>
        </div>
      </dl>
    </div>
  )
}
