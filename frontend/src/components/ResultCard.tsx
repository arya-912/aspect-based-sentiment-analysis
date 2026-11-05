import { SENTIMENTS, type PredictResponse, type Sentiment } from '../types'
import AttentionBars from './AttentionBars'
import BarRow from './BarRow'
import SentimentBadge from './SentimentBadge'

interface ResultCardProps {
  result: PredictResponse
}

/**
 * Bar colour per polarity.
 *
 * Written out rather than interpolated, for the same reason as
 * `SentimentBadge`: Tailwind only emits class names it finds literally in the
 * source.
 */
const SCORE_BARS: Record<Sentiment, string> = {
  negative: 'bg-red-400',
  neutral: 'bg-yellow-400',
  positive: 'bg-green-400',
}

/** Format a 0-1 probability as a percentage, or a dash when absent. */
function formatConfidence(confidence: number | null): string {
  if (confidence === null || Number.isNaN(confidence)) return '—'
  return `${(confidence * 100).toFixed(1)}%`
}

/** One prediction, laid out as a bordered card. */
export default function ResultCard({ result }: ResultCardProps) {
  const { aspect, sentiment, confidence, scores, attention, model_version } = result

  // Only an explicit `false` is a warning. `null` means the backend could not
  // tell, and inventing a caveat from "unknown" would cry wolf.
  const aspectMissing = result.aspect_in_text === false

  // Listed in class-id order (negative, neutral, positive) rather than by
  // value, so the rows do not reshuffle between predictions.
  const distribution = scores
    ? SENTIMENTS.filter((label) => typeof scores[label] === 'number')
    : []

  return (
    <div className="rounded-lg border border-gray-300 bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-wide text-gray-500">Aspect</p>
          <p className="text-lg font-semibold text-gray-900">{aspect}</p>
        </div>
        <SentimentBadge sentiment={sentiment} />
      </div>

      {/*
        Above the confidence on purpose. The model answers just as confidently
        for an aspect that is absent as for one that is present, so the caveat
        has to be read before the number, not after it.
      */}
      {aspectMissing && (
        <div
          role="note"
          className="mt-4 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900"
        >
          <p className="font-medium">“{aspect}” does not appear in this review</p>
          <p className="mt-1">
            Every example the model trained on used an aspect term taken
            verbatim from its sentence, so this prediction is outside what it
            learned. The confidence below is not trustworthy here.
          </p>
        </div>
      )}

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

      {scores && distribution.length > 0 && (
        <div className="mt-4 border-t border-gray-200 pt-4">
          <p className="text-xs uppercase tracking-wide text-gray-500">
            Label distribution
          </p>
          <div className="mt-2 space-y-1.5">
            {distribution.map((label) => (
              <BarRow
                key={label}
                label={label}
                value={formatConfidence(scores[label])}
                fraction={scores[label]}
                barClassName={SCORE_BARS[label]}
              />
            ))}
          </div>
        </div>
      )}

      {attention && attention.length > 0 && (
        <div className="mt-4 border-t border-gray-200 pt-4">
          <AttentionBars attention={attention} />
        </div>
      )}
    </div>
  )
}
