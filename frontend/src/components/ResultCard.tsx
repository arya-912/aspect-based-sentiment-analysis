import { asPercent } from '../lib/format'
import { SENTIMENTS, type PredictResponse, type Sentiment } from '../types'
import AttentionBars from './AttentionBars'
import BarRow from './BarRow'
import Card from './Card'
import HighlightedSentence from './HighlightedSentence'
import SentimentBadge from './SentimentBadge'
import { ChevronIcon, WarningIcon } from './icons'

interface ResultCardProps {
  result: PredictResponse
  /** The review this prediction was made about, for the attention overlay. */
  text: string
}

/**
 * Bar color per polarity: the two diverging poles and the gray midpoint.
 *
 * Written out rather than interpolated, for the same reason as
 * `SentimentBadge`: Tailwind only emits class names it finds literally in the
 * source.
 */
const SCORE_BARS: Record<Sentiment, string> = {
  negative: 'bg-neg',
  neutral: 'bg-neu',
  positive: 'bg-pos',
}

/** One prediction about one aspect. */
export default function ResultCard({ result, text }: ResultCardProps) {
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
    <Card className="overflow-hidden">
      <div className="flex items-start justify-between gap-4 p-4">
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-wide text-muted">Aspect</p>
          <p className="truncate text-lg font-semibold text-ink" title={aspect}>
            {aspect}
          </p>
        </div>
        <SentimentBadge sentiment={sentiment} />
      </div>

      {/*
        Above the confidence on purpose. The model answers just as confidently
        for an aspect that is absent as for one that is present, so the caveat
        has to be read before the number, not after it.
      */}
      {aspectMissing && (
        <div role="note" className="mx-4 mb-4 flex gap-2.5 rounded-lg border border-warning/40 bg-warning/10 p-3">
          <span className="mt-0.5 text-warning">
            <WarningIcon />
          </span>
          <div className="text-sm text-ink">
            <p className="font-medium">“{aspect}” does not appear in this review</p>
            <p className="mt-1 text-ink-2">
              Every example the model trained on used an aspect term taken verbatim
              from its sentence, so this prediction is outside what it learned. The
              confidence below is not trustworthy here.
            </p>
          </div>
        </div>
      )}

      <dl className="grid grid-cols-2 gap-4 border-t border-line px-4 py-3 text-sm">
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted">Confidence</dt>
          <dd className="tabular mt-0.5 text-xl font-semibold text-ink">
            {asPercent(confidence)}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted">Model</dt>
          <dd className="mt-0.5 truncate font-mono text-sm text-ink-2" title={model_version}>
            {model_version}
          </dd>
        </div>
      </dl>

      {scores && distribution.length > 0 && (
        <div className="border-t border-line px-4 py-3">
          <p className="mb-2 text-xs uppercase tracking-wide text-muted">
            Label distribution
          </p>
          <div className="space-y-1.5">
            {distribution.map((label) => (
              <BarRow
                key={label}
                label={label}
                value={asPercent(scores[label])}
                fraction={scores[label]}
                barClassName={SCORE_BARS[label]}
                labelClassName="w-20"
              />
            ))}
          </div>
        </div>
      )}

      {attention && attention.length > 0 && (
        <div className="border-t border-line px-4 py-3">
          <p className="mb-2 text-xs uppercase tracking-wide text-muted">
            What the model looked at
          </p>
          <HighlightedSentence text={text} attention={attention} />

          {/*
            Native <details> rather than a state-driven panel: it is keyboard
            accessible and open-by-URL-fragment searchable for free, and the
            exact figures are a detail of the highlight above rather than a
            competing view of it.
          */}
          <details className="group mt-3">
            <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-ink-2 hover:text-ink">
              <span className="transition-transform group-open:rotate-0 -rotate-90">
                <ChevronIcon className="h-3.5 w-3.5" />
              </span>
              Exact weights
            </summary>
            <div className="mt-2">
              <AttentionBars attention={attention} />
              <p className="mt-2 text-xs text-muted">
                Share of this aspect&rsquo;s attention over the review. Weights are
                reported per word, not per position, so a word occurring twice is
                highlighted above at the higher of its two figures.
              </p>
            </div>
          </details>
        </div>
      )}
    </Card>
  )
}
