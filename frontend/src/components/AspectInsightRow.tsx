import { asPercent } from '../lib/format'
import { SENTIMENTS, type AspectSummary } from '../types'
import AspectShareBar from './AspectShareBar'
import SentimentBadge from './SentimentBadge'
import { ChevronIcon } from './icons'

interface AspectInsightRowProps {
  summary: AspectSummary
  /** Show where the aspect came from. Off for the storefront, where every
   *  aspect is a fixed category sort key and the badge would be noise. */
  showSource?: boolean
}

/** How an aspect was chosen, in words. None of these is the model. */
const SOURCE_LABELS: Record<string, string> = {
  corpus: 'from the corpus vocabulary',
  'corpus-rare': 'rare corpus term',
  given: 'you named it',
}

/**
 * One aspect's verdict, its split, and the sentences behind it.
 *
 * The examples are the point of the row, not decoration: a share of 71%
 * negative is a claim, and the three sentences under it are the evidence for
 * it. They sit in a native `<details>` so the row stays scannable and the
 * evidence is one keystroke away, open-by-default for a mixed aspect -- where
 * the disagreement *is* the finding and hiding it would bury the lede.
 */
export default function AspectInsightRow({
  summary,
  showSource = true,
}: AspectInsightRowProps) {
  const { aspect, counts, shares, total, dominant, mixed, source, examples } = summary

  return (
    <div className="border-b border-line py-3 last:border-b-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <div className="flex items-baseline gap-2">
          <span className="font-medium text-ink">{aspect}</span>
          <span className="text-xs text-muted">
            {total} {total === 1 ? 'sentence' : 'sentences'}
          </span>
          {mixed && (
            <span className="rounded-full border border-line px-1.5 py-0.5 text-[11px] text-ink-2">
              mixed
            </span>
          )}
        </div>
        <SentimentBadge sentiment={dominant} size="sm" />
      </div>

      <div className="mt-2">
        <AspectShareBar shares={shares} counts={counts} aspect={aspect} />
      </div>

      {/*
        The figures in text, beside the bar rather than inside it. Nothing in
        this row is gated behind hovering a segment.
      */}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-ink-2">
        {SENTIMENTS.filter((label) => (counts[label] ?? 0) > 0).map((label) => (
          <span key={label} className="tabular">
            <span className="capitalize">{label}</span> {asPercent(shares[label] ?? 0, 0)}
            <span className="text-muted"> ({counts[label]})</span>
          </span>
        ))}
        {showSource && (
          <span className="text-muted">· {SOURCE_LABELS[source] ?? source}</span>
        )}
      </div>

      {examples.length > 0 && (
        <details className="group mt-2" open={mixed}>
          <summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-medium text-ink-2 hover:text-ink">
            <span className="-rotate-90 transition-transform group-open:rotate-0">
              <ChevronIcon className="h-3.5 w-3.5" />
            </span>
            {examples.length === 1 ? '1 example sentence' : `${examples.length} example sentences`}
          </summary>
          <ul className="mt-2 space-y-2">
            {examples.map((example, index) => (
              <li
                key={`${example.review_index}-${example.sentence}-${index}`}
                className="rounded-lg border border-line bg-plane p-2.5"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="min-w-0 text-sm text-ink">{example.sentence}</p>
                  <SentimentBadge sentiment={example.sentiment} size="sm" />
                </div>
                <p className="mt-1 text-xs text-muted">
                  about “{example.surface}”
                  {example.confidence !== null && (
                    <> · {asPercent(example.confidence)} confidence</>
                  )}{' '}
                  · review {example.review_index + 1}
                </p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}
