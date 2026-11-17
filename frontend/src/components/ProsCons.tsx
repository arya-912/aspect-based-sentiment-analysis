import type { AspectSummary } from '../types'

interface ProsConsProps {
  aspects: AspectSummary[]
  /** Minimum sentences before an aspect may be called a pro or a con. */
  minSentences?: number
}

/** The share at which an aspect counts as praised or complained about. */
const THRESHOLD = 0.6

/**
 * The one-line verdict: what this product is praised and complained about for.
 *
 * Built from the same aspect shares the bars show, not from a second
 * analysis, so the sentence and the chart can never disagree.
 *
 * Two deliberate restraints. An aspect needs `minSentences` of evidence
 * before it is named at all -- one positive sentence is not a selling point.
 * And an aspect whose shares are genuinely split appears in neither list,
 * because calling it a pro would be as wrong as calling it a con; the bars
 * below still show it, and the `mixed` tag marks it.
 */
export default function ProsCons({ aspects, minSentences = 2 }: ProsConsProps) {
  const eligible = aspects.filter((aspect) => aspect.total >= minSentences)

  const praised = eligible
    .filter((aspect) => (aspect.shares.positive ?? 0) >= THRESHOLD)
    .sort((a, b) => (b.shares.positive ?? 0) - (a.shares.positive ?? 0))
    .map((aspect) => aspect.aspect)

  const complained = eligible
    .filter((aspect) => (aspect.shares.negative ?? 0) >= THRESHOLD)
    .sort((a, b) => (b.shares.negative ?? 0) - (a.shares.negative ?? 0))
    .map((aspect) => aspect.aspect)

  if (praised.length === 0 && complained.length === 0) {
    return (
      <p className="text-sm text-ink-2">
        No aspect here is clearly praised or criticised &mdash; the reviews are
        split or too few to call.
      </p>
    )
  }

  return (
    <p className="text-sm text-ink">
      {praised.length > 0 && (
        <>
          <span className="font-medium">Praised:</span> {praised.join(', ')}.{' '}
        </>
      )}
      {complained.length > 0 && (
        <>
          <span className="font-medium">Complained about:</span> {complained.join(', ')}.
        </>
      )}
    </p>
  )
}
