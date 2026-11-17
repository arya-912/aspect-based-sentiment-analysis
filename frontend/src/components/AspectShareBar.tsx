import { asPercent } from '../lib/format'
import { SENTIMENTS, type Sentiment } from '../types'

interface AspectShareBarProps {
  /** Share per label, 0-1. Missing labels are treated as zero. */
  shares: Partial<Record<Sentiment, number>>
  /** Sentence count per label, for the hover text and the accessible name. */
  counts: Partial<Record<Sentiment, number>>
  /** What the shares are about, for the accessible name. */
  aspect: string
}

/** Fill per polarity. Literal class names: Tailwind only emits what it sees. */
const FILLS: Record<Sentiment, string> = {
  negative: 'bg-neg',
  neutral: 'bg-neu',
  positive: 'bg-pos',
}

/**
 * One aspect's sentiment split, as a segmented bar.
 *
 * A segmented bar rather than three separate bars: the three shares are parts
 * of one whole and sum to 1, which is the relationship worth showing. There is
 * no baseline and no square end, because nothing here grows from zero.
 *
 * **The thin variant, deliberately.** The tall one prints each share inside
 * its own block, which needs the label to clear contrast against the fill --
 * and on this palette it does not reliably: white on the gray midpoint is
 * 3.6:1 and near-black on the red pole is 5.3:1, so the two poles would want
 * opposite inks and neither is comfortable. So the bar stays 8px and wordless,
 * and the figures are printed beside it in text tokens. That also keeps the
 * project's own rule -- text never wears the series color.
 *
 * Segments are separated by a 2px gap in the surface color rather than a
 * border: a stroke would add ink that is not data.
 *
 * Color is never the only channel. Every row prints its counts, the dominant
 * label is spelled out in the badge beside it, each segment carries hover
 * text, and the bar has an accessible name listing all three figures.
 */
export default function AspectShareBar({ shares, counts, aspect }: AspectShareBarProps) {
  const present = SENTIMENTS.filter((label) => (shares[label] ?? 0) > 0)

  const describe = (label: Sentiment) =>
    `${counts[label] ?? 0} ${label} (${asPercent(shares[label] ?? 0, 0)})`

  return (
    <div
      className="flex h-2 w-full gap-0.5 overflow-hidden rounded-full bg-grid"
      role="img"
      aria-label={`${aspect}: ${SENTIMENTS.map(describe).join(', ')}`}
    >
      {present.map((label) => (
        <div
          key={label}
          // A share of 0 draws nothing at all -- `present` filters it out --
          // rather than a sliver that reads as a real quantity.
          style={{ width: `${(shares[label] ?? 0) * 100}%` }}
          className={`h-full first:rounded-l-full last:rounded-r-full ${FILLS[label]}`}
          title={describe(label)}
        />
      ))}
    </div>
  )
}
