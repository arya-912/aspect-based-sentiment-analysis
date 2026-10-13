import type { Sentiment } from '../types'

interface SentimentBadgeProps {
  sentiment: Sentiment
}

/**
 * Tailwind classes per polarity.
 *
 * Written out rather than built by interpolation because Tailwind scans source
 * for literal class names; a template string like `bg-${color}-100` is not in
 * the output CSS and silently renders unstyled.
 */
const STYLES: Record<Sentiment, string> = {
  positive: 'bg-green-100 text-green-800 border-green-300',
  neutral: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  negative: 'bg-red-100 text-red-800 border-red-300',
}

/**
 * A coloured pill naming a sentiment.
 *
 * The label is always spelled out: colour alone would be invisible to a
 * colour-blind or screen-reader user.
 */
export default function SentimentBadge({ sentiment }: SentimentBadgeProps) {
  return (
    <span
      className={`inline-block rounded-full border px-3 py-1 text-sm font-medium capitalize ${STYLES[sentiment]}`}
    >
      {sentiment}
    </span>
  )
}
