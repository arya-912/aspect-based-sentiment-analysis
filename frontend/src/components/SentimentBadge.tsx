import type { Sentiment } from '../types'

interface SentimentBadgeProps {
  sentiment: Sentiment
  /** `lg` for the headline badge on a result, `sm` inline in dense lists. */
  size?: 'sm' | 'lg'
}

/**
 * Tint and dot color per polarity.
 *
 * Written out rather than built by interpolation because Tailwind scans source
 * for literal class names; a template string like `bg-${token}/12` is not in
 * the output CSS and silently renders unstyled.
 *
 * Note what is *not* here: a text color. The label stays in primary ink and the
 * solid dot beside it carries the identity. That is the data-viz rule -- text
 * wears text tokens, never the series color -- and it also fixes a real
 * contrast problem, since the pole colors clear 3:1 as marks but not the 4.5:1
 * that colored body text would need.
 */
const STYLES: Record<Sentiment, { chip: string; dot: string }> = {
  negative: { chip: 'border-neg/35 bg-neg/10', dot: 'bg-neg' },
  neutral: { chip: 'border-neu/35 bg-neu/10', dot: 'bg-neu' },
  positive: { chip: 'border-pos/35 bg-pos/10', dot: 'bg-pos' },
}

const SIZES = {
  sm: 'gap-1.5 px-2 py-0.5 text-xs',
  lg: 'gap-2 px-3 py-1 text-sm',
} as const

const DOTS = {
  sm: 'h-1.5 w-1.5',
  lg: 'h-2 w-2',
} as const

/**
 * A pill naming a sentiment.
 *
 * The label is always spelled out. That is not only a preference: the red and
 * green poles sit at CVD deltaE 7.2 in light mode, inside the band that is legal
 * only when something other than color also carries the meaning. The text is
 * that something, so it cannot be dropped in favour of the dot alone.
 */
export default function SentimentBadge({ sentiment, size = 'lg' }: SentimentBadgeProps) {
  const { chip, dot } = STYLES[sentiment]

  return (
    <span
      className={`inline-flex items-center rounded-full border font-medium capitalize text-ink ${chip} ${SIZES[size]}`}
    >
      <span className={`rounded-full ${dot} ${DOTS[size]}`} aria-hidden="true" />
      {sentiment}
    </span>
  )
}
