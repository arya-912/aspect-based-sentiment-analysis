import { SENTIMENTS } from '../types'

/** Fill per polarity, written out for Tailwind's scanner. */
const DOTS = {
  negative: 'bg-neg',
  neutral: 'bg-neu',
  positive: 'bg-pos',
} as const

/**
 * The key for the share bars.
 *
 * Present once per group of bars rather than once per bar: three series need a
 * legend, and repeating it on every row would be the same information eleven
 * times. The dot carries the color and the text carries the name, which is the
 * only arrangement that survives a colorblind reader.
 */
export default function SentimentLegend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {SENTIMENTS.map((label) => (
        <li key={label} className="flex items-center gap-1.5 text-xs text-ink-2">
          <span className={`h-2 w-2 rounded-full ${DOTS[label]}`} aria-hidden="true" />
          <span className="capitalize">{label}</span>
        </li>
      ))}
    </ul>
  )
}
