import type { TokenAttention } from '../types'
import BarRow from './BarRow'

interface AttentionBarsProps {
  attention: TokenAttention[]
}

/** Format a 0-1 weight as a percentage. */
function formatWeight(weight: number): string {
  if (Number.isNaN(weight)) return '—'
  return `${(weight * 100).toFixed(1)}%`
}

/**
 * Which review words the model weighted for this aspect.
 *
 * This is the one thing that distinguishes the architecture from a plain
 * sentence-pair classifier, and it was computed on every request long before
 * anything displayed it.
 */
export default function AttentionBars({ attention }: AttentionBarsProps) {
  if (attention.length === 0) return null

  // Scale to the heaviest token so the top bar fills its track. Scaling to a
  // full 1.0 instead would leave every bar stubby, because the weights are
  // shares of a distribution spread across the whole review.
  const heaviest = Math.max(...attention.map((token) => token.weight))

  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-gray-500">
        Most attended review words
      </p>
      <div className="mt-2 space-y-1.5">
        {attention.map((token, index) => (
          // Tokens repeat within a sentence ("is" twice is normal), so the
          // index has to be part of the key.
          <BarRow
            key={`${token.token}-${index}`}
            label={token.token}
            value={formatWeight(token.weight)}
            fraction={heaviest > 0 ? token.weight / heaviest : 0}
            barClassName="bg-indigo-400"
          />
        ))}
      </div>
      <p className="mt-2 text-xs text-gray-500">
        Share of the model&rsquo;s attention over the review, for this aspect.
      </p>
    </div>
  )
}
