import { asPercent } from '../lib/format'
import type { TokenAttention } from '../types'
import BarRow from './BarRow'

interface AttentionBarsProps {
  attention: TokenAttention[]
}

/**
 * The exact attention weights, heaviest first.
 *
 * Companion to {@link HighlightedSentence}, not a replacement for it: the
 * highlight shows *where* the weight fell, this shows *how much*. It is also
 * the relief channel that lets the highlight use color -- the figures are
 * readable here without interpreting a tint, and a word repeated in the review
 * gets one row per occurrence the API reported, which the highlight cannot
 * distinguish.
 *
 * All bars share one hue. Length already encodes magnitude, and coloring them
 * by their own value would spend the identity channel re-encoding it.
 */
export default function AttentionBars({ attention }: AttentionBarsProps) {
  if (attention.length === 0) return null

  // Scale to the heaviest token so the top bar fills its track. Scaling to a
  // full 1.0 instead would leave every bar stubby, because the weights are
  // shares of a distribution spread across the whole review.
  const heaviest = Math.max(...attention.map((token) => token.weight))

  return (
    <div className="space-y-1.5">
      {attention.map((token, index) => (
        // Tokens repeat within a sentence ("is" twice is normal), so the
        // index has to be part of the key.
        <BarRow
          key={`${token.token}-${index}`}
          label={token.token}
          value={asPercent(token.weight)}
          fraction={heaviest > 0 ? token.weight / heaviest : 0}
          barClassName="bg-attn"
          labelClassName="w-24"
        />
      ))}
    </div>
  )
}
