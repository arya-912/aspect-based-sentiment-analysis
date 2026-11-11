import { attentionByWord, splitWords } from '../lib/words'
import type { TokenAttention } from '../types'

interface HighlightedSentenceProps {
  text: string
  attention: TokenAttention[]
}

interface Segment {
  text: string
  /** Attention weight, or null for punctuation, spacing and unattended words. */
  weight: number | null
}

/**
 * Split the review into word and non-word runs, tagging each word with its weight.
 *
 * The gaps matter: dropping them and joining words with a space would rewrite
 * the user's punctuation, and the sentence on screen has to be the sentence
 * they submitted.
 */
function toSegments(text: string, attention: TokenAttention[]): Segment[] {
  const weights = attentionByWord(attention)
  const words = splitWords(text)
  const segments: Segment[] = []
  let cursor = 0

  for (const word of words) {
    if (word.start > cursor) {
      segments.push({ text: text.slice(cursor, word.start), weight: null })
    }
    segments.push({ text: word.text, weight: weights.get(word.text.toLowerCase()) ?? null })
    cursor = word.end
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), weight: null })

  return segments
}

/**
 * Highest alpha any highlight reaches.
 *
 * Held well below opaque so primary ink stays legible on top in both themes --
 * the fill is behind running text, not a standalone mark, so readability of the
 * word beats saturation of the cue.
 */
const MAX_ALPHA = 48

/**
 * The review with the model's attention painted onto it.
 *
 * This is the display the architecture was built for. The attention is computed
 * *over the review's own tokens*, so showing it as a detached bar chart of
 * token names throws away the one thing that makes it readable: where in the
 * sentence the weight fell. Two aspects of the same sentence light up different
 * words, and that is visible at a glance here and nowhere else.
 *
 * Magnitude is carried by fill alpha on a single hue -- a sequential encoding,
 * since there is no length channel available inside running text. Alpha alone
 * would be color carrying meaning on its own, so every weight is also printed
 * in the exact-weights list beside this, and each highlighted word states its
 * own figure on hover.
 */
export default function HighlightedSentence({ text, attention }: HighlightedSentenceProps) {
  const segments = toSegments(text, attention)
  // Scale to the heaviest word rather than to 1.0: the weights are shares of a
  // distribution spread over the whole review, so against a full scale even the
  // top word would be a barely-tinted smudge.
  const heaviest = Math.max(...attention.map((token) => token.weight), 0)

  return (
    <p className="whitespace-pre-wrap text-[15px]/relaxed text-ink">
      {segments.map((segment, index) => {
        if (segment.weight === null || heaviest <= 0) {
          return <span key={index}>{segment.text}</span>
        }

        const share = segment.weight / heaviest
        const alpha = Math.round(share * MAX_ALPHA)
        const percent = `${(segment.weight * 100).toFixed(1)}%`

        return (
          <span
            key={index}
            title={`${segment.text}: ${percent} of this aspect's attention`}
            className="rounded-[3px] px-0.5 font-medium"
            style={{
              backgroundColor: `color-mix(in oklab, var(--color-attn) ${alpha}%, transparent)`,
            }}
          >
            {segment.text}
          </span>
        )
      })}
    </p>
  )
}
