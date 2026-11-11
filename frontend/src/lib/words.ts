/**
 * Splitting a review into word spans, and lining the model's attention up
 * against them.
 */

import type { TokenAttention } from '../types'

export interface WordSpan {
  /** The word exactly as it appears in the review, original case kept. */
  text: string
  /** Character offsets into the review, so a span can be sliced back out verbatim. */
  start: number
  end: number
}

/**
 * Letters and digits, with internal apostrophes and hyphens kept inside the
 * word -- so `hard-drive` and `doesn't` stay single words rather than
 * fragmenting into pieces nobody would click.
 *
 * Unicode property escapes rather than `\w`, which is ASCII-only and would
 * split any accented word in the corpus.
 */
const WORD_PATTERN = /[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu

/**
 * The review's words, with their offsets.
 *
 * Offsets are the point of this: an aspect built by slicing the original
 * string is guaranteed to be a verbatim span of the review, which is exactly
 * the regime the model was trained on. Every training example took its aspect
 * from a literal span of its sentence.
 */
export function splitWords(text: string): WordSpan[] {
  const spans: WordSpan[] = []
  for (const match of text.matchAll(WORD_PATTERN)) {
    // `matchAll` on a `g` regex always reports an index; the guard is for the
    // type, not for a case that happens.
    if (match.index === undefined) continue
    spans.push({ text: match[0], start: match.index, end: match.index + match[0].length })
  }
  return spans
}

/** The verbatim slice of `text` covering the words from `from` to `to` inclusive. */
export function sliceSpan(text: string, words: WordSpan[], from: number, to: number): string {
  const lo = Math.min(from, to)
  const hi = Math.max(from, to)
  const first = words[lo]
  const last = words[hi]
  if (first === undefined || last === undefined) return ''
  return text.slice(first.start, last.end)
}

/**
 * Attention weight per word, keyed by the lowercased word.
 *
 * **The API reports weights per word, not per position.** The backend merges
 * WordPiece fragments into whole words and returns the heaviest few sorted by
 * weight, so when a word occurs twice in a review ("is" twice is entirely
 * normal) two entries arrive with the same `token` and different weights, and
 * nothing says which occurrence each belongs to.
 *
 * So every occurrence of a repeated word is shown at the highest weight
 * reported for it. That is the only defensible choice available here -- the
 * alternative is to guess an order the response does not carry -- and it is
 * why the highlight is an orientation aid with the exact figures listed
 * beside it, rather than the authoritative display on its own.
 */
export function attentionByWord(attention: TokenAttention[]): Map<string, number> {
  const weights = new Map<string, number>()
  for (const { token, weight } of attention) {
    if (Number.isNaN(weight)) continue
    const key = token.toLowerCase()
    const existing = weights.get(key)
    if (existing === undefined || weight > existing) weights.set(key, weight)
  }
  return weights
}

/**
 * Whether `aspect` occurs in `text` as a contiguous run of whole words.
 *
 * Only ever used to decide whether to *offer* the "not in the review" hint
 * before a request is sent. The backend's own `aspect_in_text` is the
 * authority after one: it compares token ids, so it sees through casing,
 * punctuation and possessives, and it knows whether truncation cut the
 * occurrence off. This is a cheap pre-check, never a contradiction of that.
 */
export function looksPresent(text: string, aspect: string): boolean {
  const needle = aspect.trim().toLowerCase()
  if (needle === '') return true
  return text.toLowerCase().includes(needle)
}
