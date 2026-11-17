/**
 * `lib/words.ts` -- word spans, attention alignment, and the presence rule.
 *
 * `looksPresent` and `stemWord` are the frontend's copy of the backend's
 * 422 gate (`services/aspect_match.py`). The two must agree: the UI blocks a
 * request the server would refuse, so a disagreement means either a rejection
 * the user cannot understand or a request that dies at the API. The parity
 * table at the bottom is generated from the Python rules and is the real
 * guard; the cases above it document the intent.
 */

import { describe, expect, it } from 'vitest'

import { attentionByWord, looksPresent, sliceSpan, splitWords, stemWord } from './words'

describe('splitWords', () => {
  it('reports each word with offsets that slice it back out verbatim', () => {
    const text = 'The screen is dim.'
    const words = splitWords(text)
    expect(words.map((w) => w.text)).toEqual(['The', 'screen', 'is', 'dim'])
    for (const word of words) {
      expect(text.slice(word.start, word.end)).toBe(word.text)
    }
  })

  it('keeps an internal hyphen or apostrophe inside one word', () => {
    expect(splitWords("the hard-drive doesn't spin").map((w) => w.text)).toEqual([
      'the',
      'hard-drive',
      "doesn't",
      'spin',
    ])
  })

  it('keeps a typographic apostrophe inside the word too', () => {
    expect(splitWords('the screen’s brightness').map((w) => w.text)).toEqual([
      'the',
      'screen’s',
      'brightness',
    ])
  })

  it('splits accented words whole rather than at the accent', () => {
    // The reason for \p{L} over \w: ASCII \w would cut "café" into "caf".
    expect(splitWords('the café was loud').map((w) => w.text)).toEqual([
      'the',
      'café',
      'was',
      'loud',
    ])
  })

  it('keeps digits, which carry real aspects like 500GB', () => {
    expect(splitWords('a 500GB hard drive').map((w) => w.text)).toEqual([
      'a',
      '500GB',
      'hard',
      'drive',
    ])
  })

  it('returns nothing for text with no word characters', () => {
    expect(splitWords('   ...!  ')).toEqual([])
    expect(splitWords('')).toEqual([])
  })
})

describe('sliceSpan', () => {
  const text = 'It comes with a 500GB hard drive and a battery.'
  const words = splitWords(text)

  it('returns the review’s own characters, punctuation and all', () => {
    const from = words.findIndex((w) => w.text === 'hard')
    expect(sliceSpan(text, words, from, from + 1)).toBe('hard drive')
  })

  it('is a verbatim span, so a picked aspect is always in the review', () => {
    const from = words.findIndex((w) => w.text === '500GB')
    const span = sliceSpan(text, words, from, from + 2)
    expect(span).toBe('500GB hard drive')
    expect(text).toContain(span)
  })

  it('accepts its bounds in either order', () => {
    const a = words.findIndex((w) => w.text === 'hard')
    expect(sliceSpan(text, words, a + 1, a)).toBe(sliceSpan(text, words, a, a + 1))
  })

  it('returns one word when both bounds are the same', () => {
    const i = words.findIndex((w) => w.text === 'battery')
    expect(sliceSpan(text, words, i, i)).toBe('battery')
  })

  it('returns empty string for out-of-range bounds instead of throwing', () => {
    expect(sliceSpan(text, words, 0, 999)).toBe('')
    expect(sliceSpan(text, words, -1, 0)).toBe('')
    expect(sliceSpan(text, [], 0, 0)).toBe('')
  })
})

describe('attentionByWord', () => {
  it('keys on the lowercased word', () => {
    const weights = attentionByWord([{ token: 'Amazing', weight: 0.3 }])
    expect(weights.get('amazing')).toBe(0.3)
  })

  it('shows a repeated word at the highest weight reported for it', () => {
    // The API reports per word, not per position: two entries arrive for "is"
    // and nothing says which occurrence each belongs to.
    const weights = attentionByWord([
      { token: 'is', weight: 0.16 },
      { token: 'fast', weight: 0.13 },
      { token: 'is', weight: 0.04 },
    ])
    expect(weights.get('is')).toBe(0.16)
    expect(weights.get('fast')).toBe(0.13)
  })

  it('takes the maximum regardless of the order the entries arrive in', () => {
    const ascending = attentionByWord([
      { token: 'is', weight: 0.04 },
      { token: 'is', weight: 0.16 },
    ])
    expect(ascending.get('is')).toBe(0.16)
  })

  it('skips NaN weights rather than poisoning the map', () => {
    const weights = attentionByWord([
      { token: 'good', weight: Number.NaN },
      { token: 'bad', weight: 0.2 },
    ])
    expect(weights.has('good')).toBe(false)
    expect(weights.get('bad')).toBe(0.2)
  })

  it('is empty for an empty list', () => {
    expect(attentionByWord([]).size).toBe(0)
  })
})

describe('stemWord', () => {
  it('lowercases', () => {
    expect(stemWord('Screen')).toBe('screen')
  })

  it.each([
    ["screen's", 'screen'],
    ['screen’s', 'screen'],
  ])('strips the possessive %s', (word, expected) => {
    expect(stemWord(word)).toBe(expected)
  })

  it.each([
    ['batteries', 'battery'],
    ['screens', 'screen'],
    ['drives', 'drive'],
  ])('depluralises %s', (word, expected) => {
    expect(stemWord(word)).toBe(expected)
  })

  it.each([
    ['boxes', 'box'],
    ['dishes', 'dish'],
    ['glasses', 'glass'],
  ])('handles the -es forms: %s', (word, expected) => {
    expect(stemWord(word)).toBe(expected)
  })

  it.each([
    // The length guards: these are not plurals and must keep their s.
    ['OS', 'os'],
    ['bus', 'bus'],
    ['this', 'this'],
    ['is', 'is'],
    ['us', 'us'],
  ])('leaves %s intact', (word, expected) => {
    expect(stemWord(word)).toBe(expected)
  })

  it.each([
    // No -ing/-ed rules: over-stemming would let an aspect match a word that
    // is not it, which is worse than a visible rejection.
    ['priced', 'priced'],
    ['pricing', 'pricing'],
  ])('does not stem %s back to its verb', (word, expected) => {
    expect(stemWord(word)).toBe(expected)
  })
})

describe('looksPresent', () => {
  const review = 'The battery life is great but the screen is dim.'

  it('accepts a word that is there', () => {
    expect(looksPresent(review, 'screen')).toBe(true)
  })

  it('accepts it whatever the case', () => {
    expect(looksPresent(review, 'SCREEN')).toBe(true)
  })

  it('rejects a word that is not', () => {
    expect(looksPresent(review, 'keyboard')).toBe(false)
  })

  it('rejects a synonym, because the check is on words not meaning', () => {
    expect(looksPresent(review, 'display')).toBe(false)
  })

  it('accepts an inflected aspect against its singular in the review', () => {
    expect(looksPresent(review, 'batteries')).toBe(true)
  })

  it('accepts a possessive in the review', () => {
    expect(looksPresent("the screen's brightness is fine", 'screen')).toBe(true)
  })

  it('matches whole words, not substrings', () => {
    // "OS" lives inside "most", and a substring test would accept it.
    expect(looksPresent('most of it is fine', 'OS')).toBe(false)
  })

  it('requires a phrase to be contiguous', () => {
    expect(looksPresent('a hard case and a test drive', 'hard drive')).toBe(false)
    expect(looksPresent('it has a hard drive', 'hard drive')).toBe(true)
  })

  it('accepts a phrase whose words are each inflected', () => {
    expect(looksPresent('it has two hard drives', 'hard drive')).toBe(true)
  })

  it('accepts an aspect with no words at all, leaving it to the other checks', () => {
    // `""` and `"..."` stem to nothing; rejecting here would shadow the
    // schema's own blank-rejecting validator with a confusing message.
    expect(looksPresent(review, '')).toBe(true)
    expect(looksPresent(review, '...')).toBe(true)
  })

  it('gates presence, not aspecthood: a stopword that is there is accepted', () => {
    expect(looksPresent(review, 'the')).toBe(true)
  })
})
