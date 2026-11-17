/**
 * `lib/domains.ts` -- the UI's copy follows the served artifact.
 *
 * The failure this prevents: serving the restaurant model while offering
 * "Battery life is amazing but the screen is dim." as a one-click example,
 * which that model answers confidently and wrongly.
 */

import { describe, expect, it } from 'vitest'

import { domainFor } from './domains'
import { looksPresent } from './words'

describe('domainFor', () => {
  it.each([
    ['Laptop_Train_v2', 'Laptop reviews'],
    ['Laptops_Test_Gold', 'Laptop reviews'],
    ['Restaurants_Train_v2', 'Restaurant reviews'],
    ['Joint_Train', 'Laptop and restaurant reviews'],
    ['MAMS_Train', 'Restaurant reviews'],
  ])('maps %s to %s', (corpus, subject) => {
    expect(domainFor(corpus)?.subject).toBe(subject)
  })

  it('matches a substring of the stem, not an exact filename', () => {
    expect(domainFor('some_laptop_corpus_v9')?.subject).toBe('Laptop reviews')
  })

  it('ignores case', () => {
    expect(domainFor('LAPTOP_TRAIN_V2')?.subject).toBe('Laptop reviews')
  })

  it('checks joint before the single-domain names', () => {
    // A pooled file named for both domains must not be mislabelled as the
    // laptop-only model, which would hide half of what it serves.
    const domain = domainFor('Joint_Laptop_Restaurant')
    expect(domain?.subject).toBe('Laptop and restaurant reviews')
  })

  it.each([
    ['null', null],
    ['undefined', undefined],
    ['empty', ''],
    ['unrecognised', 'Twitter_Sentiment_2019'],
  ])('returns null for %s, so the UI falls back to generic copy', (_label, corpus) => {
    expect(domainFor(corpus)).toBeNull()
  })

  it('cites a different corpus for MAMS than for restaurants', () => {
    // MAMS inherits the restaurant examples -- it is built from restaurant
    // reviews -- but it is not the same corpus and must not claim to be.
    expect(domainFor('MAMS_Train')?.corpusName).toContain('MAMS')
    expect(domainFor('Restaurants_Train_v2')?.corpusName).not.toContain('MAMS')
  })
})

describe('every example is in regime', () => {
  const corpora = ['Laptop_Train_v2', 'Restaurants_Train_v2', 'Joint_Train', 'MAMS_Train']

  it.each(corpora)('%s offers at least one example', (corpus) => {
    expect(domainFor(corpus)?.examples.length).toBeGreaterThan(0)
  })

  /** The one example per domain that exists to demonstrate the 422. */
  const isRefusalExample = (label: string) => label.toLowerCase().includes('absent')

  it.each(corpora)('%s: every aspect is a verbatim span of its own sentence', (corpus) => {
    // The regime the model was fitted in: every SemEval aspect term is a
    // literal slice of its sentence. An example that breaks this teaches the
    // user the wrong thing about the tool on their first click. The refusal
    // example is the deliberate exception and is checked below.
    for (const example of domainFor(corpus)?.examples ?? []) {
      if (isRefusalExample(example.label)) continue
      expect(example.text).toContain(example.aspect)
    }
  })

  it.each(corpora)('%s: every ordinary example passes the presence rule', (corpus) => {
    for (const example of domainFor(corpus)?.examples ?? []) {
      if (isRefusalExample(example.label)) continue
      expect(looksPresent(example.text, example.aspect)).toBe(true)
    }
  })

  it.each(corpora)('%s: ships exactly one refusal example, and it is refused', (corpus) => {
    // This one is supposed to be absent -- it is how the UI shows that the
    // model cannot abstain. So it must fail the presence rule, or the example
    // silently stops demonstrating anything.
    const examples = domainFor(corpus)?.examples ?? []
    const refusals = examples.filter((e) => isRefusalExample(e.label))
    expect(refusals).toHaveLength(1)
    expect(refusals[0].text).not.toContain(refusals[0].aspect)
    expect(looksPresent(refusals[0].text, refusals[0].aspect)).toBe(false)
  })
})
