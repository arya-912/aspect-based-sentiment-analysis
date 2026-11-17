/**
 * `lib/weights.ts` -- the "best for me" ranking.
 *
 * The case that matters is the null: a product discussed on none of the
 * weighted aspects must not be scored 0, or it ranks below a product people
 * actively disliked. "Unknown" and "bad" are different answers.
 */

import { describe, expect, it } from 'vitest'

import { WEIGHT_STEPS, weightedScore } from './weights'

const SCORES = [
  { aspect: 'battery life', score: 0.8 },
  { aspect: 'screen', score: 0.4 },
  { aspect: 'price', score: 0.6 },
]

describe('weightedScore', () => {
  it('is the plain mean when every weight is equal', () => {
    const result = weightedScore(SCORES, { 'battery life': 1, screen: 1 })
    expect(result?.value).toBeCloseTo(0.6, 5)
    expect(result?.matched).toBe(2)
  })

  it('weights an aspect by how much it was asked for', () => {
    // battery x3, screen x1 -> (0.8*3 + 0.4*1) / 4 = 0.7
    const result = weightedScore(SCORES, { 'battery life': 3, screen: 1 })
    expect(result?.value).toBeCloseTo(0.7, 5)
  })

  it('ignores an aspect weighted zero', () => {
    const result = weightedScore(SCORES, { 'battery life': 2, screen: 0 })
    expect(result?.value).toBeCloseTo(0.8, 5)
    expect(result?.matched).toBe(1)
  })

  it('ignores an aspect with no weight at all', () => {
    const result = weightedScore(SCORES, { 'battery life': 1 })
    expect(result?.value).toBeCloseTo(0.8, 5)
  })

  it('returns null when no weights are set', () => {
    expect(weightedScore(SCORES, {})).toBeNull()
  })

  it('returns null when the product is discussed on none of the weighted aspects', () => {
    // Not zero: an unreviewed product must not rank below a disliked one.
    expect(weightedScore(SCORES, { atmosphere: 3 })).toBeNull()
  })

  it('returns null for a product with no scores at all', () => {
    expect(weightedScore([], { 'battery life': 3 })).toBeNull()
  })

  it('scores on what is known and reports how much that was', () => {
    // Two aspects weighted, one of them discussed: ranked on the one, and
    // `matched` lets the UI say so rather than implying full coverage.
    const result = weightedScore(SCORES, { 'battery life': 1, atmosphere: 1 })
    expect(result?.value).toBeCloseTo(0.8, 5)
    expect(result?.matched).toBe(1)
  })

  it('stays within 0 and 1 for any weighting', () => {
    const result = weightedScore(SCORES, { 'battery life': 3, screen: 3, price: 3 })
    expect(result?.value).toBeGreaterThanOrEqual(0)
    expect(result?.value).toBeLessThanOrEqual(1)
  })

  it('offers 0 as a real step, so an aspect can be dropped', () => {
    expect(WEIGHT_STEPS[0]).toBe(0)
    expect(WEIGHT_STEPS).toHaveLength(4)
  })
})
