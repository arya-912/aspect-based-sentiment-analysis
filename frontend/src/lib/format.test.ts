/**
 * `lib/format.ts` -- the em dash cases are the point.
 *
 * Three components used to carry their own percent helper with different
 * handling of the absent case, so one of them rendered "NaN%" on an artifact
 * that shipped without a metrics report.
 */

import { describe, expect, it } from 'vitest'

import { asCount, asPercent } from './format'

describe('asPercent', () => {
  it('renders a ratio to one decimal by default', () => {
    expect(asPercent(0.7884)).toBe('78.8%')
  })

  it('takes a digit count', () => {
    expect(asPercent(0.7884, 2)).toBe('78.84%')
    expect(asPercent(0.7884, 0)).toBe('79%')
  })

  it.each([0, 1])('renders the endpoint %s', (value) => {
    expect(asPercent(value)).toBe(value === 0 ? '0.0%' : '100.0%')
  })

  it.each([
    ['null', null],
    ['undefined', undefined],
    ['NaN', Number.NaN],
  ])('renders an em dash for %s rather than a number-shaped lie', (_label, value) => {
    expect(asPercent(value)).toBe('—')
  })

  it('rounds half away from zero as toFixed does', () => {
    expect(asPercent(0.5)).toBe('50.0%')
    expect(asPercent(0.12345, 2)).toBe('12.35%')
  })
})

describe('asCount', () => {
  it('groups thousands', () => {
    expect(asCount(1758)).toBe('1,758')
    expect(asCount(112435971)).toBe('112,435,971')
  })

  it('leaves small numbers alone', () => {
    expect(asCount(0)).toBe('0')
    expect(asCount(638)).toBe('638')
  })

  it.each([
    ['null', null],
    ['undefined', undefined],
    ['NaN', Number.NaN],
  ])('renders an em dash for %s', (_label, value) => {
    expect(asCount(value)).toBe('—')
  })
})
