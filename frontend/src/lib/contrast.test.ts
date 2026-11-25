/**
 * Contrast, computed from the tokens rather than eyeballed.
 *
 * Four of these pairs were below the floor before this suite existed, and
 * none of them was obvious by eye: `muted` carries every caption in the app
 * at 3.41:1 on the light plane, and white on the dark-mode accent measured
 * 3.64:1 because that accent is deliberately the *lighter* blue. A palette is
 * checkable arithmetic, so it is checked.
 *
 * Reads `index.css` directly so the assertion is about the tokens that ship,
 * not a copy of them that could drift.
 */

import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

const CSS = readFileSync(resolve(process.cwd(), 'src/index.css'), 'utf8')

/** The `--color-*` hex values declared inside one CSS block. */
function tokensIn(blockStart: string): Record<string, string> {
  const open = CSS.indexOf(blockStart)
  if (open < 0) throw new Error(`no ${blockStart} block in index.css`)
  const close = CSS.indexOf('\n}', open)
  const body = CSS.slice(open, close)
  const found: Record<string, string> = {}
  for (const [, name, hex] of body.matchAll(/--color-([a-z0-9-]+):\s*(#[0-9a-f]{6})/g)) {
    found[name] = hex
  }
  return found
}

const base = tokensIn('@theme {')
const light = { ...base }
const dark = { ...base, ...tokensIn(":root[data-theme='dark'] {") }

function channel(value: number): number {
  const c = value / 255
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

function luminance(hex: string): number {
  const h = hex.replace('#', '')
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16))
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
}

/** WCAG 2.x contrast ratio. */
export function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

const THEMES = [
  ['light', light],
  ['dark', dark],
] as const

/** Tokens used for body-sized text. */
const TEXT = ['ink', 'ink-2', 'muted', 'accent', 'critical'] as const
/** Tokens used only for icons and marks, which take the 3:1 graphics floor. */
const GRAPHIC = ['warning', 'good'] as const

describe('the contrast formula itself', () => {
  it('matches the known endpoints', () => {
    expect(contrast('#ffffff', '#000000')).toBeCloseTo(21, 1)
    expect(contrast('#ffffff', '#ffffff')).toBeCloseTo(1, 5)
  })

  it('is symmetric', () => {
    expect(contrast('#2a78d6', '#f9f9f7')).toBeCloseTo(contrast('#f9f9f7', '#2a78d6'), 10)
  })
})

describe.each(THEMES)('%s theme', (_name, theme) => {
  it.each(TEXT.flatMap((fg) => ['plane', 'surface'].map((bg) => [fg, bg] as const)))(
    '%s text on the %s surface clears 4.5:1',
    (fg, bg) => {
      expect(contrast(theme[fg], theme[bg])).toBeGreaterThanOrEqual(4.5)
    },
  )

  it.each(GRAPHIC)('%s clears the 3:1 graphics floor', (fg) => {
    // These only ever colour an icon or a glyph, and always alongside words.
    expect(contrast(theme[fg], theme.plane)).toBeGreaterThanOrEqual(3.0)
  })

  it.each(['accent', 'accent-hover'] as const)(
    'the on-accent ink clears 4.5:1 on %s',
    (bg) => {
      expect(contrast(theme['on-accent'], theme[bg])).toBeGreaterThanOrEqual(4.5)
    },
  )
})

describe('the validated sentiment palette is left alone', () => {
  /**
   * These three are not free to move. The pair measures CVD delta E 7.2 in
   * light and 8.6 in dark, inside the band that is legal only with a
   * secondary encoding -- which is the mandatory text label. Changing a value
   * invalidates that measurement, so it has to be a deliberate act with a new
   * measurement, not a side effect of a contrast fix.
   */
  it.each([
    ['light', light, { neg: '#e34948', neu: '#898781', pos: '#008300' }],
    ['dark', dark, { neg: '#e66767', neu: '#898781', pos: '#008300' }],
  ])('%s poles and midpoint are unchanged', (_label, theme, expected) => {
    expect({ neg: theme.neg, neu: theme.neu, pos: theme.pos }).toEqual(expected)
  })

  it('keeps the neutral midpoint distinct from the muted text token', () => {
    // They were the same hex. The text one had to darken for contrast; the
    // palette one could not follow it.
    expect(light.muted).not.toBe(light.neu)
  })
})

describe('the sequential ramp survives Tailwind pruning', () => {
  it('is declared outside @theme, in a plain :root block', () => {
    // It is only ever read through an inline var(), and Tailwind drops
    // @theme variables no utility references.
    const themeBlock = CSS.slice(CSS.indexOf('@theme {'), CSS.indexOf('\n}', CSS.indexOf('@theme {')))
    expect(themeBlock).not.toContain('--color-seq-1')
    expect(CSS).toContain('--color-seq-1:')
  })

  it('ships six steps and an ink for each, in both themes', () => {
    for (const step of [1, 2, 3, 4, 5, 6]) {
      expect(CSS).toContain(`--color-seq-${step}:`)
      expect(CSS).toContain(`--color-seq-${step}-ink:`)
    }
  })
})
