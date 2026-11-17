/**
 * `lib/theme.ts` -- three choices collapsing to two resolved values.
 *
 * The case worth testing is the hostile one: `localStorage` does not merely
 * come back empty in some privacy modes, it *throws*. A theme preference is
 * never worth taking the page down for, so every access is guarded, and these
 * tests hold that guarantee.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { applyTheme, getThemeChoice, resolveTheme, watchSystemTheme } from './theme'
import { emitSystemThemeChange, setPrefersDark } from '../test/setup'

const STORAGE_KEY = 'absa-theme'

/** Make every localStorage method throw, as a locked-down browser does. */
function breakStorage(): void {
  const boom = () => {
    throw new DOMException('The operation is insecure.', 'SecurityError')
  }
  vi.spyOn(Storage.prototype, 'getItem').mockImplementation(boom)
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(boom)
  vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(boom)
}

describe('getThemeChoice', () => {
  it('defaults to following the OS when nothing is stored', () => {
    expect(getThemeChoice()).toBe('system')
  })

  it.each(['light', 'dark'] as const)('reads a stored %s', (choice) => {
    localStorage.setItem(STORAGE_KEY, choice)
    expect(getThemeChoice()).toBe(choice)
  })

  it('treats a junk stored value as no preference', () => {
    localStorage.setItem(STORAGE_KEY, 'chartreuse')
    expect(getThemeChoice()).toBe('system')
  })

  it('never stores "system" as a value to read back', () => {
    // `system` is the absence of a stored choice, not a third stored string.
    applyTheme('system')
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('returns "system" instead of throwing when storage is blocked', () => {
    breakStorage()
    expect(getThemeChoice()).toBe('system')
  })
})

describe('resolveTheme', () => {
  it.each(['light', 'dark'] as const)('passes an explicit %s through', (choice) => {
    expect(resolveTheme(choice)).toBe(choice)
  })

  it('resolves system against the OS setting', () => {
    setPrefersDark(false)
    expect(resolveTheme('system')).toBe('light')
    setPrefersDark(true)
    expect(resolveTheme('system')).toBe('dark')
  })
})

describe('applyTheme', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute('data-theme')
  })

  it.each(['light', 'dark'] as const)('stamps data-theme=%s on <html>', (choice) => {
    expect(applyTheme(choice)).toBe(choice)
    expect(document.documentElement.dataset.theme).toBe(choice)
  })

  it('stamps a resolved value for system, never the word "system"', () => {
    // index.css expresses dark mode as one attribute selector, so the
    // attribute has to carry a resolved value or the stylesheet misses.
    setPrefersDark(true)
    expect(applyTheme('system')).toBe('dark')
    expect(document.documentElement.dataset.theme).toBe('dark')
  })

  it('persists an explicit choice', () => {
    applyTheme('dark')
    expect(localStorage.getItem(STORAGE_KEY)).toBe('dark')
  })

  it('clears the stored choice when going back to system', () => {
    applyTheme('dark')
    applyTheme('system')
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull()
  })

  it('still applies the theme when storage is blocked', () => {
    breakStorage()
    expect(() => applyTheme('dark')).not.toThrow()
    // Persistence is lost; the current session is not.
    expect(document.documentElement.dataset.theme).toBe('dark')
  })
})

describe('watchSystemTheme', () => {
  it('reports a flip to dark', () => {
    const seen: string[] = []
    watchSystemTheme((resolved) => seen.push(resolved))
    emitSystemThemeChange(true)
    expect(seen).toEqual(['dark'])
  })

  it('reports a flip back to light', () => {
    const seen: string[] = []
    watchSystemTheme((resolved) => seen.push(resolved))
    emitSystemThemeChange(true)
    emitSystemThemeChange(false)
    expect(seen).toEqual(['dark', 'light'])
  })

  it('stops reporting once unsubscribed', () => {
    const seen: string[] = []
    const stop = watchSystemTheme((resolved) => seen.push(resolved))
    stop()
    emitSystemThemeChange(true)
    expect(seen).toEqual([])
  })
})
