/**
 * Test environment setup, run before every file (`setupFiles` in
 * `vite.config.ts`).
 */

import '@testing-library/jest-dom/vitest'

import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

/**
 * jsdom implements no `matchMedia` at all, and `lib/theme.ts` calls it on
 * every resolve. A stub that reports "light" by default keeps the default
 * deterministic; `setPrefersDark` flips it for the tests that care.
 */
let prefersDark = false
const listeners = new Set<(event: MediaQueryListEvent) => void>()

export function setPrefersDark(value: boolean): void {
  prefersDark = value
}

/** Fire a system theme change at whatever `watchSystemTheme` registered. */
export function emitSystemThemeChange(dark: boolean): void {
  prefersDark = dark
  for (const listener of listeners) {
    listener({ matches: dark } as MediaQueryListEvent)
  }
}

vi.stubGlobal(
  'matchMedia',
  (query: string): MediaQueryList =>
    ({
      matches: query.includes('dark') ? prefersDark : false,
      media: query,
      onchange: null,
      addEventListener: (_: string, listener: (event: MediaQueryListEvent) => void) => {
        listeners.add(listener)
      },
      removeEventListener: (_: string, listener: (event: MediaQueryListEvent) => void) => {
        listeners.delete(listener)
      },
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList,
)

afterEach(() => {
  cleanup()
  localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
  prefersDark = false
  listeners.clear()
  vi.restoreAllMocks()
})
