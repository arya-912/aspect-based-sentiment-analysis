/**
 * Theme resolution, shared by the toggle and the pre-paint script.
 *
 * Three user-facing choices (`light`, `dark`, `system`) collapse to two
 * resolved values, and `<html data-theme>` always carries one of the two --
 * which is what lets `index.css` express dark mode as a single attribute
 * selector instead of an attribute selector plus a media query.
 */

export type ThemeChoice = 'light' | 'dark' | 'system'
export type ResolvedTheme = 'light' | 'dark'

/** Must match the key the inline script in `index.html` reads. */
const STORAGE_KEY = 'absa-theme'

const DARK_QUERY = '(prefers-color-scheme: dark)'

/**
 * Every storage access is guarded: `localStorage` is not merely empty but
 * *throws* in some privacy modes, and a theme preference is never worth
 * taking the page down for.
 */
function readStored(): ThemeChoice {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    // Unavailable storage is indistinguishable from "no preference saved".
  }
  return 'system'
}

function prefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia(DARK_QUERY).matches
}

/** The choice the user last made, defaulting to following the OS. */
export function getThemeChoice(): ThemeChoice {
  return readStored()
}

/** What that choice means right now. */
export function resolveTheme(choice: ThemeChoice): ResolvedTheme {
  if (choice === 'system') return prefersDark() ? 'dark' : 'light'
  return choice
}

/** Persist a choice and stamp the resolved value onto `<html>`. */
export function applyTheme(choice: ThemeChoice): ResolvedTheme {
  const resolved = resolveTheme(choice)
  document.documentElement.dataset.theme = resolved

  try {
    if (choice === 'system') localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, choice)
  } catch {
    // The theme still applies for this session; only persistence is lost.
  }
  return resolved
}

/**
 * Watch the OS setting, for a user sitting on `system` when their machine
 * flips at sunset. Returns an unsubscribe function.
 */
export function watchSystemTheme(onChange: (resolved: ResolvedTheme) => void): () => void {
  const query = window.matchMedia(DARK_QUERY)
  const handler = (event: MediaQueryListEvent) => onChange(event.matches ? 'dark' : 'light')
  query.addEventListener('change', handler)
  return () => query.removeEventListener('change', handler)
}
