/**
 * The handful of icons this UI needs, inline.
 *
 * Inline rather than an icon package: six glyphs do not justify a dependency,
 * and these inherit `currentColor` so they theme for free. Each is decorative
 * -- `aria-hidden` -- because every one of them sits beside its own text label.
 */

interface IconProps {
  className?: string
}

const BASE = 'h-4 w-4 shrink-0'

export function WarningIcon({ className = '' }: IconProps) {
  return (
    <svg className={`${BASE} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.515 2.625H3.72c-1.345 0-2.188-1.458-1.515-2.625L8.485 2.495ZM10 6a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 10 6Zm0 7a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

export function OfflineIcon({ className = '' }: IconProps) {
  return (
    <svg className={`${BASE} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM8.28 7.22a.75.75 0 0 0-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 1 0 1.06 1.06L10 11.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L11.06 10l1.72-1.72a.75.75 0 0 0-1.06-1.06L10 8.94 8.28 7.22Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

export function SunIcon({ className = '' }: IconProps) {
  return (
    <svg className={`${BASE} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M10 2a.75.75 0 0 1 .75.75v1a.75.75 0 0 1-1.5 0v-1A.75.75 0 0 1 10 2ZM10 15a.75.75 0 0 1 .75.75v1a.75.75 0 0 1-1.5 0v-1A.75.75 0 0 1 10 15ZM10 6a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM2.75 9.25a.75.75 0 0 0 0 1.5h1a.75.75 0 0 0 0-1.5h-1ZM15.25 9.25a.75.75 0 0 0 0 1.5h1a.75.75 0 0 0 0-1.5h-1ZM4.399 4.399a.75.75 0 0 1 1.06 0l.708.708a.75.75 0 0 1-1.061 1.06l-.707-.707a.75.75 0 0 1 0-1.06ZM13.833 13.833a.75.75 0 0 1 1.06 0l.708.707a.75.75 0 1 1-1.061 1.061l-.707-.707a.75.75 0 0 1 0-1.061ZM15.601 4.399a.75.75 0 0 1 0 1.06l-.707.708a.75.75 0 0 1-1.061-1.061l.707-.707a.75.75 0 0 1 1.061 0ZM6.167 13.833a.75.75 0 0 1 0 1.06l-.707.708a.75.75 0 0 1-1.061-1.061l.707-.707a.75.75 0 0 1 1.061 0Z" />
    </svg>
  )
}

export function MoonIcon({ className = '' }: IconProps) {
  return (
    <svg className={`${BASE} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M7.455 2.004a.75.75 0 0 1 .26.77 7 7 0 0 0 9.958 7.967.75.75 0 0 1 1.067.853A8.5 8.5 0 1 1 6.647 1.921a.75.75 0 0 1 .808.083Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

export function SystemIcon({ className = '' }: IconProps) {
  return (
    <svg className={`${BASE} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M2 4.25A2.25 2.25 0 0 1 4.25 2h11.5A2.25 2.25 0 0 1 18 4.25v8.5A2.25 2.25 0 0 1 15.75 15h-3.105l.401 1.607a.75.75 0 0 1-.728.933H7.682a.75.75 0 0 1-.728-.933L7.355 15H4.25A2.25 2.25 0 0 1 2 12.75v-8.5Zm2.25-.75a.75.75 0 0 0-.75.75v8.5c0 .414.336.75.75.75h11.5a.75.75 0 0 0 .75-.75v-8.5a.75.75 0 0 0-.75-.75H4.25Z"
        clipRule="evenodd"
      />
    </svg>
  )
}

export function ChevronIcon({ className = '' }: IconProps) {
  return (
    <svg className={`${BASE} ${className}`} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
        clipRule="evenodd"
      />
    </svg>
  )
}
