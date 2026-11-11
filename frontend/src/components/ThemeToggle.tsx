import { useEffect, useState } from 'react'

import { applyTheme, getThemeChoice, watchSystemTheme, type ThemeChoice } from '../lib/theme'
import { MoonIcon, SunIcon, SystemIcon } from './icons'

const OPTIONS: ReadonlyArray<{
  choice: ThemeChoice
  label: string
  Icon: typeof SunIcon
}> = [
  { choice: 'system', label: 'Match system', Icon: SystemIcon },
  { choice: 'light', label: 'Light', Icon: SunIcon },
  { choice: 'dark', label: 'Dark', Icon: MoonIcon },
]

/**
 * Light / dark / follow-the-system, as a segmented control.
 *
 * `system` is a real third state rather than an implied default, so a user on
 * a machine that flips at sunset keeps following it; `watchSystemTheme` is what
 * makes that live rather than only-at-load.
 */
export default function ThemeToggle() {
  const [choice, setChoice] = useState<ThemeChoice>(getThemeChoice)

  useEffect(() => {
    // Only meaningful while following the system; a pinned choice ignores the OS.
    if (choice !== 'system') return
    return watchSystemTheme(() => applyTheme('system'))
  }, [choice])

  function select(next: ThemeChoice) {
    setChoice(next)
    applyTheme(next)
  }

  return (
    <div
      className="inline-flex items-center gap-0.5 rounded-lg border border-line bg-plane p-0.5"
      role="group"
      aria-label="Color theme"
    >
      {OPTIONS.map(({ choice: option, label, Icon }) => {
        const active = choice === option
        return (
          <button
            key={option}
            type="button"
            onClick={() => select(option)}
            aria-pressed={active}
            title={label}
            className={`rounded-md p-1.5 transition-colors ${
              active
                ? 'bg-surface text-ink shadow-sm'
                : 'text-muted hover:bg-surface/60 hover:text-ink-2'
            }`}
          >
            <Icon />
            {/* The icons are decorative, so the control needs a real name here. */}
            <span className="sr-only">{label}</span>
          </button>
        )
      })}
    </div>
  )
}
