import type { ReactNode } from 'react'

interface EmptyStateProps {
  /** What the person is looking at. Sentence case, no trailing colon. */
  title: string
  /** Why it is empty and what to do about it. */
  children: ReactNode
  /** An optional action, when there is one obvious next step. */
  action?: ReactNode
  /** `warning` for a limit that was hit, `critical` for a failure. */
  tone?: 'neutral' | 'warning' | 'critical'
  icon?: ReactNode
}

/** Border and icon colour per tone. Literal class names for Tailwind. */
const TONES = {
  neutral: { ring: 'border-line', icon: 'text-muted' },
  warning: { ring: 'border-warning/40', icon: 'text-warning' },
  critical: { ring: 'border-critical/40', icon: 'text-critical' },
} as const

/**
 * The one treatment for "there is nothing here".
 *
 * Every empty, failed and not-yet state in the app routes through this, so
 * they read as the same product rather than as four developers' habits. An
 * empty screen is an instruction, not a mood: the title says what is being
 * looked at and the body says what to do, in the interface's voice.
 *
 * Tone never travels alone -- the warning and critical variants are a border
 * plus an icon plus the words, so the state survives a colourblind reader and
 * a greyscale projector, which is the one that matters in a presentation.
 */
export default function EmptyState({
  title,
  children,
  action,
  tone = 'neutral',
  icon,
}: EmptyStateProps) {
  const { ring, icon: iconTone } = TONES[tone]

  return (
    <div className={`rounded-xl border border-dashed ${ring} bg-plane px-5 py-6`}>
      <div className="flex gap-3">
        {icon && <span className={`mt-0.5 shrink-0 ${iconTone}`}>{icon}</span>}
        <div className="min-w-0">
          <p className="font-medium text-ink">{title}</p>
          <div className="mt-1 max-w-prose text-sm text-ink-2">{children}</div>
          {action && <div className="mt-3">{action}</div>}
        </div>
      </div>
    </div>
  )
}
