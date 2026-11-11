import type { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  className?: string
  /** For the cases where the card *is* the alert or status region. */
  role?: string
}

/**
 * The one raised surface in the app.
 *
 * A hairline ring rather than a drop shadow: a shadow has to be re-tuned for
 * the dark surface to not look like grime, and a 1px border reads the same in
 * both modes.
 */
export default function Card({ children, className = '', role }: CardProps) {
  return (
    <div role={role} className={`rounded-xl border border-line bg-surface ${className}`}>
      {children}
    </div>
  )
}
