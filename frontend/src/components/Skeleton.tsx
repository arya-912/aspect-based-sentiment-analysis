interface SkeletonProps {
  /** Tailwind width/height classes. Literal, because Tailwind only emits what it sees. */
  className?: string
}

/**
 * A placeholder the size of the thing that is coming.
 *
 * Sized by the caller rather than generic, because the point is to reserve
 * the real layout: a placeholder of the wrong height makes the page jump when
 * content lands, which is the problem it was meant to prevent.
 *
 * Carries no ARIA of its own. The region that owns it announces the wait once
 * -- several dozen skeleton blocks each announcing themselves is noise, and a
 * screen reader user needs one "loading", not forty.
 */
export default function Skeleton({ className = 'h-4 w-full' }: SkeletonProps) {
  return <div aria-hidden="true" className={`skeleton ${className}`} />
}
