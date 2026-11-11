interface BarRowProps {
  /** Left-hand label, e.g. a token or a sentiment name. */
  label: string
  /** Pre-formatted right-hand value, e.g. `72.3%`. */
  value: string
  /** How full the bar is, 0-1. Clamped, so a stray value cannot overflow. */
  fraction: number
  /** Literal Tailwind background class for the filled part. */
  barClassName?: string
  /** Width of the label column. Dense lists want it narrower. */
  labelClassName?: string
  /** Fuller text for the hover tooltip, when the label alone is terse. */
  title?: string
}

/**
 * One labelled horizontal bar.
 *
 * The fill width is an inline style rather than a Tailwind class: Tailwind
 * only emits classes it finds written out in the source, so a computed
 * `w-[${percent}%]` would compile to nothing and render empty.
 *
 * Mark geometry follows the project's data-viz reference: a thin 6px bar, the
 * fill anchored flush to the baseline at the left and rounded only at the data
 * end, and the value direct-labelled beside it rather than hidden in a
 * tooltip.
 */
export default function BarRow({
  label,
  value,
  fraction,
  barClassName = 'bg-muted',
  labelClassName = 'w-28',
  title,
}: BarRowProps) {
  const percent = Math.max(0, Math.min(1, fraction)) * 100

  return (
    <div className="flex items-center gap-3" title={title ?? `${label}: ${value}`}>
      <span className={`${labelClassName} shrink-0 truncate text-xs text-ink-2`}>{label}</span>

      <div className="h-1.5 flex-1 overflow-hidden rounded-[4px] bg-grid">
        {/*
          Decorative: the number to its right carries the same information, so a
          screen reader is not told it twice. A floor of 2px keeps a small but
          nonzero weight visible instead of vanishing into the track.
        */}
        <div
          className={`h-full rounded-r-[4px] ${barClassName}`}
          style={{ width: `${percent}%`, minWidth: percent > 0 ? '2px' : '0' }}
          aria-hidden="true"
        />
      </div>

      <span className="tabular w-14 shrink-0 text-right text-xs text-ink-2">{value}</span>
    </div>
  )
}
