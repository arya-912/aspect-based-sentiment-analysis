interface BarRowProps {
  /** Left-hand label, e.g. a token or a sentiment name. */
  label: string
  /** Pre-formatted right-hand value, e.g. `72.3%`. */
  value: string
  /** How full the bar is, 0-1. Clamped, so a stray value cannot overflow. */
  fraction: number
  /** Literal Tailwind background class for the filled part. */
  barClassName?: string
}

/**
 * One labelled horizontal bar.
 *
 * The fill width is an inline style rather than a Tailwind class: Tailwind
 * only emits classes it finds written out in the source, so a computed
 * `w-[${percent}%]` would compile to nothing and render empty.
 */
export default function BarRow({
  label,
  value,
  fraction,
  barClassName = 'bg-gray-400',
}: BarRowProps) {
  const percent = Math.max(0, Math.min(1, fraction)) * 100

  return (
    <div className="flex items-center gap-3">
      <span
        className="w-28 shrink-0 truncate font-mono text-xs text-gray-700"
        title={label}
      >
        {label}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-100">
        {/* Decorative: the number to its right carries the same information,
            so a screen reader is not told it twice. */}
        <div
          className={`h-full rounded-full ${barClassName}`}
          style={{ width: `${percent}%` }}
          aria-hidden="true"
        />
      </div>
      <span className="w-14 shrink-0 text-right font-mono text-xs text-gray-600">
        {value}
      </span>
    </div>
  )
}
