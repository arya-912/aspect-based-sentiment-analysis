interface StarRatingProps {
  /** Mean rating, 0-5. */
  rating: number
  /** How many reviews it averages, shown so one review cannot pose as five. */
  count: number
}

/**
 * The reviewers' own star rating.
 *
 * Shown beside the aspect breakdown, not instead of it, because the
 * comparison is the point: a 4.5-star restaurant whose `service` aspect is
 * 70% negative is exactly what a single overall score hides, and putting the
 * two side by side makes that visible rather than argued.
 *
 * The number is always printed. The stars are decoration over the top of it
 * -- a glyph count is hard to read precisely and impossible to read at all
 * for a screen reader, so they carry `aria-hidden` and the text carries the
 * value.
 */
export default function StarRating({ rating, count }: StarRatingProps) {
  const rounded = Math.round(rating * 2) / 2

  return (
    <span className="inline-flex items-baseline gap-1.5 text-sm">
      <span aria-hidden="true" className="text-warning">
        {[1, 2, 3, 4, 5].map((position) => {
          const fill = rounded >= position ? '★' : rounded >= position - 0.5 ? '⯨' : '☆'
          return <span key={position}>{fill}</span>
        })}
      </span>
      <span className="tabular font-medium text-ink">{rating.toFixed(1)}</span>
      <span className="text-xs text-muted">
        ({count} {count === 1 ? 'rating' : 'ratings'})
      </span>
    </span>
  )
}
